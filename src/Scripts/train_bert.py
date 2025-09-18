# src/train_bert.py
import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import torch.nn as nn

# ------------------------------
# 0) Load datasets
# ------------------------------
# Train: deepset/prompt-injections (PARQUET)  — only train split
# Test : keep your previous qualifire/prompt-injections-benchmark/test.csv
# --- use deepset for both train and test ---
# SPLITS = {
#     "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
#     "test":  "data/test-00000-of-00001-701d16158af87368.parquet",
# }


import pandas as pd

splits = {'train': 'data/train-00000-of-00001-9564e8b05b4757ab.parquet', 'test': 'data/test-00000-of-00001-701d16158af87368.parquet'}
train_full_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["test"])

# train_full_df = ds["train"]
# test_df  = ds["test"]
 
# train_full_df = pd.read_parquet(TRAIN_PATH)
# test_df       = pd.read_parquet(TEST_PATH)   # <-- was read_csv on qualifire


# ---- helpers: make robust to different label/text shapes ----
def normalize_labels(series: pd.Series) -> pd.Series:
    """Map various label schemes to {0,1}."""
    if series.dtype.kind in "iu":  # already ints
        return series.astype(int)

    mapping = {
        "benign": 0, "clean": 0, "safe": 0, "non_jailbreak": 0, "not_jailbreak": 0,
        "jailbreak": 1, "prompt_injection": 1, "injection": 1, "malicious": 1,
        "attack": 1, "adversarial": 1
    }
    s = series.astype(str).str.lower().map(mapping)
    # If something didn't map (NaN), try bool-like strings
    s = s.fillna(series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int))
    return s.astype(int)

def pick_text_column(df: pd.DataFrame) -> str:
    """Find the text column name; fallback to common variants."""
    for c in ["text", "content", "prompt", "input", "message", "instruction"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a text column in: {list(df.columns)}")

# ---- load train (parquet) + test (csv) ----
# train_full_df = pd.read_parquet(TRAIN_PATH)
# test_df       = pd.read_parquet(TEST_PATH)

text_col_train = pick_text_column(train_full_df)
text_col_test  = pick_text_column(test_df)

train_full_df["label"] = normalize_labels(train_full_df["label"])
test_df["label"]       = normalize_labels(test_df["label"])

# Create validation from the TRAIN split (e.g., 80/20 stratified)
train_df, val_df = train_test_split(
    train_full_df[[text_col_train, "label"]],
    test_size=0.2, stratify=train_full_df["label"], random_state=42
)

# For downstream code, standardize the text column name to "text"
train_df = train_df.rename(columns={text_col_train: "text"})
val_df   = val_df.rename(columns={text_col_train: "text"})
test_df  = test_df.rename(columns={text_col_test: "text"})[["text", "label"]]

# ------------------------------
# 1) Tokenizer + tokenization
# ------------------------------
MODEL_NAME = "distilbert-base-uncased"  # change to "bert-base-uncased" if desired
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    # Static padding makes the simple collator work reliably
    return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val_df,   preserve_index=False).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test_df,  preserve_index=False).map(tokenize, batched=True)

# rename & keep only encodings + labels
for split in [train_ds, val_ds, test_ds]:
    split = split

train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")

KEEP = {"input_ids", "attention_mask", "labels"}
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in KEEP])
val_ds   = val_ds.remove_columns([c for c in val_ds.column_names   if c not in KEEP])
test_ds  = test_ds.remove_columns([c for c in test_ds.column_names if c not in KEEP])

train_ds.set_format(type="torch")
val_ds.set_format(type="torch")
test_ds.set_format(type="torch")

ds = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)

print("TRAIN COLS:", ds["train"].column_names)
print("SAMPLE[0] keys:", list(ds["train"][0].keys()))

# ------------------------------
# 2) Model
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ------------------------------
# 3) Metrics
# ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = (preds == labels).mean().item()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc}

# ------------------------------
# 4) Class weights + WeightedTrainer
# ------------------------------
# Compute class weights on *training* labels
pos = int(train_df["label"].sum())
neg = int(len(train_df) - pos)
class_weights = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ------------------------------
# 5) TrainingArguments
# ------------------------------
# ---- TrainingArguments (Minimal disk) ----
common_args = dict(
    output_dir="models/bert-pi-detector",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=False,  # <- must be False when save_strategy="no"
    metric_for_best_model="f1",
    logging_steps=50,
    seed=42,
    report_to="none",
    fp16=False,
    overwrite_output_dir=True,
    save_total_limit=1,
)

try:
    # newer transformers
    args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="no",     # no intermediate checkpoints
        **common_args
    )
except TypeError:
    # older transformers uses eval_strategy
    args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="no",
        **common_args
    )


# ------------------------------
# 6) Train
# ------------------------------
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=default_data_collator,  # stacks our already-padded tensors
    compute_metrics=compute_metrics,
)
trainer.train()

# ------------------------------
# 7) Threshold tuning + Test eval
# ------------------------------
pred_val = trainer.predict(ds["validation"])
probs_val = torch.softmax(torch.tensor(pred_val.predictions), dim=1).numpy()[:, 1]
labels_val = pred_val.label_ids

best_f1, best_t = 0.0, 0.5
for t in np.linspace(0.05, 0.95, 19):
    p, r, f1, _ = precision_recall_fscore_support(labels_val, probs_val >= t, average="binary", zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, float(t)
print(f"\nBest threshold on validation: {best_t:.2f} (F1={best_f1:.3f})")

pred_test = trainer.predict(ds["test"])
probs_test = torch.softmax(torch.tensor(pred_test.predictions), dim=1).numpy()[:, 1]
labels_test = pred_test.label_ids
yhat = (probs_test >= best_t).astype(int)

print("\nTest classification report:")
print(classification_report(labels_test, yhat, digits=3))
print("Confusion matrix:\n", confusion_matrix(labels_test, yhat))
try:
    print(f"ROC-AUC: {roc_auc_score(labels_test, probs_test):.3f}")
except Exception:
    pass

# ---- Point estimates on test_df ----
def compute_bin_metrics(labels, probs, thr):
    preds = (probs >= thr).astype(int)
    acc = (preds == labels).mean()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return acc, p, r, f1

acc0, p0, r0, f10 = compute_bin_metrics(labels_test, probs_test, best_t)
print(f"\nTest (point estimates): Acc={acc0:.3f}  P={p0:.3f}  R={r0:.3f}  F1={f10:.3f}")

# ---- Bootstrapped mean ± std on test_df ----
B = 150  # increase to 1000 for tighter std; 500 is faster on CPU
rng = np.random.default_rng(42)
n = len(labels_test)
accs = np.empty(B); ps = np.empty(B); rs = np.empty(B); f1s = np.empty(B)

for b in range(B):
    idx = rng.integers(0, n, n)  # sample with replacement
    accs[b], ps[b], rs[b], f1s[b] = compute_bin_metrics(labels_test[idx], probs_test[idx], best_t)

def fmt(mean, std): return f"{mean:.3f} ± {std:.3f}"
print("\nBootstrapped metrics on test_df (mean ± std):")
print("Accuracy :", fmt(accs.mean(), accs.std(ddof=1)))
print("Precision:", fmt(ps.mean(),   ps.std(ddof=1)))
print("Recall   :", fmt(rs.mean(),   rs.std(ddof=1)))
print("F1       :", fmt(f1s.mean(),  f1s.std(ddof=1)))

# ------------------------------
# 8) Save
# ------------------------------
save_dir = "models/bert-pi-detector/fine_tuned"  # <-- new version path
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
with open(os.path.join(save_dir, "threshold.txt"), "w") as f:
    f.write(str(best_t))
print(f"\nSaved model to: {save_dir}")
