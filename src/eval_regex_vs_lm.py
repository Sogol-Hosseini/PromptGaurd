import os, json, torch, pandas as pd, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from rules_regex import RegexRules
from ensemble_guard import EnsembleGuard, MODEL_DIR

CSV_PATH = "hf://datasets/qualifire/prompt-injections-benchmark/test.csv"

# Load data
df = pd.read_csv(CSV_PATH)
df["y"] = df["label"].map({"benign":0, "jailbreak":1})

# --- Regex only
rr = RegexRules("src/patterns.regex.yaml")
y_rx = df["text"].apply(lambda t: 1 if rr.score(t)["level"] in ("warn","block") else 0).values

# --- LM only
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
tok.model_max_length = 256   # keep consistent with training
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

thr = 0.1
tp = os.path.join(MODEL_DIR, "threshold.txt")
if os.path.exists(tp):
    thr = float(open(tp).read().strip())

pipe = TextClassificationPipeline(
    model=mdl,
    tokenizer=tok,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
)

def pmal(t):
    scores = pipe(t, truncation=True, max_length=256)[0]
    return next(s["score"] for s in scores if s["label"].endswith("1"))


probs = df["text"].apply(pmal).values
y_lm = (probs >= thr).astype(int)

# --- Ensemble (regex + LM)
ens = EnsembleGuard(model_dir=MODEL_DIR)
def ens_decide(t): return 1 if ens.decide(t)["decision"] == "block" else 0
y_ens = df["text"].apply(ens_decide).values

y_true = df["y"].values

def show(name, y_hat):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_hat, digits=3))
    print(confusion_matrix(y_true, y_hat))

show("Regex only", y_rx)
show("LM only", y_lm)
show("Ensemble (Regex + LM)", y_ens)
