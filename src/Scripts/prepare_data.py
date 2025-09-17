import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# 1. Load dataset
df = pd.read_csv("hf://datasets/qualifire/prompt-injections-benchmark/test.csv")

print("Original labels:", df['label'].unique())

# 2. Normalize labels (benign=0, jailbreak=1)
label_map = {"benign": 0, "jailbreak": 1}
df["label"] = df["label"].map(label_map)

print("\nSample after mapping:")
print(df.head())

# 3. Train/val/test split (60/20/20)
train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

print("\nSplit sizes:")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 4. Convert to Hugging Face datasets
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_ds   = Dataset.from_pandas(val_df, preserve_index=False)
test_ds  = Dataset.from_pandas(test_df, preserve_index=False)

# 5. Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

# 6. Wrap into DatasetDict
ds = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

print("\nDataset ready with columns:", ds["train"].column_names)
print(ds)
