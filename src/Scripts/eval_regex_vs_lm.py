import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# --- Try both possible regex classes
def load_regex_detector():
    try:
        from modules.rules_regex import RegexRules as _Cls
        return _Cls
    except Exception:
        pass
    try:
        from modules.rules_regex import RegexBasedDetector as _Cls
        return _Cls
    except Exception as e:
        raise ImportError(
            f"Could not import RegexRules or RegexBasedDetector from modules.rules_regex: {e}"
        )

RegexDetectorClass = load_regex_detector()

# Ensemble (your class)
from src.modules.ensemble_guard import EnsembleGuard, MODEL_DIR

CSV_PATH = "hf://datasets/qualifire/prompt-injections-benchmark/test.csv"
PATTERNS_PATH = "src/patterns.regex.yaml"

# ---------- Load data ----------
df = pd.read_csv(CSV_PATH)
if "label" not in df.columns or "text" not in df.columns:
    raise ValueError("CSV must have 'text' and 'label' columns.")

# Map labels to 0/1
if df["label"].dtype == object:
    df["y"] = df["label"].map({"benign": 0, "jailbreak": 1})
else:
    # already numeric (0/1)
    df["y"] = df["label"].astype(int)

# ---------- Regex only ----------
try:
    # Some implementations expect path in ctor; others not. Try both.
    try:
        rx = RegexDetectorClass(PATTERNS_PATH)
    except TypeError:
        rx = RegexDetectorClass()

    def rx_pred(text: str) -> int:
        out = rx.score(text)
        # Expect dict with 'level'; be defensive:
        level = None
        if isinstance(out, dict):
            level = out.get("level")
            if level is None and "risk" in out:
                # fallback via thresholds (warn >= 3 typical)
                level = "warn" if int(out["risk"]) >= 3 else "ok"
        if level is None:
            raise RuntimeError(f"regex score() returned unexpected payload: {out}")
        return 1 if level in {"warn", "block"} else 0

    y_rx = df["text"].apply(rx_pred).astype(int).to_numpy()
except Exception as e:
    raise RuntimeError(f"Regex evaluation failed: {e}")

# ---------- LM only ----------
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
tok.model_max_length = 256  # keep consistent with training
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

thr = 0.1
tp = os.path.join(MODEL_DIR, "threshold.txt")
if os.path.exists(tp):
    try:
        thr = float(open(tp).read().strip())
    except Exception:
        pass

pipe = TextClassificationPipeline(
    model=mdl,
    tokenizer=tok,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,                 # return all scores
    truncation=True
)

def prob_mal(text: str) -> float:
    """
    Return probability/score for the positive class.
    Works whether labels are 'LABEL_0/1' or custom names; we pick the one ending with '1'
    or fall back to the max score if needed.
    """
    out = pipe(text, max_length=256)
    # `out` is a list[dict] when top_k=None; if batch, it's list[list[dict]]
    if isinstance(out, list) and out and isinstance(out[0], dict):
        scores = out
    elif isinstance(out, list) and out and isinstance(out[0], list):
        scores = out[0]
    else:
        raise RuntimeError(f"Unexpected pipeline output: {out}")

    pos = None
    for s in scores:
        if str(s.get("label", "")).endswith("1"):
            pos = float(s["score"])
            break
    if pos is None:
        # Fallback: take the highest score as positive
        pos = float(max(scores, key=lambda x: float(x["score"]))["score"])
    return pos

probs = df["text"].apply(prob_mal).astype(float).to_numpy()
y_lm = (probs >= thr).astype(int)

# ---------- Ensemble (regex + LM) ----------
ens = EnsembleGuard(model_dir=MODEL_DIR)

def ens_decide(text: str) -> int:
    out = ens.decide(text)
    if not isinstance(out, dict) or "decision" not in out:
        raise RuntimeError(f"Unexpected EnsembleGuard.decide() output: {out}")
    return 1 if out["decision"] == "block" else 0

y_ens = df["text"].apply(ens_decide).astype(int).to_numpy()

# ---------- Metrics ----------
y_true = df["y"].astype(int).to_numpy()

def show(name, y_hat):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_hat, digits=3))
    print(confusion_matrix(y_true, y_hat))

show("Regex only", y_rx)
show("LM only", y_lm)
show("Ensemble (Regex + LM)", y_ens)
