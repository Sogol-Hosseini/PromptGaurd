import os, csv, json, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from ensemble_guard import EnsembleGuard, MODEL_DIR

CSV_PATH = "hf://datasets/qualifire/prompt-injections-benchmark/test.csv"

df = pd.read_csv(CSV_PATH)
df["y"] = df["label"].map({"benign":0, "jailbreak":1})

g = EnsembleGuard(model_dir=MODEL_DIR)

rows = []
for i, r in df.iterrows():
    res = g.decide(r["text"])
    pred = 1 if res["decision"] == "block" else 0
    good = (pred == r["y"])
    rows.append({
        "idx": i,
        "text": r["text"],
        "gold": r["y"],
        "pred": pred,
        "prob": res["prob"],
        "reason": res["reason"],
        "regex_score": res["regex"]["score"],
        "regex_level": res["regex"]["level"],
        "regex_hits": " | ".join(f"{h['category']}" for h in res["regex"]["hits"]),
        "ok": good
    })

out = pd.DataFrame(rows)
out.to_csv("analysis/ensemble_error_analysis.csv", index=False)
print("Wrote analysis/ensemble_error_analysis.csv")
print("False positives:", (out["gold"]==0) & (out["pred"]==1).sum())
print("False negatives:", (out["gold"]==1) & (out["pred"]==0).sum())
