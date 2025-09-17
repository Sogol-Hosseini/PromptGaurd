import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from typing import Dict, Any, List


class LMBasedDetector:
    """
    LM-based jailbreak detector.
    Wraps a Hugging Face TextClassificationPipeline and exposes a .score() method.
    """

    def __init__(self, model_dir: str, default_thresh: float = 0.5):
        self.model_dir = model_dir

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # Load threshold (use default if file missing)
        self.threshold = default_thresh
        thr_path = os.path.join(model_dir, "threshold.txt")
        if os.path.exists(thr_path):
            try:
                self.threshold = float(open(thr_path).read().strip())
            except Exception:
                pass

        # Build pipeline
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,  # return all labels/scores
        )

    def detect(self, text: str, threshold: float = None) -> Dict[str, Any]:
        """Return raw detection result: probability + boolean flag."""
        thr = threshold if threshold is not None else self.threshold
        scores: List[Dict[str, Any]] = self.pipe(text)[0]
        # assume label ending with "1" = malicious
        p_mal = next(s["score"] for s in scores if s["label"].endswith("1"))
        return {
            "threshold": thr,
            "malicious_prob": float(p_mal),
            "is_malicious": p_mal >= thr,
            "scores": scores,
        }

    def score(self, text: str) -> Dict[str, Any]:
        """Unified API: return dict(level, score, detail, hits)."""
        raw = self.detect(text)
        p = raw["malicious_prob"]

        if p >= self.threshold:
            level = "block"
        elif p >= self.threshold * 0.7:
            level = "warn"
        else:
            level = "ok"

        return {
            "level": level,
            "score": int(round(p * 10)),  # scale 0–10
            "detail": raw,
            "hits": [
                {"category": "malicious_prob", "snippet": f"{p:.2f}"}
            ],
        }
