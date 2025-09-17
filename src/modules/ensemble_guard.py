# src/modules/ensemble_guard.py
import os, json, torch
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# ⬇️ use the class you actually have in Module 2
from modules.rules_regex import RegexBasedDetector

# keep these (you said don't skip them)
from src.modules.boundary_enforcer import wrap_prompt
from src.utils.normalizer import normalize_text

MODEL_DIR = "models/bert-pi-detector/best"
# your yaml is under utils/
CFG       = "src/utils/patterns.regex.yaml"

def _serialize_hits(rx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert hits to plain dicts so json.dumps works.
    Supports both object-style hits (h.category, ...) and dict hits.
    """
    def to_dict(h):
        if isinstance(h, dict):
            return {
                "category": h.get("category"),
                "pattern":  h.get("pattern"),
                "span":     h.get("span"),
                "snippet":  h.get("snippet"),
            }
        return {
            "category": getattr(h, "category", None),
            "pattern":  getattr(h, "pattern", None),
            "span":     getattr(h, "span", None),
            "snippet":  getattr(h, "snippet", None),
        }

    return {
        **{k: v for k, v in rx.items() if k != "hits"},
        "hits": [to_dict(h) for h in rx.get("hits", [])],
    }

class EnsembleGuard:
    def __init__(self, model_dir=MODEL_DIR, cfg=CFG, lm_threshold=None):
        # ⬇️ build Regex detector from your YAML
        # If your class exposes a different loader, adjust here (e.g., from_yaml)
        self.rules = RegexBasedDetector(cfg)


        self.max_len = 256

        # tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.model_max_length = self.max_len
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        thr = 0.5
        tp = os.path.join(model_dir, "threshold.txt")
        if os.path.exists(tp):
            with open(tp) as f:
                thr = float(f.read().strip())
        if lm_threshold is not None:
            thr = lm_threshold
        self.THRESH = thr

        device_id = 0 if torch.cuda.is_available() else -1
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_id,
            top_k=None,
        )

    def lm_prob_malicious(self, text: str) -> float:
        scores = self.pipe(text, truncation=True, max_length=self.max_len)[0]
        return next(s["score"] for s in scores if s["label"].endswith("1"))

    def decide(self, text: str) -> dict:
        norm = normalize_text(text)
        rx_raw = self.rules.score(norm)
        rx_clean = _serialize_hits(rx_raw)

        if rx_raw["level"] == "block":
            return {
                "decision": "block",
                "reason": "regex_block",
                "regex": rx_clean,
                "prob": None,
                "threshold": self.THRESH,
            }

        p = self.lm_prob_malicious(norm)
        if p >= self.THRESH or rx_raw["level"] == "warn":
            return {
                "decision": "block",
                "reason": "lm_or_regex_warn",
                "regex": rx_clean,
                "prob": p,
                "threshold": self.THRESH,
            }

        return {
            "decision": "allow",
            "reason": "clean",
            "regex": rx_clean,
            "prob": p,
            "threshold": self.THRESH,
        }

    def prepare(self, system_instructions: str, user_text: str):
        verdict = self.decide(user_text)
        if verdict["decision"] == "block":
            return verdict
        wrapped = wrap_prompt(system_instructions, user_text)
        verdict["prompt"] = wrapped.text
        verdict["fingerprint"] = wrapped.user_fingerprint
        return verdict
