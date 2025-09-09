# src/ensemble_guard.py
import os, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from rules_regex import RegexRules

from boundary_enforcer import wrap_prompt
from normalizer import normalize_text  # keep this too




MODEL_DIR = "models/bert-pi-detector/best"
CFG       = "src/patterns.regex.yaml"

def _serialize_hits(rx: dict) -> dict:
    """Convert RuleHit objects to plain dicts so json.dumps works."""
    return {
        **{k: v for k, v in rx.items() if k != "hits"},
        "hits": [
            {
                "category": h.category,
                "pattern": h.pattern,
                "span": h.span,
                "snippet": h.snippet,
            }
            for h in rx.get("hits", [])
        ],
    }

class EnsembleGuard:
    def __init__(self, model_dir=MODEL_DIR, cfg=CFG, lm_threshold=None):
        self.rules = RegexRules(cfg)

        # keep consistent with training/tokenization
        self.max_len = 256

        # tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.model_max_length = self.max_len  # enforce cap
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # threshold
        thr = 0.5
        tp = os.path.join(model_dir, "threshold.txt")
        if os.path.exists(tp):
            with open(tp) as f:
                thr = float(f.read().strip())
        if lm_threshold is not None:
            thr = lm_threshold
        self.THRESH = thr

        # device + pipeline
        device_id = 0 if torch.cuda.is_available() else -1
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_id,
            top_k=None,  # return all labels
        )

    def lm_prob_malicious(self, text: str) -> float:
        # always truncate to avoid >512-token crashes
        scores = self.pipe(text, truncation=True, max_length=self.max_len)[0]
        return next(s["score"] for s in scores if s["label"].endswith("1"))

    def decide(self, text: str) -> dict:
        norm = normalize_text(text) 
        rx_raw = self.rules.score(norm)
        rx_clean = _serialize_hits(rx_raw)

        # Regex hard block wins immediately (no LM call)
        if rx_raw["level"] == "block":
            return {
                "decision": "block",
                "reason": "regex_block",
                "regex": rx_clean,
                "prob": None,
                "threshold": self.THRESH,
            }

        # Otherwise compute LM probability; also block on 'warn'
        p = self.lm_prob_malicious(norm)
        if p >= self.THRESH or rx_raw["level"] == "warn":
            return {
                "decision": "block",
                "reason": "lm_or_regex_warn",
                "regex": rx_clean,
                "prob": p,
                "threshold": self.THRESH,
            }

        # Allow
        return {
            "decision": "allow",
            "reason": "clean",
            "regex": rx_clean,
            "prob": p,
            "threshold": self.THRESH,
        }
    def prepare(self, system_instructions: str, user_text: str):
        """
        Returns either {"decision": "block", ...} or {"decision": "allow", "prompt": <wrapped_text>, ...}
        """
        verdict = self.decide(user_text)
        if verdict["decision"] == "block":
            return verdict
        wrapped = wrap_prompt(system_instructions, user_text)
        verdict["prompt"] = wrapped.text
        verdict["fingerprint"] = wrapped.user_fingerprint
        return verdict

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="Prompt to evaluate")
    args = ap.parse_args()

    g = EnsembleGuard()
    print(json.dumps(g.decide(args.text), indent=2, ensure_ascii=False))
