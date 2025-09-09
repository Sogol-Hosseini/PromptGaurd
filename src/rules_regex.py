import re, yaml
from dataclasses import dataclass
from collections import Counter

@dataclass
class RuleHit:
    category: str
    pattern: str
    span: tuple
    snippet: str

class RegexRules:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.version = cfg.get("version", 1)
        self.risk_thresholds = cfg.get("risk_thresholds", {"hard_block":5, "warn":3})
        self.categories = []
        self.compiled = []
        for cat in cfg["categories"]:
            weight = int(cat.get("weight", 1))
            pats = []
            for p in cat.get("patterns", []):
                pats.append(re.compile(p, re.MULTILINE))
            self.categories.append({"name": cat["name"], "weight": weight})
            self.compiled.append(pats)

    def _find_hits(self, text: str):
        hits = []
        for (cat_meta, pats) in zip(self.categories, self.compiled):
            for rgx in pats:
                for m in rgx.finditer(text):
                    start, end = m.span()
                    snippet = text[max(0,start-40):min(len(text), end+40)]
                    hits.append(RuleHit(cat_meta["name"], rgx.pattern, (start,end), snippet))
        return hits

    # simple repetition detector beyond regex:
    def _repetition_score(self, text: str) -> int:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens: return 0
        counts = Counter(tokens)
        # word repeated too much and total length > threshold
        if max(counts.values()) >= 8 and len(tokens) > 40:
            return 2
        # 3-gram repeated 3+ times
        trigrams = Counter(tuple(tokens[i:i+3]) for i in range(len(tokens)-2))
        if trigrams and max(trigrams.values()) >= 3:
            return 2
        return 0

    def score(self, text: str):
        hits = self._find_hits(text)
        # sum weights by category
        cat_score = 0
        detail = {}
        for h in hits:
            w = next(c["weight"] for c in self.categories if c["name"] == h.category)
            cat_score += w
            detail.setdefault(h.category, {"weight": w, "count": 0})
            detail[h.category]["count"] += 1

        rep = self._repetition_score(text)
        if rep:
            cat_score += rep
            detail["repetition"] = {"weight": rep, "count": 1}

        level = "ok"
        if cat_score >= self.risk_thresholds.get("hard_block", 5): level = "block"
        elif cat_score >= self.risk_thresholds.get("warn", 3):      level = "warn"

        return {"score": cat_score, "level": level, "detail": detail, "hits": hits}
