import re
import yaml
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RuleHit:
    category: str
    pattern: str
    span: tuple
    snippet: str

class RegexBasedDetector:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.version = cfg.get("version", 1)
        self.risk_thresholds = cfg.get("risk_thresholds", {"hard_block": 4, "warn": 2})
        self.categories_meta = []
        self.compiled: List[List[re.Pattern]] = []

        flags = re.IGNORECASE | re.DOTALL | re.MULTILINE
        for cat in cfg["categories"]:
            weight = int(cat.get("weight", 1))
            pats = []
            for p in cat.get("patterns", []):
                pats.append(re.compile(p, flags))
            self.categories_meta.append({"name": cat["name"], "weight": weight})
            self.compiled.append(pats)

    # --- text normalization (boosts recall on weird spacing/Unicode tricks) ---
    @staticmethod
    def _normalize(text: str) -> str:
        # NFKC to fold homoglyphs; collapse whitespace; lowercasing via casefold
        t = unicodedata.normalize("NFKC", text)
        t = t.casefold()
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        # make fence markers easier to match
        t = t.replace("```", "```")
        return t

    def _find_hits(self, text: str) -> List[RuleHit]:
        hits: List[RuleHit] = []
        for (cat_meta, pats) in zip(self.categories_meta, self.compiled):
            cname = cat_meta["name"]
            for pat in pats:
                for m in pat.finditer(text):
                    start, end = m.span()
                    snippet = text[max(0, start-60):min(len(text), end+60)]
                    hits.append(RuleHit(
                        category=cname,
                        pattern=pat.pattern,
                        span=(start, end),
                        snippet=snippet
                    ))
        return hits

    def score(self, raw_text: str) -> Dict[str, Any]:
        text = self._normalize(raw_text)
        hits = self._find_hits(text)

        # dedup near-duplicate span/pattern pairs
        uniq = []
        seen = set()
        for h in hits:
            key = (h.category, h.pattern, h.span[0], h.span[1])
            if key not in seen:
                seen.add(key)
                uniq.append(h)

        # weighted score by category
        score_by_cat = {}
        total = 0
        for h in uniq:
            w = next(c["weight"] for c in self.categories_meta if c["name"] == h.category)
            score_by_cat[h.category] = score_by_cat.get(h.category, 0) + w
            total += w

        # level
        if total >= self.risk_thresholds.get("hard_block", 4):
            level = "block"
        elif total >= self.risk_thresholds.get("warn", 2):
            level = "warn"
        else:
            level = "pass"

        return {
            "version": self.version,
            "level": level,
            "score": total,
            "by_category": score_by_cat,
            "hits": [h.__dict__ for h in uniq[:50]],  # cap for readability
        }
