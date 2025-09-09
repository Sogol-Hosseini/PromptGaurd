# src/normalizer.py
import re, html, unicodedata
from urllib.parse import unquote

# Common zero-width / control chars
_ZW = [
    "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060",  # ZWSP, ZWNJ, ZWJ, BOM, WJ
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi controls
]
_ZW_RE = re.compile("|".join(map(re.escape, _ZW)))

# Minimal homoglyph map (extend as needed)
HOMO_MAP = {
    "Ｉ": "I", "Ｌ": "L", "Ｏ": "O", "Ｓ": "S", "Ａ": "A", "Ｅ": "E",
    "ａ": "a", "ｅ": "e", "ｏ": "o", "ｓ": "s", "ｌ": "l", "і": "i",  # Cyrillic i
    "е": "e",  # Cyrillic e
    "Ꮯ": "C", "Ꭱ": "R", "Ꭺ": "A",
}

_HOMO_RE = re.compile("|".join(map(re.escape, HOMO_MAP.keys()))) if HOMO_MAP else None
_WS_RE   = re.compile(r"[ \t\r\f\v]+")

def _apply_homoglyphs(s: str) -> str:
    if not _HOMO_RE:
        return s
    return _HOMO_RE.sub(lambda m: HOMO_MAP[m.group(0)], s)

def normalize_text(
    s: str,
    *,
    html_unescape: bool = True,
    url_decode: bool = True,
    unicode_norm: str = "NFKC",
    strip_controls: bool = True,
    strip_zerowidth: bool = True,
    collapse_ws: bool = True,
    lowercase: bool = False,
) -> str:
    if not isinstance(s, str):
        s = str(s)

    # 1) HTML entities → characters
    if html_unescape:
        s = html.unescape(s)

    # 2) URL percent-decoding (do twice in case of double-encoding)
    if url_decode:
        try:
            once = unquote(s)
            s = unquote(once)
        except Exception:
            pass

    # 3) Unicode normalization (canonical + compatibility)
    if unicode_norm:
        s = unicodedata.normalize(unicode_norm, s)

    # 4) Remove zero-width & bidi controls
    if strip_zerowidth:
        s = _ZW_RE.sub("", s)

    # 5) Remove other control chars (except \n, \t)
    if strip_controls:
        s = "".join(ch for ch in s if (ch.isprintable() or ch in "\n\t"))

    # 6) Basic homoglyph folding (extend map as needed)
    s = _apply_homoglyphs(s)

    # 7) Collapse horizontal whitespace
    if collapse_ws:
        s = _WS_RE.sub(" ", s)

    # 8) Lowercase (optional; keep off if case matters)
    if lowercase:
        s = s.lower()

    # Final trim
    return s.strip()
