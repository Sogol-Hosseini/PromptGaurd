import sys, json
from src.utils.normalizer import normalize_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    out = normalize_text(text)
    print(json.dumps({"original": text, "normalized": out}, ensure_ascii=False, indent=2))
