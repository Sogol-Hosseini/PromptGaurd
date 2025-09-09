import argparse, json
from rules_regex import RegexRules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Text to scan")
    ap.add_argument("--file", help="Path to a .txt with content to scan")
    ap.add_argument("--cfg", default="src/patterns.regex.yaml")
    args = ap.parse_args()

    rr = RegexRules(args.cfg)
    text = args.text or open(args.file, "r", encoding="utf-8").read()
    result = rr.score(text)
    # Convert hits to printable dicts
    out = {
        "level": result["level"],
        "score": result["score"],
        "detail": result["detail"],
        "hits": [{"category": h.category, "pattern": h.pattern, "span": h.span, "snippet": h.snippet} for h in result["hits"]],
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
