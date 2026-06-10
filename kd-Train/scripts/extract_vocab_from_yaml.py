"""data.yaml의 names 항목(dict 또는 list)을 평탄한 vocab.txt로 추출."""
import argparse
import yaml
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = yaml.safe_load(Path(args.yaml).read_text())
    names = data.get("names", {})
    if isinstance(names, dict):
        ordered = [names[i] for i in sorted(names.keys())]
    else:
        ordered = list(names)

    Path(args.out).write_text("\n".join(ordered) + "\n")
    print(f"✅ {len(ordered)} 클래스 → {args.out}")
    print(f"   처음 5개: {ordered[:5]}")


if __name__ == "__main__":
    main()
