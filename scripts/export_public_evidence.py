"""Export saved Day1 API evidence to a minimal public JSON file."""

import argparse
from pathlib import Path

from app.public_evidence import export_public_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=[
            Path("learning_log/ask_evidence.json"),
            Path("learning_log/ask_evidence_20261025_gemini38.json"),
        ],
        help="Saved source Evidence JSON files; these files are read-only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/day1_gemini_experiments.json"),
        help="Path for the newly generated public Evidence JSON",
    )
    args = parser.parse_args()
    public = export_public_evidence(args.sources, args.output)
    print(
        f"Wrote {sum(len(item['results']) for item in public['experiments'])} "
        f"public evidence rows from {len(public['experiments'])} experiments "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
