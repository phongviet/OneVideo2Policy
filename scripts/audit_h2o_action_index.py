"""Summarize official H2O Place and Pour intervals before downloading gated poses."""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

from onevideo2policy.video.h2o import read_action_index

SOURCE = (
    "https://raw.githubusercontent.com/taeinkwon/h2odataset/"
    "main/action_labels/action_train.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/h2o-action-index-audit.json"),
    )
    args = parser.parse_args()
    with urllib.request.urlopen(SOURCE, timeout=30) as response:
        text = response.read().decode("utf-8")
    temporary = args.output.with_suffix(".source.txt")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(text, encoding="utf-8")
    try:
        rows = read_action_index(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    counts = Counter(row.action for row in rows)
    candidate = next(
        row
        for row in rows
        if row.sequence == "subject1/h1/1" and row.action == "place milk"
    )
    payload = {
        "schema_version": 1,
        "source": SOURCE,
        "split": "train",
        "intervals": len(rows),
        "place_intervals": sum(value for key, value in counts.items() if key.startswith("place ")),
        "pour_intervals": counts["pour milk"],
        "place_by_object": {
            key.removeprefix("place "): value
            for key, value in sorted(counts.items())
            if key.startswith("place ")
        },
        "selected_oracle_candidate": {
            "sequence": candidate.sequence,
            "action": candidate.action,
            "start_frame": candidate.start_frame,
            "end_frame": candidate.end_frame,
            "frames": len(candidate.frame_ids),
            "reason": "short single-object Place interval with a subsequent Pour action",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
