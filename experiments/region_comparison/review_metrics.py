"""Score original exposures independently of geometric labels and difference polarity."""

import argparse
import hashlib
import json
import re
from math import comb
from pathlib import Path

from benchmark_v2 import SEQUENCES

NEGATIVE_DECLARATION = (
    "All unlisted original frames were visually classified as containing no meteor."
)
RAW_NAME = re.compile(r"`(_C\d{6}\.ORF)`")


def read_truth(source, frames, inventory):
    """Read Meteor, including explicit No or the repository's exhaustive declaration."""
    text = source.read_text()
    table = []
    column = None
    for line in text.splitlines():
        if not line.startswith("|"):
            column = None
            continue
        cells = [entry.strip() for entry in line.strip("|").split("|")]
        if "File" in cells and "Meteor" in cells:
            column = cells.index("Meteor")
        elif column is not None and (match := RAW_NAME.fullmatch(cells[0])):
            (name,) = match.groups()
            if len(cells) <= column or cells[column] not in ("Yes", "No"):
                raise ValueError(f"Expected Yes or No for {name} in {source}")
            table.append((name, cells[column] == "Yes"))
    declared = dict(table)
    if not declared or len(table) != len(declared) or not declared.keys() <= inventory:
        raise ValueError(f"Invalid or duplicate label table in {source}")
    if NEGATIVE_DECLARATION not in text and not inventory <= declared.keys():
        raise ValueError(f"Negative labels are incomplete in {source}")
    return {name: declared.get(name, False) for name in frames}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def exact_recall_interval(successes, trials):
    """Illustrative two-sided Clopper–Pearson interval under independent events."""
    if not trials:
        return None

    def inverse_upper_tail(first):
        low, high = 0.0, 1.0
        for _ in range(50):
            p = (low + high) / 2
            tail = sum(
                comb(trials, i) * p**i * (1 - p) ** (trials - i)
                for i in range(first, trials + 1)
            )
            if tail < 0.025:
                low = p
            else:
                high = p
        return (low + high) / 2

    return [
        inverse_upper_tail(successes) if successes else 0.0,
        1 - inverse_upper_tail(trials - successes) if successes < trials else 1.0,
    ]


def frame_metrics(pairs, truth):
    """Count decisions by current exposure, allowing a common warm-up subset."""
    if not pairs.keys() <= truth.keys():
        raise ValueError("A measured frame has no truth label")
    tp = [name for name, row in pairs.items() if row["candidate"] and truth[name]]
    fp = [name for name, row in pairs.items() if row["candidate"] and not truth[name]]
    fn = [name for name, row in pairs.items() if not row["candidate"] and truth[name]]
    tn = len(pairs) - len(tp) - len(fp) - len(fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn_count": tn,
        "precision": _rate(len(tp), len(tp) + len(fp)),
        "recall": _rate(len(tp), len(tp) + len(fn)),
        "recall_95_exact_if_independent": exact_recall_interval(
            len(tp), len(tp) + len(fn)
        ),
        "false_positive_rate": _rate(len(fp), len(fp) + tn),
        "candidates_per_meteor_frame": _rate(len(tp) + len(fp), len(tp)),
    }


def event_matches(pairs, frames, annotations):
    """Count one event if either scored pair has a qualifying reference-axis match."""
    events = {}
    for label in annotations:
        name = label["current"]
        index = frames.index(name)
        if index == 0 or frames[index - 1] != label["previous"]:
            raise ValueError(f"Wrong preceding frame for annotation {name}")
        if name not in pairs:
            continue
        following = frames[index + 1] if index + 1 < len(frames) else None
        outcomes = []
        for current in (name, following):
            row = pairs.get(current)
            if row is None:
                outcomes.append(None)
                continue
            coverage = row["coverage"][name]
            outcomes.append(
                {"coverage": coverage, "match": row["candidate"] and coverage >= 0.5}
            )
        appearance, next_pair = outcomes
        events[name] = {
            "appearance": appearance,
            "following": next_pair,
            "event_match": appearance["match"]
            or bool(next_pair and next_pair["match"]),
        }
    return events


def summarize_run(result, raw_root):
    name = result["sequence"]
    source = raw_root / SEQUENCES[name][0]
    frames = result["frames"]
    expected_frames = [
        f"_C{SEQUENCES[name][1]}{i:03d}.ORF"
        for i in range(SEQUENCES[name][2], SEQUENCES[name][3] + 1)
    ]
    if frames != expected_frames:
        raise ValueError(f"Unexpected frame inventory for {name}")
    readme = source / "README.md"
    checksums = source / ("SHA256SUMS" if name == "bundled" else "SHA256SUM")
    inventory = {
        line.split()[1].lstrip("*") for line in checksums.read_text().splitlines()
    }
    truth = read_truth(readme, frames, inventory)
    annotations = result["annotations"]
    if any(not truth[label["current"]] for label in annotations):
        raise ValueError(f"An axis disagrees with exposure truth for {source}")
    warmup = result["median_depth"] or 1
    scored = frames[warmup:]
    for method in result["results"].values():
        if list(method["pairs"]) != scored:
            raise ValueError(f"Pair coverage differs for {name}")
        method["frame_metrics"] = frame_metrics(method["pairs"], truth)
        method["events"] = event_matches(method["pairs"], frames, annotations)
    result["frame_truth"] = truth
    result["truth_sha256"] = hashlib.sha256(readme.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    data = {}
    for run in args.runs:
        data[run.name] = {
            "environment": json.loads((run / "environment.json").read_text()),
            "sequences": {
                path.stem: summarize_run(json.loads(path.read_text()), args.raw_root)
                for path in sorted(run.glob("*.json"))
                if path.stem != "environment"
            },
        }
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
