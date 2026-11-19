"""Compare reports for the bundled RAW sequence against reviewed meteor trails."""

import argparse
import json
from math import cos, radians, hypot
from pathlib import Path


def trail_coverage(label, segments):
    """Return the fraction of the annotated axis covered within its corridor."""
    (x0, y0), (x1, y1) = label["axis"]
    length = hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    half_width = label["half_width_px"]
    intervals = []
    for (xa, ya), (xb, yb) in segments:
        dx, dy = xb - xa, yb - ya
        along = dx * ux + dy * uy
        across = dx * -uy + dy * ux
        segment_length = hypot(dx, dy)
        if not segment_length or abs(along) < cos(radians(15)) * segment_length:
            continue

        start_along = (xa - x0) * ux + (ya - y0) * uy
        start_across = (xa - x0) * -uy + (ya - y0) * ux
        if abs(across) < 1e-9:
            if abs(start_across) > half_width:
                continue
            low_t, high_t = 0.0, 1.0
        else:
            low_t, high_t = sorted(
                (
                    (-half_width - start_across) / across,
                    (half_width - start_across) / across,
                )
            )
            low_t, high_t = max(low_t, 0.0), min(high_t, 1.0)
        if low_t >= high_t:
            continue
        low, high = sorted((start_along + low_t * along, start_along + high_t * along))
        low, high = max(low, 0.0), min(high, length)
        if low < high:
            intervals.append((low, high))

    covered = end = 0.0
    for low, high in sorted(intervals):
        covered += max(0.0, high - max(low, end))
        end = max(end, high)
    return covered / length


def load_pairs(path, frames):
    """Use the known input order to check coverage and resolve prior filenames."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda row: row["frame_index"])
    actual = [(r["frame_index"], r["prev_frame_index"], r["current"]) for r in rows]
    expected = [(index, index - 1, name) for index, name in enumerate(frames[1:], 1)]
    if actual != expected:
        raise ValueError(
            f"{path} does not contain one successful report for each expected pair"
        )
    return {row["current"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("hough_report", type=Path)
    parser.add_argument("region_report", type=Path)
    args = parser.parse_args()
    frames = sorted(path.name for path in args.raw_dir.glob("*.ORF"))
    labels = json.loads(Path(__file__).with_name("meteor_labels.json").read_text())
    hough = load_pairs(args.hough_report, frames)
    region = load_pairs(args.region_report, frames)
    selected = [
        {name for name, row in rows.items() if row["is_candidate"]}
        for rows in (hough, region)
    ]
    trails = []
    for label in labels:
        current = label["current"]
        old, new = hough[current], region[current]
        if frames[old["prev_frame_index"]] != label["previous"]:
            raise ValueError(f"Unexpected previous frame for {current}")
        region_evidence = new["extras"]["region"]
        regions = region_evidence["regions"]
        trails.append(
            {
                "current": current,
                "hough_candidate": old["is_candidate"],
                "region_candidate": new["is_candidate"],
                "hough_line_coverage": trail_coverage(
                    label, [(line[:2], line[2:]) for line in old["lines"]]
                ),
                "region_accepted_coverage": trail_coverage(
                    label,
                    [
                        span
                        for item in regions
                        if item["accepted"]
                        for span in item["axis_support"]
                    ],
                ),
                "region_measured_coverage": trail_coverage(
                    label, [span for item in regions for span in item["axis_support"]]
                ),
                "region_omitted_count": region_evidence["regions_omitted"],
            }
        )
    print(
        json.dumps(
            {
                "pairs": len(hough),
                "hough_candidates": len(selected[0]),
                "region_candidates": len(selected[1]),
                "added_candidates": sorted(selected[1] - selected[0]),
                "lost_candidates": sorted(selected[0] - selected[1]),
                "meteor_trails": trails,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
