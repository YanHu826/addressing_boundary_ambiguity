#!/usr/bin/env python3
"""Validate dataset split integrity for paper-grade experiments."""

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = PROJECT_ROOT / "airs" / "data" / "splits"

DATASETS = {
    "TN3K": {"split_dir": "tn3k", "expids": {1: "322", 2: "644", 3: "1289"}, "test": "test.txt"},
    "BUSI": {"split_dir": "BUSI", "expids": {1: "72", 2: "144", 3: "288"}, "test": "test.txt"},
    "UDIAT": {"split_dir": "UDIAT", "expids": {1: "18", 2: "36", 3: "73"}, "test": "test.txt"},
    "HC18": {"split_dir": "HC18", "expids": {1: "107", 2: "214"}, "test": "test.txt"},
    "PSFH": {"split_dir": "PSFH", "expids": {1: "320", 2: "640"}, "test": "test.txt"},
}


def canonical_dataset(name):
    key = str(name).strip().upper()
    if key == "TN3K":
        return "TN3K"
    if key in DATASETS:
        return key
    raise SystemExit(f"Unsupported dataset: {name}")


def read_split(path):
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_subset_counts(entries):
    counts = {}
    for entry in entries:
        parts = str(entry).replace("\\", "/").split("/")
        subset = "unknown"
        for idx, part in enumerate(parts):
            if part == "HC18" and idx + 1 < len(parts):
                subset = parts[idx + 1]
                break
            if part == "PSFH" and idx + 1 < len(parts):
                subset = parts[idx + 1]
                break
        counts[subset] = counts.get(subset, 0) + 1
    return dict(sorted(counts.items()))


def split_summary(path, entries):
    return {
        "path": str(path),
        "count": len(entries),
        "unique_count": len(set(entries)),
        "sha256": sha256_file(path),
        "source_subsets": source_subset_counts(entries),
    }


def fail_if_overlap(errors, left_name, left_entries, right_name, right_entries):
    overlap = set(left_entries) & set(right_entries)
    if overlap:
        examples = sorted(overlap)[:5]
        errors.append(
            {
                "type": "overlap",
                "left": left_name,
                "right": right_name,
                "count": len(overlap),
                "examples": examples,
            }
        )


def validate_one(dataset, expid, allow_missing_test=False):
    spec = DATASETS[dataset]
    split_dir = SPLIT_ROOT / spec["split_dir"]
    train_dir_name = spec["expids"].get(expid)
    if train_dir_name is None:
        raise SystemExit(f"Unsupported expID {expid} for {dataset}")

    paths = {
        "labeled": split_dir / train_dir_name / "labeled.txt",
        "unlabeled": split_dir / train_dir_name / "unlabeled.txt",
        "val": split_dir / "val.txt",
        "test": split_dir / spec["test"],
    }
    entries = {}
    errors = []
    summaries = {}

    for name, path in paths.items():
        values = read_split(path)
        if values is None:
            if name == "test" and allow_missing_test:
                entries[name] = []
                summaries[name] = {"path": str(path), "missing": True, "count": 0}
                continue
            errors.append({"type": "missing_split", "split": name, "path": str(path)})
            entries[name] = []
            continue
        entries[name] = values
        summaries[name] = split_summary(path, values)
        if len(values) != len(set(values)):
            errors.append(
                {
                    "type": "duplicate_entries",
                    "split": name,
                    "count": len(values) - len(set(values)),
                }
            )

    fail_if_overlap(errors, "labeled", entries["labeled"], "unlabeled", entries["unlabeled"])
    fail_if_overlap(errors, "labeled", entries["labeled"], "val", entries["val"])
    fail_if_overlap(errors, "unlabeled", entries["unlabeled"], "val", entries["val"])
    fail_if_overlap(errors, "labeled", entries["labeled"], "test", entries["test"])
    fail_if_overlap(errors, "unlabeled", entries["unlabeled"], "test", entries["test"])
    fail_if_overlap(errors, "val", entries["val"], "test", entries["test"])

    return {
        "dataset": dataset,
        "expID": expid,
        "split_dir": str(split_dir),
        "splits": summaries,
        "ok": not errors,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Check dataset split leakage and split hashes.")
    parser.add_argument("--dataset", default="ALL", help="Dataset name or ALL.")
    parser.add_argument("--expID", type=int, default=None, help="Experiment id. Defaults to all supported expIDs.")
    parser.add_argument("--allow-missing-test", action="store_true", help="Warn less for datasets without independent test.txt.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()

    datasets = list(DATASETS) if str(args.dataset).strip().upper() == "ALL" else [canonical_dataset(args.dataset)]
    reports = []
    for dataset in datasets:
        expids = [args.expID] if args.expID is not None else sorted(DATASETS[dataset]["expids"])
        for expid in expids:
            reports.append(validate_one(dataset, expid, allow_missing_test=args.allow_missing_test))

    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for report in reports:
            status = "OK" if report["ok"] else "FAIL"
            print(f"[{status}] {report['dataset']} exp{report['expID']}")
            for split_name, summary in report["splits"].items():
                missing = " missing" if summary.get("missing") else ""
                sources = summary.get("source_subsets") or {}
                source_text = ""
                if sources:
                    source_text = " sources=" + ",".join(f"{k}:{v}" for k, v in sources.items())
                print(f"  {split_name}: count={summary.get('count', 0)}{missing}{source_text}")
                if (
                    report["dataset"] == "HC18"
                    and split_name == "unlabeled"
                    and sources.get("test_set", 0) > 0
                ):
                    print(
                        "  NOTE HC18 unlabeled contains official test_set images; "
                        "runtime drops them to enforce the non-transductive paper protocol."
                    )
            for error in report["errors"]:
                print(f"  ERROR {error}")

    if any(not report["ok"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
