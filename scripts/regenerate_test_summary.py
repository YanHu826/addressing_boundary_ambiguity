#!/usr/bin/env python3
"""Rebuild outputs/summary/test_results.md from existing per-ckpt artifacts.

Sources of truth (per ckpt):
  - outputs/semi/results/<ckpt>/test/test_eval_manifest.json -> dataset / expID / ckpt_name
  - outputs/semi/results/<ckpt>/test/test_case_metrics.csv  -> per-case metrics
                                                              (preferred: recomputes
                                                              dataset-protocol aggregates,
                                                              e.g. HC18 uses filled-region
                                                              DSC matching BiPCC/Ours JBHI'25)
  - outputs/logs/semi/<ckpt>_test.log                       -> last "Valid Result: ..." line
                                                              (fallback for fields the CSV
                                                              does not carry)

The pipeline overwrites test_results.md on every run via initialize_test_summary(),
so any per-ckpt rows from earlier runs are lost. This rebuilder rebuilds the full
table from the on-disk artifacts.

Run:  python3 scripts/regenerate_test_summary.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "outputs" / "semi" / "results"
LOG_ROOT = REPO_ROOT / "outputs" / "logs" / "semi"
SUMMARY_PATH = REPO_ROOT / "outputs" / "summary" / "test_results.md"

HEADER = (
    "# Test Results\n\n"
    "| dataset | expid | ckpt_name | recall | specificity | precision | F1 | F2 | "
    "ACC_overall | IoU_poly | IoU_bg | IoU_mean | dice | DSC_all | Jacc_all | "
    "HD95_all | ASD_all | DSC_PS | Jacc_PS | HD95_PS | ASD_PS | "
    "DSC_FH | Jacc_FH | HD95_FH | ASD_FH |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
    "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)

ROW_KEYS = [
    "recall", "specificity", "precision", "F1", "F2",
    "ACC_overall", "IoU_poly", "IoU_bg", "IoU_mean", "dice",
    "DSC_all", "Jacc_all", "HD95_all", "ASD_all",
    "DSC_PS", "Jacc_PS", "HD95_PS", "ASD_PS",
    "DSC_FH", "Jacc_FH", "HD95_FH", "ASD_FH",
]

VALID_RESULT_PREFIX = "Valid Result:"


def parse_valid_result(line: str) -> dict[str, float]:
    payload = line.split(VALID_RESULT_PREFIX, 1)[1]
    metrics: dict[str, float] = {}
    for chunk in payload.split(","):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def last_valid_result(log_path: Path) -> dict[str, float] | None:
    if not log_path.is_file():
        return None
    last: dict[str, float] | None = None
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if VALID_RESULT_PREFIX in raw:
                last = parse_valid_result(raw)
    return last


def load_manifest(manifest_path: Path) -> dict | None:
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def hc18_filled_aggregates(csv_path: Path) -> dict[str, float]:
    """For HC18, recompute filled-region DSC/Jaccard aggregates from per-case CSV.

    The live `Valid Result:` log line was historically written using a
    boundary-band protocol that does NOT match BiPCC/Ours JBHI'25.
    `dice_union` / `jaccard_union` carry the filled-region values that do."""
    if not csv_path.is_file():
        return {}
    cols = ("dice_union", "jaccard_union", "hd95_union", "asd_union")
    bins: dict[str, list[float]] = {c: [] for c in cols}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for c in cols:
                raw = row.get(c)
                if raw in (None, ""):
                    continue
                try:
                    bins[c].append(float(raw))
                except ValueError:
                    continue
    dsc = _mean(bins["dice_union"])
    jac = _mean(bins["jaccard_union"])
    hd95 = _mean(bins["hd95_union"])
    asd = _mean(bins["asd_union"])
    out: dict[str, float] = {}
    if dsc is not None:
        out.update({"DSC_all": dsc, "DSC_FH": dsc})
    if jac is not None:
        out.update({"Jacc_all": jac, "Jacc_FH": jac})
    if hd95 is not None:
        out.update({"HD95_all": hd95, "HD95_FH": hd95})
    if asd is not None:
        out.update({"ASD_all": asd, "ASD_FH": asd})
    return out


_DATASET_ORDER = {"BUSI": 0, "TN3K": 1, "HC18": 2, "PSFH": 3}


def dataset_sort_key(name: str) -> tuple[int, str]:
    return (_DATASET_ORDER.get(name, 99), name)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def collect_rows() -> list[tuple[str, int, str, dict[str, float]]]:
    rows: list[tuple[str, int, str, dict[str, float]]] = []
    if not RESULTS_ROOT.is_dir():
        print(f"[ERROR] missing results root: {RESULTS_ROOT}", file=sys.stderr)
        return rows

    for ckpt_dir in sorted(RESULTS_ROOT.iterdir()):
        manifest = ckpt_dir / "test" / "test_eval_manifest.json"
        if not manifest.is_file():
            continue
        manifest_data = load_manifest(manifest)
        if manifest_data is None:
            print(f"[WARN] unreadable manifest: {manifest}", file=sys.stderr)
            continue
        dataset = str(manifest_data.get("dataset") or
                      manifest_data.get("args", {}).get("dataset") or "?")
        exp_id = manifest_data.get("expID") or manifest_data.get("args", {}).get("expID")
        ckpt_name = manifest_data.get("ckpt_name") or ckpt_dir.name
        try:
            exp_id = int(exp_id)
        except (TypeError, ValueError):
            exp_id = -1

        log_path = LOG_ROOT / f"{ckpt_dir.name}_test.log"
        metrics = last_valid_result(log_path)
        if metrics is None:
            print(f"[WARN] no Valid Result line in {log_path}", file=sys.stderr)
            continue

        # HC18: override boundary-band aggregates from the log with the
        # filled-region values from the per-case CSV (paper protocol).
        if str(dataset).upper() == "HC18":
            csv_path = ckpt_dir / "test" / "test_case_metrics.csv"
            metrics.update(hc18_filled_aggregates(csv_path))

        rows.append((dataset, exp_id, ckpt_name, metrics))

    rows.sort(key=lambda r: (dataset_sort_key(r[0]), r[1], r[2]))
    return rows


def build_row(dataset: str, exp_id: int, ckpt_name: str,
              metrics: dict[str, float]) -> str:
    cells = [dataset, str(exp_id), ckpt_name]
    for key in ROW_KEYS:
        cells.append(fmt(metrics.get(key)))
    return "| " + " | ".join(cells) + " |\n"


def main() -> int:
    rows = collect_rows()
    if not rows:
        print("[ERROR] no rows to write", file=sys.stderr)
        return 1
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        fh.write(HEADER)
        for dataset, exp_id, ckpt_name, metrics in rows:
            fh.write(build_row(dataset, exp_id, ckpt_name, metrics))
    print(f"[OK] wrote {len(rows)} rows to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
