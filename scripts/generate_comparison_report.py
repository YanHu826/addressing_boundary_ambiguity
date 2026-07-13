#!/usr/bin/env python3
"""
generate_comparison_report.py

Reads outputs/summary/test_results.md and generates a human-readable
comparison report (outputs/summary/comparison_report.md) that follows
the brace-experiment-results skill format:

  - BUSI & TN3K: Dice / IoU vs Shape Prior + Ours (论文)
  - HC18: DSC / Jaccard / HD95 / ASD vs BiPCC + Ours (论文)
  - PSFH: DSC(PS/FH) / Jaccard / HD95 / ASD vs BiPCC + Ours (论文)

Run standalone:  python3 scripts/generate_comparison_report.py [test_results.md] [out.md]
"""

import sys
import os
import csv
import io
import math
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# Baseline data (from brace-experiment-results skill)
# ─────────────────────────────────────────────────────────────────────────────

# BUSI / TN3K  –  exp1=1/8, exp2=1/4, exp3=1/2
BUSI_TN3K_BASELINES = {
    "Shape Prior": {
        "BUSI": {3: (80.41, 72.02), 2: (79.43, 71.17), 1: (75.19, 66.90)},
        "TN3K": {3: (84.17, 75.70), 2: (83.24, 74.47), 1: (82.21, 73.16)},
    },
    "Ours (论文)": {
        "BUSI": {3: (80.40, 71.38), 2: (80.88, 72.33), 1: (77.58, 69.11)},
        "TN3K": {3: (84.56, 75.30), 2: (84.05, 74.78), 1: (82.41, 72.83)},
    },
}

# HC18  –  exp1=10%, exp2=20%
HC18_BASELINES = {
    "BiPCC": {
        1: (83.32, 79.14, 1.94, 2.85),
        2: (89.75, 84.44, 1.31, 1.87),
    },
    "Ours (论文)": {
        1: (84.46, 80.46, 1.72, 2.32),
        2: (90.12, 85.21, 1.06, 1.56),
    },
}

# PSFH  –  exp1=10%, exp2=20%
# Each entry: (DSC_PS, Jacc_PS, HD95_PS, ASD_PS, DSC_FH, Jacc_FH, HD95_FH, ASD_FH)
PSFH_BASELINES = {
    "BiPCC": {
        1: (78.46, 73.80, 5.66, 2.77, 84.73, 81.03, 4.34, 1.89),
        2: (85.51, 81.95, 3.10, 0.57, 93.57, 90.12, 1.96, 0.33),
    },
    "Ours (论文)": {
        1: (79.39, 74.11, 4.78, 2.59, 85.23, 82.67, 4.17, 1.56),
        2: (87.54, 83.92, 2.38, 0.49, 94.41, 91.82, 1.08, 0.30),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_HEADERS = [
    "dataset", "expid", "ckpt_name",
    "recall", "specificity", "precision", "F1", "F2",
    "ACC_overall", "IoU_poly", "IoU_bg", "IoU_mean", "dice",
    "DSC_all", "Jacc_all", "HD95_all", "ASD_all",
    "DSC_PS", "Jacc_PS", "HD95_PS", "ASD_PS",
    "DSC_FH", "Jacc_FH", "HD95_FH", "ASD_FH",
]


def parse_md_table(path):
    """Return list-of-dicts from a markdown table file. Missing values -> None."""
    rows = []
    headers = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if headers is None:
                headers = [h.strip().lstrip(" ").rstrip(" ") for h in cells]
                continue
            # separator row
            if all(set(c).issubset({"-", " ", ":"}) for c in cells):
                continue
            if len(cells) != len(headers):
                continue
            row = {}
            for h, v in zip(headers, cells):
                row[h] = v if v != "-" else None
            rows.append(row)
    return rows


def f(v, scale=100.0, decimals=2):
    """Format a raw float value; scale (e.g. ×100 for percentage). Returns '???' if None."""
    if v is None:
        return "???"
    try:
        return f"{float(v) * scale:.{decimals}f}"
    except (ValueError, TypeError):
        return "???"


def fraw(v, decimals=2):
    """Format already-scaled value."""
    if v is None:
        return "???"
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return "???"


def delta(exp_str, paper_val):
    """Return Δ string (exp - paper). Both are already-scaled floats-as-string."""
    if exp_str == "???":
        return "---"
    try:
        d = float(exp_str) - float(paper_val)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}"
    except (ValueError, TypeError):
        return "---"


def best_row_for(rows, dataset, expid, variant="full"):
    """Pick most recent matching row; prefer ckpt_name containing variant."""
    candidates = [
        r for r in rows
        if r.get("dataset", "").upper() == dataset.upper()
        and str(r.get("expid", "")).strip() == str(expid)
    ]
    # prefer rows whose ckpt_name contains variant
    preferred = [r for r in candidates if variant in r.get("ckpt_name", "")]
    pool = preferred if preferred else candidates
    return pool[-1] if pool else None  # latest entry


# ─────────────────────────────────────────────────────────────────────────────
# Report sections
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 160


def section_busi_tn3k(rows):
    lines = []
    lines.append("## BUSI & TN3K  (Dice / IoU, %)")
    lines.append("")

    # Collect our experimental values
    our_exp = {}
    for ds in ("BUSI", "TN3K"):
        for exp in (1, 2, 3):
            r = best_row_for(rows, ds, exp)
            if r:
                dice_pct = f(r.get("dice"), scale=100.0)
                iou_pct  = f(r.get("IoU_poly"), scale=100.0)
            else:
                dice_pct = iou_pct = "???"
            our_exp[(ds, exp)] = (dice_pct, iou_pct)

    # Header
    hdr = (
        f"{'Method':<18} | {'BUSI 1/2':^15} | {'BUSI 1/4':^15} | {'BUSI 1/8':^15}"
        f" | {'TN3K 1/2':^15} | {'TN3K 1/4':^15} | {'TN3K 1/8':^15}"
    )
    sub = (
        f"{'':18} | {'Dice':>6}  {'IoU':>6}  | {'Dice':>6}  {'IoU':>6}  | {'Dice':>6}  {'IoU':>6}"
        f"  | {'Dice':>6}  {'IoU':>6}  | {'Dice':>6}  {'IoU':>6}  | {'Dice':>6}  {'IoU':>6}"
    )
    sep_row = SEP[:len(hdr)]

    lines.append(hdr)
    lines.append(sub)
    lines.append(sep_row)

    def row_line(method, data):
        # data: {(ds, exp): (dice_str, iou_str)}
        cols = []
        for ds in ("BUSI", "TN3K"):
            for exp in (3, 2, 1):
                d, i = data.get((ds, exp), ("???", "???"))
                cols.append(f"{d:>6}  {i:>6}")
        return f"{method:<18} | " + "  | ".join(cols)

    # Baselines
    for method, data in BUSI_TN3K_BASELINES.items():
        bdata = {}
        for ds in ("BUSI", "TN3K"):
            for exp in (1, 2, 3):
                dice_v, iou_v = data[ds][exp]
                bdata[(ds, exp)] = (fraw(dice_v), fraw(iou_v))
        lines.append(row_line(method, bdata))

    lines.append(sep_row)

    # Our experimental
    exp_line = row_line("Ours (实验)", our_exp)
    lines.append(exp_line)

    lines.append(sep_row)

    # Delta vs 论文
    paper_data = BUSI_TN3K_BASELINES["Ours (论文)"]
    delta_vals = {}
    for ds in ("BUSI", "TN3K"):
        for exp in (1, 2, 3):
            paper_dice, paper_iou = paper_data[ds][exp]
            exp_dice, exp_iou = our_exp[(ds, exp)]
            delta_vals[(ds, exp)] = (
                delta(exp_dice, paper_dice),
                delta(exp_iou, paper_iou),
            )
    lines.append(row_line("Δ(实验-论文)", delta_vals))
    lines.append("")
    return lines


def section_hc18(rows):
    lines = []
    lines.append("## HC18  (DSC% / Jaccard% / HD95 mm / ASD mm)")
    lines.append("")

    our_exp = {}
    for exp in (1, 2):
        r = best_row_for(rows, "HC18", exp)
        if r:
            dsc  = f(r.get("DSC_all"),  scale=100.0)
            jacc = f(r.get("Jacc_all"), scale=100.0)
            hd95 = fraw(r.get("HD95_all"))
            asd  = fraw(r.get("ASD_all"))
        else:
            dsc = jacc = hd95 = asd = "???"
        our_exp[exp] = (dsc, jacc, hd95, asd)

    hdr = (
        f"{'Method':<18} | {'10% labeled':^43} | {'20% labeled':^43}"
    )
    sub = (
        f"{'':18} | {'DSC%':>7} {'Jacc%':>7} {'HD95':>7} {'ASD':>7}   "
        f" | {'DSC%':>7} {'Jacc%':>7} {'HD95':>7} {'ASD':>7}"
    )
    sep_row = "─" * len(hdr)

    lines.append(hdr)
    lines.append(sub)
    lines.append(sep_row)

    def row_line(method, data):
        cols = []
        for exp in (1, 2):
            d, j, h, a = data.get(exp, ("???", "???", "???", "???"))
            cols.append(f"{d:>7} {j:>7} {h:>7} {a:>7}")
        return f"{method:<18} | " + "   | ".join(cols)

    for method, data in HC18_BASELINES.items():
        bdata = {}
        for exp in (1, 2):
            vals = data[exp]
            bdata[exp] = tuple(fraw(v) for v in vals)
        lines.append(row_line(method, bdata))

    lines.append(sep_row)
    lines.append(row_line("Ours (实验)", our_exp))
    lines.append(sep_row)

    # Delta vs 论文
    paper_data = HC18_BASELINES["Ours (论文)"]
    delta_vals = {}
    for exp in (1, 2):
        p = paper_data[exp]
        e = our_exp[exp]
        delta_vals[exp] = (
            delta(e[0], p[0]),
            delta(e[1], p[1]),
            delta(e[2], p[2]),
            delta(e[3], p[3]),
        )
    lines.append(row_line("Δ(实验-论文)", delta_vals))
    lines.append("")
    return lines


def section_psfh(rows):
    lines = []
    lines.append("## PSFH  (DSC% / Jaccard% / HD95 mm / ASD mm,  format: PS / FH)")
    lines.append("")

    our_exp = {}
    for exp in (1, 2):
        r = best_row_for(rows, "PSFH", exp)
        if r:
            dsc_ps  = f(r.get("DSC_PS"),  scale=100.0)
            jacc_ps = f(r.get("Jacc_PS"), scale=100.0)
            hd95_ps = fraw(r.get("HD95_PS"))
            asd_ps  = fraw(r.get("ASD_PS"))
            dsc_fh  = f(r.get("DSC_FH"),  scale=100.0)
            jacc_fh = f(r.get("Jacc_FH"), scale=100.0)
            hd95_fh = fraw(r.get("HD95_FH"))
            asd_fh  = fraw(r.get("ASD_FH"))
        else:
            dsc_ps = jacc_ps = hd95_ps = asd_ps = "???"
            dsc_fh = jacc_fh = hd95_fh = asd_fh = "???"
        our_exp[exp] = (dsc_ps, jacc_ps, hd95_ps, asd_ps, dsc_fh, jacc_fh, hd95_fh, asd_fh)

    hdr = (
        f"{'Method':<18}"
        f" | {'10%: DSC(PS/FH)':^15}"
        f" | {'Jacc(PS/FH)':^15}"
        f" | {'HD95(PS/FH)':^15}"
        f" | {'ASD(PS/FH)':^15}"
        f" | {'20%: DSC(PS/FH)':^15}"
        f" | {'Jacc(PS/FH)':^15}"
        f" | {'HD95(PS/FH)':^15}"
        f" | {'ASD(PS/FH)':^15}"
    )
    sep_row = "─" * len(hdr)

    lines.append(hdr)
    lines.append(sep_row)

    def fmt_pair(a, b):
        return f"{a}/{b}"

    def row_line(method, data):
        cols = []
        for exp in (1, 2):
            v = data.get(exp, ("???",) * 8)
            cols.append(fmt_pair(v[0], v[4]))   # DSC
            cols.append(fmt_pair(v[1], v[5]))   # Jacc
            cols.append(fmt_pair(v[2], v[6]))   # HD95
            cols.append(fmt_pair(v[3], v[7]))   # ASD
        return f"{method:<18} | " + " | ".join(f"{c:^15}" for c in cols)

    for method, data in PSFH_BASELINES.items():
        bdata = {}
        for exp in (1, 2):
            v = data[exp]
            bdata[exp] = tuple(fraw(x) for x in v)
        lines.append(row_line(method, bdata))

    lines.append(sep_row)
    lines.append(row_line("Ours (实验)", our_exp))
    lines.append(sep_row)

    # Delta vs 论文
    paper_data = PSFH_BASELINES["Ours (论文)"]
    delta_vals = {}
    for exp in (1, 2):
        p = paper_data[exp]
        e = our_exp[exp]
        delta_vals[exp] = tuple(delta(e[i], p[i]) for i in range(8))
    lines.append(row_line("Δ(实验-论文)", delta_vals))
    lines.append("")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    test_results_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "outputs", "summary", "test_results.md"
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(test_results_path), "comparison_report.md"
    )

    test_results_path = os.path.abspath(test_results_path)
    out_path = os.path.abspath(out_path)

    if not os.path.isfile(test_results_path):
        print(f"[ERROR] test_results.md not found: {test_results_path}", file=sys.stderr)
        sys.exit(1)

    rows = parse_md_table(test_results_path)
    print(f"[INFO] Parsed {len(rows)} test result row(s) from {test_results_path}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report_lines = []
    report_lines.append(f"# BRACE Experiment Comparison Report")
    report_lines.append(f"")
    report_lines.append(f"Generated: {ts}")
    report_lines.append(f"Source:    {test_results_path}")
    report_lines.append(f"")
    report_lines.append(f"Notation:  ??? = not yet tested  |  --- = cannot compute delta")
    report_lines.append(f"           BUSI exp1=1/8  exp2=1/4  exp3=1/2")
    report_lines.append(f"           TN3K exp1=1/8  exp2=1/4  exp3=1/2")
    report_lines.append(f"           HC18/PSFH exp1=10%  exp2=20%")
    report_lines.append(f"")
    report_lines.append("---")
    report_lines.append("")

    report_lines += section_busi_tn3k(rows)
    report_lines += section_hc18(rows)
    report_lines += section_psfh(rows)

    # Raw table summary
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Raw Results (all rows from test_results.md)")
    report_lines.append("")

    if rows:
        cols = list(rows[0].keys())
        report_lines.append("| " + " | ".join(cols) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in rows:
            vals = [r.get(c) or "-" for c in cols]
            report_lines.append("| " + " | ".join(vals) + " |")
    else:
        report_lines.append("*(no rows)*")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[INFO] Report written to: {out_path}")


if __name__ == "__main__":
    main()
