"""Fail if a number in README.md no longer matches results/."""
from __future__ import annotations

import collections
import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ol = list(csv.DictReader((ROOT / "results" / "open-loop.csv").open()))
    summ = list(csv.DictReader((ROOT / "results" / "summary.csv").open()))
    body = (ROOT / "README.md").read_text()
    # Detail moved out of the README lives in notes/METHODS.md. A figure quoted
    # there is still a quoted figure and still has to match its source.
    _methods = ROOT / "notes" / "METHODS.md"
    if _methods.exists():
        body += "\n" + _methods.read_text()
    claims, failures = [], []

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ol:
        by[r["mode"]][int(r["step"])].append(float(r["reward_mae"]))
    for mode, steps in by.items():
        for k in (1, 5, 10, 15, 20, 40):
            if k in steps:
                claims.append((f"open-loop {mode} k={k}",
                               f"{statistics.median(steps[k]):.4f}"))
    # Seed level values only where the README argues from them. It quotes all
    # three seeds at k=1 for recon and no-recon (the non-overlap claim), the
    # crossing pair at k=5, and the spans at k=10. Contrastive seed values are
    # never quoted, so demanding them would fail on numbers no one claimed.
    recon, norecon = "recon (Dreamer style)", "no-recon (MuZero style)"
    for v in sorted(by[recon][1]):
        claims.append((f"seed {recon} k=1", f"{v:.4f}"))
    for v in sorted(by[norecon][1]):
        claims.append((f"seed {norecon} k=1", f"{v:.4f}"))
    claims.append((f"worst {recon} k=5", f"{max(by[recon][5]):.4f}"))
    claims.append((f"best {norecon} k=5", f"{min(by[norecon][5]):.4f}"))
    for mode in (recon, norecon):
        claims.append((f"min {mode} k=10", f"{min(by[mode][10]):.4f}"))
        claims.append((f"max {mode} k=10", f"{max(by[mode][10]):.4f}"))

    rets = collections.defaultdict(list)
    for r in summ:
        rets[r["mode"]].append(float(r["final_return"]))
    for mode, v in rets.items():
        claims.append((f"return {mode}", f"{abs(statistics.median(v)):.2f}"))
        claims.append((f"return-min {mode}", f"{abs(min(v)):.1f}"))
        claims.append((f"return-max {mode}", f"{abs(max(v)):.1f}"))

    for label, text in claims:
        if not re.search(r"(?<![\d.])" + re.escape(text) + r"(?!\d)", body):
            failures.append(f"{label} should read {text}, not found")

    print(f"checked {len(claims)} quoted figures against results/")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures[:12]:
            print(f"  - {f}")
        if len(failures) > 12:
            print(f"  ... and {len(failures) - 12} more")
        return 1
    print("no drift")
    # What this does and does not cover, so the green line is not read as more
    # than it is: each figure is recomputed from results/ and looked for in the
    # prose. It cannot catch a wrong number that happens to appear somewhere,
    # it does not check claims written in words (ratios, multiples, ranges),
    # and it does not read notes/LOGBOOK.md.
    print("this checks quoted figures against results/, not claims written in words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
