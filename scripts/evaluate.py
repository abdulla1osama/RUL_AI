"""LOCKED official evaluator for the MKT3432 RUL project. PROTECTED: do not modify.

Usage
-----
Integrity / leakage check:
    python scripts/evaluate.py --check-split data/servo_rul_v1/split_manifest.json

Score predictions on the held-out test units:
    python scripts/evaluate.py \
        --predictions runs/student_predictions.csv \
        --split data/servo_rul_v1/split_manifest.json \
        --out runs/student_results.summary.json [--subset smoke] [--clusters clusters.csv]

Predictions CSV columns: unit_id, cycle, rul_pred.
Optional clusters CSV columns: unit_id, cycle, cluster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from servo_rul.data.loader import (
    content_hash,
    load_manifest,
    load_public,
    load_test_truth,
)
from servo_rul.eval.official_metrics import clustering_metrics, rul_metrics

SMOKE_N_UNITS = 3


def check_split(split_path: Path) -> int:
    manifest = load_manifest(split_path)
    splits = manifest["splits"]
    sets = {k: set(v) for k, v in splits.items()}
    # disjointness
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    ok = True
    for a, b in pairs:
        inter = sets[a] & sets[b]
        if inter:
            ok = False
            print(f"[check] LEAKAGE: {a} and {b} share unit_ids {sorted(inter)}")
    public = load_public(split_path)
    recomputed = content_hash(public)
    recorded = manifest["units_public_content_sha256"]
    hash_ok = recomputed == recorded
    print(f"[check] units: train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])}")
    print(f"[check] split disjoint: {ok}")
    print(f"[check] content hash match: {hash_ok}")
    if not hash_ok:
        print(f"[check]   recorded={recorded}")
        print(f"[check]   recomputed={recomputed}")
    return 0 if (ok and hash_ok) else 1


def score(split_path: Path, predictions: Path, out: Path, subset: str | None, clusters: Path | None) -> int:
    manifest = load_manifest(split_path)
    rul_cap = int(manifest["rul_cap"])
    truth = load_test_truth(split_path)

    if subset == "smoke":
        keep = sorted(truth["unit_id"].unique())[:SMOKE_N_UNITS]
        truth = truth[truth["unit_id"].isin(keep)].reset_index(drop=True)

    pred = pd.read_csv(predictions)
    result = rul_metrics(pred, truth, rul_cap)

    if clusters is not None:
        cdf = pd.read_csv(clusters)
        merged = truth.merge(cdf, on=["unit_id", "cycle"], how="inner")
        if len(merged):
            result["clustering"] = clustering_metrics(
                merged["cluster"].to_numpy(), merged["hidden_stage"].to_numpy()
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Official RUL evaluator (locked).")
    p.add_argument("--check-split", type=Path, default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/results.summary.json"))
    p.add_argument("--subset", choices=["smoke"], default=None)
    p.add_argument("--clusters", type=Path, default=None)
    args = p.parse_args()

    if args.check_split is not None:
        return check_split(args.check_split)
    if args.predictions is None or args.split is None:
        p.error("provide --check-split, or both --predictions and --split")
    return score(args.split, args.predictions, args.out, args.subset, args.clusters)


if __name__ == "__main__":
    sys.exit(main())
