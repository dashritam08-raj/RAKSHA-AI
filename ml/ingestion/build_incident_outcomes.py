"""Build a continuous observed-impact target and documented impact bands.

This script converts observed disaster consequences into an impact target for ML
DEVELOPMENT. It does not estimate future risk. Band thresholds can be supplied
as either:
  1) a dict with quantiles: low_max_quantile/moderate_max_quantile/high_max_quantile
  2) a list of {name, min_score} entries, interpreted as absolute score floors.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = {
    "score_version": "v2-log-scaled-relative-bands",
    "bands": {"low_max_quantile": 0.50, "moderate_max_quantile": 0.75, "high_max_quantile": 0.90},
    "weights": {
        "deaths": 40,
        "missing": 20,
        "injured": 10,
        "directly_affected": 10,
        "evacuated": 8,
        "relocated": 4,
        "houses_destroyed": 5,
        "houses_damaged": 2,
        "road_damage_m": 1,
    },
    "caps": {
        "deaths": 25,
        "missing": 20,
        "injured": 100,
        "directly_affected": 10000,
        "evacuated": 5000,
        "relocated": 5000,
        "houses_destroyed": 2000,
        "houses_damaged": 5000,
        "road_damage_m": 10000,
    },
}


def scaled_log(value, cap):
    value = max(float(value or 0), 0.0)
    cap = max(float(cap), 1.0)
    if value <= 0:
        return 0.0
    return min(np.log1p(value) / np.log1p(cap), 1.0) * 100.0


def impact_score(row, policy):
    total = 0.0
    weight = 0.0
    evidence = 0
    for col, w in policy["weights"].items():
        value = float(row.get(col, 0) or 0)
        if value > 0:
            evidence += 1
        total += scaled_log(value, policy["caps"].get(col, 1)) * float(w)
        weight += float(w)
    return (total / weight if weight else 0.0), evidence


def assign_relative_band(v, evidence, q50, q75, q90):
    if evidence <= 0:
        return "NO_DATA"
    if v >= q90:
        return "CRITICAL"
    if v >= q75:
        return "HIGH"
    if v >= q50:
        return "MODERATE"
    return "LOW"


def assign_absolute_band(v, evidence, bands):
    if evidence <= 0:
        return "NO_DATA"
    # Highest threshold first. This keeps a list such as
    # CRITICAL >= 80, HIGH >= 55, MODERATE >= 30, LOW >= 0 intuitive.
    ordered = sorted(
        bands,
        key=lambda x: float(x.get("min_score", 0)),
        reverse=True,
    )
    for band in ordered:
        if v >= float(band.get("min_score", 0)):
            return str(band.get("name", "LOW")).upper()
    return "LOW"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--policy", default="label_policy.json")
    return p.parse_args()


def main():
    a = parse_args()
    policy_path = Path(a.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else DEFAULT

    # Basic policy validation and compatibility with both Phase-5 policy formats.
    if not isinstance(policy.get("weights"), dict):
        raise SystemExit("Policy 'weights' must be an object/map.")
    if not isinstance(policy.get("caps"), dict):
        raise SystemExit("Policy 'caps' must be an object/map.")
    bands_cfg = policy.get("bands", DEFAULT["bands"])
    if not isinstance(bands_cfg, (dict, list)):
        raise SystemExit("Policy 'bands' must be either an object (quantiles) or a list of {name,min_score} entries.")

    df = pd.read_csv(a.input, low_memory=False)
    if "observed_at" not in df.columns:
        raise SystemExit("Input is missing observed_at")

    df["observed_at_raw"] = df["observed_at"].astype(str)
    parsed = pd.to_datetime(df["observed_at"], errors="coerce", utc=True)
    df["observed_at"] = parsed.dt.strftime("%Y-%m-%d")

    numeric_cols = list(policy["weights"])
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    scores = []
    evidence_counts = []
    for _, row in df.iterrows():
        s, e = impact_score(row, policy)
        scores.append(round(s, 3))
        evidence_counts.append(e)

    df["observed_impact_score"] = scores
    df["label_evidence_count"] = evidence_counts
    df["label_status"] = np.where(
        df["label_evidence_count"] > 0,
        "LABELED",
        "NO_IMPACT_EVIDENCE",
    )

    valid_scores = df.loc[df["label_evidence_count"] > 0, "observed_impact_score"]

    if isinstance(bands_cfg, list):
        if not all(isinstance(x, dict) and "name" in x and "min_score" in x for x in bands_cfg):
            raise SystemExit("Each policy band must contain 'name' and 'min_score'.")
        df["observed_impact_band"] = [
            assign_absolute_band(s, e, bands_cfg)
            for s, e in zip(df["observed_impact_score"], df["label_evidence_count"])
        ]
        thresholds = {
            str(x["name"]).upper(): float(x["min_score"])
            for x in bands_cfg
        }
        band_method = "absolute_min_score"
    else:
        try:
            q50 = float(valid_scores.quantile(float(bands_cfg["low_max_quantile"]))) if len(valid_scores) else 0.0
            q75 = float(valid_scores.quantile(float(bands_cfg["moderate_max_quantile"]))) if len(valid_scores) else 0.0
            q90 = float(valid_scores.quantile(float(bands_cfg["high_max_quantile"]))) if len(valid_scores) else 0.0
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid quantile band policy: {exc}")
        df["observed_impact_band"] = [
            assign_relative_band(s, e, q50, q75, q90)
            for s, e in zip(df["observed_impact_score"], df["label_evidence_count"])
        ]
        thresholds = {"q50": round(q50, 3), "q75": round(q75, 3), "q90": round(q90, 3)}
        band_method = "dataset_relative_quantiles"

    # Backwards-compatible aliases for older inspection/training helpers.
    df["impact_score"] = df["observed_impact_score"]
    df["impact_band"] = df["observed_impact_band"]

    df["label_policy_version"] = policy.get("score_version", "v2-log-scaled-relative-bands")
    if band_method == "dataset_relative_quantiles":
        df["band_q50"] = thresholds["q50"]
        df["band_q75"] = thresholds["q75"]
        df["band_q90"] = thresholds["q90"]
    else:
        df["band_q50"] = np.nan
        df["band_q75"] = np.nan
        df["band_q90"] = np.nan

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    result = {
        "output": str(out),
        "rows": int(len(df)),
        "labeled_rows": int((df["label_evidence_count"] > 0).sum()),
        "unlabeled_rows": int((df["label_evidence_count"] == 0).sum()),
        "invalid_observed_at": int(parsed.isna().sum()),
        "bands": {str(k): int(v) for k, v in df["observed_impact_band"].value_counts().to_dict().items()},
        "band_method": band_method,
        "band_thresholds": thresholds,
        "warning": "Impact bands describe observed consequences in the source dataset. They are not future-risk probabilities or universal emergency thresholds. Human/data validation is required before operational ML use.",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
