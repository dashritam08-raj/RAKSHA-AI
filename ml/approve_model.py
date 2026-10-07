"""Mark a candidate model as validated after external review.

This does not perform validation itself. It records an explicit human approval
in the model bundle so the API can distinguish validated models from demos.

Usage:
    python approve_model.py models/risk_model.joblib "Reviewer Name" "Evidence/validation reference"
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

if len(sys.argv) != 4:
    raise SystemExit("Usage: python approve_model.py model.joblib reviewer validation_reference")

path = Path(sys.argv[1])
bundle = joblib.load(path)
if not isinstance(bundle, dict) or "model" not in bundle:
    raise SystemExit("Model bundle format not recognized.")
metadata = bundle.setdefault("metadata", {})
metadata["validated_at"] = datetime.now(timezone.utc).isoformat()
metadata["validated_by"] = sys.argv[2]
metadata["validation_reference"] = sys.argv[3]
bundle["status"] = "validated"
joblib.dump(bundle, path)
print(json.dumps({"saved_to": str(path), "status": "validated", "validated_by": sys.argv[2]}, indent=2))
