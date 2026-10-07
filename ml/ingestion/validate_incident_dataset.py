"""Validate a normalized historical disaster-event CSV before ML training."""
import argparse, json, sys
from pathlib import Path
import pandas as pd

IMPACT_COLS = ["deaths","missing","injured","directly_affected","indirectly_affected","evacuated","relocated","houses_destroyed","houses_damaged","road_damage_m","economic_loss_usd","economic_loss_local"]
REQUIRED = ["incident_id","observed_at","disaster_type","country_code","region","district","location"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=False); a=p.parse_args()
    df=pd.read_csv(a.input,low_memory=False)
    missing_required=[c for c in REQUIRED if c not in df.columns]
    parsed=pd.to_datetime(df.get("observed_at"),errors="coerce",utc=True) if "observed_at" in df.columns else pd.Series([pd.NaT]*len(df))
    numeric={}
    for c in IMPACT_COLS:
        if c in df.columns:
            s=pd.to_numeric(df[c],errors="coerce")
            numeric[c]={"missing":int(s.isna().sum()),"nonzero":int((s.fillna(0)>0).sum()),"max":float(s.max()) if s.notna().any() else None}
    dup_id=int(df["incident_id"].duplicated().sum()) if "incident_id" in df.columns else None
    labeled_mask=pd.Series(False,index=df.index)
    if any(c in df.columns for c in IMPACT_COLS):
        for c in IMPACT_COLS:
            if c in df.columns:
                labeled_mask |= pd.to_numeric(df[c],errors="coerce").fillna(0)>0
    result={
      "rows":int(len(df)),"columns":list(df.columns),"missing_required":missing_required,
      "invalid_observed_at":int(parsed.isna().sum()),"duplicate_incident_id":dup_id,
      "labeled_rows_by_positive_impact":int(labeled_mask.sum()),
      "disaster_type_counts":df["disaster_type"].fillna("UNKNOWN").astype(str).value_counts().head(25).to_dict() if "disaster_type" in df.columns else {},
      "impact_columns":numeric,
    }
    print(json.dumps(result,indent=2))
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    return 1 if missing_required else 0
if __name__=="__main__": sys.exit(main())
