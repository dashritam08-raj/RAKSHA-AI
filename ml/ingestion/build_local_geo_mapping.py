"""Map DesInventar region/district/location rows to a local Nepal district gazetteer.

This avoids bulk geocoding against public geocoders. Coordinates are district-level reference
points from a cited open Nepal administrative dataset; they are NOT exact incident coordinates.
Unmatched or ambiguous rows remain REVIEW_REQUIRED.
"""
from __future__ import annotations
import argparse, csv, json, re, unicodedata
from pathlib import Path
import pandas as pd

ALIASES = {
    "chitawan": "chitawan",
    "chitwan": "chitawan",
    "terathum": "terhathum",
    "tehrathum": "terhathum",
    "okhaldhunga": "okhaldhunga",
    "sankhuwasabha": "sankhuwasabha",
    "kavre": "kavrepalanchok",
    "kavrepalanchowk": "kavrepalanchok",
    "kapilvastu": "kapilbastu",
    "kapilbastu": "kapilbastu",
    "nawalparasi": "__AMBIGUOUS_NAWALPARASI__",
    "rukum": "__AMBIGUOUS_RUKUM__",
}

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(district|dist|jilla)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def canonical_key(s: str) -> str:
    n = norm(s)
    return ALIASES.get(n, n)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--gazetteer",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--locations-output",required=True)
    a=p.parse_args()
    src=pd.read_csv(a.input,low_memory=False)
    gaz=pd.read_csv(a.gazetteer,low_memory=False)
    required={"source_region","source_district","source_location","location_id"}
    miss=required-set(src.columns)
    if miss: raise SystemExit("Mapping input missing: "+", ".join(sorted(miss)))
    for c in ["source_region","source_district","source_location"]: src[c]=src[c].fillna("").astype(str).str.strip()
    gaz["district_key"]=gaz["district_en"].map(canonical_key)
    gaz["province_key"]=gaz["province_en"].map(norm)
    exact={}
    for row in gaz.to_dict("records"): exact[(row["district_key"],row["province_key"])]=row
    district_only={}
    for row in gaz.to_dict("records"):
        district_only.setdefault(row["district_key"],[]).append(row)
    out=[]; matched=amb=unmatched=0
    for r in src.to_dict("records"):
        dkey=canonical_key(r["source_district"])
        pkey=norm(r["source_region"])
        hit=None; status="REVIEW_REQUIRED"; method=""
        if dkey.startswith("__AMBIGUOUS"):
            status="AMBIGUOUS_REVIEW_REQUIRED"
        elif (dkey,pkey) in exact:
            hit=exact[(dkey,pkey)]; status="AUTO_MAPPED_LOCAL_GAZETTEER"; method="district+province_exact"
        elif dkey in district_only and len(district_only[dkey])==1:
            hit=district_only[dkey][0]; status="AUTO_MAPPED_LOCAL_GAZETTEER"; method="district_exact"
        else:
            loc_key=canonical_key(r["source_location"])
            if loc_key in district_only and len(district_only[loc_key])==1:
                hit=district_only[loc_key][0]; status="AUTO_MAPPED_LOCAL_GAZETTEER"; method="location_equals_district"
        row=dict(r)
        if hit:
            row.update({"latitude":float(hit["latitude"]),"longitude":float(hit["longitude"]),"gazetteer_district":hit["district_en"],"gazetteer_province":hit["province_en"],"coordinate_level":"DISTRICT_REFERENCE_POINT","coordinate_source":hit["source_name"],"coordinate_source_url":hit["source_url"],"license":hit["license"]}); matched+=1
        else:
            row.update({"latitude":"","longitude":"","gazetteer_district":"","gazetteer_province":"","coordinate_level":"","coordinate_source":"","coordinate_source_url":"","license":""})
            if status.startswith("AMBIGUOUS"): amb+=1
            else: unmatched+=1
        row["review_status"]=status; row["mapping_method"]=method
        out.append(row)
    out_df=pd.DataFrame(out); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out_df.to_csv(a.output,index=False)
    approved=out_df[out_df["review_status"]=="AUTO_MAPPED_LOCAL_GAZETTEER"].copy()
    if not approved.empty:
        loc=(approved[["location_id","gazetteer_district","gazetteer_province","latitude","longitude","coordinate_source","coordinate_source_url","license"]]
             .drop_duplicates("location_id").rename(columns={"gazetteer_district":"district_en","gazetteer_province":"province_en"}))
        Path(a.locations_output).parent.mkdir(parents=True,exist_ok=True); loc.to_csv(a.locations_output,index=False)
    else:
        pd.DataFrame(columns=["location_id","district_en","province_en","latitude","longitude","coordinate_source","coordinate_source_url","license"]).to_csv(a.locations_output,index=False)
    report={"mapping_output":a.output,"locations_output":a.locations_output,"input_rows":len(src),"matched_rows":matched,"ambiguous_rows":amb,"unmatched_rows":unmatched,"unique_locations_for_weather":int(pd.read_csv(a.locations_output).shape[0]),"warning":"Coordinates are district-level reference points. They must not be presented as exact incident coordinates. Unmatched/ambiguous locations require review."}
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
