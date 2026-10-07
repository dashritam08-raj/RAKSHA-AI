"""Build a leakage-aware training table from historical disaster outcomes and environment.

Uses only environmental observations on or BEFORE the incident date. Post-event
severity/people-affected fields are deliberately excluded from model features.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--incidents',required=True); p.add_argument('--mapping',required=True); p.add_argument('--weather',required=True); p.add_argument('--earthquakes',default=None); p.add_argument('--output',required=True); p.add_argument('--weather-gap-days',type=int,default=2); args=p.parse_args()
    inc=pd.read_csv(args.incidents,low_memory=False); mp=pd.read_csv(args.mapping,low_memory=False); wt=pd.read_csv(args.weather,low_memory=False)
    need_i=['incident_id','observed_at','region','district','disaster_type','observed_impact_score']; need_m=['source_region','source_district','location_id','latitude','longitude']; need_w=['location_id','observed_at']
    for df,cols,name in [(inc,need_i,'incidents'),(mp,need_m,'mapping'),(wt,need_w,'weather')]:
        missing=set(cols)-set(df.columns)
        if missing: raise SystemExit(f'{name} missing: '+', '.join(sorted(missing)))
    for df in (inc,mp,wt):
        for c in ['region','district']:
            if c in df.columns: df[c]=df[c].fillna('').astype(str).str.strip()
    inc['observed_at']=pd.to_datetime(inc['observed_at'],errors='coerce',utc=True); wt['observed_at']=pd.to_datetime(wt['observed_at'],errors='coerce',utc=True)
    inc=inc.dropna(subset=['observed_at','observed_impact_score']); wt=wt.dropna(subset=['observed_at','location_id'])
    mp['latitude']=pd.to_numeric(mp['latitude'],errors='coerce'); mp['longitude']=pd.to_numeric(mp['longitude'],errors='coerce'); mp=mp.dropna(subset=['latitude','longitude'])
    # Prefer district mapping. This is a first-pass geographic approximation and kept explicit.
    inc=inc.merge(mp[['source_region','source_district','location_id','latitude','longitude','coordinate_level','review_status']].drop_duplicates(['source_region','source_district']),left_on=['region','district'],right_on=['source_region','source_district'],how='left')
    inc['location_match_status']=np.where(inc['location_id'].notna(),'DISTRICT_MATCH','UNMAPPED')
    inc['observed_date']=inc['observed_at'].dt.floor('D')
    wt=wt.sort_values(['location_id','observed_at'])
    weather_features=[]
    for loc,g in wt.groupby('location_id',sort=False):
        g=g.set_index('observed_at').sort_index()
        out=g.copy()
        for base,col in [('precipitation_sum','rainfall_1d_mm')]:
            if base in out: out[col]=pd.to_numeric(out[base],errors='coerce').fillna(0).rolling('1D',closed='left').sum()
        if 'precipitation_sum' in out:
            pmm=pd.to_numeric(out['precipitation_sum'],errors='coerce').fillna(0)
            out['rainfall_3d_mm']=pmm.rolling('3D',closed='left').sum(); out['rainfall_7d_mm']=pmm.rolling('7D',closed='left').sum(); out['rainfall_30d_mm']=pmm.rolling('30D',closed='left').sum()
        if 'wind_speed_10m_max' in out: out['wind_speed_3d_max_kmh']=pd.to_numeric(out['wind_speed_10m_max'],errors='coerce').rolling('3D',closed='left').max()
        if 'wind_gusts_10m_max' in out: out['wind_gust_3d_max_kmh']=pd.to_numeric(out['wind_gusts_10m_max'],errors='coerce').rolling('3D',closed='left').max()
        out=out.reset_index(); out['weather_date']=out['observed_at'].dt.floor('D'); weather_features.append(out)
    wf=pd.concat(weather_features,ignore_index=True) if weather_features else pd.DataFrame()
    keep_w=['location_id','weather_date','temperature_2m_mean','temperature_2m_max','temperature_2m_min','precipitation_sum','rain_sum','precipitation_hours','wind_speed_10m_max','wind_gusts_10m_max','rainfall_1d_mm','rainfall_3d_mm','rainfall_7d_mm','rainfall_30d_mm','wind_speed_3d_max_kmh','wind_gust_3d_max_kmh']
    keep_w=[c for c in keep_w if c in wf.columns]; wf=wf[keep_w]
    out=inc.merge(wf,left_on=['location_id','observed_date'],right_on=['location_id','weather_date'],how='left')
    # earthquake context: prior 30 days only
    out['earthquake_count_30d']=0.0; out['earthquake_max_magnitude_30d']=0.0
    if args.earthquakes:
        q=pd.read_csv(args.earthquakes,low_memory=False); q['event_time']=pd.to_datetime(q['event_time'],errors='coerce',utc=True); q['magnitude']=pd.to_numeric(q['magnitude'],errors='coerce'); q['distance_km']=pd.to_numeric(q['distance_km'],errors='coerce'); q=q.dropna(subset=['event_time','location_id'])
        vals=[]
        for r in out[['location_id','observed_at']].itertuples(index=False):
            m=(q['location_id']==r.location_id)&(q['event_time']<r.observed_at)&(q['event_time']>=r.observed_at-pd.Timedelta(days=30))&(q['distance_km']<=300)
            s=q.loc[m,'magnitude'].dropna(); vals.append((float(len(s)),float(s.max()) if not s.empty else 0.0))
        if vals:
            out['earthquake_count_30d']=[x[0] for x in vals]; out['earthquake_max_magnitude_30d']=[x[1] for x in vals]
    # Explicitly define a forecast-safe feature contract.
    out['target_observed_impact_score']=pd.to_numeric(out['observed_impact_score'],errors='coerce')
    out['log_target_impact']=np.log1p(out['target_observed_impact_score'].clip(lower=0))
    hazard=out['disaster_type'].astype(str).str.upper().str.strip(); out['hazard_family']=hazard.replace({'FOREST FIRE':'FIRE','STRUCT.COLLAPSE':'STRUCTURAL','THUNDERSTORM':'STORM','HAIL STORM':'STORM','SNOW STORM':'STORM','STRONG WIND':'STORM','COLD WAVE':'COLD','HEAT WAVE':'HEAT','BOAT CAPSIZE':'WATER_ACCIDENT'})
    out['days_since_incident']=0
    features=['incident_id','observed_at','hazard_family','location_id','latitude','longitude','rainfall_1d_mm','rainfall_3d_mm','rainfall_7d_mm','rainfall_30d_mm','temperature_2m_mean','temperature_2m_max','temperature_2m_min','precipitation_sum','precipitation_hours','wind_speed_10m_max','wind_gusts_10m_max','wind_speed_3d_max_kmh','wind_gust_3d_max_kmh','earthquake_count_30d','earthquake_max_magnitude_30d','target_observed_impact_score','log_target_impact','location_match_status','coordinate_level','review_status','dataset_type']
    out['dataset_type']='real_pending_training_review'; features=[c for c in features if c in out.columns]
    final=out[features].copy(); path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); final.to_csv(path,index=False)
    quality={'rows':int(len(final)),'mapped_rows':int((final['location_match_status']=='DISTRICT_MATCH').sum()),'unmapped_rows':int((final['location_match_status']=='UNMAPPED').sum()),'weather_joined_rows':int(final['rainfall_1d_mm'].notna().sum()),'target_rows':int(final['target_observed_impact_score'].notna().sum()),'time_min':str(final['observed_at'].min()) if len(final) else None,'time_max':str(final['observed_at'].max()) if len(final) else None,'feature_columns':[c for c in features if c not in ['target_observed_impact_score','log_target_impact']],'warning':'This is a leakage-aware candidate training table. Coordinate review, event/environment alignment, target definition, missingness and temporal validation are still required before model training or operational use.'}
    path.with_suffix('.quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8'); print(json.dumps({'output':str(path),'quality_report':str(path.with_suffix('.quality.json')),**quality},indent=2))

if __name__=='__main__': main()
