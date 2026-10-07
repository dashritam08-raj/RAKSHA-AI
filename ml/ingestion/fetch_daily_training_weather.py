"""Fetch historical daily weather for approved location points.

Designed for ML history, not live serving. Uses Open-Meteo Historical Weather API.
Multiple coordinates can be requested in one call where practical.
"""
from __future__ import annotations
import argparse, json, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

URL="https://archive-api.open-meteo.com/v1/archive"
DAILY=["temperature_2m_mean","temperature_2m_max","temperature_2m_min","precipitation_sum","rain_sum","precipitation_hours","wind_speed_10m_max","wind_gusts_10m_max"]

def fetch(rows, start, end):
    lat=','.join(str(float(r['latitude'])) for r in rows); lon=','.join(str(float(r['longitude'])) for r in rows)
    params={"latitude":lat,"longitude":lon,"start_date":start,"end_date":end,"daily":','.join(DAILY),"timezone":"UTC"}
    req=urllib.request.Request(URL+"?"+urllib.parse.urlencode(params), headers={"User-Agent":"RAKSHA-AI/phase5d-weather-builder"})
    with urllib.request.urlopen(req, timeout=120) as resp: return json.loads(resp.read().decode('utf-8'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--locations',required=True); p.add_argument('--start-date',required=True); p.add_argument('--end-date',required=True); p.add_argument('--output',required=True); p.add_argument('--batch-size',type=int,default=10); args=p.parse_args()
    loc=pd.read_csv(args.locations, low_memory=False); loc=loc.dropna(subset=['latitude','longitude','location_id']).copy(); loc['latitude']=pd.to_numeric(loc['latitude'],errors='coerce'); loc['longitude']=pd.to_numeric(loc['longitude'],errors='coerce'); loc=loc.dropna(subset=['latitude','longitude'])
    rows=[]; failures=[]
    for i in range(0,len(loc),args.batch_size):
        batch=loc.iloc[i:i+args.batch_size].to_dict('records')
        try:
            payload=fetch(batch,args.start_date,args.end_date)
            chunks=payload if isinstance(payload,list) else [payload]
            if len(chunks)!=len(batch): raise RuntimeError(f'API returned {len(chunks)} location blocks for {len(batch)} requested')
            for meta,block in zip(batch,chunks):
                times=(block.get('daily') or {}).get('time') or []
                daily=block.get('daily') or {}
                for j,t in enumerate(times):
                    out={'location_id':meta['location_id'],'latitude':meta['latitude'],'longitude':meta['longitude'],'observed_at':t,'source':'Open-Meteo Historical Weather / ERA5-family reanalysis','source_url':URL}
                    for v in DAILY:
                        arr=daily.get(v) or []; out[v]=arr[j] if j<len(arr) else None
                    rows.append(out)
        except Exception as exc: failures.append({'batch_start':i,'error':str(exc)})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    print(json.dumps({'output':str(out),'rows':len(rows),'locations':int(len(loc)),'failures':failures,'start_date':args.start_date,'end_date':args.end_date},indent=2))

if __name__=='__main__': main()
