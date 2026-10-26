#!/usr/bin/env python3
"""Read-only benchmark retrieval. Credentials never written or printed."""
from pathlib import Path
import sys, json, os
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.run_mvp_holdings_count import sha, write_json

OUT=ROOT/'output/rebalance_frequency_20261022/market'

def fetch():
    import tushare as ts
    from tushare.util import upass
    from config import get_tushare_token
    try: token=get_tushare_token()
    except ValueError: token=upass.get_token()
    if not token: raise RuntimeError('No saved Tushare credential')
    pro=ts.pro_api(token,timeout=20)
    OUT.mkdir(parents=True,exist_ok=True)
    status=[]
    for code in ['000300.SH','000016.SH','H00300.CSI','H00016.CSI']:
        frames=[]; error=None
        for year in [2012,2016,2020,2024]:
            path=OUT/f'{code}_{year}.csv'
            lo,hi=f'{year}0101',min(f'{year+3}1231','20261015')
            if path.exists():
                meta=json.loads(path.with_suffix('.json').read_text())
                if sha(path)!=meta['sha256']:raise ValueError('Changed index checkpoint')
                frame=pd.read_csv(path,dtype={'trade_date':str})
            else:
                try: frame=pro.index_daily(ts_code=code,start_date=lo,end_date=hi)
                except Exception as exc:
                    error=type(exc).__name__;break
                if frame.empty:
                    error='EMPTY_RESPONSE';break
                if len(frame)>=3000 or frame.ts_code.ne(code).any() or frame.trade_date.duplicated().any() or frame.close.le(0).any():
                    raise ValueError(f'Invalid index response {code}')
                frame.to_csv(path,index=False)
                write_json(path.with_suffix('.json'),dict(endpoint='index_daily',code=code,start=lo,end=hi,rows=len(frame),sha256=sha(path),retrieved_at=datetime.now(timezone.utc).isoformat()))
            frames.append(frame)
            print(f'{code} {year}: {len(frame)} rows',flush=True)
        if error is None:
            full=pd.concat(frames).sort_values('trade_date')
            if full.trade_date.duplicated().any():raise ValueError('Overlapping index dates')
            full.to_csv(OUT/f'{code}.csv',index=False)
        status.append(dict(code=code,status=error or 'DOWNLOADED',rows=sum(len(f) for f in frames)))
        print(status[-1],flush=True)
    write_json(OUT/'availability.json',status)

if __name__=='__main__':fetch()
