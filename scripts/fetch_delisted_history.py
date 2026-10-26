#!/usr/bin/env python3
"""Generic immutable public OTC quote/evidence downloader for research."""
from pathlib import Path
import sys,json,argparse,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from scripts.fetch_delisted_000982 import get,URL

def run():
 ap=argparse.ArgumentParser();ap.add_argument('spec');args=ap.parse_args()
 spec=json.loads(Path(args.spec).read_text());out=ROOT/spec['directory'];out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():raise ValueError('Sealed download')
 receipts=[]
 for name,url in spec['sources'].items():
  r=get(url)
  if name.endswith('.pdf') and not r.content.startswith(b'%PDF'):raise ValueError(name)
  (out/name).write_bytes(r.content);receipts.append(dict(file=name,url=r.url))
 frames=[]
 for order in [-1,1]:
  r=get(URL,dict(code=spec['code'],page=1,pagesize=1000,sortType='TRADEDATE',sortRule=order));j=r.json()
  assert j['IsSuccess'] and int(j['TotalPage'])==1
  name=f'transactions_order{order}.json';(out/name).write_bytes(r.content);receipts.append(dict(file=name,url=r.url))
  x=pd.DataFrame(j['result']);assert x.MSECUCODE.eq(spec['code']+'.NQ').all()
  x=x.rename(columns={'TRADEDATE':'date','OPEN':'open','HIGH':'high','LOW':'low','CLOSE':'close','CLOSEQF':'close_qfq',
    'VOLUMN':'volume_wan_shares','AMOUNT':'amount_wan_yuan','CHANGE':'change','PCTCHANGE':'change_pct'}).drop(columns='MSECUCODE')
  x['date']=pd.to_datetime(x.date)
  for c in x.columns[1:]:x[c]=pd.to_numeric(x[c],errors='raise')
  frames.append(x.sort_values('date').reset_index(drop=True))
 pd.testing.assert_frame_equal(*frames)
 x=frames[0].loc[frames[0].date.le('2026-10-15')].copy()
 assert not x.date.duplicated().any() and x.date.min()==pd.Timestamp(spec['first'])
 x['volume_shares']=x.volume_wan_shares*10000;x['amount_yuan']=x.amount_wan_yuan*10000
 assert (x[['open','high','low','close']]>0).all().all()
 assert x.volume_shares.ge(0).all() and x.amount_yuan.ge(0).all()
 assert (x.high>=x[['open','low','close']].max(axis=1)).all() and (x.low<=x[['open','high','close']].min(axis=1)).all()
 # Distribution differences need explicit corporate-action review, not blind use.
 np.testing.assert_allclose(x.close,x.close_qfq,atol=1e-12)
 vwap=x.amount_yuan/x.volume_shares.where(x.volume_shares>0)
 assert (((vwap>=x.low-.001)&(vwap<=x.high+.001))|vwap.isna()).all()
 x.to_csv(out/(spec['code']+'_OC_daily_raw.csv'),index=False)
 def write(n,v):(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
 write('download.json',dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),spec=spec,receipts=receipts,rows=len(x),
  first=x.date.min(),last=x.date.max(),units='unadjusted yuan/share; volume and amount multiplied by10000',
  caveat='One public vendor, ascending/descending parity is not independent exchange verification. Vendor daily records may include non-trading dates; execution requires independently bounded calendar and volume checks.'))
 write('complete.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='complete.json'})
 print('Downloaded',spec['code'],len(x));print(x.head(10).to_string(index=False))

if __name__=='__main__':run()
