#!/usr/bin/env python3
"""Fetch public OTC quotes and issuer evidence; preserve immutable raw receipts."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from scripts.fetch_delisted_000982 import get,URL
OUT=ROOT/'data/delisted_600297_20261024'
SOURCES={
 '20241030_risk.pdf':'https://static.cninfo.com.cn/finalpage/2024-10-30/1221570114.PDF',
 '20241105_limits.pdf':'https://pdf.dfcfw.com/pdf/H2_AN202411051640727286_1.pdf',
 '20250317_resume.pdf':'https://static.cninfo.com.cn/finalpage/2025-03-17/1222821222.PDF',
 '20250506_frequency.pdf':'https://static.cninfo.com.cn/finalpage/2025-05-06/1223476002.PDF',
 '20240910_confirmation.html':'https://www.citics.com/newsite/tzgg/ywgg/202409/t20240910_1191271.html',
 '20240822_delist.html':'https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10395858&stockid=600297',
}

def run():
 OUT.mkdir(exist_ok=True)
 if (OUT/'complete.json').exists():raise ValueError('Sealed download')
 receipts=[]
 for name,url in SOURCES.items():
  r=get(url)
  if name.endswith('.pdf') and not r.content.startswith(b'%PDF'):raise ValueError(name)
  (OUT/name).write_bytes(r.content);receipts.append(dict(file=name,url=r.url))
 frames=[]
 for order in [-1,1]:
  r=get(URL,dict(code='400245',page=1,pagesize=1000,sortType='TRADEDATE',sortRule=order));j=r.json()
  assert j['IsSuccess'] and int(j['TotalPage'])==1
  name=f'transactions_order{order}.json';(OUT/name).write_bytes(r.content);receipts.append(dict(file=name,url=r.url))
  x=pd.DataFrame(j['result']);assert x.MSECUCODE.eq('400245.NQ').all()
  x=x.rename(columns={'TRADEDATE':'date','OPEN':'open','HIGH':'high','LOW':'low','CLOSE':'close','CLOSEQF':'close_qfq',
    'VOLUMN':'volume_wan_shares','AMOUNT':'amount_wan_yuan','CHANGE':'change','PCTCHANGE':'change_pct'}).drop(columns='MSECUCODE')
  x['date']=pd.to_datetime(x.date)
  for c in x.columns[1:]:x[c]=pd.to_numeric(x[c],errors='raise')
  frames.append(x.sort_values('date').reset_index(drop=True))
 pd.testing.assert_frame_equal(*frames)
 x=frames[0].loc[frames[0].date.le('2026-10-15')].copy()
 assert not x.date.duplicated().any() and x.date.min()==pd.Timestamp('2024-11-01')
 x['volume_shares']=x.volume_wan_shares*10000;x['amount_yuan']=x.amount_wan_yuan*10000
 assert (x[['open','high','low','close']]>0).all().all()
 assert x.volume_shares.ge(0).all() and x.amount_yuan.ge(0).all()
 assert (x.high>=x[['open','low','close']].max(axis=1)).all() and (x.low<=x[['open','high','close']].min(axis=1)).all()
 np.testing.assert_allclose(x.close,x.close_qfq,atol=1e-12)
 vwap=x.amount_yuan/x.volume_shares.where(x.volume_shares>0)
 assert (((vwap>=x.low-.001)&(vwap<=x.high+.001))|vwap.isna()).all()
 x.to_csv(OUT/'400245_OC_daily_raw.csv',index=False)
 def write(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
 write('download.json',dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),receipts=receipts,rows=len(x),
  first=x.date.min(),last=x.date.max(),units='unadjusted yuan/share; volume ten-thousand shares and amount ten-thousand yuan converted by10000',
  caveat='Vendor daily series is not a trading calendar. Later suspension/frequency periods require separate evidence; do not treat filled zero-volume dates as executions.'))
 write('complete.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.name!='complete.json'})
 print('Downloaded',len(x));print(x.head(18).to_string(index=False))

if __name__=='__main__':run()
