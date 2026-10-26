#!/usr/bin/env python3
"""Download public evidence and unadjusted OTC bars, without touching A-share data."""
from pathlib import Path
import json,sys,time,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import requests
import pandas as pd
import numpy as np

OUT=ROOT/'data/delisted_000982_20261024'
URL='https://xinsanban.eastmoney.com/api/F10/MarketQuotation/GetTransactionDetail'
PDFS={
    '20241010_listing.pdf':'https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/SBGG/2024/2024-10/2024-10-10/10517555.PDF',
    '20241010_listing_em.pdf':'https://pdf.dfcfw.com/pdf/H2_AN202410101640245695_1.pdf',
}


def get(url,params=None):
    for attempt in range(3):
        try:
            query=url if params is None else url+'?'+'&'.join(f'{k}={v}' for k,v in params.items())
            r=requests.get(query,timeout=20);r.raise_for_status();return r
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(1)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():raise ValueError('Sealed download; choose new version')
    receipts=[]
    for name,url in PDFS.items():
        r=get(url)
        if not r.content.startswith(b'%PDF'):raise ValueError('Not a PDF')
        (OUT/name).write_bytes(r.content);receipts.append(dict(file=name,url=r.url,bytes=len(r.content)))
    datasets=[]
    for order in [-1,1]:
        r=get(URL,dict(code='400236',page=1,pagesize=1000,sortType='TRADEDATE',sortRule=order));j=r.json()
        if not j.get('IsSuccess') or int(j['TotalPage'])!=1:raise ValueError('Incomplete history page')
        name=f'transactions_order{order}.json';(OUT/name).write_bytes(r.content)
        receipts.append(dict(file=name,url=r.url,bytes=len(r.content)))
        frame=pd.DataFrame(j['result'])
        if not frame.MSECUCODE.eq('400236.NQ').all():raise ValueError('Wrong instrument')
        frame=frame.rename(columns={'TRADEDATE':'date','OPEN':'open','CLOSE':'close','HIGH':'high','LOW':'low','CLOSEQF':'close_qfq',
                                    'VOLUMN':'volume_wan_shares','AMOUNT':'amount_wan_yuan','CHANGE':'change','PCTCHANGE':'change_pct'}).drop(columns='MSECUCODE')
        frame['date']=pd.to_datetime(frame.date)
        for col in frame.columns[1:]:frame[col]=pd.to_numeric(frame[col],errors='raise')
        frame=frame.sort_values('date').reset_index(drop=True)
        if frame.empty or frame.date.duplicated().any():raise ValueError('Invalid quote dates')
        datasets.append(frame)
    pd.testing.assert_frame_equal(datasets[0],datasets[1])
    raw=datasets[0].loc[datasets[0].date.between('2024-10-11','2026-10-15')].copy()
    raw['volume_shares']=raw.volume_wan_shares*10000
    raw['amount_yuan']=raw.amount_wan_yuan*10000
    assert raw.date.min()==pd.Timestamp('2024-10-11') and raw.date.max()==pd.Timestamp('2026-10-15')
    assert raw.date.ge('2024-10-11').all() and raw.date.le('2026-10-15').all()
    assert (raw[['open','close','high','low']]>0).all().all()
    assert (raw.high>=raw[['open','close','low']].max(axis=1)).all()
    assert (raw.low<=raw[['open','close','high']].min(axis=1)).all()
    assert raw.volume_shares.ge(0).all() and raw.amount_yuan.ge(0).all()
    np.testing.assert_allclose(raw.close,raw.close_qfq,atol=1e-12)
    vwap=raw.amount_yuan/raw.volume_shares.where(raw.volume_shares>0)
    assert ((vwap>=raw.low-.001)&(vwap<=raw.high+.001)|vwap.isna()).all()
    assert .17<=raw.iloc[0]['close']<=.19
    raw.to_csv(OUT/'400236_OC_daily_raw.csv',index=False)
    write=lambda name,value:(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    write('download.json',dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),receipts=receipts,
        first=str(raw.date.min().date()),last=str(raw.date.max().date()),rows=len(raw),
        units='unadjusted yuan/share; source volume in ten-thousand shares; source amount in ten-thousand yuan',
        checks=['ascending/descending query parity','single page complete','raw versus forward-adjusted close parity','OHLC bounds','volume-value consistency','issuer first-day range'],
        caveat='Eastmoney public transaction statistics, not independently verified exchange tape; kline endpoint intermittent failure; Tencent returned no bars; NEEQ direct PDF returned403'))
    write('complete.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='complete.json'})
    print('Downloaded',len(raw),'bars',raw.date.min(),raw.date.max(),flush=True)


if __name__=='__main__':run()
