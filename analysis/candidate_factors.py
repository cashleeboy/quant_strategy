"""Research-only annual PIT candidates. No production/forward imports."""
import numpy as np
import pandas as pd
from analysis.financial_asof import _date
from factors.preprocess import preprocess_factor_panel

CANDIDATES = ['DIVIDEND_TTM','LOW_ACCRUALS','LOW_ASSET_GROWTH','GROSS_PROFIT_ASSETS','MOMENTUM_12_1']
FIELDS = {'income':['revenue','oper_cost','n_income'],
          'balance':['total_assets'], 'cashflow':['n_cashflow_act']}

def normalize_annual(raw, kind):
    cols=['ts_code','ann_date','f_ann_date','end_date','report_type','comp_type']+FIELDS[kind]
    f=raw[cols].copy()
    for c in ['ann_date','f_ann_date','end_date']: f[c]=_date(f[c])
    for c in ['report_type','comp_type']+FIELDS[kind]:f[c]=pd.to_numeric(f[c],errors='coerce')
    f['available_date']=f[['ann_date','f_ann_date']].max(axis=1)
    f=f.loc[f.report_type.eq(1)&f.end_date.dt.month.eq(12)&f.end_date.dt.day.eq(31)&
            f.available_date.notna()&f.end_date.le(f.available_date)&f.ts_code.notna()].copy()
    keys=['ts_code','end_date','available_date']
    conflict=f.groupby(keys)[FIELDS[kind]+['comp_type']].nunique(dropna=False).gt(1).any(axis=1).rename('ambiguous')
    f=f.sort_values(keys,kind='stable').drop_duplicates(keys).merge(conflict,on=keys,validate='one_to_one')
    f.loc[f.ambiguous,FIELDS[kind]]=np.nan
    return f

def annual_candidates(tables, date):
    """Latest disclosed balance year anchors fields; never backfill a missing latest report."""
    d=pd.Timestamp(date); known={}
    for k,f in tables.items():
        x=f.loc[f.available_date.lt(d)].drop_duplicates(['ts_code','end_date'],keep='last')
        known[k]=x.set_index(['ts_code','end_date'])
    b=known['balance'].reset_index().sort_values(['ts_code','end_date']).drop_duplicates('ts_code',keep='last').set_index('ts_code')
    idx=pd.MultiIndex.from_arrays([b.index,b.end_date])
    prev=pd.MultiIndex.from_arrays([b.index,b.end_date-pd.DateOffset(years=1)])
    a=b.total_assets; old=known['balance'].reindex(prev);old.index=b.index
    income=known['income'].reindex(idx);income.index=b.index
    cash=known['cashflow'].reindex(idx);cash.index=b.index
    valid=(b.comp_type.eq(1)&a.gt(0)&~b.ambiguous&((d-b.end_date).dt.days<=730))
    pair=valid&old.comp_type.eq(1)&old.total_assets.gt(0)&old.ambiguous.eq(False)
    inc=valid&income.comp_type.eq(1)&income.ambiguous.eq(False)
    acc=pair&inc&cash.comp_type.eq(1)&cash.ambiguous.eq(False)
    out=pd.DataFrame(index=b.index)
    out['LOW_ASSET_GROWTH']=(-(a/old.total_assets-1)).where(pair)
    out['GROSS_PROFIT_ASSETS']=((income.revenue-income.oper_cost)/a).where(inc&income.revenue.gt(0)&income.oper_cost.ge(0))
    out['LOW_ACCRUALS']=(-(income.n_income-cash.n_cashflow_act)/((a+old.total_assets)/2)).where(acc)
    out['report_end']=b.end_date
    out['latest_disclosure']=pd.concat([b.available_date,old.available_date,income.available_date,cash.available_date],axis=1).max(axis=1)
    out.index.name='symbol'
    return out.replace([np.inf,-np.inf],np.nan)

def dividend_asof(index, raw, max_age_days=7):
    left=index.to_frame(index=False)
    left['date']=pd.to_datetime(left.date).astype('datetime64[ns]');left['_row']=range(len(left))
    right=raw[['ts_code','trade_date','dv_ttm']].rename(columns={'ts_code':'symbol','trade_date':'snapshot_date'}).copy()
    right['snapshot_date']=_date(right.snapshot_date).astype('datetime64[ns]')
    if right.duplicated(['symbol','snapshot_date']).any():raise ValueError('Duplicate dividend snapshot')
    right.dv_ttm=pd.to_numeric(right.dv_ttm,errors='coerce')
    joined=pd.merge_asof(left.sort_values(['date','symbol']),right.sort_values(['snapshot_date','symbol']),
        left_on='date',right_on='snapshot_date',by='symbol',direction='backward',tolerance=pd.Timedelta(days=max_age_days))
    joined=joined.sort_values('_row');joined.index=index
    return pd.DataFrame({'DIVIDEND_TTM':(joined.dv_ttm/100).where(joined.dv_ttm.ge(0)),
                         'dividend_snapshot':joined.snapshot_date},index=index)

def momentum_12_1(prices):
    valid=prices.gt(0)&np.isfinite(prices)
    p=prices.where(valid)
    result=p.shift(21)/p.shift(252)-1
    return result.where(valid&p.notna().rolling(253,min_periods=253).sum().ge(np.ceil(253*.9)))

def standardize_candidates(raw,industry):
    values=raw[CANDIDATES].replace([np.inf,-np.inf],np.nan)
    # Existing standardizer fills entire small/constant groups with zero. Restore
    # raw missingness so absence of financial data is never an observed IC score.
    return preprocess_factor_panel(values,industry=industry,by_industry=True,min_industry_count=3).where(values.notna())
