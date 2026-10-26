"""Verified asset transitions in adjusted-price research units.

This does NOT elect cash options or infer merger terms from price jumps. Unknown
delisted holdings fail closed. Ordinary dividends remain in adjusted prices and
must not be credited again. Fractional units are retained, not broker lot rules.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class AssetTransitionError(RuntimeError):
    pass


def adjusted_swap_quantity(old_units, raw_ratio, old_adjustment, new_adjustment):
    values=np.asarray([old_units,raw_ratio,old_adjustment,new_adjustment],dtype=float)
    if not np.isfinite(values).all() or old_units<0 or (values[1:]<=0).any():
        raise ValueError('invalid swap units, ratio, or adjustment')
    # Price=raw_close*F, so a synthetic unit represents F current raw shares.
    return float(old_units*old_adjustment*raw_ratio/new_adjustment)


class AssetTransitions:
    def __init__(self,prices,adjustments,events,delist_dates):
        self.names=prices.columns;self.dates=prices.index
        if delist_dates.index.has_duplicates or not self.names.isin(delist_dates.index).all():
            raise ValueError('explicit unique delisting metadata required for every price symbol')
        self.delist=pd.to_datetime(delist_dates.reindex(self.names),errors='raise')
        self.events={};self.log=[];self.processed=set();self.retired={}
        ids=set();sources=set()
        for original in events:
            r=dict(original)
            if r['event_id'] in ids or r['source'] in sources:
                raise ValueError('duplicate/conflicting action')
            ids.add(r['event_id']);sources.add(r['source'])
            if r['kind']!='stock_swap' or r['source']==r['destination']:
                raise ValueError('unsupported asset transition')
            for f in ['announcement_date','record_date','effective_date']:
                r[f]=pd.Timestamp(r[f])
                if pd.isna(r[f]):raise ValueError('missing action date')
            if not r['announcement_date']<=r['record_date']<r['effective_date']:
                raise ValueError('invalid or future-known event')
            if not r.get('evidence'):raise ValueError('missing action evidence')
            numerator,denominator=float(r['ratio_numerator']),float(r['ratio_denominator'])
            if not np.isfinite([numerator,denominator]).all() or numerator<=0 or denominator<=0:
                raise ValueError('invalid raw share ratio')
            r['raw_ratio']=numerator/denominator
            self.retired[r['source']]=r['effective_date']
            if r['effective_date']<prices.index[0] or r['effective_date']>prices.index[-1]:continue
            date=r['effective_date']
            if date not in prices.index:raise ValueError('effective date must be an observed market session')
            if not {r['source'],r['destination']}.issubset(prices.columns):
                raise ValueError('missing source/destination price column')
            if r['record_date'] not in prices.index or prices.index.get_loc(date)!=prices.index.get_loc(r['record_date'])+1:
                raise ValueError('only adjacent record/listing sessions supported; separate entitlement ledger required otherwise')
            if adjustments.index.has_duplicates or adjustments.columns.has_duplicates or not adjustments.index.is_monotonic_increasing:
                raise ValueError('invalid adjustment axes')
            r['source_pos']=self.names.get_loc(r['source']);r['destination_pos']=self.names.get_loc(r['destination'])
            self.events.setdefault(date,[]).append(r)
        self.prices=prices;self.adjustments=adjustments

    def apply(self,date,shares,mark):
        delta=0.
        for r in self.events.get(date,[]):
            key=r['event_id']
            if key in self.processed:raise AssetTransitionError('action processed twice: '+key)
            self.processed.add(key)
            src,dst=r['source_pos'],r['destination_pos'];old=float(shares[src])
            if old*mark[src]<=1e-10:continue
            source_quotes=self.prices.loc[:r['record_date'],r['source']].dropna()
            if source_quotes.empty or not np.isfinite(self.prices.loc[date,r['destination']]):
                raise AssetTransitionError('missing source mark or destination listing quote: '+key)
            last=source_quotes.index[-1]
            try:
                old_f=float(self.adjustments.loc[last,r['source']])
                new_f=float(self.adjustments.loc[date,r['destination']])
            except KeyError as e:raise AssetTransitionError('missing exact adjustment observation: '+key) from e
            # A distribution during the source suspension needs its own entitlement
            # accounting; never silently reuse a stale pre-distribution share count.
            between=self.adjustments.loc[last:r['record_date'],r['source']].dropna()
            if not np.isfinite([old_f,new_f]).all() or min(old_f,new_f)<=0 or not np.allclose(between,old_f,rtol=1e-10,atol=1e-12):
                raise AssetTransitionError('missing/changed suspension adjustment: '+key)
            added=adjusted_swap_quantity(old,r['raw_ratio'],old_f,new_f)
            before_dst=float(shares[dst]);old_value=float(old*mark[src]);new_value=float(added*mark[dst])
            shares[src]=0.;shares[dst]+=added;delta+=new_value-old_value
            self.log.append(dict(date=date,event_id=key,source=r['source'],destination=r['destination'],
                source_units=old,source_adjustment=old_f,destination_adjustment=new_f,raw_ratio=r['raw_ratio'],
                source_raw_shares=old*old_f,received_raw_shares=old*old_f*r['raw_ratio'],received_units=added,
                destination_units_before=before_dst,destination_units_after=float(shares[dst]),
                last_source_quote=last,record_date=r['record_date'],source_value=old_value,received_value=new_value,
                action_pnl=new_value-old_value,fee=0.,cash_change=0.))
        expired=self.delist.notna()&self.delist.le(date)
        for s,d in self.retired.items():
            if s in self.names and d<=date:expired.loc[s]=True
        bad=(shares*mark>1e-10)&expired.to_numpy()
        if bad.any():raise AssetTransitionError('unresolved terminated holding; verified settlement required: '+','.join(self.names[bad]))
        return delta

    def validate_target(self,date,target):
        if target is None or target.empty:return
        terminal=self.delist.reindex(target.index)
        if (terminal.notna()&terminal.le(date)).any():raise AssetTransitionError('target contains terminated asset')
        if any(s in self.retired and self.retired[s]<=date for s in target.index):
            raise AssetTransitionError('target contains converted asset; stale instructions must be cancelled/rebuilt')

    def frame(self):
        columns=['date','event_id','source','destination','source_units','source_adjustment','destination_adjustment',
                 'raw_ratio','source_raw_shares','received_raw_shares','received_units','destination_units_before',
                 'destination_units_after','last_source_quote','record_date','source_value','received_value',
                 'action_pnl','fee','cash_change']
        return pd.DataFrame(self.log,columns=columns)
