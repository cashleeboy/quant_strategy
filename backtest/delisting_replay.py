"""Research replay v3: terminated A shares -> locked equity -> OTC shares.

Old replay files remain immutable. Marking and execution availability are
separate: last observable carry is explicitly not a fair-value certification.
The caller must supply a dated confirmation assumption, never an implicit default.
"""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
from backtest.corporate_actions import AssetTransitions, AssetTransitionError


def lower_limit(previous, tick=.01, rate=.05):
    p=Decimal(str(previous));t=Decimal(str(tick));r=Decimal(str(rate))
    rounded=((p*(1-r))/t).quantize(Decimal('1'),rounding=ROUND_HALF_UP)*t
    return float(max(t,min(rounded,p-t)))


class DelistingBridge:
    def __init__(self,prices,adjustments,delist_dates,events,bars,confirmations):
        self.quotes=prices.copy();self.marks=prices.ffill().fillna(0)
        self.dead=delist_dates.copy();self.events=[];self.log=[];self.processed=set()
        self.special=set();used=set();self.sell_masks={}
        for original in events:
            e=dict(original)
            for k in ['delist_announcement','delist_date','listing_announcement','listing_date','confirmation_deadline','last_source_quote']:
                e[k]=pd.Timestamp(e[k])
            if e['event_id'] in used:raise ValueError('duplicate transfer')
            used.add(e['event_id']);s,c,t=e['source'],e['claim'],e['destination']
            if len({s,c,t})!=3 or c in self.quotes or t in self.quotes:raise ValueError('conflicting asset symbols')
            if not e['delist_announcement']<=e['delist_date']<e['listing_announcement']<e['listing_date']:
                raise ValueError('invalid event information timing')
            if not e.get('evidence') or e['raw_share_ratio']!=1.:raise ValueError('only evidenced same-share transfers supported')
            if pd.Timestamp(delist_dates.loc[s])!=e['delist_date']:raise ValueError('delisting metadata mismatch')
            if not (prices.index.min()<=e['last_source_quote']<e['delist_date']<=e['listing_date']<=prices.index.max()):
                raise ValueError('this bridge requires the whole observed transfer interval')
            if e['delist_date'] not in prices.index or e['listing_date'] not in prices.index:raise ValueError('event not on market session')
            if prices.loc[:e['delist_date'],s].last_valid_index()!=e['last_source_quote']:raise ValueError('wrong last source quote')
            if prices.loc[e['delist_date']:,s].notna().any():raise ValueError('unexpected source quotes after retirement')
            factor=float(adjustments.loc[e['last_source_quote'],s])
            if not np.isfinite(factor) or factor<=0:raise ValueError('missing source adjustment')
            if not np.isclose(prices.loc[e['last_source_quote'],s]/factor,e['last_raw_close'],atol=1e-10):
                raise ValueError('raw/adjusted source unit mismatch')
            observed=adjustments.loc[e['last_source_quote']:e['delist_date'],s].dropna()
            if not np.allclose(observed,factor):raise ValueError('unhandled suspension distribution')
            confirmation=confirmations.get(e['event_id'])
            if confirmation is None:raise AssetTransitionError('explicit confirmation assumption required')
            confirmation=pd.Timestamp(confirmation)
            if pd.isna(confirmation) or not e['delist_date']<=confirmation<=e['confirmation_deadline']:
                raise AssetTransitionError('confirmation not completed by first-day deadline; later registration evidence required')
            e['confirmation_date']=confirmation;e['factor']=factor
            b=bars[t].copy()
            if b.index.has_duplicates or not b.index.is_monotonic_increasing:raise ValueError('invalid OTC quote dates')
            if not b.index.isin(prices.index).all():raise ValueError('OTC dates outside supplied market calendar')
            if b.index.min()!=e['listing_date'] or b.index.max()!=prices.index.max():raise ValueError('OTC endpoint coverage incomplete')
            required=prices.index[prices.index>=e['listing_date']]
            if not required.isin(b.index).all():raise ValueError('unexplained missing OTC sessions; suspension evidence required')
            numbers=b[['open','high','low','close','volume_shares','amount_yuan']].to_numpy(float)
            if not np.isfinite(numbers).all() or (numbers[:,:4]<=0).any() or (numbers[:,4:]<0).any():raise ValueError('invalid OTC values')
            if (b.high<b[['open','close','low']].max(axis=1)).any() or (b.low>b[['open','close','high']].min(axis=1)).any():raise ValueError('invalid OHLC')
            if not e['first_day_low']<=b.close.iloc[0]<=e['first_day_high']:raise ValueError('first-day range mismatch')
            previous=b.close.shift(1);previous.iloc[0]=e['last_raw_close']
            lower=previous.map(lambda v:lower_limit(v,e['tick_size'],e['limit_rate']))
            lower.iloc[0]=e['first_day_low']
            locked=np.isclose(b.high,b.low,atol=1e-10)&(b.close<=lower+1e-10)
            sellable=b.volume_shares.gt(0)&b.amount_yuan.gt(0)&~locked
            # No synthetic execution quote is created for the unlisted right.
            self.quotes[c]=np.nan;self.marks[c]=0.
            self.marks.loc[e['delist_date']:,c]=e['last_raw_close']
            self.quotes[t]=b.close.reindex(prices.index)
            self.marks[t]=self.quotes[t].ffill().fillna(0)
            self.sell_masks[t]=sellable.reindex(prices.index,fill_value=False)
            self.dead.loc[c]=e['listing_date'];self.dead.loc[t]=pd.NaT
            self.special.update([c,t]);self.events.append(e)
        self.sellable=self.quotes.notna()
        for t,m in self.sell_masks.items():self.sellable[t]=m
        self.names=self.quotes.columns

    def apply(self,date,units,mark):
        pnl=0.
        for e in self.events:
            if date==e['delist_date']:
                kind='delist_to_claim';source=e['source'];target=e['claim'];factor=e['factor']
            elif date==e['listing_date']:
                kind='claim_to_otc';source=e['claim'];target=e['destination'];factor=1.
            else:continue
            event=e['event_id']+'/'+kind
            if event in self.processed:raise AssetTransitionError('transfer applied twice')
            self.processed.add(event);s=self.names.get_loc(source);t=self.names.get_loc(target)
            old=float(units[s])
            if old*mark[s]<=1e-10:continue
            received=old*factor;before=float(units[t]);value=old*mark[s];newvalue=received*mark[t]
            units[s]=0.;units[t]+=received;change=newvalue-value;pnl+=change
            self.log.append(dict(date=date,event_id=event,kind=kind,source=source,destination=target,
                source_units=old,received_units=received,source_adjustment=factor,destination_adjustment=1.,raw_ratio=1.,
                source_raw_shares=received,received_raw_shares=received,destination_units_before=before,
                destination_units_after=float(units[t]),source_value=value,received_value=newvalue,
                action_pnl=change,fee=0.,cash_change=0.,last_source_quote=e['last_source_quote'],
                confirmation_date=e['confirmation_date']))
        return pnl

    def validate_target(self,target):
        if target is not None and target.index.isin(list(self.special)).any():
            raise AssetTransitionError('claims and OTC successors are exit-only; never new ranking targets')


def replay(prices,schedule,target_factory,*,actions,adjustment_factors,delist_dates,
           transfers,otc_bars,confirmations,fee=.0003):
    if (not isinstance(prices.index,pd.DatetimeIndex) or prices.empty or prices.index.hasnans or
        prices.index.has_duplicates or not prices.index.is_monotonic_increasing or prices.columns.has_duplicates):raise ValueError('invalid prices')
    values=prices.to_numpy(float)
    if np.isinf(values).any() or (values[np.isfinite(values)]<=0).any() or not np.isfinite(fee) or not 0<=fee<1:raise ValueError('invalid prices/fee')
    bridge=DelistingBridge(prices,adjustment_factors,delist_dates,transfers,otc_bars,confirmations)
    quote_frame=bridge.quotes;names=quote_frame.columns;px=quote_frame.to_numpy(float)
    marks=bridge.marks.to_numpy(float);sellability=bridge.sellable.to_numpy(bool)
    assets=AssetTransitions(quote_frame,adjustment_factors,actions,bridge.dead)
    schedule=schedule.copy();schedule.index=pd.to_datetime(schedule.index);schedule['execution_date']=pd.to_datetime(schedule.execution_date)
    if schedule.index.has_duplicates or schedule.index.hasnans or schedule.execution_date.isna().any() or schedule.execution_date.duplicated().any():raise ValueError('invalid schedule')
    for signal,row in schedule.iterrows():
        pos=prices.index.get_indexer([signal])[0]
        if pos<0 or pos+1>=len(prices) or row.execution_date!=prices.index[pos+1]:raise ValueError('execution must be next session')
    units=np.zeros(len(names));cash=1.;pending={};daily=[];executions=[];trades=[];targets=[];positions=[]
    claim_indices=[names.get_loc(e['claim']) for e in bridge.events]
    for i,date in enumerate(prices.index):
        quote,mark=px[i],marks[i];valid=np.isfinite(quote);can_sell=sellability[i];safe=np.where(valid,quote,0.)
        before_actions=float(cash+units@mark)
        action_pnl=bridge.apply(date,units,mark)+assets.apply(date,units,mark)
        pre=float(cash+units@mark)
        if not np.isclose(pre,before_actions+action_pnl,atol=1e-10,rtol=1e-10):raise AssertionError('action reconciliation')
        if date in pending:
            signal,weights=pending.pop(date);bridge.validate_target(weights);assets.validate_target(date,weights)
            before,old_cash=units.copy(),cash;buy=np.zeros(len(names));sell=buy.copy()
            action='HOLD' if weights is None else ('CASH' if weights.empty else 'TARGET')
            loc=np.array([],dtype=int) if weights is None else names.get_indexer(weights.index)
            target=np.zeros(len(names))
            if weights is not None:target[loc]=weights.to_numpy(float)
            all_missing=action=='TARGET' and not valid[loc].any()
            if action!='HOLD' and not all_missing:
                dollars=pre*target
                sell=np.maximum(units*safe-dollars,0.)*can_sell;sell[sell<=1e-9]=0.
                cash+=sell.sum()*(1-fee);units-=np.divide(sell,safe,out=np.zeros_like(sell),where=safe>0)
                need=np.maximum(dollars-units*safe,0.)*valid;need[need<=1e-9]=0.
                total=need.sum()*(1+fee);buy=need*(min(1.,cash/total) if total>0 else 1.)
                cash-=buy.sum()*(1+fee);units+=np.divide(buy,safe,out=np.zeros_like(buy),where=safe>0)
            paid=float((buy.sum()+sell.sum())*fee);post=float(cash+units@mark)
            if cash<-1e-9 or not np.isclose(post,pre-paid,rtol=1e-10,atol=1e-10):raise AssertionError('NAV reconciliation')
            if not np.isclose(cash,old_cash+sell.sum()-buy.sum()-paid,atol=1e-10):raise AssertionError('cash reconciliation')
            blocked=(before*mark>1e-10)&~can_sell&(target==0)
            executions.append(dict(date=date,signal_date=signal,action=action,pre_nav=pre,post_nav=post,cash=cash,
                buy_notional=float(buy.sum()),sell_notional=float(sell.sum()),fees=paid,two_way_turnover=float((buy.sum()+sell.sum())/pre),
                cash_fraction=cash/post,holdings=int((units*mark>1e-10).sum()),all_targets_missing_hold=all_missing,
                missing_targets=int((~valid[loc]).sum()),blocked_exit_value=float((before*mark*blocked).sum())))
            for j in np.flatnonzero(buy+sell>0):
                trades.append(dict(date=date,symbol=names[j],side='BUY' if buy[j] else 'SELL',notional=float(buy[j]+sell[j]),fee=float((buy[j]+sell[j])*fee),
                    shares_before=float(before[j]),shares_after=float(units[j]),price=float(quote[j]),pre_nav=pre))
        post=float(cash+units@mark)
        daily.append(dict(date=date,pre_nav=pre,post_nav=post,cash=cash,unquoted_value=float((units*mark*~valid).sum()),action_pnl=action_pnl,
                          locked_equity_value=float((units*mark)[claim_indices].sum())))
        for j in np.flatnonzero(units*mark>1e-10):
            positions.append(dict(date=date,symbol=names[j],units=float(units[j]),value=float(units[j]*mark[j]),weight=float(units[j]*mark[j]/post),
                                  quoted=bool(valid[j]),sellable=bool(can_sell[j])))
        if date in schedule.index:
            held=list(names[units*mark>1e-10]);target=target_factory(date,held)
            if target is not None:
                target=pd.Series(target,dtype=float)
                if target.index.has_duplicates or target.index.isna().any() or not target.index.isin(names).all() or not np.isfinite(target).all() or (target<=0).any() or target.sum()>1+1e-12:raise ValueError('invalid target')
            bridge.validate_target(target);assets.validate_target(date,target)
            execution=schedule.loc[date,'execution_date'];pending[execution]=(date,target)
            for symbol,weight in ([] if target is None else target.items()):
                targets.append(dict(date=date,execution_date=execution,symbol=symbol,weight=weight,was_actually_held=symbol in held))
    events=pd.concat([assets.frame().assign(kind='stock_swap'),pd.DataFrame(bridge.log)],ignore_index=True).sort_values('date')
    return dict(daily=pd.DataFrame(daily).set_index('date'),executions=pd.DataFrame(executions),trades=pd.DataFrame(trades,
        columns=['date','symbol','side','notional','fee','shares_before','shares_after','price','pre_nav']),
        targets=pd.DataFrame(targets,columns=['date','execution_date','symbol','weight','was_actually_held']),positions=pd.DataFrame(positions,
        columns=['date','symbol','units','value','weight','quoted','sellable']),asset_actions=events)
