#!/usr/bin/env python3
"""Create article147 figures from the sealed all-five extension."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
SRC=ROOT/'output/all5_multihorizon_extension_20261025';OUT=ROOT/'docs/articles/assets/article147_all5';OUT.mkdir(parents=True,exist_ok=True)
RESULTS=ROOT/'docs/articles/results/article147_all5';RESULTS.mkdir(parents=True,exist_ok=True)
ARMS=['ORIGINAL13_5D','ALL18_5D','ORIGINAL13_MULTI','ALL18_MULTI']
LABELS={'ORIGINAL13_5D':'13 factors / 5-day','ALL18_5D':'18 factors / 5-day','ORIGINAL13_MULTI':'13 factors / multi','ALL18_MULTI':'18 factors / multi'}
COLORS={'ORIGINAL13_5D':'#475569','ALL18_5D':'#b45309','ORIGINAL13_MULTI':'#64748b','ALL18_MULTI':'#0369a1'}


def run():
    plt.style.use('seaborn-v0_8-whitegrid');nav=pd.read_csv(SRC/'nav.csv',index_col=0,parse_dates=True)
    fig,ax=plt.subplots(figsize=(10,5.8))
    for arm in ARMS:ax.plot(nav.index,nav[f'{arm}_3bp'],label=LABELS[arm],color=COLORS[arm],lw=1.6)
    ax.set(title='Fixed-15 NAV before and after adding five factors',xlabel='Date',ylabel='Net asset value (start = 1)');ax.legend(ncol=2)
    fig.text(.01,.01,'Source: PIT A-share study, 2012-01-06 to 2026-10-15; 3 bp each buy/sell.',fontsize=8);fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'01_nav_four_arms.png',dpi=180);plt.close(fig)
    periods=pd.read_csv(SRC/'periods.csv');x=periods[periods.fee.eq(.0003)].pivot(index='period',columns='arm',values='cagr').loc[['2012-2015','2016-2020','2021-2026']]
    fig,ax=plt.subplots(figsize=(10,5.7));x.rename(columns=LABELS).plot.bar(ax=ax,color=[COLORS[c] for c in x.columns],width=.8)
    ax.axhline(0,color='black',lw=.8);ax.set(title='Annualized return by predeclared subperiod',xlabel='Subperiod',ylabel='CAGR');ax.yaxis.set_major_formatter(lambda v,p:f'{v:.0%}');ax.tick_params(axis='x',rotation=0);ax.legend(ncol=2)
    fig.text(.01,.01,'Same universe, fixed15 selection and risk budget. Only factor set and evaluation horizon differ.',fontsize=8);fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'02_subperiod_cagr.png',dpi=180);plt.close(fig)
    weights=pd.read_csv(SRC/'monthly_weights.csv');c=['DIVIDEND_TTM','LOW_ACCRUALS','LOW_ASSET_GROWTH','GROSS_PROFIT_ASSETS','MOMENTUM_12_1']
    recent=weights[weights.date.ge('2021-01-01')&weights.arm.isin(['ALL18_5D','ALL18_MULTI'])].groupby('arm')[c].mean().T
    fig,ax=plt.subplots(figsize=(9.5,5.5));recent.rename(columns=LABELS).plot.barh(ax=ax,color=[COLORS['ALL18_5D'],COLORS['ALL18_MULTI']],width=.65)
    ax.set(title='Average weights of the five added factors since 2021',xlabel='Average nominal weight',ylabel='Added factor');ax.xaxis.set_major_formatter(lambda v,p:f'{v:.0%}');ax.legend()
    fig.text(.01,.01,'All five together average24.0% under5-day and28.4% under multi-horizon weighting.',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'03_added_factor_weights.png',dpi=180);plt.close(fig)
    summary=pd.read_csv(SRC/'summary.csv');z=summary[summary.fee.eq(.0003)].set_index('arm').loc[ARMS]
    fig,axes=plt.subplots(1,2,figsize=(10,4.8));axes[0].bar([LABELS[a] for a in ARMS],z.cagr,color=[COLORS[a] for a in ARMS]);axes[0].set(title='Full-period CAGR',ylabel='CAGR');axes[0].yaxis.set_major_formatter(lambda v,p:f'{v:.0%}');axes[0].tick_params(axis='x',rotation=25)
    axes[1].bar([LABELS[a] for a in ARMS],z.max_drawdown,color=[COLORS[a] for a in ARMS]);axes[1].set(title='Full-period maximum drawdown',ylabel='Drawdown');axes[1].yaxis.set_major_formatter(lambda v,p:f'{v:.0%}');axes[1].tick_params(axis='x',rotation=25)
    fig.text(.01,.01,'Source: same fixed-15 account mechanics; 3 bp each side.',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'04_full_metrics.png',dpi=180);plt.close(fig)
    overlap=[]
    for base,added in [('ORIGINAL13_5D','ALL18_5D'),('ORIGINAL13_MULTI','ALL18_MULTI')]:
        base_decisions=pd.read_csv(SRC/f'{base}_3bp/decisions.csv.gz')
        added_decisions=pd.read_csv(SRC/f'{added}_3bp/decisions.csv.gz')
        weekly_dates=sorted(set(base_decisions.loc[base_decisions.weekly,'date']) & set(added_decisions.loc[added_decisions.weekly,'date']))
        base_targets=pd.read_csv(SRC/f'{base}_3bp/targets.csv.gz')
        added_targets=pd.read_csv(SRC/f'{added}_3bp/targets.csv.gz')
        base_targets=base_targets[base_targets.date.isin(weekly_dates)&base_targets.weight.gt(0)]
        added_targets=added_targets[added_targets.date.isin(weekly_dates)&added_targets.weight.gt(0)]
        for date in weekly_dates:
            left=set(base_targets.loc[base_targets.date.eq(date),'symbol'])
            right=set(added_targets.loc[added_targets.date.eq(date),'symbol'])
            common=len(left&right)
            overlap.append({'comparison':f'{base}_vs_{added}','date':date,'common':common,
                            'replacements_from_base':len(left-right),'identical':left==right,
                            'jaccard':common/len(left|right) if left|right else 1.0})
    pd.DataFrame(overlap).to_csv(RESULTS/'weekly_holding_overlap.csv',index=False)
    print('Created',len(list(OUT.glob('*.png'))),'article147 figures')


if __name__=='__main__':run()
