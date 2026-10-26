#!/usr/bin/env python3
"""Create article figures from the sealed multi-horizon study outputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
import pandas as pd

SRC=ROOT/'output/multihorizon_selection_ablation_v3_20261025'
OUT=ROOT/'docs/articles/assets/article146_multihorizon'
COLORS={'BASE_5D':'#334155','SHORT_REVERSAL':'#64748b','TREND_FLOOR':'#b45309','MULTI_HORIZON':'#0369a1'}
LABELS={'BASE_5D':'Unified 5-day','SHORT_REVERSAL':'Short reversal','TREND_FLOOR':'Trend floor','MULTI_HORIZON':'Multi-horizon'}


def run():
    OUT.mkdir(parents=True,exist_ok=True);plt.style.use('seaborn-v0_8-whitegrid')
    nav=pd.read_csv(SRC/'nav.csv',index_col=0,parse_dates=True)
    fig,ax=plt.subplots(figsize=(10,5.6))
    for rule in LABELS:ax.plot(nav.index,nav[f'{rule}_FULL_3bp'],label=LABELS[rule],lw=1.6,color=COLORS[rule])
    ax.set(title='Fixed-15 portfolio NAV: four factor-weight rules',xlabel='Date',ylabel='Net asset value (start = 1)');ax.legend(ncol=2)
    fig.text(.01,.01,'Source: frozen PIT A-share study, 2012-01-06 to 2026-10-15; 3 bp each buy/sell.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'01_nav_weight_rules.png',dpi=180);plt.close(fig)
    periods=pd.read_csv(SRC/'periods.csv');x=periods[(periods.fee==.0003)&(periods.ablation=='FULL')]
    pivot=x.pivot(index='period',columns='rule',values='cagr').loc[['2012-2015','2016-2020','2021-2026']]
    fig,ax=plt.subplots(figsize=(10,5.6));pivot.rename(columns=LABELS).plot.bar(ax=ax,color=[COLORS[c] for c in pivot.columns],width=.78)
    ax.axhline(0,color='black',lw=.8);ax.set(title='Annualized return by predeclared subperiod',xlabel='Subperiod',ylabel='CAGR')
    ax.yaxis.set_major_formatter(lambda v,pos:f'{v:.0%}');ax.tick_params(axis='x',rotation=0);ax.legend(ncol=2)
    fig.text(.01,.01,'Source: same fixed-15 accounts; 3 bp each buy/sell. Subperiods fixed before reading results.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'02_subperiod_cagr.png',dpi=180);plt.close(fig)
    z=periods[(periods.fee==.0003)&periods.rule.isin(['BASE_5D','MULTI_HORIZON'])&periods.period.eq('2021-2026')].pivot(index='ablation',columns='rule',values='cagr')
    z=z.loc[['FULL','NO_BUFFER','NO_CORRELATION','NO_INDUSTRY_CAP','NO_TECH_CAP']]
    fig,ax=plt.subplots(figsize=(10,5.8));z.rename(columns=LABELS).plot.bar(ax=ax,color=[COLORS['BASE_5D'],COLORS['MULTI_HORIZON']],width=.72)
    ax.axhline(0,color='black',lw=.8);ax.set(title='Selection-rule ablations: recent annualized return',xlabel='Rule configuration',ylabel='CAGR, 2021-2026')
    ax.yaxis.set_major_formatter(lambda v,pos:f'{v:.1%}');ax.tick_params(axis='x',rotation=18);ax.legend()
    fig.text(.01,.01,'Each variant removes one rule only. Effects interact and are not additive. Source: PIT study, 3 bp each side.',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/'03_ablation_recent_cagr.png',dpi=180);plt.close(fig)
    weights=pd.read_csv(SRC/'monthly_weights.csv');recent=weights[weights.date.ge('2021-01-01')].groupby('rule').mean(numeric_only=True)
    cols=['PE','CASH_PROFIT_QUALITY','PROFIT_GROWTH','MOMENTUM','MOMENTUM_60D','REVERSAL_5D','VOLATILITY']
    z=recent.loc[['BASE_5D','TREND_FLOOR','MULTI_HORIZON'],cols].T
    fig,ax=plt.subplots(figsize=(10,6));z.rename(columns=LABELS).plot.barh(ax=ax,color=[COLORS['BASE_5D'],COLORS['TREND_FLOOR'],COLORS['MULTI_HORIZON']],width=.75)
    ax.set(title='Average nominal factor weights since 2021',xlabel='Average weight',ylabel='Factor');ax.xaxis.set_major_formatter(lambda v,pos:f'{v:.0%}');ax.legend()
    fig.text(.01,.01,'Multi-horizon still assigns zero average weight to both trend factors after2021; trend-floor forces10% each.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'04_recent_factor_weights.png',dpi=180);plt.close(fig)
    print('Created',len(list(OUT.glob('*.png'))),'article figures')


if __name__=='__main__':run()
