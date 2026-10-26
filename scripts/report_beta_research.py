#!/usr/bin/env python3
"""Render fixed research results; figures and tables do not change selection."""
import json
import argparse
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.run_beta_research import OUTPUT
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.report_mvp_holdings_count import LABELS
from analysis.holdings_count import metrics
from analysis.beta_research import market_regression

NAMES={'BASE_ELIGIBLE':'基础可交易池','COMMON':'共同样本等权','BETA_Q1':'最低 Beta 组 Q1',
       'BETA_Q2':'Q2','BETA_Q3':'Q3','BETA_Q4':'Q4','BETA_Q5':'最高 Beta 组 Q5',
       'LOW_BETA50':'低 Beta 50只','LOW_VOL252_50':'低波动252日 50只','LOW_VOL20_50':'低波动20日 50只',
       'LOW_BETA126_50':'低 Beta126日 50只','LOW_BETA_CSI300_50':'沪深300 Beta 50只'}


def markdown_table(frame):
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+ '|'.join(['---']*len(frame.columns))+'|\n'+ '\n'.join('| '+' | '.join(map(str,r))+' |' for r in frame.itertuples(index=False,name=None))


def report(out=OUTPUT):
    check=json.loads((out/'delivery_validation.json').read_text())
    if check['status']!='PASS':raise ValueError('validation required')
    completed=json.loads((out/'completed.json').read_text())
    for name,digest in completed['files'].items():
        if sha(out/name)!=digest:raise ValueError('changed output '+name)
    perf=pd.read_csv(out/'performance.csv');main=perf[np.isclose(perf.fee,.0003)].set_index('variant')
    stress=perf[np.isclose(perf.fee,.0013)].set_index('variant')
    nav=pd.read_csv(out/'nav.csv',index_col='date',parse_dates=True)
    cash=pd.read_csv(out/'cash_control_nav.csv',index_col='date',parse_dates=True).post_nav
    market=pd.read_csv(out/'market/000985.CSI.csv',dtype={'trade_date':str})
    market.index=pd.to_datetime(market.trade_date,format='%Y%m%d');mr=market.close.sort_index().pct_change(fill_method=None)
    returns=nav.pct_change(fill_method=None);aligned_market=mr.reindex(nav.index)
    conditional=[]
    for key in ['COMMON','LOW_BETA50','LOW_VOL252_50','LOW_VOL20_50']:
        conditional.append(dict(variant=key,
            mean_return_on_market_up_days=returns.loc[aligned_market.gt(0),key].mean(),
            mean_return_on_market_down_days=returns.loc[aligned_market.lt(0),key].mean()))
    pd.DataFrame(conditional).to_csv(out/'conditional_returns.csv',index=False)
    cm={k:float(v[0]) for k,v in metrics(cash).items()};cm.update(market_regression(cash.pct_change(fill_method=None),mr))
    write_json(out/'cash_control_metrics.json',cm)
    diagnostics=pd.read_csv(out/'signal_diagnostics.csv');six=pd.read_csv(out/'six_strategy_risk.csv')
    sub=pd.read_csv(out/'subperiods.csv');ic=pd.read_csv(out/'ic_summary.csv')
    exp=pd.read_csv(out/'target_exposure.csv');industries=pd.read_csv(out/'industry_weights.csv')
    cov=pd.read_csv(out/'coverage.csv');validation=json.loads((out/'validation.json').read_text())
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'font.family':['Arial Unicode MS','DejaVu Sans']})
    caption=f'Local Tushare / frozen historical pool | {nav.index[0].date()}–{nav.index[-1].date()} | 3 bps each side'
    groups=[f'BETA_Q{i}' for i in range(1,6)];x=np.arange(5)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,field,title,scale in zip(axes.ravel(),['cagr','volatility','max_drawdown','beta'],
        ['Annualized return (%)','Annualized volatility (%)','Maximum drawdown (%)','Realized market beta'],[100,100,100,1]):
        vals=main.loc[groups,field]*scale
        ax.bar(x,vals,color='#397c91');ax.set_xticks(x,[f'Q{i}' for i in range(1,6)])
        ax.set(xlabel='Ascending historical beta quintile (Q1 = lowest)',ylabel=title,title=title)
        if field=='max_drawdown':ax.set_ylim(float(vals.min())*1.15,0)
        for i,v in enumerate(vals):ax.annotate(f'{v:.2f}',(i,v),xytext=(0,4 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Does lower beta mean better returns, or less market risk?\n'+caption,fontsize=12)
    fig.savefig(figures/'01_quintiles.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True,constrained_layout=True)
    curves={'COMMON':('Common universe','#777777'),'LOW_BETA50':('Low beta K50','#1b728e'),
            'LOW_VOL252_50':('Low vol 252 K50','#b46a35'),'LOW_VOL20_50':('Low vol 20 K50','#6a6295')}
    for key,(label,color) in curves.items():
        s=nav[key];axes[0].plot(s.index,s,label=label,color=color,lw=1.3)
        axes[1].plot(s.index,(s/s.cummax()-1)*100,label=label,color=color,lw=1)
    axes[0].plot(cash.index,cash,label='Common universe + cash (target beta matched)',color='#779359',ls='--')
    axes[1].plot(cash.index,(cash/cash.cummax()-1)*100,label='Common universe + cash',color='#779359',ls='--')
    axes[0].set(yscale='log',ylabel='Post-cost NAV (log scale; start = 1)',title='Same dates, weekly next-close execution')
    axes[1].set(ylabel='Drawdown (%)',xlabel='Date',title='Peak-to-trough loss')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left',ncol=2)
    fig.suptitle('Selection versus risk reduction\n'+caption,fontsize=12)
    fig.savefig(figures/'02_nav_drawdown.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    chosen=['COMMON','LOW_BETA50','LOW_VOL252_50','LOW_VOL20_50'];periods=sorted(sub.period.unique())
    for j,key in enumerate(chosen):
        g=sub[sub.variant.eq(key)&np.isclose(sub.fee,.0003)].set_index('period').reindex(periods)
        for ax,col in zip(axes,['cagr','max_drawdown']):
            ax.bar(np.arange(3)+(j-1.5)*.19,g[col]*100,width=.18,label=curves[key][0],color=curves[key][1])
    for ax,title in zip(axes,['Annualized return (%)','Maximum drawdown (%)']):
        ax.set_xticks(range(3),periods);ax.set(xlabel='Historical subperiod',ylabel=title,title=title);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Stability across already-observed historical periods\n'+caption,fontsize=12)
    fig.savefig(figures/'03_subperiods.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    keys=['overlap_vol252','overlap_vol20','overlap_short','overlap_market']
    axes[0].bar(range(4),diagnostics[keys].mean()*100,color='#397c91')
    axes[0].set_xticks(range(4),['Low vol\n252d','Low vol\n20d','Beta\n126d','CSI300\nbeta'])
    axes[0].set(ylabel='Mean common holdings / 50 (%)',xlabel='Comparator',title='Overlap with low-beta K50',ylim=(0,100))
    factors=['beta','vol252','vol20','controlled_beta']
    for j,h in enumerate([5,20]):
        y=ic[ic.horizon.eq(h)].set_index('factor').reindex(factors).mean_ic
        axes[1].bar(np.arange(4)+(j-.5)*.35,y,width=.34,label=f'{h}-session future return')
    axes[1].set_xticks(range(4),['Low beta','Low vol\n252d','Low vol\n20d','Low beta\ncontrolled'])
    axes[1].axhline(0,color='gray',lw=.7);axes[1].set(ylabel='Mean weekly cross-sectional RankIC',xlabel='Score (lower risk ranks higher)',title='Return information, not risk prediction');axes[1].legend(fontsize=8)
    fig.suptitle('Overlap and future-return diagnostics (IC before costs; overlapping labels)\n'+caption,fontsize=12)
    fig.savefig(figures/'04_overlap_ic.png',dpi=150);plt.close(fig)
    rolling=pd.read_csv(out/'six_strategy_rolling_beta.csv',index_col='date',parse_dates=True)
    weekly=pd.read_csv(out/'six_strategy_target_beta.csv',parse_dates=['date'])
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,(key,(_,label)) in zip(axes.ravel(),LABELS.items()):
        ax.plot(rolling.index,rolling[key],label='Realized NAV beta (252d)',lw=1,color='#397c91')
        w=weekly[weekly.strategy.eq(key)]
        ax.plot(w.date,w.target_beta_known_normalized,label='Target beta (known-stock normalized)',lw=.8,alpha=.65,color='#b46a35')
        ax.axhline(1,color='gray',ls='--',lw=.6);ax.set(title=label,xlabel='Date',ylabel='Beta to CSI All Share',ylim=(.4,1.7));ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Six unchanged FIXED K50 historical strategies: target estimates are NOT actual holdings\n'+caption,fontsize=12)
    fig.savefig(figures/'05_six_strategy_beta.png',dpi=150);plt.close(fig)
    top=industries[industries.variant.isin(['LOW_BETA50','COMMON'])].groupby(['variant','industry']).weight.mean().unstack(0,fill_value=0)
    # Mean includes zero-weight dates, not only dates present in a portfolio.
    top=industries[industries.variant.isin(['LOW_BETA50','COMMON'])].groupby(['variant','industry']).weight.sum().unstack(0,fill_value=0)/validation['signal_count']
    top=top.sort_values('LOW_BETA50').tail(10)
    fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True);y=np.arange(len(top))
    ax.barh(y-.18,top.LOW_BETA50*100,height=.34,label='Low beta K50',color='#397c91')
    ax.barh(y+.18,top.COMMON*100,height=.34,label='Common universe',color='#999999')
    ax.set_yticks(y,top.index);ax.set(xlabel='Mean target industry weight (%)',ylabel='Historical industry label',title='Low beta can carry large industry tilts');ax.legend();ax.grid(axis='x',alpha=.2)
    fig.suptitle(caption+'\nIndustry classification/version coverage remains a limitation',fontsize=10)
    fig.savefig(figures/'06_industry_weights.png',dpi=150);plt.close(fig)
    dest=ROOT/'docs/articles/assets/article145_beta';dest.mkdir(parents=True,exist_ok=True)
    for f in figures.glob('*.png'):shutil.copy2(f,dest/f.name)
    rows=[]
    for name,r in main.iterrows():
        rows.append([NAMES[name],f'{100*r.total_return:.2f}%',f'{100*r.cagr:.2f}%',f'{100*r.volatility:.2f}%',f'{r.sharpe:.2f}',f'{100*r.max_drawdown:.2f}%',f'{r.beta:.3f}',f'{100*stress.loc[name,"cagr"]:.2f}%'])
    table=markdown_table(pd.DataFrame(rows,columns=['组合','累计收益','年化收益','年化波动','夏普','最大回撤','实际Beta','13bp年化收益']))
    rows=[]
    for _,r in six.iterrows():
        rows.append([LABELS[r.strategy][0],f'{100*r.cagr:.2f}%',f'{100*r.max_drawdown:.2f}%',f'{r.beta:.3f}',f'{100*r.r_squared:.1f}%',f'{100*r.annual_intercept:.2f}%',f'{r.mean_target_beta:.3f}',f'{100*r.mean_beta_known_weight:.2f}%'])
    sixtable=markdown_table(pd.DataFrame(rows,columns=['六策略原版本','年化收益','最大回撤','实际Beta','R²','年化回归截距','平均目标Beta','Beta已知权重']))
    summary={'first_signal':validation['first_signal'],'first_execution':validation['first_execution'],'end':validation['last_date'],
        'signals':validation['signal_count'],'base_mean':cov.loc[cov.included,'base_names'].mean(),
        'common_mean':cov.loc[cov.included,'common_names'].mean(),'common_min':int(cov.loc[cov.included,'common_names'].min()),
        'mean_beta_vol_rank_corr':diagnostics.beta_vol_rank_corr.mean(),
        'mean_overlaps':diagnostics[['overlap_vol252','overlap_vol20','overlap_short','overlap_market']].mean().to_dict(),
        'controlled_coverage':diagnostics.controlled_coverage.mean(),'cash_control':cm,
        'cash_control_mean_stock_budget':pd.read_csv(out/'cash_control_stock_budget.csv').stock_budget.mean(),
        'max_terminal_unquoted_fraction':main.terminal_unquoted_fraction.max(),
        'target_exposure':exp.groupby('variant')[['target_beta','beta_weight_coverage','max_industry_weight','unknown_industry_weight']].mean().to_dict(),
        'top_low_beta_industries':top.LOW_BETA50.sort_values(ascending=False).to_dict()}
    write_json(out/'report_summary.json',summary)
    lines=['# Beta 长历史研究：数据结果','',f'信号 {summary["first_signal"]}，首次执行 {summary["first_execution"]}，截至 {summary["end"]}；共 {summary["signals"]} 次周频信号。','',
           '## 组合结果','',table,'','## 六策略风险诊断','',sixtable,'',
           'Beta 与截距为原始日收益回归；不是超额收益 CAPM，也不证明独立选股 alpha。夏普无风险收益设零。历史已被观察；不是新样本外验收。',
           '','## 因子 IC','',markdown_table(ic.round(5)),'','## 子区间','',markdown_table(sub[np.isclose(sub.fee,.0003)].round(5))]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(table);print(sixtable)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT)
    report(ap.parse_args().output)
