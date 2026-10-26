#!/usr/bin/env python3
"""Build the new article from sealed candidate-factor experiments."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scripts.run_candidate_factors import OUT
from scripts.run_mvp_holdings_count import sha
from analysis.candidate_factors import CANDIDATES

NAMES={'BASE':'原13因子','DIVIDEND_TTM':'＋股息率','LOW_ACCRUALS':'＋低应计利润率',
       'LOW_ASSET_GROWTH':'＋低资产增长','GROSS_PROFIT_ASSETS':'＋总资产毛利率','MOMENTUM_12_1':'＋跳月动量',
       'ALL5':'＋全部五个','COMMON_BASE':'共同样本／原13因子','COMMON_ALL5':'共同样本／加五个'}
SHORT={'DIVIDEND_TTM':'Dividend','LOW_ACCRUALS':'Low accruals','LOW_ASSET_GROWTH':'Low asset growth',
       'GROSS_PROFIT_ASSETS':'Gross profit/assets','MOMENTUM_12_1':'Momentum12-1','ALL5':'All five'}
FREQ={'WEEKLY':'每周调仓','MONTHLY':'每月调仓'}
OLD_NAMES={'PE':'PE估值','ROE':'ROE','CASH_PROFIT_QUALITY':'现金利润质量','REVENUE_GROWTH':'营收增长','PROFIT_GROWTH':'利润增长'}
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def build():
    v=json.loads((OUT/'independent_validation.json').read_text());assert v['status']=='PASS'
    for name,digest in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/name)==digest,name
    perf=pd.read_csv(OUT/'performance.csv');sub=pd.read_csv(OUT/'subperiods.csv')
    nav=pd.read_csv(OUT/'nav.csv',index_col='date',parse_dates=True)
    weights=pd.read_csv(OUT/'factor_weights.csv',parse_dates=['date']);coverage=pd.read_csv(OUT/'coverage.csv',parse_dates=['date'])
    dailyic=pd.read_csv(OUT/'daily_ic.csv',parse_dates=['date']).set_index('date')
    redundancy=pd.read_csv(OUT/'redundancy.csv.gz');annual=pd.read_csv(OUT/'annual.csv')
    pct=lambda x:f'{x:.2%}'
    def row(arm,freq,fee=.0003,period='FULL'):
        return sub[sub.arm.eq(arm)&sub.frequency.eq(freq)&np.isclose(sub.fee,fee)&sub.period.eq(period)].iloc[0]
    def fulltable(freq):
        rows=[]
        for a in ['BASE']+CANDIDATES+['ALL5']:
            r=row(a,freq);r2=row(a,freq,period='RECENT');p=perf[perf.arm.eq(a)&perf.frequency.eq(freq)&np.isclose(perf.fee,.0003)].iloc[0]
            rows.append([NAMES[a],pct(r.total_return),pct(r.cagr),pct(r.volatility),f'{r.sharpe:.3f}',pct(r.max_drawdown),f'{p.annual_turnover:.2f}倍',pct(r2.cagr)])
        return table(['版本','全期累计','全期年化','波动','夏普','最大回撤','年均双边换手','2021年以来年化'],rows)
    costs=table(['频率','版本','零成本全期年化','3bp全期年化','13bp全期年化','3bp近年年化','13bp近年年化'],[
        [FREQ[f],NAMES[a]]+[pct(row(a,f,fee,period).cagr) for period,fee in [('FULL',0),('FULL',.0003),('FULL',.0013),('RECENT',.0003),('RECENT',.0013)]]
        for f in FREQ for a in ['BASE']+CANDIDATES+['ALL5']])
    matched=table(['频率','样本与评分','全期年化','2021年以来年化','全期最大回撤'],[
        [FREQ[f],NAMES[a],pct(row(a,f).cagr),pct(row(a,f,period='RECENT').cagr),pct(row(a,f).max_drawdown)]
        for f in FREQ for a in ['BASE','ALL5','COMMON_BASE','COMMON_ALL5']])
    empty_description='、'.join(f'{FREQ[f]}{int(perf[perf.arm.eq("COMMON_BASE")&perf.frequency.eq(f)].iloc[0].empty_signal_pools)}次' for f in FREQ)
    covtable=table(['新增因子','决策日平均有效覆盖','早期2012—2015','近期2021年以来'],[
        [NAMES[c].removeprefix('＋')]+[pct(g[c].mean()) for g in [coverage,coverage[coverage.date.dt.year<=2015],coverage[coverage.date.dt.year>=2021]]] for c in CANDIDATES])
    ic_table=table(['新增因子','全期日度RankIC均值','近年日度RankIC均值','全期有效观测日','与原因子最大绝对平均相关'],[
        [NAMES[c].removeprefix('＋'),f'{dailyic.loc["2012":,c].mean():.4f}',f'{dailyic.loc["2021":,c].mean():.4f}',dailyic.loc['2012':,c].count(),
         f'{OLD_NAMES.get(r.old_factor,r.old_factor)}：{r.correlation:+.3f}']
        for c in CANDIDATES for mean in [redundancy[redundancy.candidate.eq(c)].groupby('old_factor').correlation.mean()]
        for r in [mean.reindex(mean.abs().sort_values(ascending=False).index).reset_index().iloc[0]]])
    w=weights[weights.arm.eq('ALL5')&weights.date.ge('2012')]
    wt=table(['新增因子','月度更新点平均名义权重','最高权重','零权重更新点比例','近年平均名义权重'],[
        [NAMES[c].removeprefix('＋'),pct(g.weight.mean()),pct(g.weight.max()),pct(g.weight.le(1e-10).mean()),pct(g.loc[g.date.ge('2021'),'weight'].mean())]
        for c in CANDIDATES for g in [w[w.factor.eq(c)]]])
    overlap=[]
    for f in FREQ:
        signals=pd.DatetimeIndex(json.loads((OUT/'schedules.json').read_text())[f]);sets={}
        for a in ['BASE','ALL5']:
            t=pd.read_csv(OUT/f'{a}_{f}_3bp'/'targets.csv.gz',parse_dates=['date'])
            sets[a]={d:set(g.symbol) for d,g in t[t.date.isin(signals)].groupby('date')}
        common=[len(sets['BASE'].get(d,set())&sets['ALL5'].get(d,set())) for d in signals]
        for a in ['BASE','ALL5']:
            r=perf[perf.arm.eq(a)&perf.frequency.eq(f)&np.isclose(perf.fee,.0003)].iloc[0]
            overlap.append([FREQ[f],NAMES[a],pct(r.average_equity),int(r.actual_trade_days),int(r.risk_only_days),f'{np.mean(common):.2f}只' if a=='ALL5' else '对照'])
    holdings=table(['频率','版本','平均股票仓位','实际成交日数','仅风控成交日','定期目标与原方案平均重合'],overlap)
    outcome=[]
    for f in FREQ:
        baseline=row('BASE',f);joint=row('ALL5',f);recent=row('ALL5',f,period='RECENT');oldrecent=row('BASE',f,period='RECENT')
        singles=[(c,row(c,f)) for c in CANDIDATES];best=max(singles,key=lambda x:x[1].cagr)
        outcome.append(f'- **{FREQ[f]}**：全部加入后，全期年化由{pct(baseline.cagr)}变为{pct(joint.cagr)}，最大回撤由{pct(baseline.max_drawdown)}变为{pct(joint.max_drawdown)}；2021年以来年化由{pct(oldrecent.cagr)}变为{pct(recent.cagr)}。单独新增版本中，全期年化最高的是{NAMES[best[0]]}，为{pct(best[1].cagr)}。这只是本轮已测试版本的历史排序，不是新的上线标准。')
    interpretation=f'''这组结果不是“五个因子全好”或“新增因子全没用”，而是不同信息与组合频率出现了不同的配合：

- **股息率**：周度全期年化为{pct(row('DIVIDEND_TTM','WEEKLY').cagr)}，最大回撤为{pct(row('DIVIDEND_TTM','WEEKLY').max_drawdown)}，相比原方案有改善；但月度近年年化为{pct(row('DIVIDEND_TTM','MONTHLY',period='RECENT').cagr)}，不能据此说高股息已经解决近年收益。
- **低应计利润率**：周度近年年化为{pct(row('LOW_ACCRUALS','WEEKLY',period='RECENT').cagr)}，13bp下仍为{pct(row('LOW_ACCRUALS','WEEKLY',.0013,'RECENT').cagr)}；月度近年却为{pct(row('LOW_ACCRUALS','MONTHLY',period='RECENT').cagr)}。频率不同，组合效果明显不同。
- **低资产增长**：周、月版本的全期和近年3bp年化均高于各自原方案。月度近年13bp年化为{pct(row('LOW_ASSET_GROWTH','MONTHLY',.0013,'RECENT').cagr)}；周度同口径仍为{pct(row('LOW_ASSET_GROWTH','WEEKLY',.0013,'RECENT').cagr)}，仍没有摆脱成本压力。
- **总资产毛利率**：月度全期年化达到{pct(row('GROSS_PROFIT_ASSETS','MONTHLY').cagr)}，但最大回撤为{pct(row('GROSS_PROFIT_ASSETS','MONTHLY').max_drawdown)}。收益增加与回撤改善不是同一件事。
- **跳月动量**：本次单独加入后，周、月两个版本的全期3bp年化都低于原方案，最大回撤也更深。本轮没有依据把它作为确定的改进，不能仅凭它是常用因子就强行上生产。

这些是结果描述，不是已经完成的因果归因。特别是低资产增长：全期平均IC即使不强，也可能在部分历史窗口获得正权重；它加入后还会改变其他因子的配比。不能用一行全期平均IC解释所有交易结果。'''
    phase=table(['频率','阶段','原13因子年化','加五个年化','年化差'],[
        [FREQ[f],{'EARLY':'2012—2015','MID':'2016—2020','RECENT':'2021—2026.10.15'}[period],pct(row('BASE',f,period=period).cagr),pct(row('ALL5',f,period=period).cagr),f'{100*(row("ALL5",f,period=period).cagr-row("BASE",f,period=period).cagr):+.2f}个百分点']
        for f in FREQ for period in ['EARLY','MID','RECENT']])
    byyear=annual[np.isclose(annual.fee,.0003)&annual.arm.isin(['BASE','ALL5'])].pivot(index='year',columns=['frequency','arm'],values='return_')
    yearstable=table(['年份','周／原方案','周／加五个','月／原方案','月／加五个'],[[str(y)+('（截至10月15日）' if y==2026 else '')]+[pct(r[(f,a)]) for f in FREQ for a in ['BASE','ALL5']] for y,r in byyear.iterrows()])
    indices=[]
    from analysis.holdings_count import metrics
    for code,name in [('H00300.CSI','沪深300全收益'),('H00016.CSI','上证50全收益'),('POOL_EQ_3bp','原池每周等权，3bp')]:
        s=nav[code];recent=s.loc['2021':];pos=s.index.get_loc(recent.index[0]);recent=pd.concat([s.iloc[pos-1:pos],recent]);a=metrics(s);b=metrics(recent)
        indices.append([name,pct(a['total_return'][0]),pct(a['cagr'][0]),pct(b['cagr'][0]),pct(a['max_drawdown'][0])])
    benchmarks=table(['参考线','全期累计','全期年化','近年年化','全期最大回撤'],indices)
    assets=ROOT/'docs/research/assets/candidate_factors';assets.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#355b83','#be7544','#7b8993','#699380']
    for period,filename in [('FULL','01_full_nav_drawdown.png'),('RECENT','02_recent_nav_drawdown.png')]:
        fig,axs=plt.subplots(2,2,figsize=(13,8),sharex='col')
        for col,f in enumerate(FREQ):
            keys=[f'BASE_{f}_3bp',f'ALL5_{f}_3bp','H00300.CSI','H00016.CSI'];labels=['Original13','Add all five','CSI300 gross TR','SSE50 gross TR']
            for key,label,color in zip(keys,labels,colors):
                s=nav[key]
                if period=='RECENT':s=s.iloc[np.flatnonzero(s.index.year>=2021)[0]-1:]
                s=s/s.iloc[0];axs[0,col].plot(s.index,s,label=label,color=color,lw=1.2)
                axs[1,col].plot(s.index,s/s.cummax()-1,color=color,lw=1)
            axs[0,col].set_title(f'{f}: {period}, 3bp per side');axs[0,col].set_ylabel('NAV (log scale)');axs[0,col].set_yscale('log');axs[0,col].legend(fontsize=8)
            axs[1,col].set_ylabel('Drawdown');axs[1,col].set_xlabel('Date');axs[1,col].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axs.flat:ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(assets/filename,dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5.5));added=CANDIDATES+['ALL5']
    for ax,period in zip(axs,['FULL','RECENT']):
        y=np.arange(len(added))
        for k,f in enumerate(FREQ):
            vals=[100*(row(a,f,period=period).cagr-row('BASE',f,period=period).cagr) for a in added]
            ax.barh(y+(k-.5)*.35,vals,height=.35,label=f,color=colors[k])
        ax.set_yticks(y,labels=[SHORT[a] for a in added]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7)
        ax.set_xlabel('Annualized return change vs original13 (percentage points)');ax.set_title(f'{period}: same pool, 3bp');ax.legend();ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(assets/'03_incremental_returns.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,period in zip(axs,['FULL','RECENT']):
        for f,color in zip(FREQ,colors):
            for a,style in [('BASE','--'),('ALL5','-')]:
                ax.plot([0,3,13],[row(a,f,bp/10000,period).cagr for bp in [0,3,13]],ls=style,color=color,marker='o',label=f'{f} / {a}')
        ax.set_title(f'{period}: trading cost sensitivity');ax.set_xlabel('Cost per buy or sell (bp)');ax.set_ylabel('Annualized return');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xticks([0,3,13]);ax.axhline(0,color='black',lw=.5);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(assets/'04_cost_sensitivity.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5));wp=w.pivot(index='date',columns='factor',values='weight')[CANDIDATES];wp['Original13']=1-wp.sum(axis=1)
    ax.stackplot(wp.index,*[wp[c] for c in wp],labels=[SHORT.get(c,c) for c in wp],alpha=.8,step='post')
    ax.set_ylim(0,1);ax.set_ylabel('Nominal factor weight');ax.set_xlabel('Monthly update date');ax.set_title('ALL5: realized historical RankIC allocation (not stock holdings)');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncol=3,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.0))
    fig.tight_layout();fig.savefig(assets/'05_actual_factor_weights.png',dpi=160);plt.close(fig)
    article=f'''# 给15只多因子组合增加五个因子：股息、利润兑现、资产效率与长期动量的历史验证

研究日期：2026-10-22。回测范围：2012-01-06—2026-10-15；首次模拟成交：2012-01-09收盘。状态：代码、54组回放与独立校验完成；未更改六策略前向纸面账户。

## 一、背景：原方案并非毫无价值，但还没有解决所有问题

前面的研究已经得到两类不能混淆的结论。

一方面，固定15只组合通过分散、持仓缓冲和仓位管理降低了风险。在较低成本下，它也跑赢了本次对照的沪深300和上证50全收益指数。这是实际得到的历史结果，不能因为仍有不足，就把它说成“什么都没解决”。这里的上证50并不是上证综指，不能混用名称。

另一方面，近年的绝对收益偏弱，换手带来的损耗仍然明显；整体RankIC为正，也没有保证最终15只股票持续优于自身股票池。因此，下一步不是不断添加交易开关，而是检查：**原有因子是否缺少一些真正有用的信息？新增信息能不能通过同一套组合规则，转化为扣费后的收益？**

我们此前列出六个候选，其中Beta已经单独研究。本轮不重复加入Beta，只测试其余五个：股息率、应计利润率、总资产增长率、总资产毛利率，以及跳过最近一个月的中长期动量。

## 二、这五个因子分别看什么

| 新增因子 | 本轮计算口径与方向 | 与已有信息的区别 |
| --- | --- | --- |
| 股息率 | Tushare历史`dv_ttm÷100`，数值较高得分较高 | 公司已经实施派现的历史信息，不是承诺将来分红 |
| 低应计利润率 | `－（年度合并净利润－同年度经营现金流净额）÷平均总资产` | 比较账面利润与现金兑现，并按资产规模缩放；不是当前现金利润质量的简单改名 |
| 低资产增长 | `－（本年末总资产÷上一年末总资产－1）` | 测试较保守的资本扩张，而不是偏好资产增长越快越好 |
| 总资产毛利率 | `（年度营业收入－营业成本）÷同年末总资产`，高为优 | 已有毛利率按收入缩放，这里按资产缩放，观察资产创造毛利的效率 |
| 跳月动量 | `复权价[t－21]÷复权价[t－252]－1`，高为优 | 约12个月前到1个月前的趋势，排除最近21个交易日的价格变化 |

应计利润率使用的平均资产，是本年末和**精确上一年末**总资产的平均；不是随便找最近两份报表，也不是原始Sloan论文公式的完整复刻。总资产毛利率则沿用已有年度研究的期末资产口径，不混称平均资产口径。

净利润使用包含少数股东损益的合并口径`n_income`，与合并现金流和总资产对应，不混用归母利润。原始字段含义分别核对了[Tushare利润表](https://tushare.pro/document/2?doc_id=33)、[现金流量表](https://tushare.pro/document/2?doc_id=44)和[资产负债表](https://tushare.pro/document/2?doc_id=36)。

举个简单例子：平均资产100亿元、净利润10亿元、经营现金流15亿元，应计利润率是－5%，取负后的因子分数为＋5%；如果现金流只有5亿元，取负后则是－5%。它表达的是利润与现金流之间的差异，不是直接判定企业是否造假。同样创造30亿元毛利，使用100亿元资产和300亿元资产，对应的总资产毛利率分别是30%与10%，这就与销售毛利率区别开了。

三个财务候选只在非金融公司计算。银行等公司的资产与现金流性质不同，本轮不给它们套用这三个工业企业口径；但在主实验中，**没有因此把金融股踢出股票池**。低资产增长暂放入成长组，是资本扩张的反向补充信号；归组是一项实验设计，不意味着“低资产增长等于高成长”。

### 股息率需要特别说明

依据[Tushare每日指标文档](https://tushare.pro/document/2?doc_id=32)，本轮使用供应商的TTM股息率，涉及近12个月除息及分红报告期的限制。它不是预测股息率，也不是我们从每笔派息重新构建的无限制过去一年现金总和。

本地`daily_basic`是周频采样。每个信号只能读取不晚于当天、最多7个自然日之前的最新快照，不向未来找值、不将旧股息率按新股价重新估算。周末信号通常能取得当天值，其他日子可能滞后；月末也遵守相同规则。字段为空时，无法仅凭这张表区分“不分红”和“缺记录”，因此**不把空值冒充零股息率**。这意味着本轮股息研究还不是逐事件核验完备的最终版本。

所以，也不能把本轮结果直接解释成“高股息一定战胜不分红”：部分空值在组合评分中只按中性贡献处理，并没有被证明是真实零股息。

## 三、接到哪套策略里，哪些规则没有改变

接入的是研究组合：**历史RankIC调权＋分散约束＋持仓缓冲＋固定15只＋总仓位管理**，不是日报正在运行的50只组合。

### 1. 实验分组：从同一个原方案出发，平行比较，不是逐个累加

**下面七个主版本各自独立回测。每一个“＋”都表示在原13因子基础上新增，不是在上一行的版本上继续新增。**

| 主版本 | 实际参与研究的因子配方 | 配方中的因子数量 |
| --- | --- | --- |
| 原方案 | 原13因子 | 13个 |
| 单独加股息率 | 原13因子＋股息率 | 14个 |
| 单独加低应计利润率 | 原13因子＋低应计利润率，不包含新增股息率 | 14个 |
| 单独加低资产增长 | 原13因子＋低资产增长 | 14个 |
| 单独加总资产毛利率 | 原13因子＋总资产毛利率 | 14个 |
| 单独加跳月动量 | 原13因子＋跳月动量 | 14个 |
| 五个全部加入 | 原13因子＋上述五个新增因子 | 18个 |

例如，“＋低应计利润率”不是“原13因子＋股息率＋低应计利润率”。它与“＋股息率”是两个平行版本，分别与相同频率、相同成本下的原方案比较。

本轮**没有测试先加一个、再加第二个、依次加到五个的递进过程，也没有穷举两两、三个或四个新增因子的组合**。因此，我们可以观察每个因子单独加入后的增量，以及五个联合加入的整体效果；但还不能据此判断哪一对子组合最好，也不能把几个单独版本的收益改善直接相加。

### 2. 为什么共有54组回测

七个主版本之外，还有两组“共同样本”诊断：在全部五项新增数据都有效的同一股票集合内，分别使用原13因子评分和加入五个后的评分，具体规则见第四、七节。

因此，实验数量是：**（7个主版本＋2个共同样本对照）×2种调仓频率（周、月）×3档单边成本（0、3、13bp）＝54组。** 沪深300、上证50等参考线不算在这54个策略账户内。

### 3. 每个版本独立调权，其他持仓规则保持一致

每个主版本都根据自己包含的因子，独立生成一条历史RankIC权重路径。**不是把原13因子的历史权重锁住，再额外给新因子一份预算**：加入一个因子后，原有因子的权重也可能发生变化。原方案自身的权重路径仍作为不变的对照；周、月调仓版本则共用对应配方的同一条调权路径，只改变定期换股日期。

所以，“单独加一个因子的效果”，准确地说是“新增这一项信息，并按既定规则重新分配全部因子权重后，整套组合的变化”，不是在其他权重完全不变时测得的孤立因子收益。配方写着14个或18个，也不意味着这些因子每个月都具有非零权重。


| 环节 | 本轮保持的规则 |
| --- | --- |
| 基础股票池 | 原有点时点可交易池及硬风险排除；主实验不因新增字段缺失再筛股票 |
| 股票打分 | 原13因子分别加入一个候选，或同时加入五个；保留原来的两层因子组标准化 |
| 调权证据 | 过去252个已成熟日度观测，评价下一交易日收盘买入后5个交易日收益；至少120个有效IC观测 |
| 因子权重 | 每月首个周度信号日更新；正平均RankIC分配，单因子最高20%，每次最多变5个百分点 |
| 股票数量 | 固定目标15只；从前100名按限制选，选不满的槽位留现金 |
| 分散与缓冲 | 行业最多3只，指定科技主题合计最多5只，历史相关系数不高于0.8；旧持仓在前30名内优先保留但仍须满足限制 |
| 股票仓位 | 拟持有组合过去60日波动估计，15%目标年化波动，预算20%—100%；每日检查，但有10个百分点幅度与5交易日冷却要求 |
| 交易时点 | 信号后的下一交易日收盘模拟；不是看到当日收盘价再按该收盘价买入 |
| 换股频率 | 同时保留周度、月度两个版本；月度版本仍可能发生单独的风险仓位调整 |
| 交易成本 | 每笔买入或卖出分别收0、3、13bp；3bp为0.03%，不是整次往返总费用 |

新增因子初始化为零，原13因子保留原来的起始配方；利用2011年历史逐月预热。新增因子有足够的正向历史证据后，才通过同一调权算法获得权重。没有强行保证“加了就必须买”，也没有另开一套事后挑赢家的准入规则。

五组预算并非永远各20%。更新后，各组预算是组内名义权重之和。因子权重影响打分，**不等于股票资金权重**；选满15只时，股票仍平分实际股票仓位。

新因子缺失时，在完成截面标准化后给其评分贡献中性零值；计算IC时则保留缺失，不伪造有效观测。原13因子的缺失处理和历史IC缓存保持原状。这使主实验不额外换股票池，但中性填充本身仍是一项建模选择，不是对真实数据缺口的修复。

## 四、先看数据覆盖，再看收益

财报使用`ann_date`与`f_ann_date`中的较晚日期，并要求**严格早于信号日**。只取当时已知的年度合并报表版本；同日冲突版本置缺失，以最新已知资产年报期锚定同报告期利润与现金流，不因为新一期不齐就退回上一套完整年报。报表年龄超过730天不用于这些新财务因子。

这些规则防止已知的未来披露泄漏，但供应商今天返回的历史版本，仍不等于完整的当年不可变数据库快照。历史更正与行业覆盖的限制继续保留。

{covtable}

以上是每个周/月联合决策截面内原比较池的有效比例，再对日期取平均，不是全市场全部股票的比例。全部五项同时有效的股票，平均约占原比较池的**{pct((coverage.complete/coverage.pool).mean())}**；原池每个信号平均{coverage.pool.mean():.0f}只，共同有效池平均{coverage.complete.mean():.0f}只。

因此另加两组诊断：同样限定全部五项有原始数据，一组仍按原13因子评分，一组按加入五个后的评分。两者使用各自已在原池学得的权重和标准化结果，只过滤可选股票；不重新训练一套共同池IC模型。

## 五、回测结果：单独加入和全部加入必须分开看

**读表提示：以下“＋某因子”均是“原13因子＋该因子”的简称，不承接上一行；“＋全部五个”才是18因子的联合版本。比较新增效果时，应对照同一张表中的“原13因子”行。**

以下默认每笔交易3bp；“全期”为2012年初至2026年10月15日，“近年”为2021年起至同一终点。累计收益是整个区间净值变化，年化采用252交易日口径；夏普的无风险利率设为0。

### 1. 每周调仓

{fulltable('WEEKLY')}

### 2. 每月调仓

{fulltable('MONTHLY')}

{chr(10).join(outcome)}

### 3. 怎样读这些差异

{interpretation}

![全期净值与回撤](assets/candidate_factors/01_full_nav_drawdown.png)

图1：左列周度、右列月度；上排是累计净值，下排是回撤。蓝线是原13因子，橙线是加入全部五个。指数使用全收益口径；策略已经扣3bp交易费用，而指数未扣购买基金等产品所需费用。图中没有把不同资产配置的收益差直接叫作风险调整后的alpha。

![新增因子的年化收益变化](assets/candidate_factors/03_incremental_returns.png)

图2：每根柱子是新增版本相对**同频率原13因子**的年化收益变化，单位为百分点；左侧全期，右侧近年。正数表示这次历史回放改善，负数表示变弱。它衡量的是“加入因子并重新调权”这一整项改动，不能全部归因于新因子自身的独立预测能力。

## 六、跑赢指数值得承认，但近年与费用不能省略

{benchmarks}

上表中的原池等权，是每个时点把池内股票按相同比例配置、每周重平衡的参考组合，不是固定15只，也不采用同样的风险仓位。它帮助辨别是否只是股票池和风格暴露带来的收益，但也不是与策略风险完全相同的基准。

我们保留两种判断：对投资结果而言，跑赢全收益指数有意义；对研究归因而言，还要知道这种领先是否只是不同股票池、风格和仓位造成。两者不是互相否定。

{phase}

![2021年以来净值与回撤](assets/candidate_factors/02_recent_nav_drawdown.png)

图3：各线从2020年末最后一个交易日重新归一化，避免2012—2020年积累的收益掩盖近年变化。这里并未在2021年重置策略、资金或历史因子权重，只是重新展示这一段真实持有路径。

### 成本压力测试

{costs}

![成本敏感性](assets/candidate_factors/04_cost_sensitivity.png)

图4：虚线原13因子，实线加五个；每种颜色代表一个换股频率。新增因子若让股票排名更不稳定，可能提高毛收益却增加交易损耗。因此既看低成本结果，也保留13bp压力测试；13bp仍不能代替真实最低佣金、税费和成交约束。

## 七、是不是只因为换了一批有数据的股票

{matched}

主实验的原方案与加五个版本，始终使用原固定评分有效池；这组对照并未通过“删除缺财务数据的股票”制造差异。

月度共同样本对照中，年化由{pct(row('COMMON_BASE','MONTHLY').cagr)}变为{pct(row('COMMON_ALL5','MONTHLY').cagr)}，近年年化由{pct(row('COMMON_BASE','MONTHLY',period='RECENT').cagr)}变为{pct(row('COMMON_ALL5','MONTHLY',period='RECENT').cagr)}。在这项诊断下，月度改善并未因固定数据可用样本而消失；但这仍不是对所有混杂因素的彻底排除。

共同样本组则回答另一个问题：在能够完整观察五个新增指标的同一截面里，改评分是否仍有帮助？比较时要看**共同样本原13因子与共同样本加五个**，而不是将共同样本新方案直接对上完整池旧方案，把样本筛选的效果也算成新因子的功劳。完整共同样本偏向有股息率记录、年报齐全和价格历史充足的非金融公司，它本身也会改变风格。

还有一个不能隐藏的现象：共同样本池为空的信号日分别是{empty_description}。长假之后，最近的周频股息率快照可能已超过7个自然日。这些时点两个共同样本组都按原来的“空槽留现金”规则提交空仓目标，直到下一次有效定期选股；主实验没有因此清仓。共同样本结果因此还包含数据可用性带来的现金阶段，不能冒充无缺口的纯因子实验。

## 八、新因子真的参与了吗，还是只加了名字

{ic_table}

这里的IC是日度截面Spearman相关，即RankIC；前面已按因子预设方向处理，所以正值才是支持预设方向的证据。相关表是决策截面上与原13因子的平均Spearman相关，先按因子对平均，再找绝对值最大的一对；它不是独立信息量的完整检验。5日收益标签相互重叠，不能把有效日数当作同等数量的独立样本，也没有在本轮宣称统计显著。

{wt}

权重表仅统计“全部五个同时加入”版本的月度更新点，周/月换股共用同一条权重路径。单独新增版本的权重另存于完整结果文件，不能把上表套用到单独新增实验。

![实际因子权重](assets/candidate_factors/05_actual_factor_weights.png)

图5：彩色区域是五个新增因子的名义权重，剩余区域是原13因子合计。某因子权重为零，说明这套历史调权机制当时没有分配预算，不等于已经证明该因子永远无效。这里仍然使用5日收益评价所有因子，没有因为某个长期财务信号表现不好就临时换评价期限。

### 评分改变后，持仓究竟变了多少

{holdings}

重合数量比较同一信号日的两套目标名单，不是实际成交后永远保持的仓位；未成交、缺报价和价格漂移仍由原执行模型处理。仓位管理规则虽没改，拟持有股票的波动发生变化后，算出来的仓位与额外风控交易次数也可能变化。因此，新增因子影响的不只是股票名字，还会沿着既有风险规则传导到组合路径。

## 九、逐年记录：避免只记住一行总收益

{yearstable}

2026年是不完整年度。上述结果只描述这套公式、缺失处理、调权和组合规则在已经观察过的历史上的表现，不能用“每隔一段时间选一次当时最优版本”的方式拼接成另一条更好曲线。

## 十、这轮实验解决了什么，下一步不做什么

本轮完成的是五个候选信息进入既有15只研究组合的可复现验证：有公式、有时点约束、有实际权重、有单独和联合消融，也有同频率、同成本及共同样本对照。它比单看某个因子IC或一张净值图更接近我们真正关心的组合效果。

**最值得保留的进展，是“月度换股＋五个新增因子”的整体结果。**全期3bp年化由{pct(row('BASE','MONTHLY').cagr)}提高到{pct(row('ALL5','MONTHLY').cagr)}，最大回撤由{pct(row('BASE','MONTHLY').max_drawdown)}收窄到{pct(row('ALL5','MONTHLY').max_drawdown)}；近年3bp年化由{pct(row('BASE','MONTHLY',period='RECENT').cagr)}提高到{pct(row('ALL5','MONTHLY',period='RECENT').cagr)}，13bp下近年年化仍为{pct(row('ALL5','MONTHLY',.0013,'RECENT').cagr)}。这不是毫无改善，值得继续保留为研究候选。

但同样加五个，周度近年3bp年化却降到{pct(row('ALL5','WEEKLY',period='RECENT').cagr)}。因此结论不是“因子越多越好”，而是**信息、换股频率、调权和成本必须作为一套组合来验证**。月度候选的近年收益也仍不高，回撤仍超过三成，不能称为已经能稳定盈利或足以实盘。

月度加五个的近年年化也仍低于上表原池每周等权参考线。两者的股票数量、仓位和风险不同，不能直接把这段差距解释成负alpha，但它提醒我们：尚不能宣称已经解决近年的选股超额问题。更合理的后续是保留原月度方案、月度加五个和较简洁的月度低资产增长版本作为少量冻结候选；先核对股息数据与真实交易约束，再用新增数据观察，而不是继续在这些历史年份上搜索更多组合。

但没有一次性解决最终持仓、集中风险和成本。新增因子主要改变选股信息；行业与相关限制仍负责分散，缓冲仍负责减少无谓交易，风险预算仍负责投入多少。不同因子可能继续偏向相同股票，不能因为因子名称变多就认为风险自然分散。

原13因子的周/月版本在三档成本下都与上一轮逐日净值和交易记录一致。独立校验覆盖{v['target_checks']}组目标持仓、{v['industry_theme_selection_checks']}次3bp版本定期选择，以及现金、费用、下一日成交、标签成熟度、权重上限与披露时点；工程测试506项通过。六策略前向账户的输入与账本哈希前后一致。

回放仍使用可分割复权股数，未完整模拟整手、最低佣金、实际印花税、涨跌停排队和容量；不是小账户实盘验收。复权价格已反映公司行动，不再额外把股息率加到收益上，避免重复计算分红。

另外，原13因子中名为`FREE_CASH_FLOW_YIELD`的历史实现，实际是每股经营现金流与价格的比值代理，不是严格扣除资本开支后的自由现金流收益率。本轮为保护对照没有顺便修正这个旧名称和公式，不能把它作为已经准确具备自由现金流因子的证据。

**我们现在可以根据结果缩小下一步候选，但不能把本轮历史最高收益直接当成正式策略。**尤其是股息字段的缺失语义与周频滞后，需要先补齐逐事件或更完整日频验证；真正要改日报配方，还应冻结一个候选，在后续未参与设计的数据上继续前向记录。

### 复现与结果文件

- 协议：`research_protocols/20261022_candidate_factors_v1.json`
- 因子实现：`analysis/candidate_factors.py`
- 完整运行：`.venv/bin/python scripts/run_candidate_factors.py --stage all`（已封存目录禁止覆盖；复跑应使用新的输出目录与协议版本）
- 独立校验：`.venv/bin/python scripts/validate_candidate_factors.py`
- 本文生成：`.venv/bin/python scripts/report_candidate_factors.py`
- 明细：`output/candidate_factors_20261022/`，包含54组逐日净值、交易汇总、决策、目标持仓、因子权重、日度IC、覆盖率与输入/结果哈希。
'''
    path=ROOT/'docs/research/CANDIDATE_FACTOR_EXTENSION.md';path.write_text(article)
    print(path)

if __name__=='__main__':build()
