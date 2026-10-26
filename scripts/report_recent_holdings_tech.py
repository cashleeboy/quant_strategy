#!/usr/bin/env python3
"""Build a readable holdings-based investigation from sealed results."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from analysis.holdings_count import metrics
from scripts.run_mvp_holdings_count import sha

OUT=ROOT/'output/recent_holdings_tech_20261023'
DOC=ROOT/'docs/research/RECENT_HOLDINGS_TECH_DIAGNOSIS.md'
ASSET=DOC.parent/'assets/recent_holdings_tech'
FREQ={'WEEKLY':'周度','MONTHLY':'月度'}
RECIPE={'BASE':'原13因子','ALL5':'加入五个新因子'}
PARENTS=[f'{a}_{f}_3bp' for a in RECIPE for f in FREQ]
TECHS=[f'{a}_TECH15_{f}_3bp' for a in RECIPE for f in FREQ]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def name(key):
    parts=key.split('_');a=RECIPE.get(parts[0],parts[0])
    f=next((FREQ[x] for x in parts if x in FREQ),'')
    area='科技15只' if 'TECH15' in key else '科技最多5只' if 'TECH_LITERAL' in key else '全市场15只'
    return f'{a}／{area}／{f}'


def build():
    validation=json.loads((OUT/'independent_validation.json').read_text());assert validation['status']=='PASS'
    for f,h in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/f)==h,f
    perf=pd.read_csv(OUT/'performance.csv');recent=perf[perf.period.eq('RECENT')].set_index('key')
    annual=pd.read_csv(OUT/'annual.csv').pivot(index='key',columns='year',values='return_')
    exposure=pd.read_csv(OUT/'exposure_attribution.csv');exp=exposure[exposure.period.eq('RECENT')].set_index('key')
    industry=pd.read_csv(OUT/'industry_attribution.csv')
    nav=pd.read_csv(OUT/'nav.csv',index_col='date',parse_dates=True)
    pct=lambda x:f'{x:.2%}'
    pp=lambda x:f'{x*100:+.2f}个百分点'
    def stockdata(key):return pd.read_csv(OUT/('holdings_'+key)/'RECENT_stocks.csv')
    def stocks(key,which='worst',n=10):
        x=stockdata(key)
        if which=='best':x=x.sort_values('contribution',ascending=False)
        if which=='weight':x=x.sort_values('average_weight',ascending=False)
        return table(['股票','代码','实际持有交易日数','全区间平均账户权重','对区间累计收益的净贡献'],[
            [r['name'],r['symbol'],int(r['held_days']),pct(r['average_weight']),pp(r['contribution'])] for _,r in x.head(n).iterrows()])
    def performance(keys):
        return table(['版本','累计收益','年化收益','年化波动','最大回撤','平均股票仓位'],[
            [name(k)]+[pct(recent.loc[k,c]) for c in ['total_return','cagr','volatility','max_drawdown','average_equity']] for k in keys])
    def years(keys,names=None):
        return table(['版本']+[str(y)+('截至10月15日' if y==2026 else '') for y in range(2021,2027)],
                     [[(names or {}).get(k,name(k))]+[pct(annual.loc[k,y]) for y in range(2021,2027)] for k in keys])
    six=pd.read_csv(ROOT/'output/six_strategy_ic_comparison_20261018/nav.csv',index_col='date',parse_dates=True)
    six_rows=[]
    for code,label in [('BASE_V2_MULTI_FACTOR','基础池'),('LARGE_CAP_MULTI_FACTOR','大盘'),('MID_SMALL_CAP_MULTI_FACTOR','中小盘'),('VALUE_FOCUSED','价值'),('GROWTH_FOCUSED','成长'),('TECH_GROWTH_THEME_MULTI_FACTOR','科技主题')]:
        v=six[code+'__FIXED__K50'];x=v.loc['2021':];pos=v.index.get_loc(x.index[0]);x=pd.concat([v.iloc[pos-1:pos],x])
        m=metrics(x);six_rows.append([label]+[pct(m[k][0]) for k in ['total_return','cagr','max_drawdown']])
    six_table=table(['原六策略固定权重Top50历史回放','2021年以来累计','年化','最大回撤'],six_rows)
    old=pd.read_csv(ROOT/'output/candidate_factors_20261022/subperiods.csv')
    costs=table(['全市场15只版本','零成本年化','3bp年化','13bp年化','3bp下年均双边换手'],[
        [name(k)]+[pct(old[old.arm.eq(a)&old.frequency.eq(f)&old.period.eq('RECENT')&np.isclose(old.fee,fee)].iloc[0].cagr) for fee in [0,.0003,.0013]]+
        [f'{recent.loc[k,"annual_turnover"]:.2f}倍'] for a in RECIPE for f in FREQ for k in [f'{a}_{f}_3bp']])
    theme_table=table(['版本','科技平均账户权重','科技净贡献','其他股票净贡献','合计累计收益','期间先后持有股票数'],[
        [name(k),pct(exp.loc[k,'technology_average_weight']),pp(exp.loc[k,'technology_contribution']),
         pp(recent.loc[k,'total_return']-exp.loc[k,'technology_contribution']),pct(recent.loc[k,'total_return']),int(exp.loc[k,'unique_names'])] for k in PARENTS])
    ig=industry[industry.key.eq('BASE_WEEKLY_3bp')&industry.period.eq('RECENT')].sort_values('net_contribution')
    indtable=table(['行业','平均账户权重','累计净贡献'],[[r.industry,pct(r.average_weight),pp(r.net_contribution)]
        for r in pd.concat([ig.head(8),ig.tail(5)]).itertuples()])
    compares={'TECH_EQ_WEEKLY_3bp':'科技池全体等权／周度','TECH_EQ_MONTHLY_3bp':'科技池全体等权／月度',
              'OLD_TECH_FIXED50':'原六策略科技Top50历史回放','POOL_EQ_3bp':'基础池全体等权／周度',
              'H00300.CSI':'沪深300全收益指数','H00016.CSI':'上证50全收益指数'}
    references=table(['参考线（结构不同，不是纯因果对照）','累计收益','年化收益','年化波动','最大回撤'],[
        [label]+[pct(recent.loc[k,c]) for c in ['total_return','cagr','volatility','max_drawdown']] for k,label in compares.items()])
    techcost=table(['科技15只版本','3bp累计','3bp年化','13bp累计','13bp年化'],[
        [name(k),pct(recent.loc[k,'total_return']),pct(recent.loc[k,'cagr']),pct(recent.loc[k.replace('_3bp','_13bp'),'total_return']),pct(recent.loc[k.replace('_3bp','_13bp'),'cagr'])] for k in TECHS])
    full=perf[perf.period.eq('FULL')].set_index('key')
    fulltable=table(['版本','2012年以来累计','全期年化','2021年以来年化'],[
        [name(k),pct(full.loc[k,'total_return']),pct(full.loc[k,'cagr']),pct(recent.loc[k,'cagr'])] for k in PARENTS+TECHS])
    # End-of-history quote audit: do not silently treat terminal stale value as cash.
    z=np.load(ROOT/'output/mvp_holdings_count_20261017_complete_factors/prices.npz')
    px=pd.DataFrame(z['values'],index=pd.DatetimeIndex(z['dates']),columns=z['symbols'])
    stale=[]
    for k in PARENTS+TECHS:
        last=pd.read_csv(OUT/('holdings_'+k)/'last_holdings.csv')
        for r in last.itertuples():
            lastquote=px[r.symbol].last_valid_index()
            if lastquote<px.index[-1]:stale.append([name(k),r.name,r.symbol,str(lastquote.date()),pct(r.weight)])
    stale_table=table(['账户','显示名称','代码','本地最后有效报价日','期末账面占比'],stale)
    latest=pd.read_csv(OUT/'holdings_BASE_TECH15_WEEKLY_3bp/last_holdings.csv').sort_values('weight',ascending=False)
    latest_table=table(['科技原13因子周度版期末实际持仓','代码','账户权重'],[[r['name'],r.symbol,pct(r.weight)] for _,r in latest.iterrows()])

    ASSET.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    first=nav.index.get_loc(nav.loc['2021':].index[0]);n=nav.iloc[first-1:];n=n/n.iloc[0]
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for ax,(a,f) in zip(axs.flat,[(a,f) for a in RECIPE for f in FREQ]):
        for k,label,color in [(f'{a}_{f}_3bp','Broad universe','#64748b'),(f'{a}_TECH15_{f}_3bp','Technology15','#0369a1'),(f'TECH_EQ_{f}_3bp','Technology pool equal weight','#d97706')]:
            ax.plot(n.index,n[k],label=label,color=color,lw=1.3)
        ax.set_title(f'{a} / {f}, 3bp per side');ax.axhline(1,color='black',lw=.5);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('2021 to 2026-10-15: actual post-fee NAV (all inherit pre-2021 positions)')
    fig.tight_layout();fig.savefig(ASSET/'01_recent_nav.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    for ax,k,title in [(axs[0],'BASE_WEEKLY_3bp','Broad universe / BASE weekly'),(axs[1],'BASE_TECH15_WEEKLY_3bp','Technology15 / BASE weekly')]:
        x=stockdata(k);x=pd.concat([x.head(8),x.tail(8)]).sort_values('contribution')
        ax.barh(x.symbol,x.contribution*100,color=np.where(x.contribution>0,'#0284c7','#dc2626'))
        ax.set_title(title);ax.set_xlabel('Net contribution (percentage points of initial NAV)');ax.axvline(0,color='black',lw=.6)
    fig.tight_layout();fig.savefig(ASSET/'02_stock_contributions.png',dpi=170);plt.close(fig)
    keys=['BASE_WEEKLY_3bp','BASE_TECH15_WEEKLY_3bp','ALL5_WEEKLY_3bp','ALL5_TECH15_WEEKLY_3bp','ALL5_MONTHLY_3bp','ALL5_TECH15_MONTHLY_3bp','OLD_TECH_FIXED50']
    matrix=annual.loc[keys].to_numpy();fig,ax=plt.subplots(figsize=(11,5))
    im=ax.imshow(matrix,cmap='RdYlGn',vmin=-.4,vmax=.4,aspect='auto')
    ax.set_yticks(range(len(keys)),['BASE broad weekly','BASE tech15 weekly','ALL5 broad weekly','ALL5 tech15 weekly','ALL5 broad monthly','ALL5 tech15 monthly','Original tech50'])
    ax.set_xticks(range(6),['2021','2022','2023','2024','2025','2026 YTD'])
    for i in range(len(keys)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:+.1%}',ha='center',va='center',fontsize=9)
    ax.set_title('Annual returns: technology is not a permanent downside hedge')
    fig.colorbar(im,ax=ax,format=PercentFormatter(1));fig.tight_layout();fig.savefig(ASSET/'03_annual_returns.png',dpi=170);plt.close(fig)

    text=f'''# 2021年以来收益为什么变弱：逐股持仓归因与科技股票池对照

> 2026-10-24更新：本文发现的两项换股资产残留已在新的研究回放入口修复，并完成20账户重跑。本文保留修复前诊断与历史数字；修复范围、独立核验和新结果见[公司行动修复记录](ASSET_TRANSITION_FIX.md)。

研究日期：2026-10-23。数据截止：2026-10-15。本文是工程内研究记录，不是新增实盘方案。已完成22个账户的复现或回放、8个账户的逐日持仓归因与独立核验；六策略前向纸面账户未修改。

## 一、先回答问题：不能把“近年收益弱”扩大成所有策略都一样弱

之前说的年化0.94%、1.09%、0.21%、2.70%，指的是**全市场股票池里的15只研究组合**，不代表日报六策略全部只有这样的收益。

这次查到三件明确的事：

1. 原13因子周度组合的收益，被许多股票的正负贡献相互抵消。科技持仓本身并非拖累：它贡献约＋12.54个百分点，其他股票合计－7.19个百分点，最后剩＋5.34%。
2. 同一套原13因子调权和组合规则，限制在科技池内、允许科技占满15个目标名额后，近年累计＋78.69%、年化11.02%，明显好于原来的＋5.34%、年化0.94%。但加入五个新因子后的科技周度版只有年化4.14%，不能说换科技池就什么配方都有效。
3. 持仓穿透还发现两个账户有跨多年的终止报价残留持仓。这是公司行动与估值处理尚未闭环的具体证据，不能因为账面核验通过就把全部结果认定为可实盘。

**这次得到的是“问题在哪些持仓和规则中显现”的证据，不是已经证明某一个因子失败的唯一经济原因。**

## 二、到底比较哪套方案

主研究方案仍是：历史RankIC调权、分散约束、持仓缓冲、目标15只、总仓位管理。

| 项目 | 本次口径 |
| --- | --- |
| 因子配方 | 原13因子，以及原13因子＋五个候选因子，共两种；没有按这次结果再挑子组合 |
| RankIC | 过去252个已成熟观测，五日收益标签，每月更新；保留原权重上限和变化幅度限制 |
| 选股 | 综合排名前100名内筛选，单行业最多3只，相关系数上限0.8，旧持仓前30名缓冲 |
| 仓位 | 15%目标年化波动、20%—100%预算；选不满15只的份额留现金 |
| 调仓 | 周度和月度分别比较；两者都保留每日风险检查和额外风控调仓 |
| 成交 | 信号后下一交易日收盘，复权价格、可分割股数、先卖后买；现金不计息 |
| 成本 | 买卖每边3bp为主，13bp为压力情景；不是完整印花税、最低佣金或冲击模型 |
| 时间 | 从2012-01-06启动回放，延续持仓到2021，评价2021首个交易日—2026-10-15；不在2021年清仓重置 |
| 年化 | 按252个交易日计算；2026年不是完整年度 |

原13因子与五个候选的公式、缺失值处理和财务可得时间继承[上一轮记录](CANDIDATE_FACTOR_EXTENSION.md)。这次没有补取10月15日之后的数据，没有用今天的科技成分股倒推历史。

## 三、先拆开全市场15只组合：损失不是平均分布在每一年

{performance(PARENTS)}

{years(PARENTS)}

原13因子周度版2021年－2.85%、2022年－18.82%，2023年仅＋2.17%；2024、2025年恢复上涨。六年合在一起只剩较低年化，不能理解成“从2021开始每年都不赚钱”。加入五个因子的周度版在2026年前三季又有明显回落，而月度版情况不同。

## 四、它到底买了什么：先看真实持仓，而不是排名名单

### 1. 15只是同时持仓目标，不是这几年只交易15家公司

原13因子周度账户2021年以来先后持有过596只股票，月度账户363只；加五因子的周度394只、月度256只。换股、持仓延续和现金比例共同决定了结果。

下面是原13因子周度版按“全区间平均账户权重”排序的前十只。这个平均包含没有持仓的日期，所以1%不代表买入时只配1%；实际持有时可能占账户4%—6%。持有天数也可以分布在多个不连续区间。

{stocks('BASE_WEEKLY_3bp','weight')}

名单说明这并不是长期持有一篮子市场熟悉的科技龙头。很多资金在农业、家电、周期制造、医药和新能源等股票之间轮换。它们是否因为周期高点的财务指标被打了高分，仍要进一步追踪当时的逐因子得分，不能只凭公司名称下定论。

### 2. 哪些股票拉低了收益

{stocks('BASE_WEEKLY_3bp')}

“英科医疗－4.48个百分点”的意思是：把2021开始时账户净值看作100，整个研究区间中这只股票的实际持仓价格损益减去分摊费用，累计约减少4.48。**不是英科医疗自身只跌了4.48%，也不是把它从2021买到现在的收益。**

原13因子周度版2022年主要负贡献包括蓝帆医疗、明微电子、新乡化纤、英科医疗、TCL科技、博汇纸业等。账户确实在它们部分下跌期间持有仓位；这是交易与持仓证据。是否属于财务信号滞后、行业集中、买入时点不佳，需要逐次信号解释才可以定因。

几个具体时间窗口可以帮助理解：英科医疗的持仓记录分布在2021-05-06至2022-05-05之间，合计182个持有交易日；蓝帆医疗在2022-01-17至2022-04-25之间，合计63日；通威股份的持仓记录分布在2022-08-29至2024-05-06之间，合计207日。这里是首次至最后一次持仓或退出记录的边界，不代表全程连续持有。完整逐日记录可以核查中间的卖出、重新买入和加减仓。

### 3. 也有赚钱的股票，但没有形成足够大的净优势

{stocks('BASE_WEEKLY_3bp','best')}

把每只股票在整个区间的净贡献先合并，再分正负：原13因子周度版正贡献合计＋146.85个百分点，负贡献合计－141.51个百分点，净额约＋5.34%。这是所有分批交易和复利路径下的累计金额归因，并非可以实际分别持有的两个独立组合收益率。

这比“所有因子都失效了”更准确：**有赢的交易，但输的交易、持仓结构和成本抵消掉了大部分成果。**

![实际持仓的个股贡献对照](assets/recent_holdings_tech/02_stock_contributions.png)

图左是全市场原13因子周度组合，图右是科技15只原13因子周度组合；均展示贡献最差和最好的各8只。横轴是对账户累计收益的贡献百分点，不是个股涨跌幅。

## 五、科技到底有没有配置，是否拖累了全市场组合

工程的科技主题定义是历史申万一级行业中的**电子、计算机、通信、传媒、电力设备**。它是一个较宽的可配置主题，不等于纯半导体、纯AI、纯计算机，也包含电力设备中的新能源方向。

{theme_table}

对原13因子而言，科技部分总体贡献为正，非科技部分总体贡献为负。但对加入五因子的周度版本，科技部分却贡献－3.97个百分点，说明“买科技”不是一个足够精确的策略描述，买了什么股票、何时持有和用了哪套配方都重要。

原13因子周度版进一步按历史行业拆分如下，展示较大的负贡献和正贡献行业：

{indtable}

计算机、电子、通信总体为正，不意味着科技五个子行业都为正；电力设备和传媒在这个账户内反而是负贡献。行业归因使用持仓日之前一个收盘时点的历史行业，不按今天行业分类重写过去。

这些是现有仓位的盈亏分账。**不能把＋12.54个百分点直接按比例放大，当作如果全仓科技就会赚多少**：候选排序、行业上限、相关性和仓位管理都会随组合改变。

## 六、交易成本确实有影响，但不是唯一原因

{costs}

原13因子周度版本在零成本下，近年年化也只有1.81%；所以不能把低收益全部归咎于手续费。3bp后降到0.94%，13bp后变成－1.91%，也说明高换手的损耗不能忽略。年均双边换手28.63倍是买入与卖出成交金额相加，再按账户净值归一，不是每年换28.63次完整股票名单。

该账户逐笔实际费用占2021期初净值约4.34个百分点。零成本与3bp最终累计收益差约5.15个百分点，二者不完全一样：成本改变资金规模及后续复利路径，不能拿静态费用金额冒充完整反事实差异。

## 七、原六策略中的科技策略，近年究竟怎样

这是既有六策略固定因子权重、Top50的历史回放，不是把2026年才开始的前向纸面账户伪装成2021年就存在。六个策略各自独立，不能将收益相加，当成一个已有资金配置的总账户。

{six_table}

原科技Top50累计约62.25%、年化9.11%，并不是接近零收益；但最大回撤约－41.97%。同期其科技池全体等权累计约62.14%、年化9.10%，说明这条Top50策略的终点收益几乎没有超过自身科技池等权，不宜把大部分涨幅归功于选股超额。

## 八、把15只研究方案放进科技池：到底改了什么

不能机械地把原规则复制过去。原方案规定“科技主题最多5只”，如果候选池全部是科技，却不解除这一限制，最多只能选5只，每只仍分到目标预算的1/15，至少三分之二名额留现金。

因此本次同时保留两个版本：

| 版本 | 科技总上限 | 其他规则 | 用途 |
| --- | --- | --- | --- |
| 科技最多5只 | 原0.35上限，取整最多5只 | 不变；剩余名额现金，不重分给5只 | 检查原规则机械照搬会怎样 |
| 科技15只 | 放开到1.0，允许15只都在科技池内 | 行业最多3只、相关性、前30缓冲、风险预算均不变 | 检查专门的科技组合 |

科技池只有这五个一级行业，因此单行业最多3只意味着：**如果目标选满15只，五个行业各3只**；不是让模型把15只全部集中到当时最热的半导体或AI子行业。

评分先在原基础池内完成标准化，继续使用原全市场历史RankIC权重；然后按当时行业筛出科技候选，再取前100名。**不是只在旧的全市场前100名里找科技股，也没有重新在科技池内训练RankIC权重。**

这控制住了很多变量，但“全市场原方案→科技15只”仍同时改变了候选池和科技总上限，不能声称是纯股票池单变量效应。两种上限版本就是为了把这个差别显示出来。

### 主结果：允许科技池内选满15只

{performance(TECHS)}

原13因子周度科技版是本轮这四个科技组合中近年收益最高的，但这是看完这段历史后的比较，不是一个未见过数据的胜出证明。加入五个候选并未进一步改善科技组合：周度年化从11.02%变为4.14%，月度从8.51%变为7.68%。

### 机械保留科技最多5只，会变成低仓位组合

{performance([f'{a}_TECH_LITERAL_{f}_3bp' for a in RECIPE for f in FREQ])}

这些版本平均股票仓位只有约18%—24%，因此不能拿它们较小的回撤当成选股更优秀的证据。这里“最多5只”是决策目标；停牌等未成交残留可能使实际账户暂时超过目标只数。

### 与科技池、原六策略科技Top50和指数对照

{references}

科技全体等权通常持有很多股票、接近满仓；科技15只组合只有约64%—73%平均股票仓位。二者并非同风险或同资金利用率的纯选股对照。这次没有进一步回放“完全跟随15只账户仓位路径的科技等权影子账户”，因此不能把年化差直接叫纯Alpha。

![2021年以来同配方不同股票池的净值](assets/recent_holdings_tech/01_recent_nav.png)

图中BASE表示原13因子，ALL5表示加入五个新因子；灰线为全市场15只，蓝线为科技15只，橙线为科技池全体等权。四个图分别拆开配方和周/月频率，避免只展示最好的版本。

## 九、科技组合不是每一年都好，也没有解决抗跌保证

{years(['BASE_TECH15_WEEKLY_3bp','BASE_TECH15_MONTHLY_3bp','ALL5_TECH15_WEEKLY_3bp','ALL5_TECH15_MONTHLY_3bp','OLD_TECH_FIXED50'],{'OLD_TECH_FIXED50':'原六策略科技Top50'})}

原13因子科技周度版2022年仍亏21.36%，比全市场原13因子周度的－18.82%更差；2026截至10月15日亏5.84%，同期全市场原方案反而盈利4.03%。其累计优势来自整个历史路径，并非每次下跌都更抗跌。原科技Top50在2025年涨幅很大，2022年和2026年却也有明显损失。

![各版本分年度收益](assets/recent_holdings_tech/03_annual_returns.png)

### 成本提高之后，优势有没有完全消失

{techcost}

原13因子科技周度版在13bp下近年年化仍为8.53%，月度为7.58%；但加五因子的科技周度版只剩2.45%。这再次表明股票池、配方、持有频率与成本之间有交互，不能只看“科技”两个字。

## 十、科技15只究竟赚在什么股票上

以原13因子周度科技版为例，以下是主要正贡献：

{stocks('BASE_TECH15_WEEKLY_3bp','best')}

主要负贡献也必须一起看：

{stocks('BASE_TECH15_WEEKLY_3bp','worst',8)}

它并不是简单地买入并一直持有当下熟悉的明星科技股：期间先后持有364只，神工股份、道通科技、佰维存储、指南针等贡献较大，德明利、江波龙、通威股份、吉比特等在实际持有区间贡献为负。同一只股票长期可能上涨，也可能在策略的进出区间造成损失；不能用它最后的总涨幅替代我们的持仓盈亏。

原全市场版中的寒武纪贡献约－2.26个百分点，就说明“买到科技明星”和“在它身上赚到钱”不是一回事。它在该账户中的持仓记录分布在2025-05-19至2026-02-24之间，合计44个持有交易日，并非从2021长期持有；本次没有把卖出后的上涨计入账户收益。

截至2026-10-15收盘，这个科技研究账户的实际持仓如下；这是历史实验记录，不是今天的买入建议。股票合计约{pct(latest.weight.sum())}，剩余为现金，平均仓位与期末仓位不是一回事。

{latest_table}

## 十一、穿透持仓发现的工程问题：终止报价的老股票没有完成处置

{stale_table}

本地股票清单显示，小天鹅A已于2019年终止上市、长城信息已于2017年终止上市。这里列的是**本地最后有效报价日**，不与正式终止上市日混用。

当前研究撮合器的处理是：没有可交易报价就不虚构成交，已有股票用最后价格估值。对短期停牌，这是一个明确的估值约定；但对跨多年的终止报价股票，缺少公司行动后续处理就可能让账面资产永久挂着。

因此：

- 加五因子的全市场周度账户，期末除了15只正常目标股，还挂着小天鹅A残留，账面占比约3.77%。
- 原13因子的科技月度账户，期末还挂着长城信息残留，账面占比约3.21%。
- 必须核实历史公司行动对应的换股、现金结算、其他权益转换或确无回收等情形，再决定正确处理；不能凭名称直接清零，也不能强行在最后一天按最后价卖出。
- 这些残留在2021期初就可能存在，影响起始净值、可用资金与后续交易路径。**不能从区间收益简单减掉期末3.77%或3.21%就声称完成修正。**

原13因子的全市场周度与科技周度账户期末没有这类无报价残留，因此这个问题并不能解释它们之间的全部差距。但它明确降低了另外两个账户结果的经济完整性。所有数字仍是原撮合与估值规则下的结果，尚未做公司行动修复后的重新回放。

独立校验PASS只意味着账户、成交、价格损益和费用可以对账，**不代表供应商原始数据、公司行动、涨跌停、整手和最低佣金这些实盘问题已经通过验收**。股票展示名称取自本地最新清单，带ST或“退”字不等于历史买入当时就是该名称；入池仍使用当时可得规则。

## 十二、现在能确认什么，不能确认什么

| 问题 | 本轮证据 | 结论边界 |
| --- | --- | --- |
| 近年是不是所有版本都差 | 原六策略科技Top50年化9.11%，新科技原13因子周度年化11.02% | 不能把全市场15只的弱收益推广到所有策略 |
| 原15只账户是否没有买科技 | 科技平均占账户约19.7%，且原13因子周度科技净贡献＋12.54个百分点 | 科技有配置；不是完全错过，但也不是长期纯科技策略 |
| 是不是个股全部亏钱 | 596只先后持有股票中有大量正负贡献，二者几乎抵消 | 问题是净效果不足，不是每笔交易都失败 |
| 是不是只要降费就好 | 原全市场周度零成本年化也只有1.81% | 成本重要，但不是唯一解释 |
| 科技池能否改善本段历史 | 原13因子科技周/月明显改善，加五因子效果较弱 | 依赖配方和组合限制，不能承诺未来重复 |
| 能否证明纯因子超额 | 参考池风险与仓位不同，且科技上限有变化 | 只能评价当前组合结果，尚不能精确拆成纯选股Alpha |
| 回测账户是否完全可实盘 | 查到两类历史终止报价残留，且未模拟真实交易约束 | 不能据此直接批准实盘 |

下一步不应立刻把日报六策略全部换成科技15只。更合理的是：**先补齐这次已经查实的公司行动与终止报价处理；修复后保持本轮参数不动重跑对照，再用真正未参与设计的前向纸面账户考察候选。** 至于为什么个别周期或医药股票被打出高分，可以从贡献最差的实际交易开始逐笔解释，而不是再无边界地搜索更多开关。

这次最重要的修正是：我们的问题并非“所有传统因子从2021开始都没用了”，而是**同一套因子在不同候选池、持仓约束和交易路径下，净收益差异很大；全市场15只组合里的收益抵消和高换手确实存在，部分历史资产处理也还不完整。**

## 附录：完整区间、可复现文件与核验

{fulltable}

所有区间都不是新的未见样本；这是对已研究历史的进一步诊断。相对参考线更好的收益不等于因果验证或未来收益承诺。

本轮累计22个账户：4个旧账户精确还原，16个科技配方／频率／上限／成本组合，2个科技池全体等权参考账户。旧六策略Top50和指数引用已冻结历史结果，不重复计入22个账户。

核验覆盖{validation['trade_rows']:,}条逐股成交、{validation['technology_selection_checks']:,}次科技目标选择、{validation['historical_industry_rows']:,}个日期—股票历史行业映射、{validation['attribution_period_checks']}组收益归因；最大加总误差{validation['maximum_attribution_error']:.2e}。测试套件509项通过。一次开发预检中的持仓前值字段类型错误已修复并补测试，预检输出单独保留，不用于本文。

文件入口：

- [实验协议](../../research_protocols/20261023_recent_holdings_tech_v1.json)
- [逐股归因模块](../../analysis/recent_holdings.py)
- [回放入口](../../scripts/run_recent_holdings_tech.py)
- [独立核验](../../scripts/validate_recent_holdings_tech.py)
- [结果总表](../../output/recent_holdings_tech_20261023/performance.csv)
- [分年度收益](../../output/recent_holdings_tech_20261023/annual.csv)
- [行业与持仓暴露归因](../../output/recent_holdings_tech_20261023/industry_attribution.csv)
- [原13因子周度全部596只股票贡献](../../output/recent_holdings_tech_20261023/holdings_BASE_WEEKLY_3bp/RECENT_stocks.csv)
- [原13因子周度逐日实际持仓](../../output/recent_holdings_tech_20261023/holdings_BASE_WEEKLY_3bp/daily_holdings.csv.gz)
- [原13因子周度全部逐股成交](../../output/recent_holdings_tech_20261023/holdings_BASE_WEEKLY_3bp/trades_recent.csv.gz)
- [科技15只周度全部364只股票贡献](../../output/recent_holdings_tech_20261023/holdings_BASE_TECH15_WEEKLY_3bp/RECENT_stocks.csv)
- [独立核验结果](../../output/recent_holdings_tech_20261023/independent_validation.json)

其他账户在同一结果目录下，`holdings_账户名`目录包含逐日实际持仓、全部成交、分年股票贡献和期末股票明细。归因包含未实现浮盈亏与交易费用，全部股票贡献加总等于区间累计收益；没有用目标仓位假装实际成交。
'''
    DOC.write_text(text)
    print(DOC)

if __name__=='__main__':build()
