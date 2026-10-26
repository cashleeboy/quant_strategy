#!/usr/bin/env python3
"""Report an honestly blocked experiment and its common pre-suspension prefix."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from analysis.holdings_count import metrics
from scripts.run_mvp_holdings_count import sha
from scripts.run_fixed15_ml import OUT as TRAIN
from scripts.audit_fixed15_ml import OUT

NAMES={'BASE':'原RankIC','lightgbm':'LightGBM','catboost':'CatBoost'}
AREAS={'BROAD':'基础池','TECH15':'科技池'}
ASSET=ROOT/'docs/research/assets/fixed15_ml'
DOC=ROOT/'docs/research/FIXED15_ML_COMPARISON.md'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def run():
    for f,h in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/f)==h,f
    valid=json.loads((OUT/'independent_validation.json').read_text());assert valid['status']=='PASS'
    states=json.loads((OUT/'status.json').read_text());refs=json.loads((OUT/'accounts.json').read_text())
    problems=[(key,asset,s) for key,s in states.items() if s['status']=='BLOCKED' for asset in s['blocked_assets']]
    cutoff=min(pd.Timestamp(asset['last_quote']) for _,asset,_ in problems)
    rows=[];navs={};annual=[];trades=[]
    for key,path in refs.items():
        backend,area,bp=key.split('_');folder=ROOT/path
        daily=pd.read_csv(folder/'daily.csv.gz',index_col='date',parse_dates=True).loc[:cutoff]
        n=daily.post_nav;navs[key]=n
        ex=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date'])
        for period,start in [('COMMON_FULL','2012'),('COMMON_RECENT','2021')]:
            v=n.loc[start:];pos=n.index.get_loc(v.index[0]);values=pd.concat([n.iloc[pos-1:pos],v]) if pos else v
            m={name:float(z[0]) for name,z in metrics(values).items()};e=ex[ex.date.between(v.index[0],cutoff)]
            rows.append(dict(key=key,backend=backend,area=area,bp=bp,period=period,end=cutoff,**m,
                average_equity=(1-daily.loc[v.index,'cash']/v).mean(),
                annual_two_way_turnover=e.two_way_turnover.sum()/((len(values)-1)/252)))
        for year in range(2021,cutoff.year+1):
            v=n.loc[str(year)];i=n.index.get_loc(v.index[0]);base=n.iloc[i-1] if i else 1.
            annual.append(dict(key=key,year=year,return_value=v.iloc[-1]/base-1))
        if backend!='BASE':
            t=pd.read_csv(folder/'trades.csv.gz');t=t[t.symbol.eq('000982.SZ')&t.date.ge('2024-05-01')]
            trades.append(t.assign(key=key))
    data=pd.DataFrame(rows);nav=pd.DataFrame(navs);annual=pd.DataFrame(annual)
    # Derived diagnostics are deliberately outside the sealed source manifests.
    data.to_csv(OUT/'common_prefix_comparison.csv',index=False)
    nav.to_csv(OUT/'common_prefix_nav.csv');annual.to_csv(OUT/'common_prefix_annual.csv',index=False)
    pd.concat(trades).to_csv(OUT/'blocked_stock_recent_trades.csv',index=False)
    pct=lambda v:f'{v:.2%}'
    def perf(period):
        return table(['股票池','评分','累计收益','年化收益','年化波动','夏普','最大回撤'],[
            [AREAS[a],NAMES[m]]+[pct(r[c]) if c!='sharpe' else f'{r[c]:.2f}' for c in ['total_return','cagr','volatility','sharpe','max_drawdown']]
            for a in AREAS for m in NAMES
            for _,r in data[data.area.eq(a)&data.backend.eq(m)&data.bp.eq('3bp')&data.period.eq(period)].iterrows()])
    costs=[]
    for area in AREAS:
        for model in NAMES:
            x=data[data.area.eq(area)&data.backend.eq(model)&data.period.eq('COMMON_RECENT')].set_index('bp')
            costs.append([AREAS[area],NAMES[model],pct(x.loc['3bp','cagr']),pct(x.loc['13bp','cagr']),
                          pct(x.loc['3bp','average_equity']),f"{x.loc['3bp','annual_two_way_turnover']:.2f}倍"])
    ASSET.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'BASE':'#64748b','lightgbm':'#0369a1','catboost':'#c05621'}
    for filename,start in [('01_common_full.png','2012'),('02_common_recent.png','2021')]:
        v=nav.loc[start:];i=nav.index.get_loc(v.index[0]);v=nav.iloc[max(0,i-1):];v=v/v.iloc[0]
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
        for j,area in enumerate(AREAS):
            for model in NAMES:
                s=v[f'{model}_{area}_3bp'];axes[0,j].plot(s.index,s,label=model,color=colors[model],lw=1.2)
                axes[1,j].plot(s.index,s/s.cummax()-1,color=colors[model],lw=1.1)
            axes[0,j].set_title(area+' / fixed15 / 3bp per side');axes[0,j].set_ylabel('Post-fee NAV (start = 1)')
            axes[0,j].legend();axes[1,j].set_ylabel('Within-period drawdown (%)');axes[1,j].set_xlabel('Date')
            axes[1,j].yaxis.set_major_formatter(PercentFormatter(1))
            for ax in axes[:,j]:ax.grid(alpha=.2)
        fig.suptitle(f'{start} to {cutoff.date()}: common diagnostic prefix, NOT completed 2026 comparison\nSource: frozen model scores and independently reconciled account ledgers')
        fig.tight_layout();fig.savefig(ASSET/filename,dpi=160);plt.close(fig)
    training=pd.read_csv(TRAIN/'training.csv');diag=pd.read_csv(TRAIN/'diagnostics.csv')
    diag=diag[diag.date.ge('2021-01-01')&diag.date.le(str(cutoff.date()))]
    diagtable=table(['股票池','模型','周度平均RankIC','原始Top15五日均值','全池五日均值'],[
        [AREAS[a],NAMES[m],f'{g.rank_ic.mean():.4f}',pct(g.raw_top15_return.mean()),pct(g.pool_return.mean())]
        for (a,m),g in diag.groupby(['area','backend'])])
    failures=table(['账户','停止日期','最后报价','旧价格下账面占比'],[
        [k,str(pd.Timestamp(s['failure_date']).date()),str(pd.Timestamp(a['last_quote']).date()),pct(a['stale_weight'])] for k,a,s in problems])
    yearly=table(['年份']+[AREAS[a]+'/'+NAMES[m] for a in AREAS for m in NAMES],[
        [str(y)+('（截至6月21日）' if y==2024 else '')]+[pct(annual[annual.key.eq(f'{m}_{a}_3bp')&annual.year.eq(y)].return_value.iloc[0]) for a in AREAS for m in NAMES]
        for y in range(2021,2025)])
    text=f'''# 固定15只组合接入LightGBM与CatBoost：初步对比与退市资产阻断记录

实验日期：2026-10-24。**状态：模型接入和全部滚动训练已完成；原计划截至2026-10-15的完整账户对比未完成。** 8个机器学习账户都在2024-08-12被同一只未解决结算的退市资产阻断。本文严格分开已经验证的代码、故障前共同区间的表现，以及尚不能发布的完整收益。

## 一、研究背景和本轮问题

最近的研究组合是“历史RankIC调权＋分散约束＋持仓缓冲＋固定15只＋总仓位管理”。它不是六策略日报的50只版本。前面的研究表明：风险控制能够降低波动，但2021年以来基础池收益偏弱；增加传统因子也没有稳定解决问题。前一项工作又修复了两项换股吸收合并的资产处理。

本轮不继续增加因子，而是问：**把现有13个因子交给LightGBM或CatBoost学习，是否能比原RankIC评分更好地选股？**

工程已有两种模型的构造代码，但此前研究组合和日报并没有因此自动使用它们。本次实际安装了LightGBM4.6.0与CatBoost1.2.8，复用构造器，新增严格训练入口。请求哪种后端就必须运行哪种；不允许缺依赖后偷偷回退成其他模型。

## 二、三种方案是独立对照，不是累积加模型

| 方案 | 股票评分方法 | 后续持仓规则 |
| --- | --- | --- |
| 原RankIC | 按历史正平均RankIC逐月更新因子权重，沿用原上限和变化幅度限制 | 原固定15只方案 |
| LightGBM | 用13因子预测未来可执行五日收益，按预测值排序 | 同一固定15只方案 |
| CatBoost | 用相同特征和标签，独立训练另一种提升树模型 | 同一固定15只方案 |

没有先加LightGBM再叠加CatBoost，也没有把模型分数混入原RankIC分数。两种模型分别替换股票评分。模型的特征重要性不等于资金权重，也不能直接解释为因果关系。

原13个因子是PE、现金流收益率代理、ROE、毛利率、净利率、低负债率、现金利润质量、营收增长、利润增长、原动量、60日动量、5日反转、波动率。未加入新闻公告、股票代码、行业作为模型特征，也未加入最近五个候选因子。现金流收益率仍是原工程的经营现金流代理，不冒称已完整扣除资本开支的自由现金流。

分别在基础池和科技池上回放。模型只在基础池历史候选样本上训练一次，科技池沿用同一预测，先筛当时属于电子、计算机、通信、传媒、电力设备的股票，再选Top100。它不是独立科技模型，也不是纯AI池。

## 三、如何训练，哪些地方没有使用未来

| 项目 | 冻结规则 |
| --- | --- |
| 数据终点 | 与前几轮相同，2026-10-15；没有下载新行情 |
| 历史窗口 | 最近252个市场交易日内的周度因子截面，不是252周 |
| 预测目标 | 信号后第1交易日收盘买入，到第6交易日收盘的收益，即持有5个交易日 |
| 标签成熟 | 训练标签结束日严格早于训练日，当天刚结束的也排除 |
| 重训 | 每月第一个周度信号日训练，月内其他周只预测 |
| 最少数据 | 26个历史周度截面、1000条股票记录 |
| 缺失特征 | 仅用训练样本中位数填充；训练列全缺失才补0 |
| 预测成员 | 按当时原固定融合分数有效性确定，不根据未来标签是否可用筛股票 |
| 复杂度 | 80棵树、学习率0.05、深度4、随机种子42、CPU两线程 |
| 调参 | 无收益导向参数搜索，无测试集早停，无模型之间择优切换 |

实际完成{len(training)}次月度模型训练，单次有效样本{training.train_rows.min():,}—{training.train_rows.max():,}条。这里一条是“一个日期的一只股票”，不是一次独立时间实验。

LightGBM继承15叶节点和80%列采样；虽然原构造器有subsample=0.8，subsample_freq=0意味着本轮没有行抽样。CatBoost使用数值输入、RMSE损失，不使用类别特征。训练日志和参数逐月保存。

原RankIC方案用每日IC记录调权，而模型用每周因子截面训练；原方案关注排序相关性，模型拟合原始收益误差。所以这是三套评分配方的实际对照，不能把差异全部解释成算法类别本身。训练按时间滚动不等于这段已研究多次的历史重新变成未见过的盲测区间。

## 四、持仓与执行保持哪些规则

每周从前100名按约束挑选15只，原持仓在前30名内有优先保留资格。单行业最多3只，历史正相关系数不高于0.8；基础池科技最多5只，科技池允许15只全部来自科技但仍保留单行业上限。

入选股票等分股票仓位，不足15只的空位留现金。总仓位按过去60日波动估计调整，目标年化波动15%，仓位预算20%—100%，每日检查但保留幅度门槛和冷却期。信号下一交易日收盘模拟成交。目标15只与行业上限不保证实际成交和价格漂移后绝不短暂超出；15%波动目标也不保证回撤。

买卖分别收3bp或13bp综合成本。所有新回放调用已经修复换股处理的资产入口，不使用旧的永久挂账回放逻辑。原方案引用封存的同日期、同成本修复后账户。

## 五、为什么完整回测停下：模型选中了中银绒业

不是程序没运行，也不是模型训练失败。8个模型账户均在中银绒业（000982.SZ）退市时仍有持仓，触发“需要已核实资产结算”的异常。现有两项换股登记不能用于这只股票。

深交所公告确认，该股触及连续20个交易日收盘价低于1元的交易类强制退市条件，且不进入退市整理期；[终止上市公告](https://www.szse.cn/disclosure/notice/company/t20240802_608510.html)提供了这一依据。[发行人收到终止上市决定公告](https://static.cninfo.com.cn/finalpage/2024-08-03/1220789319.PDF)也作出了相同提示。其退市后仍涉及股份确权和转入退市板块，而不是收到确定现金或自动换成另一只A股；[券商关于确权的公告](https://www.cjsc.com.cn/main/a/20240820/33172.shtml)给出的确权代码为400236。

现有回测价格矩阵没有400236的报价列。不能事后把000982从历史股票池删掉、假装最后一天一定卖得掉，不能沿用旧价格直到2026年，也不能不作说明地把股东权益全部清零。

{failures}

这些百分比只是停止当日按旧报价计算的账面占比，不是已验证的可变现价值。它们并非浮点残留。以基础池LightGBM、3bp账户为例，2024-06-03、06-11和06-17都出现买入，之后最后报价为06-21。最后三次买入已经写入单独的逐笔故障审计文件。

股票为什么在科技池？本轮“科技”按当期历史行业划分，并不按公司简称是否像科技公司判断；该股对应信号的行业被数据源归到电力设备。这个现象提醒我们：行业筛选和模型预测都不等于退市风险过滤。以后若要新增风险门槛，必须用当时可知的规则另设策略版本，不能针对已看到的失败股票事后补黑名单。

## 六、先看故障前的共同区间，而不是冒充完整回测

为避免只比较某个账户提前结束的收益，三种方案统一截到 **{cutoff.date()}**，即阻断资产最后有报价的日期。没有把6月下旬到8月按旧价格挂账的尾段加入这张共同对比表。

这是发现阻断后产生的诊断性截断，不是预注册的完整终点。它能回答“故障前模型表现怎样”，不能回答“截至2026年最后谁最好”。账户仍从2012年运行，2021年没有重新建仓；年化按252交易日，现金利息及无风险收益为0。

### 6.1 2012-01-06—2024-06-21，买卖各3bp

{perf('COMMON_FULL')}

![故障前共同长区间净值及回撤](assets/fixed15_ml/01_common_full.png)

### 6.2 2021年以来至2024-06-21，买卖各3bp

{perf('COMMON_RECENT')}

![故障前共同近年区间净值及回撤](assets/fixed15_ml/02_common_recent.png)

这两个图的左列是基础池、右列是科技池；上排为净值，下排为区间内回撤。图二用2020年末净值归一为1，只改变展示起点，不改变历史持仓。

**在这个共同区间，两种模型都没有提高原方案的年化收益。** 基础池的近年收益更弱；科技池CatBoost仍有正收益，回撤较小，但收益也低于原RankIC。不能把“所有指标都更差”和“收益没有提高”混为一谈，也不能据此证明所有机器学习方法都无效。

逐年对比：

{yearly}

2024年仅截至6月21日，没有将这段收益冒充全年收益。

## 七、成本和风险投入是否解释了差别

同样只比较2021—2024-06-21：

{table(['股票池','评分','3bp年化','13bp年化','3bp平均股票仓位','3bp年化双边换手'],costs)}

双边换手包括买入和卖出金额；1倍表示累计买卖总额相当于账户资产的100%。13bp是每边总成本，不是在3bp基础上额外增加13bp。实际规则相同也不代表实际仓位相同：股票选择不同，会改变波动估计和仓位预算，所以净收益不是纯粹的同风险选股Alpha。

## 八、排序诊断和仍未查明的问题

以下为2021年至共同截止日期的信号诊断，五日标签可能结束在信号之后，因此它不是截止日已实现账户收益：

{diagtable}

原始Top15没有经过分散、缓冲、仓位管理，五日均值也未扣成本；不能拿它代替实际组合。缺失标签只参与可配对统计，已经记录覆盖数量，选择性缺失仍是限制。

目前能够确认的是：固定配置模型并没有在共同区间自动改善结果，而且都选中了这只后来无法在A股市场正常退出的资产。不能直接下结论说“树模型就是追涨杀跌”或“某个因子导致了退市股持仓”。究竟是收益标签、反转信号、训练窗口、极端样本还是缺少硬风险门槛造成，还需要专项归因，不能用猜测代替证据。

## 九、哪些验收完成，哪些没有

- 已完成：两种真实后端、{len(training)}次严格滚动训练、未来标签排除、训练期中位数填充、完整参数与输入哈希、独立账户现金股数核对。
- 已核对：{valid['accounts']}个账户路径，包括{valid['complete_accounts']}个原方案完整账户、{valid['blocked_accounts']}个模型故障前前缀；共{valid['trade_rows']:,}条成交记录。校验PASS仅表示这些账目与时序检查通过，不表示8个模型账户完成了全期回测。
- 本轮全套单元测试527项通过，纸面账户哈希未改变。没有修改六策略日报、旧文章、旧模型默认配置或已封存的旧回测。
- 未完成：8个模型账户2024年退市后到2026年的资产路径、截至2026-10-15的三方案完整收益比较、整手与真实税费及涨跌停成交验证、真正新日期的前向验证。

下一步应先解决退市权益的迁移、后续价格和可交易性，或者事先约定明确的保守减值压力情景并另开实验版本。两条路径的含义不同：前者力求还原真实资产，后者只是情景假设。**不能为了让程序跑通，事后把它当成已按最后价格卖掉。** 本轮不自动采用清零假设、不修改股票池过滤、不继续搜索更好看的模型参数。

## 十、代码和复现

- 协议：`research_protocols/20261024_fixed15_ml_v1.json`。
- 训练工具：`analysis/fixed15_ml.py`；可选依赖：`requirements-ml-research.txt`。
- 原实验入口：`.venv/bin/python scripts/run_fixed15_ml.py`，遇到未处理退市资产会按协议报错。
- 分账户故障审计：`.venv/bin/python scripts/audit_fixed15_ml.py`，捕获异常前的原始账本并标记BLOCKED，不改变任何成交或资产规则；审计完成不等于回测完成。
- 独立核对：`.venv/bin/python scripts/validate_fixed15_ml.py --audit`。
- 生成本记录：`.venv/bin/python scripts/report_fixed15_ml.py`。
- 训练及评分：`output/fixed15_ml_20261024/`；账户状态、前缀账本、共同区间表格及问题股票交易：`output/fixed15_ml_20261024_audit/`。

输入及结果封存后拒绝覆盖；新的结算规则或策略变更必须使用新协议及输出目录。训练数据和模型构造仍继承原工程的数据修订、现金流代理和历史行业覆盖限制；更复杂的模型不会自动消除这些限制。
'''
    DOC.write_text(text)
    print(DOC)


if __name__=='__main__':run()
