#!/usr/bin/env python3
"""Render the reproducible research record; never overwrite earlier experiments."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scripts.run_fixed15_ml import OUT
from scripts.run_mvp_holdings_count import sha

DOC=ROOT/'docs/research/FIXED15_ML_COMPARISON.md'
ASSETS=ROOT/'docs/research/assets/fixed15_ml'
LABELS={'BASE':'原RankIC评分','lightgbm':'LightGBM评分','catboost':'CatBoost评分'}
AREAS={'BROAD':'基础池','TECH15':'科技池'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def run():
    if not (OUT/'completed.json').exists():
        from scripts.report_fixed15_ml_audit import run as report_blocked
        return report_blocked()
    valid=json.loads((OUT/'independent_validation.json').read_text());assert valid['status']=='PASS'
    for f,h in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/f)==h,f
    data=pd.read_csv(OUT/'comparison.csv');nav=pd.read_csv(OUT/'nav.csv',index_col='date',parse_dates=True)
    train=pd.read_csv(OUT/'training.csv');diag=pd.read_csv(OUT/'diagnostics.csv')
    annual=pd.read_csv(OUT/'annual.csv');pct=lambda v:f'{v:.2%}'
    def performance(period):
        rows=[]
        for area in AREAS:
            for model in LABELS:
                r=data[data.period.eq(period)&data.area.eq(area)&data.backend.eq(model)&data.fee.eq(.0003)].iloc[0]
                rows.append([AREAS[area],LABELS[model],pct(r.total_return),pct(r.cagr),pct(r.volatility),f'{r.sharpe:.2f}',pct(r.max_drawdown)])
        return table(['股票池','评分方法','累计收益','年化收益','年化波动','夏普（无风险0）','最大回撤'],rows)
    costs=[];operations=[];changes=[]
    for area in AREAS:
        for model in LABELS:
            x=data[data.period.eq('RECENT')&data.area.eq(area)&data.backend.eq(model)].set_index('fee')
            a,b=x.loc[.0003],x.loc[.0013]
            costs.append([AREAS[area],LABELS[model],pct(a.cagr),pct(b.cagr),pct(b.total_return),pct(b.max_drawdown)])
            operations.append([AREAS[area],LABELS[model],pct(a.average_equity),f'{a.annual_two_way_turnover:.2f}倍',f'{a.annual_action_dates:.1f}天'])
            if model!='BASE':
                base=data[data.period.eq('RECENT')&data.area.eq(area)&data.backend.eq('BASE')&data.fee.eq(.0003)].iloc[0]
                stressbase=data[data.period.eq('RECENT')&data.area.eq(area)&data.backend.eq('BASE')&data.fee.eq(.0013)].iloc[0]
                changes.append(f'- {AREAS[area]}／{LABELS[model]}：近年年化相对原方案变化 **{(a.cagr-base.cagr)*100:+.2f}个百分点**；最大回撤从{pct(base.max_drawdown)}变为{pct(a.max_drawdown)}；13bp下年化相对变化{(b.cagr-stressbase.cagr)*100:+.2f}个百分点。')
    diagnostic=table(['股票池','模型','每周平均RankIC','有效标签股票数均值','原始Top15五日平均收益','全池五日平均收益'],[
        [AREAS[area],LABELS[model],f'{g.rank_ic.mean():.4f}',f'{g.label_pairs.mean():.0f}',pct(g.raw_top15_return.mean()),pct(g.pool_return.mean())]
        for (area,model),g in diag[diag.date.ge('2021')].groupby(['area','backend'])])
    yeartable=table(['年份']+[AREAS[a]+'/'+LABELS[m] for a in AREAS for m in LABELS],[
        [str(y)+('（截至10月15日）' if y==2026 else '')]+[pct(annual[(annual.key==f'{m}_{a}_3bp')&(annual.year==y)].return_value.iloc[0]) for a in AREAS for m in LABELS]
        for y in range(2021,2027)])
    ASSETS.mkdir(parents=True,exist_ok=True)
    colors={'BASE':'#64748b','lightgbm':'#0369a1','catboost':'#c05621'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for filename,start,title in [('01_full_nav.png','2012','2012-01-06 to 2026-10-15'),('02_recent_nav.png','2021','2021 to 2026-10-15 (inherited holdings)')]:
        sub=nav.loc[start:];loc=nav.index.get_loc(sub.index[0]);sub=nav.iloc[max(0,loc-1):];sub=sub/sub.iloc[0]
        fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
        for j,area in enumerate(AREAS):
            for model in LABELS:
                v=sub[f'{model}_{area}_3bp'];axs[0,j].plot(v.index,v,label=model,color=colors[model],lw=1.2)
                dd=v/v.cummax()-1;axs[1,j].plot(dd.index,dd,label=model,color=colors[model],lw=1.)
            axs[0,j].set_title(f'{area} / fixed15 / 3bp per side');axs[0,j].set_ylabel('Net asset value (start = 1)')
            axs[1,j].set_ylabel('Drawdown within displayed period (%)');axs[1,j].yaxis.set_major_formatter(PercentFormatter(1))
            axs[1,j].set_xlabel('Date');axs[0,j].legend();axs[0,j].grid(alpha=.2);axs[1,j].grid(alpha=.2)
        fig.suptitle(title+' | Source: frozen, asset-aware research replay')
        fig.tight_layout();fig.savefig(ASSETS/filename,dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for ax,area in zip(axs,AREAS):
        x=np.arange(3)
        for offset,fee,color in [(-.18,.0003,'#0369a1'),(.18,.0013,'#94a3b8')]:
            values=[data[data.period.eq('RECENT')&data.area.eq(area)&data.backend.eq(m)&data.fee.eq(fee)].cagr.iloc[0] for m in LABELS]
            ax.bar(x+offset,values,width=.36,color=color,label=f'{round(fee*10000)}bp per side')
        ax.set_xticks(x,list(LABELS));ax.set_title(area);ax.set_ylabel('Annualized net return (%)');ax.set_xlabel('Scoring method')
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.axhline(0,color='black',lw=.6);ax.legend()
    fig.suptitle('2021 to 2026-10-15: execution-cost sensitivity | Source: frozen research replay')
    fig.tight_layout();fig.savefig(ASSETS/'03_cost_sensitivity.png',dpi=160);plt.close(fig)
    text=f'''# 固定15只组合接入LightGBM与CatBoost：滚动训练和同期回测

实验日期：2026-10-24。数据和回放终点仍为2026-10-15。本轮只改研究评分入口，不修改六策略日报，不增加新下载数据。

## 一、为什么现在做这次实验

最近几轮研究的核心问题不是“因子名称不够多”，而是：在历史RankIC调权、分散约束、持仓缓冲、固定15只和总仓位管理这套组合中，2021年以来的基础池收益偏弱。科技池曾有改善，但不能据此认定选股能力普遍恢复。

前一项工作又发现并修复了换股吸收合并的资产处理问题。本次必须站在修复后的账户账本上比较，不能让模型变化与资产记账变化混在一起。

这次提出一个有限的问题：**不增加数据和因子，改用模型学习原13个因子的非线性关系和相互作用，是否比原RankIC评分更有帮助？**

工程原本有两种模型构造代码，但有代码不等于此前日报正在使用；旧函数在依赖缺失时还可能回退到另一模型。本次安装固定版本LightGBM4.6.0、CatBoost1.2.8，强制核对真实后端，训练失败直接停止。

## 二、怎样接入：三组独立比较，不是逐个叠加

| 方案 | 股票怎么打分 | 股票怎么持有 |
| --- | --- | --- |
| 原RankIC | 历史正平均RankIC逐月调节因子权重，保留原有约束和分组标准化 | 原固定15只组合规则 |
| LightGBM | 用13因子预测下一交易日收盘买入、持有五个交易日的收益，按预测值排序 | 同一套固定15只规则 |
| CatBoost | 同样输入和目标，由另一种梯度提升树模型评分 | 同一套固定15只规则 |

这不是“先加LightGBM，再加CatBoost”，也不是在原分数上再各加一票。两种模型分别替代选股评分，彼此不融合。模型的重要性数值不是资金权重，更不是固定的因子配置比例。

只用原来的13个传统因子：PE、现金流收益率代理、ROE、毛利率、净利率、低资产负债率、现金利润质量、营收增长、利润增长、原动量、60日动量、5日反转、波动率。不加入消息、公告、股票代码、行业标签作为特征，也不加入最近新增的五个候选因子。现金流收益率保留原工程的经营现金流代理口径，不能冒称已扣资本开支的完整自由现金流。

两种模型都在基础池历史样本上训练；科技方案使用同一个模型，在当期历史行业为电子、计算机、通信、传媒、电力设备的候选中排序。它不是专门训练的科技模型，也不是纯AI主题池。

## 三、训练如何避免看到未来

| 环节 | 本次固定规则 |
| --- | --- |
| 历史窗口 | 最近252个市场交易日中的每周因子截面，不是252周 |
| 标签 | 信号后第1个交易日收盘到第6个交易日收盘的复权收益，即持有5个交易日 |
| 标签成熟 | 标签结束日必须严格早于训练日，当天才结束的也不纳入 |
| 更新频率 | 每月第一个周度信号日重新训练，月内其他周只预测 |
| 样本下限 | 至少26个历史周度截面、1000条有效股票样本 |
| 缺失值 | 只用训练样本的中位数填充；训练列全缺失才补0；特征名单预先固定 |
| 预测股票池 | 只按当时原固定评分是否有效确定，不以未来收益是否有数据筛股票 |
| 参数 | 80棵树、学习率0.05、深度4、随机种子42、CPU两线程；无调参和测试集早停 |
| 成交 | 信号之后的下一交易日收盘模拟成交；修复后的公司行动账本 |

LightGBM继承原构造器的15个叶节点和80%列抽样。原参数虽然写了subsample=0.8，但subsample_freq=0，本轮没有启用行抽样，不能把它说成已经使用了80%行随机采样。CatBoost使用数值特征与RMSE，不启用类别特征处理。

实际完成 **{len(train)}次月度训练**，单次有效样本为{train.train_rows.min():,}—{train.train_rows.max():,}条。每个日期包含许多股票，这些行不是彼此独立的时间样本，不能把几十万行解释成几十万次独立验证。

还有一项比较边界：原RankIC估计使用每日IC记录，模型使用每周截面；原方案优化排序相关性，模型最小化原始收益预测误差。本次比较的是三种完整评分配方，而不是只有“模型类别”一个数学差别的消融。

历史在过去多轮研究中已经看过。因此，“训练不读未来”成立，并不等于这段历史重新变成了未参与研究的盲测样本。

## 四、哪些持仓规则保持不变

每周从分数前100名中选择目标15只；原持仓仍在前30名时有保留资格，但仍须满足其他限制。单行业最多3只、历史正相关系数不能超过0.8；基础池科技合计最多5只，科技专用池允许15只全部来自科技，但单行业上限不变。

入选股票平分目标股票仓位；若不足15只，缺额留现金。总仓位由过去60日组合波动估计确定，目标年化波动15%，预算限制20%—100%，每天检查但有幅度门槛和冷却期。因此“每周调仓”不等于一周只能交易一次；风险调整仍可能在周中发生。

这些是信号时的目标限制，不保证实际成交后及价格漂移中的持仓永远正好15只或行业占比绝不越界。15%目标波动也不是回撤上限。

## 五、全期结果：2012-01-06—2026-10-15

每次买入和卖出各收3bp综合成本。下表年化按252个交易日计算，现金利息为0；原方案直接引用已封存、修复后同日期同费率账户，不另改一套基准。

{performance('FULL')}

![全期净值及回撤](assets/fixed15_ml/01_full_nav.png)

上排是扣成本后的净值，下排是从区间内高点计算的回撤。左右分别为基础池与科技池，不能只根据不同池子的净值高低把收益归因于模型。

## 六、核心检验：2021年以来有没有改善

近年区间延续2012年开始运行的真实研究持仓，不在2021年重新空仓启动。累计收益使用2020年末净值作为起点。

{performance('RECENT')}

{chr(10).join(changes)}

![近年净值及回撤](assets/fixed15_ml/02_recent_nav.png)

图中把2020年最后交易日净值归一为1，只有展示起点重设，没有重新建仓。区间最大回撤也按这个展示区间计算。

逐年看，避免把某一段上涨误认成每年都有效：

{yeartable}

2026年只是截至10月15日的累计收益，不是全年收益，更没有将当年这段收益再年化放进年度表。

## 七、加大交易成本之后

13bp仍是买卖各自承担的综合压力成本，不是额外再加13bp，也不是完整真实佣金和税费账单。

{table(['股票池','评分方法','3bp近年年化','13bp近年年化','13bp近年累计','13bp近年最大回撤'],costs)}

![近年成本敏感性](assets/fixed15_ml/03_cost_sensitivity.png)

这些账户的实际风险投入和交易频率也应当一起看：

{table(['股票池','评分方法','近年平均股票仓位','年化双边换手','年均动作日期'],operations)}

双边换手是买入和卖出金额之和除以当时账户资产后再年化，1倍表示相当于账户规模的100%买卖总额。动作日期包含周度选股和额外风险调仓，并不只统计更换股票名单的日期。

即使风险规则相同，模型选出的股票不同，估计波动和最终仓位仍会不同。因此账户收益差异不是完全相同风险暴露下的纯选股Alpha。

## 八、模型排序的辅助检查

以下只统计2021年以来的周度信号，收益标签依然是后续可执行的五日收益：

{diagnostic}

“原始Top15”是未经分散约束、持仓缓冲和仓位管理的最高预测分15只，不是实际账户持仓；其五日平均收益不是账户年化，也没有扣交易成本。遇到缺失标签只对可观测配对计算，并记录数量，这种选择性缺失本身仍是限制。

这些统计用于检查模型有没有截面信息，不能把正RankIC直接等同于最终净收益有效，也不能把某种模型的特征重要性当成因果解释。

## 九、验证、边界和如何使用

独立核对 **{valid['accounts']}个账户、{valid['trade_rows']:,}条买卖记录、{valid['asset_actions']}笔公司行动**：按成交重新累加现金和股数，每日重建净值；检查信号必须下一交易日成交、没有遗留已终止上市的旧代码、模型训练标签严格成熟、科技筛选在Top100之前完成。纸面账户文件哈希保持不变。

这次新增的是可重复运行的研究入口，不是把模型自动接管日报。执行结果应按“近年改善是否存在、回撤是否付出更大代价、13bp是否保留、收益是否只集中一两年”联合判断，不能只选全期收益最高的一条线。

未完成的实盘条件仍包括整手交易、最低佣金、真实税费、涨跌停和流动性容量。训练标签还继承复权价格与缺失报价的限制；既有财务历史修订、行业历史覆盖不足也没有被机器学习自动修好。已知资产转换使用现有两项核实事件，其他未知终止上市事件若实际被持有会中止回放，而不是自动按最后价格永久挂账。

如果某模型值得继续，也应先冻结规则并开展新日期的前向纸面验证；本轮不追加搜索训练窗口、树深和TopK，不直接替换六策略日报。

## 十、复现位置

- 协议：`research_protocols/20261024_fixed15_ml_v1.json`。
- 严格训练工具：`analysis/fixed15_ml.py`；复用`factors/factor_ml.py`模型构造器，但不使用旧的滚动训练和静默回退行为。
- 运行：`.venv/bin/python scripts/run_fixed15_ml.py`；每月训练及每个账户有哈希检查点，完成后拒绝覆盖同目录。
- 校验：`.venv/bin/python scripts/validate_fixed15_ml.py`。
- 报告：`.venv/bin/python scripts/report_fixed15_ml.py`。
- 数据、评分、训练中位数、模型参数、特征重要性、账户路径及核对日志：`output/fixed15_ml_20261024/`。不保存重复的原始数据或全部模型二进制；用固定代码、输入哈希、版本及随机种子复现。
- 依赖：`requirements-ml-research.txt`；模型官方说明：[LightGBM Python包](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst)、[CatBoost Python安装](https://catboost.ai/docs/en/installation/python-installation-method-pip-install?lang=en)。

旧实验输出、旧文章、原ML默认配置和六策略纸面交易记录均未改写。
'''
    DOC.write_text(text)
    print(DOC)


if __name__=='__main__':run()
