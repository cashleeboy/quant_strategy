#!/usr/bin/env python3
"""Insert the missing-cell results into the existing report, without replacing old results."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from scripts.run_mvp_holdings_count import sha

OUT=ROOT/'output/single_account_fixed15_risk_20261021'
REPORT=ROOT/'docs/research/SINGLE_ACCOUNT_ADAPTIVE_RESEARCH.md'
NEW='IC_CONSTRAINED15_RISK'
LABELS={
    'IC_CONSTRAINED15':'IC＋分散缓冲固定15只（无仓位管理）',
    NEW:'IC＋分散缓冲固定15只＋仓位管理（补测）',
    'IC_DYNAMIC_RISK':'IC＋分散缓冲动态10–20只＋仓位管理',
    'FIXED15_RISK':'固定因子权重15只＋仓位管理',
}
ENGLISH={
    'IC_CONSTRAINED15':'IC + constraints / K15, no risk budget',
    NEW:'IC + constraints / K15 + risk budget (NEW)',
    'IC_DYNAMIC_RISK':'IC + constraints / dynamic K + risk budget',
    'FIXED15_RISK':'Fixed weights / K15 + risk budget',
}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def extend_report(content,out=OUT):
    validation=json.loads((out/'independent_validation.json').read_text())
    if validation['status']!='PASS':
        raise ValueError('supplement has not passed verification')
    for name,digest in json.loads((out/'completed.json').read_text()).items():
        if sha(out/name)!=digest:
            raise ValueError(f'changed supplement result: {name}')
    perf=pd.read_csv(out/'performance.csv')
    base=perf[perf.fee.eq(.0003)].set_index('arm')
    stress=perf[perf.fee.eq(.0013)].set_index('arm')
    sub=pd.read_csv(out/'subperiods.csv')
    audit=pd.read_csv(out/'audit_summary.csv').query('fee == .0003').set_index('arm')
    parity=pd.read_csv(out/'control_parity.csv')
    nav=pd.read_csv(out/'nav.csv',index_col='date',parse_dates=True)
    pct=lambda x:f'{x:.2%}'
    money=lambda x:f'{30*(1+x):.2f}万元'
    summary=table(['版本','累计收益','年化收益','年化波动','夏普','最大回撤','平均股票仓位'],[
        [LABELS[a],pct(r.total_return),pct(r.cagr),pct(r.volatility),f'{r.sharpe:.3f}',pct(r.max_drawdown),pct(r.average_equity)]
        for a,r in base.iterrows()])
    costs=table(['版本','13bp累计收益','13bp年化收益','13bp夏普','13bp最大回撤'],[
        [LABELS[a],pct(r.total_return),pct(r.cagr),f'{r.sharpe:.3f}',pct(r.max_drawdown)] for a,r in stress.iterrows()])
    periods=table(['版本','2012–2015年化','2016–2020年化','2021–2026年化'],[
        [LABELS[a]]+[pct(sub[sub.arm.eq(a)&sub.fee.eq(.0003)&sub.period.eq(period)].iloc[0].cagr)
                     for period in ['2012-2015','2016-2020','2021-2026']] for a in LABELS])
    recent=sub[sub.period.eq('2021-2026')]
    recent_table=table(['版本','近段3bp累计收益','近段13bp累计收益'],[
        [LABELS[a]]+[pct(recent[recent.arm.eq(a)&recent.fee.eq(fee)].iloc[0].total_return)
                     for fee in [.0003,.0013]] for a in LABELS])
    n=base.loc[NEW];old=base.loc['IC_CONSTRAINED15'];dynamic=base.loc['IC_DYNAMIC_RISK'];ns=stress.loc[NEW]
    operations=table(['版本','年化双边换手','额外风控交易日','平均实际持股数'],[
        [LABELS[a],f'{base.loc[a,"annual_two_way_turnover"]:.2f}倍',int(audit.loc[a,'extra_risk_actions']),
         f'{base.loc[a,"average_execution_holdings"]:.2f}'] for a in LABELS])
    assets=ROOT/'docs/research/assets/single_account_adaptive'
    assets.mkdir(parents=True,exist_ok=True)
    colors=['#7f7f7f','#157f78','#b66c31','#8073ac']
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for arm,color in zip(LABELS,colors):
        s=nav[f'{arm}_3bp'];lw=2.4 if arm==NEW else 1.2
        axes[0,0].plot(s.index,s,color=color,lw=lw,label=ENGLISH[arm])
        axes[0,1].plot(s.index,s/s.cummax()-1,color=color,lw=lw)
        for ax,fee in zip(axes[1],[3,13]):
            full=nav[f'{arm}_{fee}bp']
            start=full.index.get_loc(full.loc['2021':].index[0])-1
            tail=full.iloc[start:]/full.iloc[start]-1
            ax.plot(tail.index,tail,color=color,lw=lw)
    axes[0,0].set_yscale('log');axes[0,0].set_title('2012-2026 NAV, 3 bp (log scale)')
    axes[0,0].legend(fontsize=8,loc='upper left')
    axes[0,1].set_title('2012-2026 drawdown, 3 bp')
    axes[1,0].set_title('2021 onward cumulative return, 3 bp')
    axes[1,1].set_title('2021 onward cumulative return, 13 bp')
    for ax in [axes[0,1],*axes[1]]: ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Missing-cell completion: fixed K15 with the SAME volatility-budget rule')
    fig.tight_layout();fig.savefig(assets/'04_fixed15_risk_extension.png',dpi=160);plt.close(fig)
    section=f'''## 十一、补齐固定15只的仓位管理对照（2026-10-21）

### 1. 这次补的不是新因子，而是原实验漏掉的一格

第一轮确实没有运行“IC＋分散缓冲固定15只＋仓位管理”，这个遗漏现在已经补齐。它与已有版本的区别很简单：

- 相对“IC＋分散缓冲固定15只”：只增加原来那套波动仓位管理。
- 相对“IC＋分散缓冲动态10–20只＋仓位管理”：只把目标数量固定为15只，不再按分散指标切换数量。
- 相对“固定因子权重15只＋仓位管理”：增加IC权重、分散限制和持仓缓冲。这一比较同时改变多项机制，不能把全部差异归功于IC。

新增版本每周仍然选15只，但不一定满仓；例如30万元账户、60%股票仓位时，目标为每只约1.2万元，另外12万元留现金。RankIC调整因子权重，分散规则限制集中风险，缓冲减少无谓换股，总仓位管理控制股票投入的总比例。

数据、IC历史窗口、因子权重上限、行业与主题限额、缓冲范围、波动目标、调整门槛及成本全部不变。仍为2012-01-06至2026-10-15，约14年9个月，不追加10月21日的数据，也不根据新结果调参。IC继续使用上限和变化速度约束，不是另换成无约束纯IC。

新旧风控版使用的是**同一套仓位规则**，不是强行使用同一条仓位曲线。股票篮子不同，估计波动和具体仓位自然可能不同；额外交易还可能改变后续缓冲所读取的实际持仓。

### 2. 低成本口径：固定15只加入仓位管理后的结果

每笔买卖按成交额扣3bp，即0.03%，结果如下：

{summary}

新版本累计收益为**{pct(n.total_return)}**。若仅将回测净值等比例换算成30万元本金，期末约为**{money(n.total_return)}**，其中盈利约{30*n.total_return:.2f}万元。不是只剩本金的204.17%，也不是一年赚204.17%；这仍不代表按30万元整手限制做过真实账户回测。

与相同选股规则、没有仓位管理的15只版本相比：

- 年化收益：{pct(old.cagr)} → {pct(n.cagr)}。
- 年化波动：{pct(old.volatility)} → {pct(n.volatility)}。
- 夏普：{old.sharpe:.3f} → {n.sharpe:.3f}。
- 最大回撤：{pct(old.max_drawdown)} → {pct(n.max_drawdown)}。

这说明减仓主要换来了更低的波动和回撤，而不是让绝对收益也同步上升。新版本平均股票仓位约{pct(n.average_equity)}，其余主要留现金。

与动态10–20只风控版相比，新版本年化收益高约{(n.cagr-dynamic.cagr)*100:.2f}个百分点，最大回撤幅度小约{(n.max_drawdown-dynamic.max_drawdown)*100:.2f}个百分点，夏普也略高。两者平均股票仓位分别为{pct(n.average_equity)}和{pct(dynamic.average_equity)}，并不是固定15只靠明显更低的平均股票投入取得这一结果。

不过，这些历史差异并未经过独立样本显著性验证，不能写成“固定15只一定优于动态数量”。

### 3. 加大交易成本后，固定15只仍优于动态版本，但收益明显缩水

将每笔买卖成本提高到13bp，即0.13%，所有账户重新回放：

{costs}

新增版本的年化收益从{pct(n.cagr)}降至{pct(ns.cagr)}，最大回撤从{pct(n.max_drawdown)}扩大至{pct(ns.max_drawdown)}。它仍优于动态风控版，但不能忽略超过四成的历史回撤。

{operations}

“交易日更多”不等于“总交易金额更多”。新版本增加了部分减仓和回补操作，但由于平均股票投入更低，年化双边换手低于不做仓位管理的版本。双边换手按买卖金额相对账户净值累计，不能只看成交笔数或调仓次数判断成本。

### 4. 近年表现依然是主要短板

{periods}

分段沿用持续账户，只切片观察，没有在每段开头重置持仓。重点看2021年至2026-10-15这一段累计收益：

{recent_table}

新版本在低成本口径下，这近六年累计只赚5.34%，年化约0.94%；高成本口径下累计亏10.16%，年化约-1.91%。而简单的“固定因子权重15只＋仓位管理”，同段低成本累计收益为12.46%，高成本累计亏3.77%，两种口径都好于新版本。

因此，**补测改善了这组长期风险收益对比，但没有解决近年盈利薄弱、对交易成本敏感的问题。** 全期领先不能代替分阶段检查，更不能直接升级为实盘默认策略。

![固定15只仓位管理补测：长期净值、回撤及近年成本对比](assets/single_account_adaptive/04_fixed15_risk_extension.png)

图中绿色粗线是本次补测版本。左上看全期净值，右上看回撤；下方分别用3bp和13bp展示2021年以来的累计收益。下方曲线以2020年最后一个交易日净值为基点，用来保留2021年第一个交易日的收益。长期绿色线略好，并不意味着近几年也形成了足够强的盈利能力。

### 5. 这次如何保证是补齐对照，而不是另换一套实验

- 月度因子权重、每周股票排名和固定配方核对记录，与上一轮逐表一致。
- 三个已有对照在两档成本下全部重跑，共六组复现；每日净值最大绝对误差为{parity.max_daily_nav_error.max():.2e}，目标持仓、执行记录和逐笔交易也核对通过。
- 新增版本756个周度信号均固定目标15只，未选满周数为{int(audit.loc[NEW,'underfilled_weeks'])}；额外风控交易日为{int(audit.loc[NEW,'extra_risk_actions'])}。
- 本次新增4项接线测试，核实固定15只、真实持仓缓冲、仓位调整和未来数据不影响过去目标；全仓库486项测试通过。
- 独立复核费用、现金、下一交易日执行、权重与相关性约束通过，原始结果文件和六策略纸面账户没有改变。

这仍不等于实盘验收。新增版本实际持股最高达到{int(audit.loc[NEW,'max_actual_holdings'])}只，原因是旧持仓缺价时不能伪造卖出；单日未报价资产最高占净值{pct(audit.loc[NEW,'max_unquoted_fraction'])}。在{int(audit.loc[NEW,'risk_review_days'])}个风险检查日中，{int(audit.loc[NEW,'insufficient_risk_history_days'])}日共同历史不足而保留上一预算。风控估计的仍是拟持有篮子，没有把无法退出的额外旧仓逐一并入。这些限制没有在补测时被暗中修掉，因此对照保持一致，但生产可用性仍有限。

### 6. 补测后的结论

现在可以把结论说完整：**在本轮同数据、同成本、同风控规则的历史对照中，固定15只比动态10–20只略好；暂没有必要为了“更智能”而增加动态K这一层。** 不加仓位管理和加仓位管理的两种条件，都没有看到动态数量带来的优势。

但应保留两条边界。第一，新版本全期收益仍低于没有仓位管理的固定15只，风险改善有代价。第二，近年低收益、高成本下亏损以及36.15%的低成本最大回撤，说明它尚未解决稳定盈利问题。

因此，这次补测支持把“固定数量＋明确约束＋独立仓位管理”保留为更简单的研究候选，不支持宣称它已经能稳定抵抗下跌。它没有自动替换六策略日报，也没有获得实盘批准。下一步若推进实际账户，仍应先明确真实本金、可承受亏损，补足整手、真实费用与可成交性检查，再用新数据前向验证；不能继续用同一段历史反复挑赢家。

'''
    start=content.index('## 十一、')
    stop=content.index('## 十二、',start)
    content=content[:start]+section+content[stop:]
    notice='**2026-10-21补充：已完成“IC＋分散缓冲固定15只＋仓位管理”。前十节保留首轮七组结果，第十一节给出第八组补测、相同风控下的对照及更新结论。**'
    if notice not in content:
        content=content.replace('## 一、先说结果',notice+'\n\n## 一、先说结果',1)
    marker='### 固定15只仓位管理补测的复现'
    if marker not in content:
        content+='''\n### 固定15只仓位管理补测的复现

补充协议：`research_protocols/20261021_fixed15_risk_extension_v1.json`

独立结果目录：`output/single_account_fixed15_risk_20261021/`。包含新组与三组重跑对照的绩效、分段收益、每日账本、目标、逐笔交易及`control_parity.csv`复现核对表；原10月20日结果不覆盖。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/quant-mpl .venv/bin/python scripts/run_fixed15_risk_extension.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/quant-mpl .venv/bin/python scripts/report_single_account_adaptive.py
.venv/bin/python -m unittest tests.test_fixed15_risk_extension -v
```

生成原报告时会自动读取已通过核验的补测结果，保留本节内容，不再恢复“该组合尚未测试”的旧结论。
'''
    return content


if __name__=='__main__':
    REPORT.write_text(extend_report(REPORT.read_text()),encoding='utf-8')
    print(REPORT)
