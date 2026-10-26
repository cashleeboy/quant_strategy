# Beta 长历史研究运行记录

日期：2026-10-19。仅研究，不改变生产因子、股票池或前向纸面账本。

## 结果入口

- 完整文章：[第145篇](/Users/dev/Documents/quant_strategy/docs/articles/145_Beta到底能不能选股_一次跨越十五年的A股因子研究.md)。
- 数据和结果：`output/beta_research_20261019/`。
- 协议：`research_protocols/20261019_beta_research_v1.json`。
- 原因子库候选讨论：`docs/research/FACTOR_LIBRARY_EXTENSION_DISCUSSION.md`。

2010年行情预热，2011-01-14首个信号、2011-01-17首次执行，截至2026-10-15，共806次周频信号。主基准中证全指，252日、至少200个配对收益；126日窗口和沪深300基准作为预先指定对照。所有版本使用共同有效样本，另保留完整基础池等权。

低Beta50只年化3.20%、波动17.01%、最大回撤-52.69%；共同样本等权分别为3.37%、25.98%、-63.30%。126日和沪深300版本收益转负。结论为风险诊断有价值，尚无充分证据默认加入选股融合。不是一次新的未知样本外验收。

## 复现

```bash
# 读取环境变量 TUSHARE_TOKEN，或 Tushare 已保存凭据；不在命令中写 token。
.venv/bin/python scripts/run_beta_research.py --stage fetch

MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_beta_research.py --stage run

MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/diagnose_beta_controls.py

MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/validate_beta_research.py

MPLCONFIGDIR=/tmp/quant_strategy_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/report_beta_research.py

.venv/bin/python -m unittest tests.test_beta_research tests.test_holdings_count tests.test_season02_integration -v
```

读者解释补充：运行 `.venv/bin/python scripts/describe_beta_groups.py` 可从封存的个股Beta与目标名单生成 `beta_group_ranges.csv` 和 `beta_group_range_summary.json`，逐周核对Q1—Q5成员后统计真实范围。此步骤不重新选股、不改回测，并支持 `--output` 指定研究目录。第145篇已补入理论取值含义、全样本分布及2026-10-09的真实分组例子。

已完成的输出禁止直接覆盖。重跑计算需用 `--output` 指定新目录，并先在同一目录运行 fetch；诊断、验收和绘图也使用相同的 `--output`。各命令默认指向本次封存目录。重复执行补充诊断应得到相同内容；不得改动已封存主结果。绘图脚本还会更新第145篇对应的图片目录，新实验若不用于更新本文，应先修改该输出位置。

## 验收

- 475项全量测试通过。
- 128个真实数据OLS核对通过。
- 完整低Beta50路径与独立批量回放引擎一致。
- 所有绩效由净值重新计算核对。
- IC区间未成熟不得输出值。
- 源价格、年度输入、市场数据、研究代码和主输出记录校验和。
- 前向账本执行前后哈希不变。

现金对照按当时目标Beta近似匹配，非实际等风险保证；六策略橙线为已知股票归一化目标Beta，不是实际持仓加权Beta。完整费用、市场冲击、整手交易、历史财务原始版本等仍未解决，见文章限制章节。

研究输出、文章和图片遵循仓库既有忽略规则；本次未执行提交或推送。
