"""回测模块：单因子 / 多因子与工具函数。

新增逐账户研究使用 replay_with_asset_transitions，显式提供公司行动和
终止上市元数据。research_replay/compact_replay 保留用于冻结历史结果复现。
"""

from backtest.action_replay import replay as replay_with_asset_transitions

__all__ = ["replay_with_asset_transitions"]
