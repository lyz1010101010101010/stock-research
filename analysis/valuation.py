"""
基本面估值分析 —— PE-TTM / PB / ROE / 历史分位 / 估值区间判定
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from data.fetch_fundamental import (
    fetch_valuation_indicators,
    fetch_pe_pb_history,
    fetch_roe_history,
)


class ValuationResult:
    """估值分析结果容器"""

    def __init__(self):
        self.pe_ttm: Optional[float] = None
        self.pb: Optional[float] = None
        self.ps: Optional[float] = None
        self.market_cap: Optional[float] = None
        self.roe_list: list[float] = []
        self.roe_avg: Optional[float] = None

        self.pe_percentile: Optional[float] = None   # 0~100
        self.pb_percentile: Optional[float] = None

        self.valuation_range: str = "数据不足"       # 低估 / 合理偏低 / 合理 / 偏高 / 泡沫 / 数据不足

    def summary_dict(self) -> dict:
        return {
            "pe_ttm": self.pe_ttm,
            "pb": self.pb,
            "ps": self.ps,
            "market_cap": self.market_cap,
            "roe_avg": self.roe_avg,
            "pe_percentile": self.pe_percentile,
            "pb_percentile": self.pb_percentile,
            "valuation_range": self.valuation_range,
        }


def analyze_valuation(symbol: str) -> ValuationResult:
    """
    对单个股票执行估值分析。

    步骤：
    1. 获取 PE-TTM / PB / PS / 市值
    2. 获取近 5 年 PE / PB 日频序列并计算当前分位
    3. 获取近 3 年 ROE
    4. 综合判定估值区间
    """
    result = ValuationResult()

    # ---------- 1. 最新估值指标 ----------
    val = fetch_valuation_indicators(symbol)
    result.pe_ttm = val["pe_ttm"]
    result.pb = val["pb"]
    result.ps = val["ps"]
    result.market_cap = val["market_cap"]

    # ---------- 2. ROE 历史 ----------
    roe_data = fetch_roe_history(symbol, years=3)
    result.roe_list = roe_data
    if roe_data:
        result.roe_avg = round(sum(roe_data) / len(roe_data), 2)

    # ---------- 3. PE / PB 历史分位 ----------
    hist = fetch_pe_pb_history(symbol, years=cfg.VALUATION_HISTORY_YEARS)
    result.pe_percentile = hist["pe_ttm_percentile"]
    result.pb_percentile = hist["pb_percentile"]

    # ---------- 4. 估值区间判定 ----------
    result.valuation_range = _judge_valuation_range(
        pe_ttm=result.pe_ttm,
        pe_pct=result.pe_percentile,
        pb=result.pb,
        pb_pct=result.pb_percentile,
        roe_avg=result.roe_avg,
    )

    return result


def _judge_valuation_range(
    pe_ttm: Optional[float],
    pe_pct: Optional[float],
    pb: Optional[float],
    pb_pct: Optional[float],
    roe_avg: Optional[float],
) -> str:
    """
    综合多因子判定估值区间。

    逻辑：
    - 优先看分位：
        pe_pct >= 80 或 pb_pct >= 80 → 偏高 / 泡沫
        pe_pct <= 20 或 pb_pct <= 20 → 低估
    - 若分位缺失，拍 PE-TTM 绝对值 + 行业常识（不同行业合理 PE 不同）：
        PE > 60 → 泡沫（非成长股）
        PE < 10 且 ROE > 15% → 低估
        PE 10~20 → 合理偏低
        PE 20~40 → 合理
        PE 40~60 → 偏高
    - 有 ROE 辅助：
        ROE > 20% 可容忍更高的 PE
    """
    # --- 分位法（有数据时优先） ---
    if pe_pct is not None and pb_pct is not None:
        avg_pct = (pe_pct + pb_pct) / 2
        if avg_pct >= 90:
            return "泡沫"
        elif avg_pct >= 75:
            return "偏高"
        elif avg_pct >= 40:
            return "合理"
        elif avg_pct >= 20:
            return "合理偏低"
        else:
            return "低估"

    # 只有一列分位
    single_pct = pe_pct if pe_pct is not None else pb_pct
    if single_pct is not None:
        if single_pct >= 90:
            return "泡沫"
        elif single_pct >= 75:
            return "偏高"
        elif single_pct >= 40:
            return "合理"
        elif single_pct >= 20:
            return "合理偏低"
        else:
            return "低估"

    # --- 绝对值法（无分位时兜底） ---
    if pe_ttm is not None:
        # 高 ROE 可以给予更高 PE 容忍度
        premium = 0
        if roe_avg is not None and roe_avg > 20:
            premium = 15
        elif roe_avg is not None and roe_avg > 15:
            premium = 8

        pe = pe_ttm
        if pe > 60 + premium:
            return "泡沫"
        elif pe > 40 + premium:
            return "偏高"
        elif pe > 20 + premium:
            return "合理"
        elif pe > 10:
            return "合理偏低"
        elif pe > 0:
            return "低估"

    # --- 数据不足 ---
    return "数据不足"


if __name__ == "__main__":
    r = analyze_valuation("600519")
    from pprint import pprint
    pprint(r.summary_dict())
