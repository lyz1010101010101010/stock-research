"""
趋势分析 —— MA 多头/空头排列、是否站上年线、EMA 计算
"""

import sys
import os
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class TrendResult:
    """趋势分析结果"""

    def __init__(self):
        self.mas: dict[int, float] = {}            # {周期: 数值}
        self.ema_fast: Optional[float] = None      # EMA12
        self.ema_slow: Optional[float] = None      # EMA26
        self.trend_direction: str = "数据不足"      # 多头排列 / 空头排列 / 震荡 / 数据不足
        self.above_ma250: Optional[bool] = None     # 是否站上年线
        self.latest_price: Optional[float] = None


def _calc_ema(series: pd.Series, span: int) -> pd.Series:
    """计算指数移动平均（EMA）"""
    return series.ewm(span=span, adjust=False).mean()


def analyze_trend(df: pd.DataFrame) -> TrendResult:
    """
    对日线 DataFrame 执行趋势分析。

    需要 columns: close
    返回 TrendResult：MA、EMA、多空排列判定、年线位置
    """
    result = TrendResult()
    if df is None or df.empty or "close" not in df.columns:
        return result

    close = df["close"].astype(float)
    result.latest_price = float(close.iloc[-1])

    # ---------- 1. 计算 MA ----------
    for period in cfg.MA_PERIODS:
        if len(close) >= period:
            ma_val = close.rolling(window=period).mean().iloc[-1]
            result.mas[period] = round(float(ma_val), 2)
        else:
            result.mas[period] = None

    # ---------- 2. 计算 EMA (MACD 前置) ----------
    if len(close) >= cfg.MACD_SLOW:
        ema_fast = _calc_ema(close, cfg.MACD_FAST)
        ema_slow = _calc_ema(close, cfg.MACD_SLOW)
        result.ema_fast = round(float(ema_fast.iloc[-1]), 2)
        result.ema_slow = round(float(ema_slow.iloc[-1]), 2)

    # ---------- 3. 年线 ----------
    if result.mas.get(250) is not None and result.latest_price is not None:
        result.above_ma250 = result.latest_price > result.mas[250]

    # ---------- 4. 判定多空排列 ----------
    result.trend_direction = _judge_ma_arrangement(result.mas, result.latest_price)

    return result


def _judge_ma_arrangement(
    mas: dict[int, Optional[float]],
    latest_price: Optional[float],
) -> str:
    """
    判定均线排列状态。

    多头排列：MA5 > MA10 > MA20 > MA60 > MA250，且价格站上 MA5
    空头排列：MA5 < MA10 < MA20 < MA60 < MA250，且价格低于 MA5
    震荡：不符合上述两种
    数据不足：缺少足够均线
    """
    required = [5, 10, 20, 60, 250]
    values = {}
    for p in required:
        v = mas.get(p)
        if v is None or pd.isna(v):
            return "数据不足"
        values[p] = v

    price = latest_price
    if price is None:
        return "数据不足"

    # 多头
    if (values[5] > values[10] > values[20] > values[60] > values[250]) and price > values[5]:
        return "多头排列"

    # 空头
    if (values[5] < values[10] < values[20] < values[60] < values[250]) and price < values[5]:
        return "空头排列"

    # 部分多头（短期均线虽未严格排列，但价格在 MA60 之上且站上年线）
    if price > values[60] and price > values[250]:
        return "震荡偏多"

    # 部分空头
    if price < values[60] and price < values[250]:
        return "震荡偏空"

    return "震荡"


if __name__ == "__main__":
    from data.fetch_price import fetch_daily_price
    df = fetch_daily_price("600519")
    r = analyze_trend(df)
    print(f"趋势方向: {r.trend_direction}")
    print(f"MA: {r.mas}")
    print(f"站上年线: {r.above_ma250}")
