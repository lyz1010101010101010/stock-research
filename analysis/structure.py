"""
支撑 / 阻力 以及 风险指标 —— 近60日高低点、年线、最大回撤、年化波动率、年线偏离度
"""

import sys
import os
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class StructureResult:
    """结构 + 风险分析结果"""

    def __init__(self):
        # 支撑 / 阻力
        self.support_levels: list[float] = []       # 参考支撑
        self.resistance_levels: list[float] = []    # 参考压力
        self.recent_high: Optional[float] = None
        self.recent_low: Optional[float] = None
        self.ma250_value: Optional[float] = None

        # 风险指标
        self.max_drawdown: Optional[float] = None    # 近250日最大回撤（%）
        self.annual_volatility: Optional[float] = None  # 年化波动率（%）
        self.ma250_deviation: Optional[float] = None  # 当前价格偏离年线幅度（%）


def analyze_structure(df: pd.DataFrame) -> StructureResult:
    """
    分析支撑/阻力及风险指标。

    参数
    -----
    df : pd.DataFrame
        必须含 columns: close, high, low

    返回 StructureResult
    """
    result = StructureResult()
    if df is None or df.empty:
        return result

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    # ========== 1. 近期高低点（60日内） ==========
    lookback = min(cfg.STRUCTURE_LOOKBACK, len(df))
    recent_high_vals = high.iloc[-lookback:]
    recent_low_vals = low.iloc[-lookback:]

    result.recent_high = round(float(recent_high_vals.max()), 2)
    result.recent_low = round(float(recent_low_vals.min()), 2)

    # ========== 2. MA250 年线 ==========
    if len(close) >= 250:
        ma250 = close.rolling(250).mean()
        result.ma250_value = round(float(ma250.iloc[-1]), 2)

    # ========== 3. 支撑 / 压力 ==========
    latest_close = float(close.iloc[-1])

    # 支撑：
    supports = []
    # - 近期低点
    supports.append(result.recent_low)
    # - MA60
    if len(close) >= 60:
        ma60 = float(close.rolling(60).mean().iloc[-1])
        if ma60 < latest_close:
            supports.append(round(ma60, 2))
    # - MA250（如果在当前价格之下）
    if result.ma250_value is not None and result.ma250_value < latest_close:
        supports.append(result.ma250_value)

    # 去重、排序、取两个
    supports = sorted(set(s for s in supports if s is not None and s < latest_close), reverse=True)[:2]
    # 按升序（支撑从低到高）
    supports.sort()
    result.support_levels = supports

    # 压力：
    resistances = []
    # - 近期高点
    if result.recent_high > latest_close:
        resistances.append(result.recent_high)
    # - MA60（如果在当前价格之上）
    if len(close) >= 60:
        ma60 = float(close.rolling(60).mean().iloc[-1])
        if ma60 > latest_close:
            resistances.append(round(ma60, 2))
    # - MA250（如果在当前价格之上）
    if result.ma250_value is not None and result.ma250_value > latest_close:
        resistances.append(result.ma250_value)

    resistances = sorted(set(r for r in resistances if r is not None and r > latest_close))[:2]
    result.resistance_levels = resistances

    # ========== 4. 最大回撤（近250日） ==========
    lookback_risk = min(cfg.RISK_LOOKBACK, len(close))
    if lookback_risk >= 20:
        segment = close.iloc[-lookback_risk:]
        rolling_max = segment.expanding().max()
        drawdown = (segment - rolling_max) / rolling_max * 100
        result.max_drawdown = round(float(drawdown.min()), 2)

    # ========== 5. 年化波动率 ==========
    if lookback_risk >= 20:
        daily_returns = close.iloc[-lookback_risk:].pct_change().dropna()
        daily_std = float(daily_returns.std())
        result.annual_volatility = round(daily_std * np.sqrt(252) * 100, 2)

    # ========== 6. 年线偏离度 ==========
    if result.ma250_value is not None and result.ma250_value > 0:
        deviation = (latest_close - result.ma250_value) / result.ma250_value * 100
        result.ma250_deviation = round(float(deviation), 2)
    else:
        result.ma250_deviation = None

    return result


if __name__ == "__main__":
    from data.fetch_price import fetch_daily_price
    df = fetch_daily_price("600519")
    r = analyze_structure(df)
    print(f"支撑: {r.support_levels}")
    print(f"压力: {r.resistance_levels}")
    print(f"最大回撤: {r.max_drawdown}%")
    print(f"年化波动率: {r.annual_volatility}%")
    print(f"年线偏离度: {r.ma250_deviation}%")
