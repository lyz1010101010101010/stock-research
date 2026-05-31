"""
量价分析 —— 成交量均线、量能放大/缩小、放量上涨/下跌
"""

import sys
import os
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class VolumeResult:
    """量价分析结果"""

    def __init__(self):
        self.vol_ma5: Optional[float] = None
        self.vol_ma20: Optional[float] = None
        self.volume_ratio: Optional[float] = None     # 当前成交量 / MA5
        self.volume_status: str = "数据不足"           # 放量 / 缩量 / 正常
        self.price_volume_signal: str = "数据不足"    # 放量上涨 / 缩量上涨 / 放量下跌 / 缩量下跌 / 正常


def analyze_volume(df: pd.DataFrame) -> VolumeResult:
    """
    对日线 DataFrame 执行量价分析。

    需要 columns: close, volume, pct_chg
    """
    result = VolumeResult()
    if df is None or df.empty:
        return result
    if "volume" not in df.columns or "close" not in df.columns:
        return result

    volume = df["volume"].astype(float)
    close = df["close"].astype(float)
    pct_chg = df["pct_chg"].astype(float) if "pct_chg" in df.columns else close.pct_change() * 100

    min_vol = len(volume)
    ma5_period = cfg.VOL_MA_PERIODS[0]  # 5
    ma20_period = cfg.VOL_MA_PERIODS[1]  # 20

    # ---------- 1. 量均线 ----------
    if min_vol >= ma5_period:
        result.vol_ma5 = round(float(volume.rolling(ma5_period).mean().iloc[-1]), 0)
    if min_vol >= ma20_period:
        result.vol_ma20 = round(float(volume.rolling(ma20_period).mean().iloc[-1]), 0)

    # ---------- 2. 量比 ----------
    if result.vol_ma5 is not None and result.vol_ma5 > 0:
        current_vol = float(volume.iloc[-1])
        result.volume_ratio = round(current_vol / result.vol_ma5, 2)
    else:
        current_vol = 0

    # ---------- 3. 量能状态 ----------
    if result.volume_ratio is not None:
        if result.volume_ratio >= 1.5:
            result.volume_status = "显著放量"
        elif result.volume_ratio >= 1.2:
            result.volume_status = "温和放量"
        elif result.volume_ratio <= 0.6:
            result.volume_status = "显著缩量"
        elif result.volume_ratio <= 0.8:
            result.volume_status = "温和缩量"
        else:
            result.volume_status = "正常"

    # ---------- 4. 量价配合 ----------
    if len(pct_chg) >= 2 and result.volume_ratio is not None:
        cur_pct = float(pct_chg.iloc[-1])
        is_up = cur_pct > 0.5          # 涨 >0.5%
        is_down = cur_pct < -0.5       # 跌 >0.5%
        is_high_vol = result.volume_ratio >= 1.3
        is_low_vol = result.volume_ratio <= 0.7

        if is_up and is_high_vol:
            result.price_volume_signal = "放量上涨 ✅"
        elif is_up and is_low_vol:
            result.price_volume_signal = "缩量上涨 ⚠️"
        elif is_down and is_high_vol:
            result.price_volume_signal = "放量下跌 ❌"
        elif is_down and is_low_vol:
            result.price_volume_signal = "缩量下跌 🔶"
        else:
            result.price_volume_signal = "量价配合正常"
    else:
        result.price_volume_signal = "数据不足"

    return result


if __name__ == "__main__":
    from data.fetch_price import fetch_daily_price
    df = fetch_daily_price("600519")
    r = analyze_volume(df)
    print(f"量MA5: {r.vol_ma5}, 量MA20: {r.vol_ma20}")
    print(f"量比: {r.volume_ratio}, 量能: {r.volume_status}")
    print(f"量价信号: {r.price_volume_signal}")
