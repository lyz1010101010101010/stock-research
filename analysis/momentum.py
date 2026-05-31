"""
动能分析 —— MACD（含顶/底背离）、RSI、KDJ
"""

import sys
import os
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class MomentumResult:
    """动能指标综合结果"""

    def __init__(self):
        # MACD
        self.macd_dif: Optional[float] = None
        self.macd_dea: Optional[float] = None
        self.macd_hist: Optional[float] = None
        self.macd_signal: str = "数据不足"      # 金叉 / 死叉 / 动能增强 / 动能减弱 / 数据不足
        self.macd_divergence: str = "未检测"     # 顶背离 / 底背离 / 未检测

        # RSI
        self.rsi: Optional[float] = None
        self.rsi_signal: str = "中性"           # 超买 / 超卖 / 中性 / 数据不足

        # KDJ
        self.kdj_k: Optional[float] = None
        self.kdj_d: Optional[float] = None
        self.kdj_j: Optional[float] = None
        self.kdj_signal: str = "中性"


def _calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def analyze_momentum(df: pd.DataFrame) -> MomentumResult:
    """
    对日线 DataFrame 执行 MACD / RSI / KDJ 分析。

    DataFrame 必须含 columns: close(收盘价) 和 high/low（用于 KDJ）。
    """
    result = MomentumResult()
    if df is None or df.empty:
        return result

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    # ========== 1. MACD ==========
    result.macd_signal, result.macd_divergence = _analyze_macd(close, result)

    # ========== 2. RSI ==========
    result.rsi, result.rsi_signal = _analyze_rsi(close)

    # ========== 3. KDJ ==========
    result.kdj_k, result.kdj_d, result.kdj_j, result.kdj_signal = _analyze_kdj(
        high, low, close
    )

    return result


# ----- MACD -----

def _analyze_macd(close: pd.Series, result: MomentumResult):
    """MACD 计算 + 金叉/死叉 + 简单背离检测。"""
    n_fast = cfg.MACD_FAST
    n_slow = cfg.MACD_SLOW
    n_signal = cfg.MACD_SIGNAL

    if len(close) < n_slow + n_signal:
        return "数据不足", "数据不足"

    ema_fast = _calc_ema(close, n_fast)
    ema_slow = _calc_ema(close, n_slow)
    dif = ema_fast - ema_slow
    dea = _calc_ema(dif, n_signal)
    hist = dif - dea

    result.macd_dif = round(float(dif.iloc[-1]), 4)
    result.macd_dea = round(float(dea.iloc[-1]), 4)
    result.macd_hist = round(float(hist.iloc[-1]), 4)

    # 判断金叉/死叉
    signal = "中性"
    if len(dif) >= 2:
        prev_dif = dif.iloc[-2]
        prev_dea = dea.iloc[-2]
        # 金叉：DIF 上穿 DEA
        if prev_dif <= prev_dea and dif.iloc[-1] > dea.iloc[-1]:
            signal = "金叉"
        # 死叉：DIF 下穿 DEA
        elif prev_dif >= prev_dea and dif.iloc[-1] < dea.iloc[-1]:
            signal = "死叉"
        # 动能增强：hist 变长
        elif len(hist) >= 2:
            prev_hist = hist.iloc[-2]
            if abs(hist.iloc[-1]) > abs(prev_hist):
                signal = "动能增强"
            else:
                signal = "动能减弱"

    # 简单顶/底背离检测
    divergence = _detect_divergence(close, dif)
    return signal, divergence


def _detect_divergence(close: pd.Series, dif: pd.Series, lookback: int = 60) -> str:
    """
    简单版顶背离/底背离检测。

    顶背离：价格创 N 日新高，但 DIF 未创新高
    底背离：价格创 N 日新低，但 DIF 未创新低
    """
    if len(close) < lookback or len(dif) < lookback:
        return "数据不足"

    recent_close = close.iloc[-lookback:]
    recent_dif = dif.iloc[-lookback:]

    # 当前值
    cur_close = recent_close.iloc[-1]
    cur_dif = recent_dif.iloc[-1]

    # 前 lookback 日内最高/最低（不含当前）
    before_close = recent_close.iloc[:-1]
    before_dif = recent_dif.iloc[:-1]

    # 顶背离：当前价格接近前高（>= 前高的 98%），但 DIF 明显低于前高
    high_close = before_close.max()
    high_dif = before_dif.max()
    if cur_close >= high_close * 0.98 and cur_dif < high_dif * 0.95:
        return "顶背离"

    # 底背离：当前价格接近前低（<= 前低的 102%），但 DIF 明显高于前低
    low_close = before_close.min()
    low_dif = before_dif.min()
    if cur_close <= low_close * 1.02 and cur_dif > low_dif * 1.05:
        return "底背离"

    return "未检测"


# ----- RSI -----

def _analyze_rsi(close: pd.Series, period: int = None) -> tuple:
    """计算 RSI(14) 并给出超买/超卖判断。"""
    if period is None:
        period = cfg.RSI_PERIOD
    if len(close) < period + 1:
        return None, "数据不足"

    # 转为 numpy 数组，避免 pandas index 对齐问题
    close_vals = close.values.astype(np.float64)
    n = len(close_vals)

    delta = np.diff(close_vals, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # 用 list 存结果，从 period+1 开始（跳过首元素 diff NaN 的影响）
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # 第一个有效 avg 在 period+1 位置（第 period 个有效 delta 之后）
    # 因为 diff[0] = NaN，有效 diff 从 index 1 开始
    start_idx = period + 1
    if n <= start_idx:
        return None, "数据不足"

    avg_gain[start_idx] = np.mean(gain[1:start_idx + 1])   # gain[1]~gain[start_idx]
    avg_loss[start_idx] = np.mean(loss[1:start_idx + 1])

    # Wilder 平滑
    for i in range(start_idx + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    last_gain = avg_gain[-1]
    last_loss = avg_loss[-1]
    if np.isnan(last_gain) or np.isnan(last_loss):
        return None, "数据不足"

    if last_loss == 0:
        current_rsi = 100.0
    else:
        rs = last_gain / last_loss
        current_rsi = float(100.0 - (100.0 / (1.0 + rs)))

    if np.isnan(current_rsi):
        return None, "数据不足"

    current_rsi = round(current_rsi, 1)

    if current_rsi > 80:
        signal = "严重超买"
    elif current_rsi > 70:
        signal = "超买"
    elif current_rsi < 20:
        signal = "严重超卖"
    elif current_rsi < 30:
        signal = "超卖"
    else:
        signal = "中性"

    return current_rsi, signal


# ----- KDJ -----

def _analyze_kdj(
    high: pd.Series, low: pd.Series, close: pd.Series,
) -> tuple:
    """计算 KDJ(9,3,3) 并给出信号。"""
    n = cfg.KDJ_N
    m1 = cfg.KDJ_M1
    m2 = cfg.KDJ_M2

    if len(close) < n + m1 + m2:
        return None, None, None, "数据不足"

    # RSV
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100

    # K、D、J
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d

    current_k = float(k.iloc[-1])
    current_d = float(d.iloc[-1])
    current_j = float(j.iloc[-1])

    # 简单信号
    signal = "中性"
    if current_k > 80 and current_d > 80 and current_j > 100:
        signal = "高位钝化，警惕回调（J值严重偏高）"
    elif current_k > 80 and current_d > 80:
        signal = "高位钝化，警惕回调"
    elif current_k < 20 and current_d < 20 and current_j < 0:
        signal = "低位超卖（J值严重偏低）"
    elif current_k < 20 and current_d < 20:
        signal = "低位超卖"
    elif current_j < 0:
        signal = "J值进入负值区间，超跌信号"
    elif current_j > 100:
        signal = "J值进入超涨区间，注意回调"
    elif current_k > current_d and k.iloc[-2] <= d.iloc[-2]:
        signal = "K线金叉"
    elif current_k < current_d and k.iloc[-2] >= d.iloc[-2]:
        signal = "K线死叉"
    elif current_k > current_d:
        signal = "短线偏多"
    elif current_k < current_d:
        signal = "短线偏空"

    return (
        round(current_k, 2),
        round(current_d, 2),
        round(current_j, 2),
        signal,
    )


if __name__ == "__main__":
    from data.fetch_price import fetch_daily_price
    df = fetch_daily_price("600519")
    r = analyze_momentum(df)
    print(f"MACD: DIF={r.macd_dif}, DEA={r.macd_dea}, HIST={r.macd_hist}")
    print(f"MACD信号: {r.macd_signal}, 背离: {r.macd_divergence}")
    print(f"RSI: {r.rsi} ({r.rsi_signal})")
    print(f"KDJ: K={r.kdj_k}, D={r.kdj_d}, J={r.kdj_j} -> {r.kdj_signal}")
