#!/usr/bin/env python3
"""
可视化模块 —— 生成价格+KDJ图、PE/PB历史分位图
自动保存为 PNG，Windows 下适配中文字体确保不乱码。

包含：
  - 价格走势 + MA 均线 + KDJ 双面板图
  - 支撑/压力线（60日高低点）
  - MACD 金叉/死叉、KDJ 超卖反弹 买卖点标注（最近 30 根 K 线）
  - PE/PB 历史分布分位图
"""

import os
import sys
import platform
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from data.fetch_fundamental import fetch_pe_pb_history


# ==================== 中文字体适配 ====================

def _setup_chinese_font():
    """配置中文字体，优先适配 Windows，其次 Linux/macOS。"""
    if platform.system() == "Windows":
        font_candidates = [
            "Microsoft YaHei",
            "SimHei",
            "KaiTi",
            "FangSong",
            "STSong",
        ]
        for font_name in font_candidates:
            try:
                matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "Arial"]
                break
            except Exception:
                continue
        else:
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    else:
        plt.rcParams["font.sans-serif"] = [
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "SimHei",
            "DejaVu Sans",
        ]

    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


# ==================== 内部工具 ====================

def _calc_kdj_series(df: pd.DataFrame):
    """
    计算完整 KDJ 序列（用于绘图，复刻 momentum.py 中的算法）。

    返回 (k_series, d_series, j_series)，均为与 df 等长的 pd.Series。
    """
    n = cfg.KDJ_N
    m1 = cfg.KDJ_M1
    m2 = cfg.KDJ_M2

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()

    denominator = high_n - low_n
    denominator = denominator.replace(0, np.nan)
    rsv = (close - low_n) / denominator * 100

    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


def _calc_macd_series(df: pd.DataFrame):
    """
    计算完整 MACD 序列（用于绘图 + 买卖点检测）。

    返回 (dif_series, dea_series, hist_series)，均为与 df 等长的 pd.Series。
    """
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def _get_save_dir():
    """返回 PNG 文件默认保存目录（脚本所在目录）。"""
    return os.path.dirname(os.path.abspath(__file__))


# ==================== 图表 1：价格 + KDJ（含支撑/压力 + 买卖点）====================

def plot_price_kdj(
    df: pd.DataFrame,
    symbol: str,
    stock_name: str,
    trend_result=None,
    momentum_result=None,
    save_dir: Optional[str] = None,
) -> str:
    """
    生成「价格走势 + KDJ 指标」双面板图。

    上方：收盘价 + MA 均线 + 支撑/压力线 + 买卖点标注
    下方：KDJ (K / D / J) 指标，含超买(80) / 超卖(20) 参考线

    参数
    -----
    df : pd.DataFrame
        日线数据，含 date, close（必选），high/low（可选，用于 KDJ / 支撑压力 / 买卖点）
    symbol : str
        股票代码
    stock_name : str
        股票简称
    trend_result : TrendResult, optional
        趋势分析结果（MA 数值标注用）
    momentum_result : MomentumResult, optional
        动能结果（KDJ 数值标注用）
    save_dir : str, optional
        图片保存目录，默认为脚本所在目录

    返回
    -----
    str
        保存成功的 PNG 绝对路径
    """
    if save_dir is None:
        save_dir = _get_save_dir()

    filepath = os.path.join(save_dir, f"chart_{symbol}.png")

    if df is None or df.empty or "close" not in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"{symbol} {stock_name}\n行情数据不足，无法绘图",
                transform=ax.transAxes, ha="center", va="center", fontsize=16, color="gray")
        ax.set_title(f"{symbol} {stock_name} — 价格走势 & KDJ", fontsize=14)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return filepath

    df = df.copy()
    dates = pd.to_datetime(df["date"])
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    # ---------- 计算 MA ----------
    mas = {}
    for period in cfg.MA_PERIODS:
        if len(close) >= period:
            ma = close.rolling(window=period).mean()
            if not ma.isna().all():
                mas[period] = ma

    # ---------- 计算 KDJ ----------
    k_series, d_series, j_series = _calc_kdj_series(df)

    # ---------- 计算 MACD（用于买卖点）----------
    dif_series, dea_series, hist_series = _calc_macd_series(df)

    # ---------- 创建图形 ----------
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

    # ===== 上方面板：价格 + 均线 + 支撑/压力 + 买卖点 =====
    ax1 = fig.add_subplot(gs[0])

    # 收盘价
    ax1.plot(dates, close, color="#1a1a2e", linewidth=1.3, label="收盘价", zorder=3)

    # 均线
    ma_colors = {
        5: "#e74c3c",
        10: "#e67e22",
        20: "#2ecc71",
        60: "#3498db",
        250: "#9b59b6",
    }
    for period in cfg.MA_PERIODS:
        ma = mas.get(period)
        if ma is not None:
            color = ma_colors.get(period, "#888888")
            lw = 1.1 if period == 250 else 0.8
            alpha = 1.0 if period in (60, 250) else 0.65
            ax1.plot(dates, ma, color=color, linewidth=lw, alpha=alpha,
                     label=f"MA{period}", zorder=2)

    latest_price = float(close.iloc[-1])

    # ---------- 🔧 支撑线 / 压力线（60日高低点）----------
    lookback_sr = min(60, len(df))
    recent_60_low = float(low.iloc[-lookback_sr:].min())
    recent_60_high = float(high.iloc[-lookback_sr:].max())

    # 支撑线（绿色虚线）
    ax1.axhline(y=recent_60_low, color="#27ae60", linestyle="--", linewidth=1.3, alpha=0.65)
    ax1.text(
        dates.iloc[-1], recent_60_low,
        f" 支撑 {recent_60_low:.2f}",
        fontsize=8.5, color="#27ae60", fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # 压力线（红色虚线）
    ax1.axhline(y=recent_60_high, color="#e74c3c", linestyle="--", linewidth=1.3, alpha=0.65)
    ax1.text(
        dates.iloc[-1], recent_60_high,
        f" 压力 {recent_60_high:.2f}",
        fontsize=8.5, color="#e74c3c", fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # 当前价格水平线
    ax1.axhline(y=latest_price, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)

    # 价格标注
    ax1.annotate(
        f" {latest_price:.2f}",
        xy=(dates.iloc[-1], latest_price),
        xytext=(6, 0), textcoords="offset points",
        fontsize=9, color="#1a1a2e", fontweight="bold", va="center",
    )

    # ---------- 🔧 买卖点标注（最近 30 根 K 线）----------
    n_markers = min(30, len(df))
    idx_start = len(df) - n_markers

    dates_30 = dates.iloc[idx_start:]
    close_30 = close.iloc[idx_start:]
    high_30 = high.iloc[idx_start:]
    low_30 = low.iloc[idx_start:]
    dif_30 = dif_series.iloc[idx_start:]
    dea_30 = dea_series.iloc[idx_start:]
    j_30 = j_series.iloc[idx_start:]

    # 价格波动幅度（用于合理放置标记偏移）
    price_range = high_30.max() - low_30.min()
    if price_range <= 0:
        price_range = close_30.mean() * 0.05

    # --- MACD 金叉 / 死叉 ---
    for i in range(1, len(dif_30)):
        di = idx_start + i                     # 原始索引
        prev_dif = dif_30.iloc[i - 1]
        prev_dea = dea_30.iloc[i - 1]
        cur_dif = dif_30.iloc[i]
        cur_dea = dea_30.iloc[i]

        if pd.isna(prev_dif) or pd.isna(prev_dea) or pd.isna(cur_dif) or pd.isna(cur_dea):
            continue

        # 金叉：DIF 上穿 DEA
        if prev_dif <= prev_dea and cur_dif > cur_dea:
            y_pos = float(low_30.iloc[i]) - price_range * 0.06
            ax1.scatter(dates.iloc[di], y_pos, marker="^", color="#27ae60",
                        s=90, zorder=10, edgecolors="white", linewidths=0.5)
            ax1.annotate(
                "BUY", (dates.iloc[di], y_pos),
                xytext=(0, -10), textcoords="offset points",
                fontsize=7.5, color="#27ae60", fontweight="bold", ha="center",
            )

        # 死叉：DIF 下穿 DEA
        if prev_dif >= prev_dea and cur_dif < cur_dea:
            y_pos = float(high_30.iloc[i]) + price_range * 0.06
            ax1.scatter(dates.iloc[di], y_pos, marker="v", color="#e74c3c",
                        s=90, zorder=10, edgecolors="white", linewidths=0.5)
            ax1.annotate(
                "SELL", (dates.iloc[di], y_pos),
                xytext=(0, 10), textcoords="offset points",
                fontsize=7.5, color="#e74c3c", fontweight="bold", ha="center",
            )

    # --- KDJ J<20 且拐头向上（超卖反弹信号）---
    for i in range(2, len(j_30)):
        di = idx_start + i                     # 原始索引
        j_val = j_30.iloc[i]
        j_prev = j_30.iloc[i - 1]
        j_prev2 = j_30.iloc[i - 2]

        if pd.isna(j_val) or pd.isna(j_prev):
            continue

        # J 值在 20 以下，且从下降转为上升（V 型拐头）
        turning_up = j_val < 20 and j_prev < j_val
        # 也检测：J 从很低开始连续反弹
        if not turning_up and not pd.isna(j_prev2):
            turning_up = (
                j_val < 20
                and j_prev2 > j_prev
                and j_prev < j_val
            )

        if turning_up:
            y_pos = float(low_30.iloc[i]) - price_range * 0.06
            ax1.scatter(dates.iloc[di], y_pos, marker="D", color="#2980b9",
                        s=65, zorder=10, edgecolors="white", linewidths=0.5)
            ax1.annotate(
                "超卖反弹", (dates.iloc[di], y_pos),
                xytext=(0, -10), textcoords="offset points",
                fontsize=6.5, color="#2980b9", fontweight="bold", ha="center",
            )

    # 格式
    title = f"{symbol} {stock_name}  —  价格走势"
    ax1.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax1.set_ylabel("价格（元）", fontsize=11)
    # 图例合并（排除买卖点标记）
    ax1.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.75)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.tick_params(axis="x", labelbottom=False)

    # ===== 下方面板：KDJ =====
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax2.plot(dates, k_series, color="#e74c3c", linewidth=0.9, label="K", alpha=0.92)
    ax2.plot(dates, d_series, color="#3498db", linewidth=0.9, label="D", alpha=0.92)
    ax2.plot(dates, j_series, color="#9b59b6", linewidth=0.7, label="J", alpha=0.75)

    # 超买 / 超卖区域
    ax2.axhline(y=80, color="red", linestyle=":", linewidth=0.8, alpha=0.35)
    ax2.axhline(y=20, color="green", linestyle=":", linewidth=0.8, alpha=0.35)
    ax2.fill_between(dates, 80, 120, alpha=0.04, color="red")
    ax2.fill_between(dates, -20, 20, alpha=0.04, color="green")

    # 标注 80/20 文字
    ax2.text(dates.iloc[-1], 80, " 超买线 80", fontsize=7, color="red", alpha=0.55, va="bottom")
    ax2.text(dates.iloc[-1], 20, " 超卖线 20", fontsize=7, color="green", alpha=0.55, va="top")

    # 最新 KDJ 数值标注
    if not k_series.isna().all():
        last_k = float(k_series.iloc[-1])
        last_d = float(d_series.iloc[-1])
        last_j = float(j_series.iloc[-1])
        if not np.isnan(last_k):
            ax2.annotate(f"K={last_k:.1f}", xy=(dates.iloc[-1], last_k),
                         fontsize=7, color="#e74c3c", fontweight="bold",
                         xytext=(12, -4), textcoords="offset points")
        if not np.isnan(last_d):
            ax2.annotate(f"D={last_d:.1f}", xy=(dates.iloc[-1], last_d),
                         fontsize=7, color="#3498db", fontweight="bold",
                         xytext=(12, -4), textcoords="offset points")
        if not np.isnan(last_j):
            ax2.annotate(f"J={last_j:.1f}", xy=(dates.iloc[-1], last_j),
                         fontsize=7, color="#9b59b6", fontweight="bold",
                         xytext=(12, 6), textcoords="offset points")

    ax2.set_ylabel("KDJ", fontsize=11)
    ax2.set_ylim(-20, 120)
    ax2.set_yticks([0, 20, 50, 80, 100])
    ax2.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.75)
    ax2.grid(True, alpha=0.25, linestyle="--")

    # 日期格式化
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")

    # 紧缩 layout → 保存
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return filepath


# ==================== 图表 2：PE / PB 历史分位 ====================

def plot_valuation_percentile(
    symbol: str,
    stock_name: str,
    valuation_result=None,
    save_dir: Optional[str] = None,
) -> str:
    """
    生成「PE / PB 历史分位图」。

    左侧：PE-TTM 历史分布直方图 + 当前位置标记
    右侧：PB 历史分布直方图 + 当前位置标记

    参数
    -----
    symbol : str
        股票代码
    stock_name : str
        股票简称
    valuation_result : ValuationResult, optional
        估值分析结果（pe_ttm, pb, pe_percentile, pb_percentile, valuation_range）
    save_dir : str, optional
        图片保存目录，默认为脚本所在目录

    返回
    -----
    str
        保存成功的 PNG 绝对路径
    """
    if save_dir is None:
        save_dir = _get_save_dir()

    filepath = os.path.join(save_dir, f"valuation_{symbol}.png")

    # ---------- 获取历史分位数据 ----------
    hist = fetch_pe_pb_history(symbol, years=cfg.VALUATION_HISTORY_YEARS)
    pe_list = hist.get("pe_ttm_list", [])
    pb_list = hist.get("pb_list", [])

    cur_pe = valuation_result.pe_ttm if valuation_result else None
    cur_pb = valuation_result.pb if valuation_result else None
    pe_pct = valuation_result.pe_percentile if valuation_result else None
    pb_pct = valuation_result.pb_percentile if valuation_result else None
    val_range = valuation_result.valuation_range if valuation_result else "N/A"

    # ---------- 创建图形 ----------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    _plot_single_percentile(
        ax=axes[0],
        history_values=pe_list,
        current_value=cur_pe,
        percentile=pe_pct,
        label="PE-TTM",
        color="#3498db",
    )

    _plot_single_percentile(
        ax=axes[1],
        history_values=pb_list,
        current_value=cur_pb,
        percentile=pb_pct,
        label="PB",
        color="#e67e22",
    )

    # 总标题
    fig.suptitle(
        f"{symbol} {stock_name}  —  估值历史分位（近{cfg.VALUATION_HISTORY_YEARS}年）  |  综合判定：{val_range}",
        fontsize=15, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return filepath


def _plot_single_percentile(
    ax,
    history_values: list,
    current_value: Optional[float],
    percentile: Optional[float],
    label: str,
    color: str,
):
    """
    在单个 Axes 上绘制估值指标的历史分布直方图 + 当前位置标记。

    - 直方图展示历史分布
    - 红色竖线 + 箭头标记当前值
    - 统计信息框（均值 / 中位数 / P25 / P75）
    - 估值区间颜色标识
    """
    # ---- 无数据兜底 ----
    if not history_values:
        ax.text(0.5, 0.5, "暂无历史数据", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title(label, fontsize=13, fontweight="bold")
        return

    values = np.array(history_values, dtype=np.float64)
    # 去掉极端异常值（负值 / inf）
    values = values[(~np.isnan(values)) & (~np.isinf(values)) & (values > 0)]

    if len(values) == 0:
        ax.text(0.5, 0.5, "暂无有效数据", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title(label, fontsize=13, fontweight="bold")
        return

    # ---- 直方图 ----
    # bins 数量自适应
    n_bins = max(8, min(35, len(values) // 3))
    n, bins_edge, patches = ax.hist(
        values, bins=n_bins, color=color, alpha=0.55, edgecolor="white",
        linewidth=0.6, label="历史分布", density=False,
    )

    # ---- 当前值标记 ----
    if current_value is not None and not np.isnan(current_value) and not np.isinf(current_value) and current_value > 0:
        ax.axvline(x=current_value, color="#e74c3c", linewidth=2.8, linestyle="-",
                   label=f"当前{label}={current_value:.2f}", alpha=0.9)

        ymax = ax.get_ylim()[1]
        pct_text = f"分位: {percentile:.1f}%" if percentile is not None else ""

        # 根据分位决定标注方向
        ha = "left"
        offset_sign = 1
        if percentile is not None and percentile > 50:
            ha = "right"
            offset_sign = -1

        ax.annotate(
            f"← {label}={current_value:.2f}  {pct_text}",
            xy=(current_value, ymax * 0.92),
            xytext=(30 * offset_sign, 0), textcoords="offset points",
            fontsize=10, color="#e74c3c", fontweight="bold", ha=ha,
        )

    # ---- 统计信息 ----
    mean_val = np.mean(values)
    median_val = np.median(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    vmin, vmax = np.min(values), np.max(values)

    # P25 / P50 / P75 虚线
    for pv, pname, ls in [
        (p25, "P25", (0, (3, 5))),
        (median_val, "P50", (0, (5, 3))),
        (p75, "P75", (0, (3, 5))),
    ]:
        ax.axvline(x=pv, color="gray", linewidth=0.9, linestyle=ls, alpha=0.5)
        ax.annotate(f"{pname}={pv:.2f}", xy=(pv, ax.get_ylim()[1] * 0.15),
                    fontsize=7, color="gray", ha="center", rotation=90, alpha=0.7)

    stats_text = (
        f"样本数: {len(values)}\n"
        f"均值: {mean_val:.2f}\n"
        f"中位数: {median_val:.2f}\n"
        f"P25: {p25:.2f}\n"
        f"P75: {p75:.2f}\n"
        f"范围: [{vmin:.2f}, {vmax:.2f}]"
    )
    ax.text(
        0.96, 0.95, stats_text, transform=ax.transAxes,
        fontsize=8, va="top", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85,
                  edgecolor="gray", linewidth=0.7),
    )

    # ---- 估值区间色标 ----
    zone_color_map = {
        (0, 20): ("低估区间", "#27ae60"),
        (20, 40): ("合理偏低", "#2ecc71"),
        (40, 60): ("合理区间", "#f39c12"),
        (60, 80): ("偏高区间", "#e67e22"),
        (80, 101): ("泡沫区间", "#e74c3c"),
    }
    if percentile is not None:
        for (lo, hi), (zone_label, zone_color) in zone_color_map.items():
            if lo <= percentile < hi:
                ax.text(0.03, 0.95, f"● {zone_label}", transform=ax.transAxes,
                        fontsize=12, color=zone_color, fontweight="bold", va="top")
                break

    # ---- 轴标签 ----
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("出现频次", fontsize=11)
    ax.set_title(f"{label} 历史分位分布", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.75)
    ax.grid(True, alpha=0.2, axis="y")
