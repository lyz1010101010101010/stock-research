#!/usr/bin/env python3
"""
低估 + 趋势向上 筛选器

条件：
  - PE 历史分位 ≤ 30%
  - PB 历史分位 ≤ 40%
  - 最新收盘价 > MA60（站上季线）

输出筛选结果到 screen_result.txt。
"""

import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from data.fetch_price import fetch_daily_price
from data.fetch_fundamental import fetch_pe_pb_history, fetch_stock_name


def screen_stocks(
    stock_codes: Optional[list[str]] = None,
    pe_threshold: float = 30.0,
    pb_threshold: float = 40.0,
) -> list[dict]:
    """
    对股票池执行低估 + 趋势向上筛选。

    参数
    -----
    stock_codes : list[str], optional
        待筛选的股票代码列表。None 时使用 config.STOCK_LIST。
    pe_threshold : float
        PE 分位阈值（≤ 此值判定为低估），默认 30。
    pb_threshold : float
        PB 分位阈值（≤ 此值判定为低估），默认 40。

    返回
    -----
    list[dict]
        每项包含 symbol, name, close, pe_pct, pb_pct, ma60, ma250,
        above_ma60, above_ma250, trend_brief, valuation_range, market_cap。
    """
    # ---------- 确定股票列表 ----------
    if stock_codes:
        stock_list = [(code, None) for code in stock_codes]
    else:
        stock_list = cfg.STOCK_LIST

    if not stock_list:
        print("❌ 股票池为空，请在 config.py 中配置 STOCK_LIST 或通过命令行传入股票代码。")
        return []

    print(f"\n{'='*55}")
    print(f"  🔍 低估 + 趋势向上 筛选器")
    print(f"  条件: PE分位 ≤ {pe_threshold}%  |  PB分位 ≤ {pb_threshold}%  |  收盘 > MA60")
    print(f"  股票池: {len(stock_list)} 只")
    print(f"{'='*55}\n")

    results = []

    for i, (symbol, name) in enumerate(stock_list, 1):
        print(f"  [{i}/{len(stock_list)}] 检查 {symbol} ...", end=" ", flush=True)

        # ---- 1. 获取名称 ----
        if not name:
            try:
                name = fetch_stock_name(symbol)
            except Exception:
                name = symbol

        # ---- 2. 获取价格数据 ----
        df = fetch_daily_price(symbol)
        if df is None or df.empty:
            print("❌ 行情获取失败")
            continue

        close = df["close"].astype(float)
        latest_close = float(close.iloc[-1])

        # ---- 3. 计算 MA60 / MA250 ----
        if len(close) < 60:
            print("⚠️ 数据不足（<60日）")
            continue

        ma60 = float(close.rolling(window=60).mean().iloc[-1])
        above_ma60 = latest_close > ma60

        ma250 = None
        above_ma250 = None
        if len(close) >= 250:
            ma250 = float(close.rolling(window=250).mean().iloc[-1])
            above_ma250 = latest_close > ma250

        # ---- 4. 获取 PE / PB 分位 ----
        try:
            hist = fetch_pe_pb_history(symbol, years=cfg.VALUATION_HISTORY_YEARS)
            pe_pct = hist.get("pe_ttm_percentile")
            pb_pct = hist.get("pb_percentile")
        except Exception:
            pe_pct = None
            pb_pct = None

        if pe_pct is None or pb_pct is None:
            print("⚠️ 分位数据缺失")
            continue

        # ---- 5. 判断条件 ----
        pe_ok = pe_pct <= pe_threshold
        pb_ok = pb_pct <= pb_threshold

        if not pe_ok:
            print(f"❌ PE分位={pe_pct:.1f}%（>{pe_threshold}%）")
            continue
        if not pb_ok:
            print(f"❌ PB分位={pb_pct:.1f}%（>{pb_threshold}%）")
            continue
        if not above_ma60:
            print(f"❌ 收盘{latest_close:.2f} ≤ MA60({ma60:.2f})")
            continue

        # ---- 6. 判定趋势简述 ----
        trend_brief = _describe_trend(df, close, latest_close, ma60, ma250, above_ma250)

        # ---- 7. 估值区间 ----
        # 用分位快速判定
        avg_pct = (pe_pct + pb_pct) / 2
        if avg_pct <= 20:
            val_range = "低估"
        elif avg_pct <= 40:
            val_range = "合理偏低"
        elif avg_pct <= 60:
            val_range = "合理"
        elif avg_pct <= 80:
            val_range = "偏高"
        else:
            val_range = "泡沫"

        # 总市值（可选）
        market_cap = _get_market_cap(symbol, latest_close)

        result = {
            "symbol": symbol,
            "name": name,
            "close": latest_close,
            "pe_pct": pe_pct,
            "pb_pct": pb_pct,
            "ma60": round(ma60, 2),
            "ma250": round(ma250, 2) if ma250 else None,
            "above_ma60": above_ma60,
            "above_ma250": above_ma250,
            "trend_brief": trend_brief,
            "valuation_range": val_range,
            "market_cap": market_cap,
        }
        results.append(result)

        print(f"✅ 通过! PE分位={pe_pct:.1f}% PB分位={pb_pct:.1f}% {trend_brief}")

    return results


def _describe_trend(
    df: pd.DataFrame,
    close: pd.Series,
    latest_close: float,
    ma60: float,
    ma250: Optional[float],
    above_ma250: Optional[bool],
) -> str:
    """生成趋势简述文字。"""
    parts = []

    # 均线排列
    mas = {}
    for p in [5, 10, 20, 60]:
        if len(close) >= p:
            mas[p] = float(close.rolling(window=p).mean().iloc[-1])

    if all(k in mas for k in [5, 10, 20, 60]):
        if mas[5] > mas[10] > mas[20] > mas[60]:
            parts.append("多头排列")
        elif mas[5] < mas[10] < mas[20] < mas[60]:
            parts.append("空头排列")
        elif latest_close > mas[60]:
            parts.append("震荡偏多")
        elif latest_close < mas[60]:
            parts.append("震荡偏空")
        else:
            parts.append("震荡")

    # 站上年线？
    if above_ma250 is True:
        parts.append("站上年线")
    elif above_ma250 is False and ma250 is not None:
        parts.append("年线下方")

    # MA60 位置关系
    if latest_close > ma60:
        parts.append("站上季线")
    else:
        parts.append("季线下方")

    return "，".join(parts) if parts else "数据不足"


def _get_market_cap(symbol: str, latest_price: float) -> Optional[float]:
    """尝试获取总市值（亿）。"""
    try:
        from data.fetch_fundamental import _fetch_total_shares
        shares = _fetch_total_shares(symbol)
        if shares:
            return round((latest_price * shares) / 1e8, 2)
    except Exception:
        pass
    return None


def format_screen_report(results: list[dict]) -> str:
    """将筛选结果列表格式化为可读文本报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 70

    lines = []
    lines.append("")
    lines.append(sep)
    lines.append(f"  低估 + 趋势向上 筛选结果    时间: {now}")
    lines.append(sep)
    lines.append("")
    lines.append(f"  筛选条件: PE分位 ≤ 30%  |  PB分位 ≤ 40%  |  收盘 > MA60")
    lines.append(f"  通过数量: {len(results)}")
    lines.append("")

    if not results:
        lines.append("  ⚠️ 无股票通过筛选，建议放宽条件或等待市场调整。")
        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    # 表头
    header = (
        f"  {'代码':<8s} {'名称':<10s} {'收盘':>8s} {'PE分位':>8s} {'PB分位':>8s} "
        f"{'市值(亿)':>10s} {'趋势简述'}"
    )
    line_sep = "  " + "-" * 100

    lines.append(header)
    lines.append(line_sep)

    for r in results:
        mc_str = f"{r['market_cap']:.0f}" if r["market_cap"] else "N/A"
        row = (
            f"  {r['symbol']:<8s} {r['name']:<10s} {r['close']:>8.2f} "
            f"{r['pe_pct']:>7.1f}% {r['pb_pct']:>7.1f}% "
            f"{mc_str:>10s} {r['trend_brief']}"
        )
        lines.append(row)

    lines.append(line_sep)
    lines.append("")

    # 逐个明细
    for idx, r in enumerate(results, 1):
        lines.append(f"  ┌─ #{idx}  [{r['symbol']}] {r['name']} ─────────────────────────────")
        lines.append(f"  │  最新价: {r['close']:.2f}")
        lines.append(f"  │  PE-TTM 分位: {r['pe_pct']:.1f}%     PB 分位: {r['pb_pct']:.1f}%")
        lines.append(f"  │  估值区间: {r['valuation_range']}")
        lines.append(f"  │  MA60: {r['ma60']:.2f}     {'站上 ✅' if r['above_ma60'] else '下方 ❌'}")
        if r["ma250"] is not None:
            yn = "站上 ✅" if r["above_ma250"] else "下方 ❌"
            lines.append(f"  │  MA250: {r['ma250']:.2f}    {yn}")
        if r["market_cap"]:
            lines.append(f"  │  总市值: {r['market_cap']:.1f} 亿")
        lines.append(f"  │  趋势: {r['trend_brief']}")
        lines.append(f"  └{'─'*55}")

    lines.append("")
    lines.append(f" ⚠️ 本结果仅供参考，不构成投资建议")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


def run_screen(stock_codes: Optional[list[str]] = None):
    """执行筛选并保存结果到 screen_result.txt。"""
    results = screen_stocks(stock_codes)

    # 输出摘要
    print(f"\n{'─'*55}")
    if results:
        print(f"  ✅ 共 {len(results)} 只股票通过筛选：")
        for r in results:
            print(f"     [{r['symbol']}] {r['name']}  "
                  f"PE分位={r['pe_pct']:.1f}%  PB分位={r['pb_pct']:.1f}%  "
                  f"{r['trend_brief']}")
    else:
        print(f"  ⚠️ 无股票通过筛选")
    print(f"{'─'*55}")

    # 生成报告
    report = format_screen_report(results)

    # 保存
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_result.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [保存] 筛选报告已写入 {save_path}")

    return results


if __name__ == "__main__":
    run_screen()
