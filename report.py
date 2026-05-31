"""
报告生成 —— 将各分析模块的结果汇总为格式化的中文研判报告
"""

import sys
import os
from typing import Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from analysis.valuation import ValuationResult
from analysis.trend import TrendResult
from analysis.momentum import MomentumResult
from analysis.volume import VolumeResult
from analysis.structure import StructureResult


def _val_str(v, suffix="") -> str:
    """安全格式化可选数值"""
    if v is None:
        return "N/A"
    return f"{v}{suffix}"


def _pct(v) -> str:
    """百分比格式化"""
    if v is None:
        return "N/A"
    return f"{v:.1f}%"


def _rjust(s, width=8) -> str:
    return str(s).rjust(width)


def generate_report(
    symbol: str,
    stock_name: str,
    latest_price: Optional[float],
    valuation: ValuationResult,
    trend: TrendResult,
    momentum: MomentumResult,
    volume: VolumeResult,
    structure: StructureResult,
) -> str:
    """
    汇总生成完整的文字报告。

    返回格式化的字符串，可直接 print 到控制台。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "─" * 50

    lines = []
    lines.append("")
    lines.append(sep)
    lines.append(f" [{symbol}] {stock_name}    报告时间: {now}")
    lines.append(sep)

    # ========================= 估值 =========================
    lines.append("")
    lines.append(" 📊 估值")

    pe_str = _val_str(valuation.pe_ttm)
    pe_pct_str = _pct(valuation.pe_percentile) if valuation.pe_percentile is not None else "N/A"
    pb_str = _val_str(valuation.pb)
    pb_pct_str = _pct(valuation.pb_percentile) if valuation.pb_percentile is not None else "N/A"
    roe_str = _pct(valuation.roe_avg) if valuation.roe_avg is not None else "N/A"

    lines.append(f"   PE-TTM：{pe_str}（分位 {pe_pct_str}）")
    lines.append(f"   PB：{pb_str}（分位 {pb_pct_str}）")
    lines.append(f"   ROE(近3年均)：{roe_str}")
    lines.append(f"   估值区间：{valuation.valuation_range}")

    # 如果 PE / PB 分位差距过大，加提示
    if valuation.pe_percentile is not None and valuation.pb_percentile is not None:
        spread = abs(valuation.pe_percentile - valuation.pb_percentile)
        if spread > 50:
            pe_higher = valuation.pe_percentile > valuation.pb_percentile
            if pe_higher:
                lines.append(f"   ⚠️ PE分位({valuation.pe_percentile:.0f}%)远高于PB分位({valuation.pb_percentile:.0f}%)，盈利能力或有波动")
            else:
                lines.append(f"   ⚠️ PB分位({valuation.pb_percentile:.0f}%)远高于PE分位({valuation.pe_percentile:.0f}%)，净资产估值偏贵")
        elif spread > 30:
            lines.append(f"   ℹ️ PE分位({valuation.pe_percentile:.0f}%)与PB分位({valuation.pb_percentile:.0f}%)差距较大，建议结合行业特征判断")

    if valuation.market_cap:
        lines.append(f"   总市值：{valuation.market_cap:.1f} 亿")

    # ========================= 技术面 =========================
    lines.append("")
    lines.append(" 📈 技术面")

    # --- 趋势 ---
    trend_icon = {"多头排列": "✅", "震荡偏多": "🟢", "震荡": "🔶", "震荡偏空": "🟡", "空头排列": "❌", "数据不足": "⚪"}
    icon = trend_icon.get(trend.trend_direction, "⚪")
    above_line = "站上年线" if trend.above_ma250 else "年线下方"
    trend_summary = f"{trend.trend_direction}，{above_line}"
    lines.append(f"   趋势：{icon} {trend_summary}")

    # MA 明细
    ma_parts = []
    for p in cfg.MA_PERIODS:
        v = trend.mas.get(p)
        if v is not None:
            ma_parts.append(f"MA{p}={v:.2f}")
    lines.append(f"   MA：{', '.join(ma_parts)}")

    # --- MACD ---
    macd_info = []
    if momentum.macd_dif is not None:
        macd_info.append(f"DIF={momentum.macd_dif:.2f}")
    if momentum.macd_dea is not None:
        macd_info.append(f"DEA={momentum.macd_dea:.2f}")
    if momentum.macd_hist is not None:
        macd_info.append(f"HIST={momentum.macd_hist:.2f}")
    macd_summary = f"MACD：{', '.join(macd_info)} —— {momentum.macd_signal}"
    if momentum.macd_divergence and momentum.macd_divergence not in ("未检测", "数据不足"):
        macd_summary += f" ⚠️ {momentum.macd_divergence}"
    lines.append(f"   {macd_summary}")

    # --- RSI ---
    rsi_str = _val_str(momentum.rsi)
    rsi_icon = "⚠️" if "超买" in (momentum.rsi_signal or "") else ("✅" if "超卖" in (momentum.rsi_signal or "") else "●")
    lines.append(f"   RSI(14)：{_rjust(rsi_str)} {rsi_icon} {momentum.rsi_signal}")

    # --- KDJ ---
    kdj_parts = []
    if momentum.kdj_k is not None: kdj_parts.append(f"K={momentum.kdj_k:.2f}")
    if momentum.kdj_d is not None: kdj_parts.append(f"D={momentum.kdj_d:.2f}")
    if momentum.kdj_j is not None: kdj_parts.append(f"J={momentum.kdj_j:.2f}")
    kdj_summary = f"KDJ：{', '.join(kdj_parts)} —— {momentum.kdj_signal}"
    lines.append(f"   {kdj_summary}")

    # --- 量价 ---
    vol_parts = []
    if volume.vol_ma5 is not None:
        vol_parts.append(f"VOL_MA5={volume.vol_ma5:.0f}")
    if volume.vol_ma20 is not None:
        vol_parts.append(f"VOL_MA20={volume.vol_ma20:.0f}")
    vol_line = "   " + " | ".join(vol_parts)
    if vol_parts:
        lines.append(vol_line)
    if volume.volume_ratio is not None:
        lines.append(f"   量比：{volume.volume_ratio:.2f}（{volume.volume_status}）")
    lines.append(f"   量价信号：{volume.price_volume_signal}")

    # ========================= 结构 =========================
    lines.append("")
    lines.append(" 📍 结构")
    support_str = " / ".join(f"{s:.2f}" for s in structure.support_levels) if structure.support_levels else "未识别"
    resistance_str = " / ".join(f"{r:.2f}" for r in structure.resistance_levels) if structure.resistance_levels else "未识别"
    lines.append(f"   参考支撑：{support_str}")
    lines.append(f"   参考压力：{resistance_str}")

    if structure.recent_high:
        lines.append(f"   近60日高点：{structure.recent_high:.2f}  低点：{structure.recent_low:.2f}")

    # ========================= 风险 =========================
    lines.append("")
    lines.append(" ⚠️ 风险指标")
    lines.append(f"   近250日最大回撤：{_pct(structure.max_drawdown)}")
    lines.append(f"   年化波动率：{_pct(structure.annual_volatility)}")
    if structure.ma250_deviation is not None:
        dev = structure.ma250_deviation
        dev_str = f"{dev:+.2f}%"
        if abs(dev) > 30:
            dev_note = "（偏离过大，注意风险）" if dev > 0 else "（大幅低于年线）"
        elif abs(dev) > 15:
            dev_note = "（价格偏高，注意回调）" if dev > 0 else "（偏低，可能存在机会）"
        else:
            dev_note = "（正常范围）"
        lines.append(f"   价格偏离年线：{dev_str} {dev_note}")

    # ========================= 综合建议 =========================
    lines.append("")
    lines.append(" 🧭 综合建议")

    advice_mid, advice_short, risk_note = _generate_advice(
        valuation_range=valuation.valuation_range,
        trend_direction=trend.trend_direction,
        above_ma250=trend.above_ma250,
        macd_signal=momentum.macd_signal,
        rsi_signal=momentum.rsi_signal,
        kdj_signal=momentum.kdj_signal,
        price_volume_signal=volume.price_volume_signal,
        volume_ratio=volume.volume_ratio,
        max_drawdown=structure.max_drawdown,
        ma250_deviation=structure.ma250_deviation,
        pe_percentile=valuation.pe_percentile,
        latest_price=latest_price,
    )

    for line in advice_mid:
        lines.append(f"   {line}")
    for line in advice_short:
        lines.append(f"   {line}")

    lines.append("")
    lines.append(f"   ⚠️ {risk_note}")

    # ========================= 免责声明 =========================
    lines.append("")
    lines.append(f" ⚠️ 本结果仅供参考，不构成投资建议")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


def _generate_advice(
    valuation_range: str,
    trend_direction: str,
    above_ma250: Optional[bool],
    macd_signal: str,
    rsi_signal: str,
    kdj_signal: str,
    price_volume_signal: str,
    volume_ratio: Optional[float],
    max_drawdown: Optional[float],
    ma250_deviation: Optional[float],
    pe_percentile: Optional[float],
    latest_price: Optional[float],
) -> tuple:
    """生成中长线建议、短线建议和风险提示。"""
    advice_mid = []
    advice_short = []
    risk_items = []

    # ------ 中长线 ------
    val_score = 0
    if valuation_range in ("低估", "合理偏低"):
        val_score = 1
        advice_mid.append("• 中长线：估值偏低，具备安全边际，可重点关注")
    elif valuation_range == "合理":
        advice_mid.append("• 中长线：估值合理，结合趋势判断")
    else:
        val_score = -1
        advice_mid.append("• 中长线：估值偏高，需警惕估值回归风险")

    trend_up = trend_direction in ("多头排列", "震荡偏多")
    trend_down = trend_direction in ("空头排列", "震荡偏空")

    if trend_up and above_ma250:
        advice_mid.append("  趋势向上且站上年线，中长线持有逻辑未破坏")
    elif trend_down:
        advice_mid.append("  趋势偏空，中长线建议等待企稳信号")

    # MACD 与趋势协同
    if macd_signal == "金叉" and trend_up:
        advice_mid.append("  MACD 金叉 + 多头趋势，中线积极信号")
    elif macd_signal == "死叉" and trend_down:
        advice_mid.append("  MACD 死叉 + 空头趋势，中线回避")

    # ------ 短线 ------
    short_score = 0

    # RSI
    if "超买" in rsi_signal:
        advice_short.append("• 短线：RSI 超买，不宜追高，等待回调")
        short_score -= 1
    elif "超卖" in rsi_signal:
        advice_short.append("• 短线：RSI 超卖，可能出现超跌反弹机会")
        short_score += 1
    else:
        advice_short.append("• 短线：RSI 中性，方向未明")

    # KDJ
    if "高位钝化" in kdj_signal:
        advice_short.append("  KDJ 高位钝化，短线警惕回调")
        short_score -= 1
    elif "低位超卖" in kdj_signal:
        advice_short.append("  KDJ 低位超卖，短线或有反弹")
        short_score += 1
    elif "J值进入负值区间" in kdj_signal:
        advice_short.append("  KDJ J值进入负值区间，超跌反弹概率较大")
        short_score += 1
    elif "J值进入超涨区间" in kdj_signal:
        advice_short.append("  KDJ J值进入超涨区间，短线注意回调")
        short_score -= 1
    elif "金叉" in kdj_signal:
        advice_short.append("  KDJ 金叉，短期动能偏多")
        short_score += 1
    elif "死叉" in kdj_signal:
        advice_short.append("  KDJ 死叉，短期动能偏空")
        short_score -= 1
    elif "短线偏多" in kdj_signal:
        advice_short.append("  KDJ 短线偏多")
        short_score += 0.5
    elif "短线偏空" in kdj_signal:
        advice_short.append("  KDJ 短线偏空")
        short_score -= 0.5

    # 量价
    if "放量上涨" in price_volume_signal:
        advice_short.append("  量价配合良好（放量上涨），短线动能充足")
        short_score += 1
    elif "缩量上涨" in price_volume_signal:
        advice_short.append("  缩量上涨，注意上涨动能衰竭")
        short_score -= 1
    elif "放量下跌" in price_volume_signal:
        advice_short.append("  放量下跌，短线抛压较大，建议观望")
        short_score -= 2

    # 综合短线判断
    if short_score >= 2:
        advice_short.append("  综合判断：短线动能偏多，可小仓位参与")
    elif short_score <= -2:
        advice_short.append("  综合判断：短线信号偏空，建议观望回避")
    else:
        advice_short.append("  综合判断：短线信号不明朗，等待明确方向")

    # ------ 风险 ------
    if "偏高" in valuation_range or "泡沫" in valuation_range:
        risk_items.append("估值处于高位区间")
    if trend_down:
        risk_items.append("处于空头趋势中")
    if "死叉" in macd_signal:
        risk_items.append("MACD 死叉")
    if "超买" in rsi_signal:
        risk_items.append("RSI 超买，短期回调压力大")
    if "放量下跌" in price_volume_signal:
        risk_items.append("出现放量下跌，抛压明显")
    if max_drawdown is not None and abs(max_drawdown) > 40:
        risk_items.append(f"近一年最大回撤已达 {abs(max_drawdown):.1f}%，波动剧烈")
    if ma250_deviation is not None and abs(ma250_deviation) > 30:
        risk_items.append(f"价格偏离年线 {ma250_deviation:+.1f}%，偏离度过大")

    if risk_items:
        risk_note = "风险提示：" + "；".join(risk_items)
    else:
        risk_note = "请自行控制仓位，做好风险管理"

    risky_keywords = ["泡沫", "空头排列", "放量下跌", "顶背离"]
    if any(kw in risk_note for kw in risky_keywords):
        risk_note = "⚠️ " + risk_note

    # 加入操作建议
    if latest_price:
        advice_mid.append(f"  当前价格：{latest_price:.2f}")

    return advice_mid, advice_short, risk_note


if __name__ == "__main__":
    # 示例：直接运行可查看报告结构
    from data.fetch_price import fetch_daily_price
    from data.fetch_fundamental import fetch_stock_name
    from analysis.valuation import analyze_valuation
    from analysis.trend import analyze_trend
    from analysis.momentum import analyze_momentum
    from analysis.volume import analyze_volume
    from analysis.structure import analyze_structure

    symbol = "600519"
    df = fetch_daily_price(symbol)
    name = fetch_stock_name(symbol)
    price = float(df["close"].iloc[-1]) if df is not None and "close" in df.columns else None

    val = analyze_valuation(symbol)
    trend = analyze_trend(df)
    mom = analyze_momentum(df)
    vol = analyze_volume(df)
    struct = analyze_structure(df)

    report = generate_report(symbol, name, price, val, trend, mom, vol, struct)
    print(report)
