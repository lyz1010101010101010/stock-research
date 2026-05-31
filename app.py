#!/usr/bin/env python3
"""
A 股智能分析工具 —— Streamlit Web 应用

用法:
    streamlit run app.py

依赖:
    pip install streamlit akshare pandas numpy matplotlib
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

# ---------- 现有模块 ----------
from main import analyze_stock
from screen import screen_stocks

# ---------- 自选股看板依赖 ----------
import akshare as ak
from data.fetch_price import fetch_daily_price
from data.fetch_fundamental import fetch_stock_name

# ---------- 批量分析依赖 ----------
from analysis.valuation import analyze_valuation
from analysis.trend import analyze_trend
from analysis.momentum import analyze_momentum
from analysis.volume import analyze_volume
from analysis.structure import analyze_structure

# ==================== 【修复】Streamlit Cloud RemoteDisconnected ====================
# 1) 全局 requests User-Agent 补丁，避免被远端拒绝
import requests as _requests

_orig_get = _requests.get

def _patched_get(*args, **kwargs):
    kwargs.setdefault("headers", {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    })
    return _orig_get(*args, **kwargs)

_requests.get = _patched_get

# 2) akshare 重试装饰器（将在 fetch_price 中使用）
import time as _time

# ==================== 常量 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")


# ==================== 自选股持久化 ====================
def _load_watchlist() -> list:
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _save_watchlist(codes: list):
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(codes, f, ensure_ascii=False)


# ==================== 缓存数据获取 ====================
@st.cache_data(ttl=30, show_spinner=False)
def _fetch_spot_em():
    """全市场实时行情 —— 缓存 30 秒，带 3 次重试"""
    for i in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                return df
        except Exception:
            if i == 2:
                return None
            _time.sleep(2)
    return None


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_tech_signals(symbol: str):
    """
    获取单只股票的简版技术信号：MA 排列 + KDJ 状态。
    基于日线数据计算，缓存 10 分钟（日线盘中不变）。
    返回 (ma_signal: str, kdj_signal: str)
    """
    try:
        df = fetch_daily_price(symbol)
        if df is None or df.empty or "close" not in df.columns:
            return "—", "—"

        close = df["close"].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close

        if len(close) < 20:
            return "—", "—"

        # ---- MA 排列 ----
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
            ma_signal = "—"
        elif ma5 > ma10 > ma20:
            ma_signal = "✅ 多头"
        elif ma5 < ma10 < ma20:
            ma_signal = "❌ 空头"
        else:
            ma_signal = "➖ 震荡"

        # ---- KDJ 简版 (N=9, M1=3, M2=3) ----
        n = 9
        low_n = low.rolling(n).min()
        high_n = high.rolling(n).max()
        denom = high_n - low_n
        denom = denom.replace(0, np.nan)
        rsv = (close - low_n) / denom * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        k_now, d_now = k.iloc[-1], d.iloc[-1]
        k_prev, d_prev = k.iloc[-2], d.iloc[-2]
        j_now = j.iloc[-1]

        if pd.isna(k_now) or pd.isna(d_now):
            kdj_signal = "—"
        elif k_prev <= d_prev and k_now > d_now:
            kdj_signal = "🟢 KDJ金叉"
        elif k_prev >= d_prev and k_now < d_now:
            kdj_signal = "🔴 KDJ死叉"
        elif j_now < 20:
            kdj_signal = "🔵 超卖"
        elif j_now > 100:
            kdj_signal = "🟣 超买"
        elif k_now > d_now:
            kdj_signal = "🟢 偏多"
        else:
            kdj_signal = "🔴 偏空"

        return ma_signal, kdj_signal
    except Exception:
        return "—", "—"


# ==================== 结构化分析（批量用） ====================
def _analyze_stock_summary(code: str) -> dict:
    """
    【新增】对单只股票执行完整分析，返回结构化 dict。
    包含：代码、名称、估值、技术指标、综合评分。
    用于 sidebar 批量分析的结果展示。
    """
    # 获取名称
    try:
        name = fetch_stock_name(code)
    except Exception:
        name = code

    result = {
        "代码": code,
        "名称": name,
        "最新价": "—",
        "估值区间": "—",
        "PE分位": None,
        "PB分位": None,
        "趋势": "—",
        "MACD": "—",
        "KDJ": "—",
        "RSI": None,
        "RSI信号": "—",
        "量价信号": "—",
        "最大回撤": "—",
        "评分": 50,
        "建议": "—",
    }

    try:
        df = fetch_daily_price(code)
        if df is None or df.empty:
            result["建议"] = "❌ 数据不足"
            return result

        close = df["close"].astype(float)
        if close.empty:
            result["建议"] = "❌ 数据不足"
            return result

        latest_price = float(close.iloc[-1])
        result["最新价"] = f"{latest_price:.2f}"

        # ---- 估值 ----
        valuation = analyze_valuation(code)
        result["PE分位"] = valuation.pe_percentile
        result["PB分位"] = valuation.pb_percentile
        result["估值区间"] = valuation.valuation_range

        # ---- 趋势 ----
        trend = analyze_trend(df)
        result["趋势"] = trend.trend_direction

        # ---- 动能 (MACD / KDJ / RSI) ----
        momentum = analyze_momentum(df)
        result["MACD"] = momentum.macd_signal
        result["KDJ"] = momentum.kdj_signal
        result["RSI"] = round(momentum.rsi, 1) if momentum.rsi is not None else None
        result["RSI信号"] = momentum.rsi_signal or "—"

        # ---- 量价 ----
        volume = analyze_volume(df)
        result["量价信号"] = volume.price_volume_signal

        # ---- 结构 ----
        structure = analyze_structure(df)
        if structure.max_drawdown is not None:
            result["最大回撤"] = f"{structure.max_drawdown:.1f}%"

        # ======== 综合评分 (0-100) ========
        score = 50  # 中性基准

        # 估值评分 (±25)
        vr = valuation.valuation_range
        if vr == "低估":
            score += 25
        elif vr == "合理偏低":
            score += 15
        elif vr == "合理":
            score += 5
        elif vr == "偏高":
            score -= 15
        elif vr == "泡沫":
            score -= 25

        # 趋势评分 (±15)
        td = trend.trend_direction
        if td == "多头排列":
            score += 15
        elif td == "震荡偏多":
            score += 8
        elif td == "震荡偏空":
            score -= 8
        elif td == "空头排列":
            score -= 15

        # MACD 评分 (±10)
        if momentum.macd_signal == "金叉":
            score += 10
        elif momentum.macd_signal == "死叉":
            score -= 10

        # KDJ 评分 (±5)
        kdj = momentum.kdj_signal or ""
        if "金叉" in kdj:
            score += 5
        elif "死叉" in kdj:
            score -= 5

        # 量价评分 (±5)
        pv = volume.price_volume_signal or ""
        if "放量上涨" in pv:
            score += 5
        elif "放量下跌" in pv:
            score -= 5

        result["评分"] = max(0, min(100, score))

        # 综合建议
        if score >= 75:
            result["建议"] = "🟢 强烈关注"
        elif score >= 60:
            result["建议"] = "🟢 可以关注"
        elif score >= 40:
            result["建议"] = "🟡 中性观望"
        elif score >= 25:
            result["建议"] = "🟠 谨慎回避"
        else:
            result["建议"] = "🔴 建议回避"

        return result

    except Exception as e:
        result["建议"] = f"❌ 异常: {e}"
        return result


# ==================== 【新增】AI 评语 + 综合评分信号灯 ====================

def _map_valuation_category(info: dict) -> str:
    """将 '估值区间' 映射为三档：低估 / 合理 / 偏高"""
    val = str(info.get("估值区间", ""))
    if "低估" in val or "偏低" in val:
        return "低估"
    elif "合理" in val:
        return "合理"
    else:
        return "偏高"  # 泡沫也归入偏高


def _map_tech_category(info: dict) -> str:
    """将 '趋势' 映射为三档：偏多 / 震荡 / 偏空"""
    tech = str(info.get("趋势", ""))
    if "多头" in tech or "偏多" in tech:
        return "偏多"
    elif "偏空" in tech or "空头" in tech:
        return "偏空"
    else:
        return "震荡"


def calc_score(info: dict) -> int:
    """
    【新增】根据估值 + 技术面计算综合评分 (0–100)。
    评分规则：
        估值=低估 → +40；合理 → +25；偏高 → +10
        技术=偏多 → +30；震荡 → +15；偏空 → +5
        基准分 = 30，上限 100
    """
    BASE = 30

    # ---- 估值评分 ----
    val_cat = _map_valuation_category(info)
    val_bonus = {"低估": 40, "合理": 25, "偏高": 10}
    val_score = val_bonus.get(val_cat, 10)

    # ---- 技术评分 ----
    tech_cat = _map_tech_category(info)
    tech_bonus = {"偏多": 30, "震荡": 15, "偏空": 5}
    tech_score = tech_bonus.get(tech_cat, 5)

    return min(100, BASE + val_score + tech_score)


def show_score(score: int):
    """
    【新增】信号灯展示。
    ≥80：st.success + 进度条（🟢 强烈关注）
    ≥60：st.warning（🟡 中性观察）
    <60：st.error（🔴 谨慎回避）
    """
    if score >= 80:
        st.success(f"🟢 综合评分: {score}/100 — 强烈关注")
        st.progress(score / 100)
    elif score >= 60:
        st.warning(f"🟡 综合评分: {score}/100 — 中性观察")
    else:
        st.error(f"🔴 综合评分: {score}/100 — 谨慎回避")


def ai_comment(name: str, code: str, info: dict, score: int) -> str:
    """
    【新增】基于规则生成 AI 一句结论（不调用外部 API）。
    语气按分数档位：
        高分 → 积极偏多
        中分 → 中性跟踪
        低分 → 谨慎回避
    """
    val_cat = _map_valuation_category(info)
    tech_cat = _map_tech_category(info)

    # 高分 (≥80)
    if score >= 80:
        pool = [
            f"估值{val_cat}，技术面{tech_cat}，具备中长期关注价值，可考虑分批布局。",
            f"综合评分优秀，{name}({code}) 估值与技术共振向上，建议纳入核心观察池。",
            f"当前{name}处于估值洼地且趋势向好，中线持有逻辑清晰，值得重点跟踪。",
        ]
        return pool[hash(code + "high") % len(pool)]

    # 中高分 (70-79)
    if score >= 70:
        pool = [
            f"估值{val_cat}、技术{tech_cat}，整体偏积极但非最优，可小仓位试探。",
            f"{name}方向偏多但仍有不确定性，建议结合大盘节奏灵活应对。",
        ]
        return pool[hash(code + "midhigh") % len(pool)]

    # 中分 (60-69)
    if score >= 60:
        return "方向尚不明朗，估值与技术未能形成共振，建议持续跟踪，等待更明确的入场信号。"

    # 低分 (<60)
    pool = [
        f"风险偏高（估值{val_cat}、技术{tech_cat}），建议暂时观望或回避。",
        f"{name}({code}) 当前性价比不突出，不宜追高，等待回调或趋势明朗后再评估。",
    ]
    return pool[hash(code + "low") % len(pool)]


# ==================== 页面配置 ====================
st.set_page_config(page_title="A 股智能分析工具", page_icon="📊", layout="wide")
st.title("📊 A 股智能分析工具")
st.caption("基于估值分位 + 多维度技术指标的综合研判工具")

# ==================== 【新增】Sidebar: 自选股 + 批量分析 ====================
with st.sidebar:
    st.header("📋 自选股 + 批量分析")

    st.markdown("**多只 A 股代码**（逗号分隔）")
    batch_input = st.text_area(
        "股票代码",
        placeholder="例如：600887,000001,600519",
        height=100,
        label_visibility="collapsed",
        key="sidebar_batch_input",
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        parse_btn = st.button("🔍 解析代码", use_container_width=True)
    with col_s2:
        analyze_all_btn = st.button("🚀 批量分析", type="primary", use_container_width=True)

    if parse_btn:
        raw = batch_input.replace("，", ",").replace("、", ",").replace("\n", ",")
        parsed = [c.strip() for c in raw.split(",") if c.strip()]
        # 过滤出 6 位数字代码
        valid_codes = [c for c in parsed if c.isdigit() and len(c) == 6]
        invalid = [c for c in parsed if not (c.isdigit() and len(c) == 6)]
        if invalid:
            st.warning(f"以下代码格式不正确已忽略：{', '.join(invalid)}")
        if valid_codes:
            st.session_state.batch_codes = valid_codes
            st.success(f"✅ 已解析 {len(valid_codes)} 只股票：{' · '.join(valid_codes)}")
        else:
            st.warning("未识别到有效的 6 位股票代码")

    if analyze_all_btn:
        raw = batch_input.replace("，", ",").replace("、", ",").replace("\n", ",")
        parsed = [c.strip() for c in raw.split(",") if c.strip()]
        valid_codes = [c for c in parsed if c.isdigit() and len(c) == 6]
        if not valid_codes:
            # 尝试从已解析的代码中获取
            if st.session_state.get("batch_codes"):
                valid_codes = st.session_state.batch_codes
        if not valid_codes:
            st.error("请先输入有效代码并点击「解析代码」")
        else:
            st.session_state.batch_codes = valid_codes
            st.session_state.batch_results = None  # 触发分析
            st.rerun()

    # 如果 codes 已解析但尚未分析，给出提示
    if st.session_state.get("batch_codes") and st.session_state.get("batch_results") is None:
        st.info(f"📌 待分析 {len(st.session_state.batch_codes)} 只股票，点击「🚀 批量分析」开始")

    st.markdown("---")
    st.caption("💡 提示：分析过程需要获取行情、估值及技术指标，每只股票约需 3~5 秒。")

# ---- 初始化 session_state ----
if "watchlist" not in st.session_state:
    st.session_state.watchlist = _load_watchlist()
if "jump_code" not in st.session_state:
    st.session_state.jump_code = ""
# ---- 【新增】批量分析 session_state ----
if "batch_codes" not in st.session_state:
    st.session_state.batch_codes = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

# ==================== 【新增】批量分析结果展示（主区域） ====================
# 当 sidebar 触发了 st.rerun() 后，在此处执行分析并展示结果
if st.session_state.batch_results is None and st.session_state.batch_codes:
    # 需要执行分析
    codes = st.session_state.batch_codes
    results = []
    progress = st.progress(0, text="正在批量分析…")
    for i, code in enumerate(codes):
        progress.progress((i) / len(codes), text=f"正在分析 [{i+1}/{len(codes)}] {code} …")
        results.append(_analyze_stock_summary(code))
    progress.progress(1.0, text=f"✅ 全部完成！共分析 {len(codes)} 只股票")
    st.session_state.batch_results = results
    st.rerun()

# 展示已缓存的结果
if st.session_state.batch_results:
    results = st.session_state.batch_results
    codes = st.session_state.batch_codes

    st.markdown("---")
    st.markdown("### 📊 批量分析结果")

    # ---- 3 列指标卡片 ----
    col_total, col_avg, col_best = st.columns(3)
    scores = [r["评分"] for r in results]
    with col_total:
        st.metric("分析股票数", len(results))
    with col_avg:
        avg_score = sum(scores) / len(scores) if scores else 0
        st.metric("平均评分", f"{avg_score:.0f}/100")
    with col_best:
        if scores:
            best_idx = scores.index(max(scores))
            best = results[best_idx]
            st.metric("最高评分", f"{best['代码']} {best['评分']}/100")
        else:
            st.metric("最高评分", "—")

    st.markdown("---")

    # ---- DataFrame 总览表 ----
    st.markdown("#### 📋 总览表")
    df = pd.DataFrame(results)
    display_cols = [
        "代码", "名称", "最新价", "估值区间", "PE分位", "PB分位",
        "趋势", "MACD", "KDJ", "RSI", "量价信号", "评分", "建议",
    ]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PE分位": st.column_config.NumberColumn(format="%.1f%%"),
            "PB分位": st.column_config.NumberColumn(format="%.1f%%"),
            "RSI": st.column_config.NumberColumn(format="%.1f"),
            "评分": st.column_config.ProgressColumn(
                format="%d/100", min_value=0, max_value=100,
            ),
        },
    )

    # ---- 3 列卡片详情（AI 评语 + 信号灯） ----
    st.markdown("---")
    st.markdown("#### 🃏 个股卡片 · AI 评语")

    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            # ---- 【新增】用 calc_score 重新计算信号灯评分 ----
            ai_score = calc_score(r)
            comment = ai_comment(r["名称"], r["代码"], r, ai_score)

            with st.container(border=True):
                # ① 股票名称 + 代码
                st.markdown(f"**{r['代码']}** {r['名称']}")
                st.caption(f"最新价: {r['最新价']}")

                # ② 估值 / 技术 精简标签
                val_cat = _map_valuation_category(r)
                tech_cat = _map_tech_category(r)
                col_v, col_t = st.columns(2)
                with col_v:
                    v_icon = {"低估": "🟢", "合理": "🟡", "偏高": "🔴"}.get(val_cat, "⚪")
                    st.caption(f"{v_icon} 估值: {val_cat}")
                with col_t:
                    t_icon = {"偏多": "🟢", "震荡": "🟡", "偏空": "🔴"}.get(tech_cat, "⚪")
                    st.caption(f"{t_icon} 技术: {tech_cat}")

                # ③ 评分信号灯
                show_score(ai_score)

                # ④ AI 评语
                st.caption(f"💬 {comment}")

    # 清除按钮
    if st.button("🗑️ 清除批量结果"):
        st.session_state.batch_codes = []
        st.session_state.batch_results = None
        st.rerun()

    st.markdown("---")

# ---- 标签页 ----
tabs = st.tabs(["🔍 单股分析", "📋 批量分析", "🎯 低估筛选", "⭐ 自选股看板"])

# ====================================================================
#  Tab 1 —— 单股分析
# ====================================================================
with tabs[0]:
    st.subheader("单股深度分析")

    # 支持从自选股看板跳转（预填代码）
    default_code = st.session_state.get("jump_code", "")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        stock_code = st.text_input(
            "股票代码",
            value=default_code,
            placeholder="例如：600887",
            label_visibility="collapsed",
            key="single_code",
        )
    with col_btn:
        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    # 使用后清除跳转标记
    if default_code and stock_code:
        st.session_state.jump_code = ""

    if analyze_btn and stock_code:
        code = stock_code.strip()
        with st.spinner(f"正在分析 {code}，获取行情、估值、技术指标中…"):
            try:
                report = analyze_stock(code)
                st.markdown("### 📝 分析报告")
                st.code(report, language=None)

                chart_path = os.path.join(BASE_DIR, f"chart_{code}.png")
                val_path = os.path.join(BASE_DIR, f"valuation_{code}.png")

                col_a, col_b = st.columns(2)
                with col_a:
                    if os.path.exists(chart_path):
                        st.image(chart_path, caption=f"{code} — 价格走势 & KDJ", use_container_width=True)
                with col_b:
                    if os.path.exists(val_path):
                        st.image(val_path, caption=f"{code} — 估值历史分位", use_container_width=True)
            except Exception as e:
                st.error(f"分析失败: {e}")

# ====================================================================
#  Tab 2 —— 批量分析
# ====================================================================
with tabs[1]:
    st.subheader("批量股票分析")

    stock_list_input = st.text_area(
        "股票代码（逗号 / 顿号 / 换行 分隔均可）",
        placeholder="例如：600036,600601,600887",
        height=90,
    )

    if st.button("📊 批量分析", type="primary"):
        raw = stock_list_input.replace("，", ",").replace("、", ",").replace("\n", ",")
        codes = [c.strip() for c in raw.split(",") if c.strip()]

        if not codes:
            st.warning("请输入至少一个股票代码")
        else:
            st.info(f"共 **{len(codes)}** 只股票：{' · '.join(codes)}")
            progress = st.progress(0)
            status = st.empty()

            for i, code in enumerate(codes):
                status.text(f"正在分析 [{i + 1}/{len(codes)}] {code} …")
                with st.expander(f"📈 [{code}] 分析结果", expanded=(len(codes) <= 2)):
                    try:
                        report = analyze_stock(code)
                        st.code(report, language=None)

                        chart_path = os.path.join(BASE_DIR, f"chart_{code}.png")
                        val_path = os.path.join(BASE_DIR, f"valuation_{code}.png")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if os.path.exists(chart_path):
                                st.image(chart_path, caption="走势图", use_container_width=True)
                        with col_b:
                            if os.path.exists(val_path):
                                st.image(val_path, caption="估值分位", use_container_width=True)
                    except Exception as e:
                        st.error(f"{code} 分析异常: {e}")

                progress.progress((i + 1) / len(codes))

            status.text(f"✅ 全部完成！共分析 {len(codes)} 只股票")

# ====================================================================
#  Tab 3 —— 低估筛选
# ====================================================================
with tabs[2]:
    st.subheader("低估 + 趋势向上 筛选器")
    st.markdown(
        "> **筛选条件**：PE 分位 ≤ 阈值 · PB 分位 ≤ 阈值 · 收盘价 > MA60（季线）"
    )

    col_pe, col_pb, col_pool = st.columns(3)
    with col_pe:
        pe_threshold = st.slider("PE 分位阈值 (%)", 10, 50, 30, 5)
    with col_pb:
        pb_threshold = st.slider("PB 分位阈值 (%)", 10, 60, 40, 5)
    with col_pool:
        custom_pool = st.text_input(
            "自定义股票池（留空 = 使用 config 默认池）",
            placeholder="如 600036,000333,600887",
        )

    if st.button("🔍 开始筛选", type="primary"):
        codes = None
        if custom_pool.strip():
            raw = custom_pool.replace("，", ",").replace("、", ",")
            codes = [c.strip() for c in raw.split(",") if c.strip()]

        with st.spinner("正在筛选，请稍候…"):
            try:
                results = screen_stocks(
                    stock_codes=codes,
                    pe_threshold=pe_threshold,
                    pb_threshold=pb_threshold,
                )

                if not results:
                    st.warning("⚠️ 无股票通过筛选，建议放宽条件或等待市场调整。")
                else:
                    st.success(f"✅ 共 **{len(results)}** 只股票通过筛选！")

                    df = pd.DataFrame(results)
                    display_cols = [
                        "symbol", "name", "close", "pe_pct", "pb_pct",
                        "market_cap", "valuation_range", "trend_brief",
                    ]
                    display_df = df[[c for c in display_cols if c in df.columns]].copy()
                    display_df.rename(
                        columns={
                            "symbol": "代码", "name": "名称",
                            "close": "最新价", "pe_pct": "PE分位(%)",
                            "pb_pct": "PB分位(%)", "market_cap": "市值(亿)",
                            "valuation_range": "估值区间", "trend_brief": "趋势简述",
                        },
                        inplace=True,
                    )
                    st.dataframe(
                        display_df, use_container_width=True, hide_index=True,
                        column_config={
                            "最新价": st.column_config.NumberColumn(format="%.2f"),
                            "PE分位(%)": st.column_config.NumberColumn(format="%.1f"),
                            "PB分位(%)": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )

                    st.markdown("---")
                    st.markdown("### 📋 详细结果")
                    for idx, r in enumerate(results, 1):
                        with st.expander(f"#{idx}  [{r['symbol']}] {r['name']} — {r['valuation_range']}"):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("最新价", f"{r['close']:.2f}")
                                st.metric("PE 分位", f"{r['pe_pct']:.1f}%")
                            with c2:
                                st.metric("MA60", f"{r['ma60']:.2f}",
                                          delta="站上 ✅" if r["above_ma60"] else "下方 ❌")
                                st.metric("PB 分位", f"{r['pb_pct']:.1f}%")
                            with c3:
                                if r.get("ma250"):
                                    st.metric("MA250", f"{r['ma250']:.2f}",
                                              delta="站上 ✅" if r.get("above_ma250") else "下方 ❌")
                                if r.get("market_cap"):
                                    st.metric("总市值(亿)", f"{r['market_cap']:.1f}")
                            st.caption(f"趋势：{r['trend_brief']}")

                    st.markdown("---")
                    st.caption("⚠️ 本结果仅供参考，不构成投资建议。")
            except Exception as e:
                st.error(f"筛选失败: {e}")

# ====================================================================
#  Tab 4 —— 自选股看板
# ====================================================================
with tabs[3]:
    st.subheader("⭐ 自选股看板")

    # ========== 1. 自选股管理 ==========
    watchlist_str = st.text_input(
        "自选股代码（逗号分隔）",
        value=",".join(st.session_state.watchlist),
        placeholder="例如：600887,600036,600519",
        key="watchlist_input",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💾 保存自选", use_container_width=True):
            codes = [c.strip() for c in watchlist_str.replace("，", ",").split(",") if c.strip()]
            st.session_state.watchlist = codes
            _save_watchlist(codes)
            st.success(f"已保存 {len(codes)} 只自选股")
            st.rerun()

    with col2:
        refresh_clicked = st.button("🔄 刷新实盘", use_container_width=True, type="primary")

    # ========== 2. 实时行情 + 技术信号 ==========
    wl = st.session_state.watchlist
    if not wl:
        st.info("👆 请先输入自选股代码并点击「💾 保存自选」")
    else:
        # 获取实时行情（容错 + 重试）
        spot_df = None
        try:
            spot_df = _fetch_spot_em()
        except Exception:
            spot_df = None

        if spot_df is None or spot_df.empty:
            st.warning("⚠️ 实时行情获取失败（可能非交易时间或服务受限），已跳过")

        if spot_df is not None and not spot_df.empty:
            # 匹配自选股
            matched = spot_df[spot_df["代码"].isin(wl)].copy()
            if matched.empty:
                st.warning("未能在实时行情中匹配到自选股代码")
            else:
                # 获取每只自选股的技术信号
                signals = {}
                for code in wl:
                    if code in matched["代码"].values:
                        signals[code] = _fetch_tech_signals(code)

                # 构建展示表格
                rows = []
                for _, r in matched.iterrows():
                    code = r["代码"]
                    pct = r.get("涨跌幅", 0)
                    pct = float(pct) if pd.notna(pct) else 0.0
                    amount = r.get("成交额", None)
                    amt_str = f"{amount / 1e8:.2f}亿" if pd.notna(amount) and amount else "—"

                    emoji = "🔴" if pct < -0.01 else ("🟢" if pct > 0.01 else "➖")
                    ma_sig, kdj_sig = signals.get(code, ("—", "—"))

                    rows.append({
                        "代码": code,
                        "名称": r.get("名称", code),
                        "最新价": f"{r.get('最新价', 0):.2f}",
                        "涨跌幅": f"{emoji} {pct:+.2f}%",
                        "成交额(亿)": amt_str,
                        "MA排列": ma_sig,
                        "KDJ信号": kdj_sig,
                    })

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

                # 选中某只 → 补算完整技术信号（显示更多细节）
                st.markdown("---")
                detail_code = st.selectbox(
                    "选择股票查看详情或跳转分析",
                    [c for c in wl if c in matched["代码"].values],
                    key="detail_select",
                )

                col_d1, col_d2 = st.columns([1, 3])
                with col_d1:
                    if st.button("🔍 跳转单股分析", use_container_width=True):
                        st.session_state.jump_code = detail_code
                        st.success(f"✅ 已选定 {detail_code}，请切换到 **【单股分析】** Tab 查看完整报告")

                with col_d2:
                    if st.button("📊 内联展开简版", use_container_width=True):
                        st.session_state.pop("_inline_code", None)  # 切换展开
                        st.session_state._inline_code = detail_code

                if st.session_state.get("_inline_code"):
                    code = st.session_state._inline_code
                    with st.expander(f"📈 [{code}] 简版分析", expanded=True):
                        try:
                            report = analyze_stock(code)
                            st.code(report, language=None)

                            chart_path = os.path.join(BASE_DIR, f"chart_{code}.png")
                            val_path = os.path.join(BASE_DIR, f"valuation_{code}.png")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if os.path.exists(chart_path):
                                    st.image(chart_path, caption="走势图", use_container_width=True)
                            with col_b:
                                if os.path.exists(val_path):
                                    st.image(val_path, caption="估值分位", use_container_width=True)
                        except Exception as e:
                            st.error(f"分析失败: {e}")
