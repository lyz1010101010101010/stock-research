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
    """全市场实时行情 —— 缓存 30 秒"""
    return ak.stock_zh_a_spot_em()


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


# ==================== 页面配置 ====================
st.set_page_config(page_title="A 股智能分析工具", page_icon="📊", layout="wide")
st.title("📊 A 股智能分析工具")
st.caption("基于估值分位 + 多维度技术指标的综合研判工具")

# ---- 初始化 session_state ----
if "watchlist" not in st.session_state:
    st.session_state.watchlist = _load_watchlist()
if "jump_code" not in st.session_state:
    st.session_state.jump_code = ""

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
        # 获取实时行情（容错）
        spot_df = None
        spot_error = None
        try:
            spot_df = _fetch_spot_em()
        except Exception as e:
            spot_error = str(e)

        if spot_error:
            st.warning(f"⚠️ 实时行情获取失败: {spot_error}")

        if spot_df is not None:
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
