#!/usr/bin/env python3
"""
A 股智能分析工具 —— Streamlit Web 应用（约牛风格最终版）
"""

import sys
import os
import json
import time as _time
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

# ==================== 路径 & 模块 ====================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import analyze_stock
from screen import screen_stocks

import akshare as ak
import baostock as bs
import atexit
from data.fetch_price import fetch_daily_price
from data.fetch_fundamental import fetch_stock_name

from analysis.valuation import analyze_valuation
from analysis.trend import analyze_trend
from analysis.momentum import analyze_momentum
from analysis.volume import analyze_volume
from analysis.structure import analyze_structure

# ==================== 修复 Streamlit Cloud 请求 ====================
import requests as _requests
_orig_get = _requests.get

def _patched_get(*args, **kwargs):
    kwargs.setdefault("headers", {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return _orig_get(*args, **kwargs)

_requests.get = _patched_get

# ==================== Baostock 初始化 ====================
_BS_READY = False

def _init_bs():
    """一次性初始化 baostock 登录，Streamlit 进程生命周期内有效"""
    global _BS_READY
    if not _BS_READY:
        try:
            lg = bs.login()
            if lg.error_code == '0':
                _BS_READY = True
                atexit.register(bs.logout)
        except Exception:
            pass

def _to_bs_code(code: str) -> str:
    """将 6 位数字代码转为 baostock 格式（sh.600519 / sz.000001）"""
    code = code.strip()
    return f"sh.{code}" if code.startswith("6") else f"sz.{code}"


# ==================== 常量 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")

# ==================== 自选股持久化 ====================
def _load_watchlist():
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                return json.load(f) if isinstance(json.load(f), list) else []
        except Exception:
            return []
    return []

def _save_watchlist(codes):
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(codes, f, ensure_ascii=False)

# ==================== 行情兜底 ====================
_SPOT_COLS = ["代码", "名称", "最新价", "涨跌幅", "成交额(亿)"]

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_spot_em(codes):
    for _ in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                if "成交额" in df.columns:
                    df["成交额(亿)"] = df["成交额"] / 1e8
                df = df[df["代码"].isin(codes)]
                return df[_SPOT_COLS]
        except Exception:
            _time.sleep(2)

    # 降级：历史价
    rows = []
    for code in codes:
        try:
            price = fetch_daily_price(code)["close"].iloc[-1]
            name = fetch_stock_name(code)
        except Exception:
            price, name = 0.0, code
        rows.append({"代码": code, "名称": name, "最新价": price,
                     "涨跌幅": 0.0, "成交额(亿)": None})
    return pd.DataFrame(rows, columns=_SPOT_COLS)

# ==================== 四维共振（三色版） ====================
def _state(val, pos, neg):
    return 1 if pos in val else (-1 if neg in val else 0)

@st.cache_data(ttl=600, show_spinner=False)
def calc_resonance(code):
    try:
        df = fetch_daily_price(code)
        if df is None or len(df) < 60:
            raise ValueError("数据不足")

        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        # 趋势
        ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
        if close.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
            trend = "🟢 多头"
        elif close.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
            trend = "🔴 空头"
        else:
            trend = "🟡 震荡"

        # 量能
        v5, v60 = vol.rolling(5).mean(), vol.rolling(60).mean()
        vol_state = "🔥 放量" if v5.iloc[-1] > v60.iloc[-1] else "❄️ 缩量"

        # MACD
        ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        macd = "🟢 走强" if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2] else \
               "🔴 走弱" if hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2] else "⚪ 平衡"

        # KDJ
        low, high = df["low"].astype(float), df["high"].astype(float)
        rsv = (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min()) * 100
        k, d = rsv.ewm(com=2).mean(), rsv.ewm(com=2).mean().ewm(com=2).mean()
        j = 3 * k - 2 * d
        kdj = "🟢 偏多" if j.iloc[-1] < 20 or (k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]) \
              else "🔴 偏空" if j.iloc[-1] > 80 or (k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2]) \
              else "⚪ 震荡"

        return {"趋势": trend, "量能": vol_state, "中期": macd, "短期": kdj}
    except Exception:
        return {"趋势": "⚪", "量能": "⚪", "中期": "⚪", "短期": "⚪"}

def _resonance_lights(res):
    m = {"🟢": "🟢", "🔴": "🔴", "🟡": "🟡", "❄️": "🔴", "⚪": "⚪"}
    return f"{m.get(res['趋势'][0], '⚪')} {m.get(res['量能'][0], '⚪')} {m.get(res['中期'][0], '⚪')} {m.get(res['短期'][0], '⚪')}"


def calc_daily_resonance(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 df 每一行逐日计算四维共振状态。
    输入需包含 close, high, low, volume 列（或对应中文列名）。
    返回副本，新增 _trend, _volume, _medium, _short 四列（文字标签）。
    """
    out = df.copy()
    c = pd.to_numeric(out.get("close", out.get("收盘", pd.Series(dtype=float))), errors="coerce")
    v = pd.to_numeric(out.get("volume", out.get("成交量", pd.Series(dtype=float))), errors="coerce")
    h = pd.to_numeric(out.get("high", out.get("最高", c)), errors="coerce")
    l = pd.to_numeric(out.get("low", out.get("最低", c)), errors="coerce")

    # 趋势：close vs MA20
    ma20 = c.rolling(20).mean()
    out["_trend"] = "🟡 震荡"
    out.loc[c > ma20, "_trend"] = "🟢 多头"
    out.loc[c < ma20, "_trend"] = "🔴 空头"

    # 量能：volume vs VOL_MA5
    v5 = v.rolling(5).mean()
    out["_volume"] = "❄️ 缩量"
    out.loc[v > v5, "_volume"] = "🔥 放量"

    # 中期：MACD Hist > 0
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    out["_medium"] = "🔴 走弱"
    out.loc[hist > 0, "_medium"] = "🟢 走强"

    # 短期：KDJ
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    denom = high9 - low9
    rsv = (c - low9) / denom.replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    out["_short"] = "⚪ 震荡"
    out.loc[j < 20, "_short"] = "🟢 偏多"
    out.loc[j > 80, "_short"] = "🔴 偏空"
    # 金叉/死叉辅助
    golden = (k.shift(1) <= d.shift(1)) & (k > d)
    death = (k.shift(1) >= d.shift(1)) & (k < d)
    out.loc[golden, "_short"] = "🟢 偏多"
    out.loc[death, "_short"] = "🔴 偏空"

    return out

# ==================== K线 + 四维共振（全历史 + 逐日交互） ====================
@st.cache_data(ttl=600, show_spinner=False)
def _fetch_kline(code):
    """
    获取全历史日线。
    优先 Baostock → 失败降级 AKShare。
    返回列：date, open, high, low, close, volume
    """
    end_str = datetime.today().strftime("%Y-%m-%d")

    # ── 1. Baostock（主数据源） ──
    _init_bs()
    if _BS_READY:
        try:
            bs_code = _to_bs_code(code)
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date="1990-01-01",
                end_date=end_str,
                frequency="d",
                adjustflag="3",
            )
            if rs.error_code == '0':
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
                    df["date"] = pd.to_datetime(df["date"])
                    for col in ["open","high","low","close","volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    # ── 2. AKShare（降级） ──
    try:
        df = ak.stock_zh_a_hist(code, "daily", "19900101",
                                 datetime.today().strftime("%Y%m%d"), "qfq")
        if df is None or df.empty:
            return pd.DataFrame()
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)
        for cn, en in {"日期":"date","开盘":"open","收盘":"close",
                        "最高":"high","最低":"low","成交量":"volume"}.items():
            if cn in df.columns:
                df[en] = pd.to_numeric(df[cn], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

# ==================== Session State ====================
for k, v in {"watchlist": _load_watchlist(), "batch_codes": [], "batch_results": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== UI ====================
st.set_page_config("A 股智能分析工具", "📊", layout="wide")
st.title("📊 A 股智能分析工具")

tabs = st.tabs(["🔍 单股分析", "📋 批量分析", "🎯 低估筛选", "⭐ 自选股看板"])

# ==================== Tab 1: 单股分析（交互式 K线 + 四维共振看板） ====================
with tabs[0]:
    col_inp, _ = st.columns([2, 3])
    with col_inp:
        code = st.text_input("股票代码（6位）", placeholder="600887", key="stock_code",
                             label_visibility="collapsed")

    if code and len(code) == 6 and code.isdigit():
        df = _fetch_kline(code)
        if df.empty:
            st.error("❌ 获取数据失败，请检查代码")
        else:
            df_res = calc_daily_resonance(df)

            # ========== 统一图：K 线（上）+ 每日四维方块（下） ==========
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                vertical_spacing=0.03,
            )

            # ── Row 1: K 线 + MA ──
            fig.add_trace(go.Candlestick(
                x=df["date"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
                name="K线"), row=1, col=1)
            for p, c in [(20, "#ff9800"), (60, "#1565c0")]:
                fig.add_trace(go.Scatter(x=df["date"], y=df["close"].rolling(p).mean(),
                                        line=dict(color=c, width=1), name=f"MA{p}"),
                             row=1, col=1)

            # ── Row 2: 每日四维状态方块（4 行散点，每行一个维度） ──
            def _dim_color(val):
                s = str(val)
                if "多头" in s or "偏多" in s or "放量" in s or "走强" in s:
                    return "#00c853"
                if "空头" in s or "偏空" in s or "缩量" in s or "走弱" in s:
                    return "#ff1744"
                return "#bdbdbd"

            dim_config = [("趋势", "_trend", 3), ("量能", "_volume", 2),
                          ("中期", "_medium", 1), ("短期", "_short", 0)]

            for dname, dcol, ypos in dim_config:
                colors = []
                htext = []
                for _, rr in df_res.iterrows():
                    v = rr[dcol]
                    colors.append(_dim_color(v))
                    htext.append(f"{dname}<br>{v}")
                fig.add_trace(go.Scatter(
                    x=df["date"], y=[ypos] * len(df),
                    mode="markers",
                    marker=dict(size=16, color=colors, symbol="square",
                                line=dict(width=0.3, color="white")),
                    name=dname,
                    text=htext,
                    hoverinfo="text+x",
                    showlegend=False,
                ), row=2, col=1)

            fig.update_xaxes(range=[df["date"].min(), df["date"].max()],
                            rangeslider_visible=True, row=1, col=1)

            try:
                stock_name = fetch_stock_name(code)
            except Exception:
                stock_name = code

            fig.update_layout(
                title=dict(text=f"{code} {stock_name}  —  K线 & 每日四维共振",
                           x=0.5, font=dict(size=15)),
                height=680,
                template="plotly_white",
                hovermode="x unified",
                margin=dict(l=40, r=20, t=50, b=10),
                xaxis2=dict(showgrid=False),
                yaxis2=dict(tickvals=[3, 2, 1, 0],
                           ticktext=["趋势", "量能", "中期", "短期"],
                           range=[-0.5, 3.5],
                           side="left"),
            )

            # ── 点击交互 ──
            clicked = plotly_events(fig, click_event=True, override_height=720,
                                    override_width="100%", key=f"kc_{code}")

            # ---------- 确定选中的日期 ----------
            if clicked and len(clicked) > 0:
                pt = clicked[0]
                if "x" in pt:
                    sel_date = pd.to_datetime(pt["x"])
                elif "pointNumber" in pt:
                    sel_date = df_res.iloc[pt["pointNumber"]]["date"]
                else:
                    sel_date = df_res["date"].iloc[-1]
            else:
                sel_date = df_res["date"].iloc[-1]

            # ---------- 查找该日四维状态 ----------
            row = df_res[df_res["date"] == sel_date]
            if row.empty:
                row = df_res.iloc[[-1]]
                sel_date = row["date"].iloc[-1]

            r = row.iloc[0]

            # ---------- 当日快照文字 ----------
            date_str = sel_date.strftime("%Y-%m-%d") if hasattr(sel_date, "strftime") else str(sel_date)
            trend_s = r["_trend"]
            vol_s = r["_volume"]
            med_s = r["_medium"]
            short_s = r["_short"]
            st.info(f"📅 **{date_str}**　｜　趋势 {trend_s}　｜　量能 {vol_s}　｜　MACD {med_s}　｜　KDJ {short_s}")

# -------- 自选股 --------
with tabs[3]:
    wl = st.multiselect("自选股", options=st.session_state.watchlist, default=st.session_state.watchlist)
    if st.button("保存自选"):
        st.session_state.watchlist = wl
        _save_watchlist(wl)

    df = _fetch_spot_em(wl)
    if not df.empty:
        df["四维"] = df["代码"].apply(lambda c: _resonance_lights(calc_resonance(c)))
        st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("⚠️ 数据仅供参考，不构成投资建议")