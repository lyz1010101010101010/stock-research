"""
数据获取 —— 日线行情
接口（按优先级）：
  1. ak.stock_zh_a_daily  (新浪源，需带 sh/sz 前缀)
  2. ak.stock_zh_a_hist   (东方财富源，兜底)
"""

import sys
import os
from datetime import datetime, timedelta

import pandas as pd

# 确保项目根目录在 path 中，以便 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


def _symbol_with_prefix(symbol: str) -> str:
    """
    将纯数字代码转为带交易所前缀的格式。

    规则：
      - 6xxxxx → sh600xxx（上海主板 / 科创板）
      - 0xxxxx → sz000xxx（深圳主板）
      - 3xxxxx → sz300xxx（创业板）
      - 4xxxxx → sh400xxx（老三板）
      - 8xxxxx → sh800xxx（科创板）
    """
    symbol = symbol.strip()
    if symbol.startswith(("6", "9")):
        return f"sh{symbol}"
    elif symbol.startswith(("0", "3", "2")):
        return f"sz{symbol}"
    else:
        # 兜底：默认深市
        return f"sz{symbol}"


def fetch_daily_price(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str | None = None,
) -> pd.DataFrame | None:
    """
    从 akshare 获取个股日线行情。

    参数
    -----
    symbol : str
        股票代码，如 "600519"
    start_date : str, optional
        开始日期 "YYYYMMDD"，默认 5 年前
    end_date : str, optional
        结束日期 "YYYYMMDD"，默认今天
    adjust : str, optional
        复权方式: "qfq" / "hfq" / None，默认 cfg.ADJUST

    返回
    -----
    pd.DataFrame | None
        包含 columns: date, open, close, high, low, volume, amount, pct_chg
        失败返回 None
    """
    import akshare as ak

    if adjust is None:
        adjust = cfg.ADJUST
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * cfg.VALUATION_HISTORY_YEARS)).strftime("%Y%m%d")

    # ---------- 优先使用 stock_zh_a_daily（新浪源，更稳定） ----------
    prefixed = _symbol_with_prefix(symbol)
    df = _try_fetch_via_daily(ak, prefixed, start_date, end_date, adjust)

    # ---------- 兜底：stock_zh_a_hist（东方财富源） ----------
    if df is None:
        df = _try_fetch_via_hist(ak, symbol, start_date, end_date, adjust)

    if df is None:
        print(f"  [错误] 获取 {symbol} 行情失败（两个接口均不可用）")
        return None

    # ---------- 后处理 ----------
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 确保 numeric 类型
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算涨跌幅（如果源未提供）
    if "pct_chg" not in df.columns:
        df["pct_chg"] = df["close"].pct_change() * 100

    return df


def _try_fetch_via_daily(
    ak,
    prefixed_symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame | None:
    """
    尝试通过 stock_zh_a_daily（新浪）接口获取。
    返回列：date, open, close, high, low, volume, amount, pct_chg
    """
    try:
        df = ak.stock_zh_a_daily(
            symbol=prefixed_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # 统一列名
    df = df.rename(columns={
        "date": "date",
        "open": "open",
        "high": "high",
        "close": "close",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
    })
    return df


def _try_fetch_via_hist(
    ak,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame | None:
    """
    尝试通过 stock_zh_a_hist（东方财富）接口获取。
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # 中文列名 → 英文
    df = df.rename(columns=cfg.COLUMN_MAP)
    return df


def get_latest_price(df: pd.DataFrame) -> float:
    """取最新收盘价"""
    if df is None or df.empty:
        return 0.0
    return float(df["close"].iloc[-1])


def get_latest_volume(df: pd.DataFrame) -> float:
    """取最新成交量"""
    if df is None or df.empty:
        return 0.0
    return float(df["volume"].iloc[-1])


if __name__ == "__main__":
    # 简单测试
    df = fetch_daily_price("600519")
    if df is not None:
        print(df.tail(5))
