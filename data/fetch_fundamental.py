"""
数据获取 —— 基本面指标
接口：
  - ak.stock_financial_abstract    → EPS, BVPS, ROE 等（同花顺源）
  - ak.stock_zh_a_daily            → 日线价格（配合计算 PE / PB）
"""

import sys
import os
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from data.fetch_price import fetch_daily_price, _symbol_with_prefix


# ==================== 指标常量 ====================
# stock_financial_abstract 中 指标列 的中文关键词
INDICATOR_EPS = "基本每股收益"           # 每股收益（元）
INDICATOR_BVPS = "每股净资产"            # 每股净资产（元）
INDICATOR_ROE = "净资产收益率(ROE)"       # 净资产收益率（%）
INDICATOR_REVENUE_PS = "每股营业收入"     # 每股营业收入（元）


def _parse_abstract(symbol: str):
    """
    调用 stock_financial_abstract，返回 (categories, indicators, df_raw)，
    其中 df_raw 的列是时间点。
    失败返回 (None, None, None)。
    """
    import akshare as ak
    try:
        df = ak.stock_financial_abstract(symbol=symbol)
    except Exception as e:
        print(f"  [错误] 获取 {symbol} 财务摘要失败: {e}")
        return None, None, None

    if df is None or df.empty:
        return None, None, None

    categories = df.iloc[:, 0].tolist()   # 分类（利润指标/每股指标/盈利指标...）
    indicators = df.iloc[:, 1].tolist()   # 指标名称
    return categories, indicators, df


def _find_indicator(indicators: list[str], keyword: str) -> int | None:
    """按关键词查找指标行号"""
    for i, name in enumerate(indicators):
        if keyword in name:
            return i
    return None


def _parse_timeseries(row: pd.Series) -> dict[str, float]:
    """
    将一行数据解析为 {period: value} 字典。
    period 格式如 "20251231"。
    """
    result = {}
    for col in row.index:
        try:
            val = float(row[col])
            result[str(col)] = val
        except (ValueError, TypeError):
            continue
    return result


def _ttm_eps(eps_by_period: dict[str, float]) -> float | None:
    """
    从季度 EPS 数据计算 TTM（最近四个季度）EPS。

    eps_by_period 包含 "YYYYMMDD": value 格式的键。
    最新年度的年报 EPS 可直接作为 TTM（年报已包含整年）。
    """
    # 取最新四个季度的数据
    periods = sorted(eps_by_period.keys(), reverse=True)

    # 找最近完整年度（1231结尾）
    annual_eps = None
    quarterly_sum = 0.0
    q_count = 0

    for p in periods:
        if p.endswith("1231"):
            annual_eps = eps_by_period[p]
            break

    # 如果是年报数据且是最新的，直接用年报 EPS
    if annual_eps is not None and periods[0].endswith("1231"):
        return annual_eps

    # 否则累计最近4个季度的数据（Q1+Q2+Q3+Q4）
    # 但这里的数据是累计数据，不是单季度数据
    # 比如 20250630 是上半年的累计EPS
    # 所以 TTM = 最新年报 - 上年同期 + 最新累计
    return annual_eps  # fallback to latest annual


def fetch_valuation_indicators(symbol: str, price_df=None) -> dict:
    """
    获取股票最新估值指标。

    通过 stock_financial_abstract 获取 EPS / BVPS，
    结合最新股价计算 PE-TTM / PB。

    返回 dict:
      {
        "pe_ttm": float | None,
        "pb": float | None,
        "ps": float | None,
        "eps_ttm": float | None,
        "bvps": float | None,
        "market_cap": float | None,
        "date": str | None,
      }
    """
    result = {"pe_ttm": None, "pb": None, "ps": None,
              "eps_ttm": None, "bvps": None,
              "market_cap": None, "date": None}

    # ---------- 1. 获取最新收盘价 ----------
    if price_df is None:
        price_df = fetch_daily_price(symbol, start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"))
    if price_df is None or price_df.empty:
        print(f"  [警告] 获取 {symbol} 价格数据失败，无法计算 PE/PB")
        return result

    latest_price = float(price_df["close"].iloc[-1])

    # ---------- 2. 获取财务摘要 ----------
    categories, indicators, df_raw = _parse_abstract(symbol)
    if df_raw is None:
        return result

    # ---------- 3. 提取 EPS ----------
    eps_row_idx = _find_indicator(indicators, INDICATOR_EPS)
    bvps_row_idx = _find_indicator(indicators, INDICATOR_BVPS)
    roe_row_idx = _find_indicator(indicators, INDICATOR_ROE)
    rev_ps_row_idx = _find_indicator(indicators, INDICATOR_REVENUE_PS)

    # 提取 TTM EPS
    if eps_row_idx is not None:
        eps_series = _parse_timeseries(df_raw.iloc[eps_row_idx])
        eps_ttm = _extract_ttm(eps_series)
        result["eps_ttm"] = eps_ttm
        if eps_ttm and eps_ttm > 0:
            result["pe_ttm"] = round(latest_price / eps_ttm, 2)

    # 提取 BVPS
    if bvps_row_idx is not None:
        bvps_series = _parse_timeseries(df_raw.iloc[bvps_row_idx])
        sorted_periods = sorted(bvps_series.keys(), reverse=True)
        for p in sorted_periods:
            latest_bvps = bvps_series[p]
            if latest_bvps and latest_bvps > 0:
                result["bvps"] = round(latest_bvps, 3)
                result["pb"] = round(latest_price / latest_bvps, 2)
                break

    # 提取每股营业收入（PS）
    if rev_ps_row_idx is not None:
        rev_series = _parse_timeseries(df_raw.iloc[rev_ps_row_idx])
        rev_ttm = _extract_ttm(rev_series)
        if rev_ttm and rev_ttm > 0:
            result["ps"] = round(latest_price / rev_ttm, 2)

    # 估算总市值（亿）= 最新收盘价 × 总股本
    # 用 BVPS/EPS 反推：总市值 = PE × 净利润
    # 简单方法：用价格 × 每股收益 / 每股收益率...
    # 从股本角度：如果知道 EPS 和净利润，可以算股本
    # 这里简化：从 akshare 获取总股本信息
    total_shares = _fetch_total_shares(symbol)
    if total_shares and latest_price:
        market_cap_yi = (latest_price * total_shares) / 1e8  # 亿
        result["market_cap"] = round(market_cap_yi, 2)

    return result


def _extract_ttm(series: dict[str, float]) -> float | None:
    """
    从季度时间序列中计算 TTM 值。

    series 格式: {"20251231": 65.66, "20250930": 51.53, "20250630": 36.18, ...}
    数据是累计值（如 20250630 是上半年累计值）。

    TTM = 最近年报值 - 上年同期累计值 + 最近累计值
         = 20241231_full_year - 20240630_half_year + 20250630_half_year
    """
    if not series:
        return None

    periods = sorted(series.keys(), reverse=True)

    # 策略1：最新数据点是年报（1231结尾）→ 直接使用
    if periods[0].endswith("1231"):
        return series[periods[0]]

    # 策略2：计算 TTM
    latest_val = series[periods[0]]

    # 找最近完整年报
    last_annual = None
    last_annual_period = None
    for p in periods:
        if p.endswith("1231"):
            last_annual = series[p]
            last_annual_period = p
            break

    if last_annual is None:
        return latest_val  # 没有年报数据，直接用最新值

    # 找上年同期
    # 如果最新是 20250930，找 20240930
    latest_year = periods[0][:4]
    latest_month = periods[0][4:]
    same_period_last_year = f"{int(latest_year) - 1}{latest_month}"

    same_period_val = series.get(same_period_last_year)
    if same_period_val is not None:
        ttm = last_annual - same_period_val + latest_val
        return round(ttm, 3)

    # 兜底：用最新值
    return latest_val


def _fetch_total_shares(symbol: str) -> float | None:
    """
    获取总股本（股数）。
    使用 stock_zh_a_daily 最后一行的 outstanding_share 字段。
    """
    try:
        import akshare as ak
        prefixed = _symbol_with_prefix(symbol)
        df = ak.stock_zh_a_daily(
            symbol=prefixed,
            start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        if df is not None and not df.empty and "outstanding_share" in df.columns:
            return float(df["outstanding_share"].iloc[-1])
    except Exception:
        pass
    return None


def fetch_pe_pb_history(symbol: str, years: int = 5) -> dict:
    """
    获取历史 PE-TTM / PB 序列（季度频次），用于计算历史分位。

    方法：
      - 从 stock_financial_abstract 提取每个季度末的 EPS、BVPS
      - 从 stock_zh_a_daily 获取对应日期的收盘价
      - PE = Price / EPS（TTM），PB = Price / BVPS

    返回:
      {
        "pe_ttm_list": list[float],
        "pb_list": list[float],
        "pe_ttm_percentile": float | None,
        "pb_percentile": float | None,
      }
    """
    result = {"pe_ttm_list": [], "pb_list": [],
              "pe_ttm_percentile": None, "pb_percentile": None}

    # ---------- 1. 获取财务摘要 ----------
    categories, indicators, df_raw = _parse_abstract(symbol)
    if df_raw is None:
        return result

    eps_row_idx = _find_indicator(indicators, INDICATOR_EPS)
    bvps_row_idx = _find_indicator(indicators, INDICATOR_BVPS)

    if eps_row_idx is None:
        print(f"  [警告] {symbol} 未找到 {INDICATOR_EPS} 指标")
        return result

    # ---------- 2. 解析 EPS / BVPS 时间序列 ----------
    eps_data = _parse_timeseries(df_raw.iloc[eps_row_idx])
    bvps_data = _parse_timeseries(df_raw.iloc[bvps_row_idx]) if bvps_row_idx is not None else {}

    # 只保留最近 years 年的数据
    cutoff_date = datetime.now() - timedelta(days=int(365 * years))
    cutoff_str = cutoff_date.strftime("%Y%m%d")
    eps_data = {k: v for k, v in eps_data.items() if k >= cutoff_str}
    bvps_data = {k: v for k, v in bvps_data.items() if k >= cutoff_str}

    if not eps_data:
        return result

    # ---------- 3. 获取价格数据 ----------
    price_df = fetch_daily_price(symbol)
    if price_df is None or price_df.empty:
        return result

    price_df = price_df.copy()
    price_df["date_str"] = price_df["date"].dt.strftime("%Y%m%d")
    price_map = dict(zip(price_df["date_str"], price_df["close"]))

    # ---------- 4. 计算每个报告期的 TTM EPS 和 PB ----------
    all_periods = sorted(eps_data.keys())

    pe_values = []
    pb_values = []

    for i, period in enumerate(all_periods):
        # 找报告期末最近交易日价格
        price = _find_nearest_price(price_map, period)
        if price is None or price <= 0:
            continue

        # TTM EPS：用 _extract_ttm 但只用到当前 period 为止的数据
        historical_eps = {k: v for k, v in eps_data.items() if k <= period}
        ttm_eps = _extract_ttm(historical_eps)
        if ttm_eps and ttm_eps > 0:
            pe_values.append(round(price / ttm_eps, 4))

        # BVPS
        bvps = bvps_data.get(period)
        if bvps and bvps > 0:
            pb_values.append(round(price / bvps, 4))

    # ---------- 5. 计算分位 ----------
    result["pe_ttm_list"] = pe_values
    result["pb_list"] = pb_values

    if pe_values:
        current_pe = pe_values[-1]
        pct = sum(1 for v in pe_values if v <= current_pe) / len(pe_values) * 100
        result["pe_ttm_percentile"] = round(pct, 1)

    if pb_values:
        current_pb = pb_values[-1]
        pct = sum(1 for v in pb_values if v <= current_pb) / len(pb_values) * 100
        result["pb_percentile"] = round(pct, 1)

    return result


def _find_nearest_price(price_map: dict[str, float], date_str: str) -> float | None:
    """
    在价格字典中查找最接近 date_str 的收盘价。
    先找 exact match，再往前找最多 10 个交易日。
    """
    # Exact match
    if date_str in price_map:
        return price_map[date_str]

    # 往前找（报告日通常在非交易日）
    from datetime import datetime, timedelta
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None

    for offset in range(1, 15):
        prev = (dt - timedelta(days=offset)).strftime("%Y%m%d")
        if prev in price_map:
            return price_map[prev]

    return None


def fetch_roe_history(symbol: str, years: int = 3) -> list[float]:
    """
    获取近 N 年 ROE（净资产收益率）数据。

    从 stock_financial_abstract 提取 净资产收益率(ROE) 指标。
    返回 list[float]：最近 N 年年度 ROE（%）。
    """
    categories, indicators, df_raw = _parse_abstract(symbol)
    if df_raw is None:
        return []

    roe_idx = _find_indicator(indicators, INDICATOR_ROE)
    if roe_idx is None:
        print(f"  [警告] {symbol} 未找到 {INDICATOR_ROE} 指标")
        return []

    roe_data = _parse_timeseries(df_raw.iloc[roe_idx])

    # 只取年报（1231结尾），取最近 years 条
    annual_roe = {k: v for k, v in roe_data.items() if k.endswith("1231")}
    sorted_periods = sorted(annual_roe.keys(), reverse=True)

    roe_list = [annual_roe[p] for p in sorted_periods[:years] if annual_roe[p] is not None]
    return [round(v, 2) for v in roe_list]


def fetch_stock_name(symbol: str) -> str:
    """根据代码获取股票简称。"""
    import akshare as ak

    # 根据代码前缀判断市场：6/9 开头为上海，0/3/2 开头为深圳
    is_shanghai = symbol.strip().startswith(("6", "9"))

    if is_shanghai:
        try:
            df = ak.stock_info_sh_name_code()
            if df is not None and not df.empty:
                # 列名: 证券代码, 证券简称, ...
                match = df[df.iloc[:, 0].astype(str).str.strip() == symbol.strip()]
                if not match.empty:
                    return str(match.iloc[0, 1]).replace(" ", "").replace("　", "")
        except Exception:
            pass
    else:
        try:
            df = ak.stock_info_sz_name_code()
            if df is not None and not df.empty:
                # 列名: 板块, A股代码, A股简称, ...  → 代码在第2列(index 1)
                match = df[df.iloc[:, 1].astype(str).str.strip() == symbol.strip()]
                if not match.empty:
                    return str(match.iloc[0, 2]).replace(" ", "").replace("　", "")   # 简称在第3列(index 2)
        except Exception:
            pass

    # 兜底：用 fetch_daily_price 返回的 df 中的名称字段
    try:
        df = fetch_daily_price(symbol, start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"))
        if df is not None and not df.empty and "name" in df.columns:
            return str(df["name"].iloc[-1])
    except Exception:
        pass

    return symbol


# ----- 内部工具 -----

def _safe_float(row: pd.Series, col: str) -> float | None:
    val = row.get(col)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    symbol = "600519"
    print(f"=== {symbol} 基本面测试 ===\n")

    val = fetch_valuation_indicators(symbol)
    print(f"估值指标: {json.dumps(val, ensure_ascii=False, indent=2)}\n")

    hist = fetch_pe_pb_history(symbol)
    print(f"PE分位: {hist['pe_ttm_percentile']}% (共 {len(hist['pe_ttm_list'])} 期)")
    print(f"PB分位: {hist['pb_percentile']}% (共 {len(hist['pb_list'])} 期)\n")

    roe = fetch_roe_history(symbol)
    print(f"ROE历史: {roe}\n")

    name = fetch_stock_name(symbol)
    print(f"股票名称: {name}")
