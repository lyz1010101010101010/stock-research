#!/usr/bin/env python3
"""
A 股 估值 + 多维度技术分析 综合研判工具 —— 主入口

用法：
    python main.py                         # 分析 config.py 中配置的全部股票
    python main.py 600519                  # 分析单只股票
    python main.py 600519 000858 000333    # 分析多只股票

输出到控制台，并可选保存到 result_{symbol}.txt 文件。
"""

import sys
import io
import os
from datetime import datetime
from typing import Optional

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ----- Windows 下强制 UTF-8 输出 -----
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfg
from data.fetch_price import fetch_daily_price
from data.fetch_fundamental import fetch_stock_name
from analysis.valuation import analyze_valuation, ValuationResult
from analysis.trend import analyze_trend, TrendResult
from analysis.momentum import analyze_momentum, MomentumResult
from analysis.volume import analyze_volume, VolumeResult
from analysis.structure import analyze_structure, StructureResult
from report import generate_report
from visualize import plot_price_kdj, plot_valuation_percentile


def analyze_stock(symbol: str, stock_name: Optional[str] = None) -> str:
    """
    对单只股票执行全部分析，返回格式化报告字符串。

    参数
    -----
    symbol : str
        股票代码，如 "600519"
    stock_name : str, optional
        股票名称，不传则自动获取

    返回
    -----
    str
        完整的文字报告
    """
    # ---------- 1. 获取股票名称 ----------
    if not stock_name:
        try:
            stock_name = fetch_stock_name(symbol)
        except Exception:
            stock_name = symbol

    print(f"\n{'='*55}")
    print(f"  🔍 正在分析 [{symbol}] {stock_name} ...")
    print(f"{'='*55}")

    # ---------- 2. 获取价格数据 ----------
    df = fetch_daily_price(symbol)
    latest_price = float(df["close"].iloc[-1]) if df is not None and not df.empty and "close" in df.columns else None

    if df is None or df.empty:
        err_msg = (
            f"\n{'─'*50}\n"
            f" [{symbol}] {stock_name or symbol}\n"
            f"  ❌ 获取行情数据失败，跳过本股票\n"
            f"{'─'*50}\n"
        )
        print(err_msg)
        return err_msg

    # ---------- 3. 逐一运行分析模块 ----------
    print(f"   ├─ 估值分析...")
    valuation: ValuationResult = analyze_valuation(symbol)

    print(f"   ├─ 趋势分析...")
    trend: TrendResult = analyze_trend(df)

    print(f"   ├─ 动能分析...")
    momentum: MomentumResult = analyze_momentum(df)

    print(f"   ├─ 量价分析...")
    volume: VolumeResult = analyze_volume(df)

    print(f"   └─ 结构 & 风险分析...")
    structure: StructureResult = analyze_structure(df)

    # ---------- 4. 生成报告 ----------
    report = generate_report(
        symbol=symbol,
        stock_name=stock_name,
        latest_price=latest_price,
        valuation=valuation,
        trend=trend,
        momentum=momentum,
        volume=volume,
        structure=structure,
    )

    # ---------- 5. 生成可视化图表 ----------
    try:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        chart_path = plot_price_kdj(df, symbol, stock_name, trend, momentum, save_dir)
        print(f"   📊 价格走势图 → {os.path.basename(chart_path)}")
        val_path = plot_valuation_percentile(symbol, stock_name, valuation, save_dir)
        print(f"   📊 估值分位图 → {os.path.basename(val_path)}")
    except Exception as e:
        print(f"   ⚠️ 生成可视化图表失败: {e}")

    return report


def main():
    """主函数"""

    # ----- 筛选模式 -----
    if len(sys.argv) > 1 and sys.argv[1] == "--screen":
        from screen import run_screen
        stock_codes = sys.argv[2:] if len(sys.argv) > 2 else []
        run_screen(stock_codes if stock_codes else None)
        return

    # 1. 确定股票列表
    if len(sys.argv) > 1:
        # 从命令行参数读股票代码
        stock_list = [(code, None) for code in sys.argv[1:]]
    else:
        # 从 config.py 读取
        stock_list = cfg.STOCK_LIST

    if not stock_list:
        print("❌ 未指定任何股票代码。请在 config.py 中配置 STOCK_LIST 或通过命令行参数传入。")
        sys.exit(1)

    print(f"\n{'#'*55}")
    print(f"#   A 股 综合研判工具")
    print(f"#   分析股票数: {len(stock_list)}")
    print(f"#   报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*55}")

    # 2. 逐一分析
    all_reports = []
    for symbol, name in stock_list:
        try:
            report = analyze_stock(symbol, stock_name=name)
            all_reports.append((symbol, report))

            # 3. 打印报告
            print(report)
        except KeyboardInterrupt:
            print("\n⚠️  用户中断，退出。")
            sys.exit(0)
        except Exception as e:
            err = (
                f"\n{'─'*50}\n"
                f" [{symbol}] 分析异常: {e}\n"
                f"{'─'*50}\n"
            )
            print(err)
            all_reports.append((symbol, err))

    # 4. 可选：保存到文件
    for symbol, report in all_reports:
        try:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"result_{symbol}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"  [保存] 报告已写入 {filepath}")
        except Exception as e:
            print(f"  [警告] 保存 {symbol} 报告失败: {e}")

    print(f"\n{'='*55}")
    print(f"  全部分析完成！共分析 {len(all_reports)} 只股票")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
