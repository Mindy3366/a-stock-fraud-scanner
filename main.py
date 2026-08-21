#!/usr/bin/env python3
"""
财务造假识别 Agent v2.0 — 主入口
升级: 会计勾稽验证 + 商业逻辑验证 + LLM增强分析

用法:
  # 规则引擎模式 (默认，不依赖外部AI)
  python main.py --manual sample_data.json

  # LLM增强模式 (Ollama本地模型)
  python main.py --manual sample_data.json --llm ollama --model qwen3

  # LLM增强模式 (API)
  python main.py --manual sample_data.json --llm openai --model deepseek-chat --apikey sk-xxx

  # 用 akshare 拉取真实数据
  python main.py --stock 600519 --llm ollama
"""
import argparse
import json
import os
import sys

from agent_pipeline import FraudDetectionAgent
from report import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="财务造假识别 Agent v2.0 — 三维度思维链",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --manual sample_data.json                           # 规则引擎
  python main.py --stock 600519                                      # akshare拉取
  python main.py --manual sample_data.json --llm ollama --model qwen3 # LLM增强
  python main.py --stock 600519 --llm openai --model deepseek-chat --apikey sk-xxx
        """
    )
    parser.add_argument("--stock", type=str, help="股票代码")
    parser.add_argument("--source", type=str, default="akshare",
                        choices=["akshare", "tushare", "manual"])
    parser.add_argument("--token", type=str, default="", help="Tushare token")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--industry", type=str, default="", help="行业覆盖 (如: 白酒, 科技, 医药制造)")
    parser.add_argument("--manual", type=str, help="手动输入JSON文件路径")
    parser.add_argument("--output", type=str, default="./output", help="输出目录")
    parser.add_argument("--no-viz", action="store_true", help="跳过可视化")
    parser.add_argument("--skip-realtime", action="store_true", help="跳过实时行情获取（批量回测加速）")
    parser.add_argument("--quiet", action="store_true", help="静默模式")

    # LLM选项
    parser.add_argument("--llm", type=str, default="heuristic",
                        choices=["heuristic", "ollama", "openai"],
                        help="LLM后端 (默认: heuristic 规则引擎, 可选: ollama, openai)")
    parser.add_argument("--model", type=str, default="",
                        help="LLM模型名 (ollama: qwen3/deepseek-r1, openai: deepseek-chat/gpt-4)")
    parser.add_argument("--apikey", type=str, default="",
                        help="API Key (openai模式)")
    parser.add_argument("--apibase", type=str, default="",
                        help="API Base URL (openai兼容模式)")

    args = parser.parse_args()

    if not args.stock and not args.manual:
        parser.print_help()
        print("\n[ERROR] 请提供 --stock 或 --manual 参数")
        sys.exit(1)

    # 加载手动数据
    manual_data = None
    if args.manual:
        if not os.path.exists(args.manual):
            print(f"[ERROR] 文件不存在: {args.manual}")
            sys.exit(1)
        with open(args.manual, "r", encoding="utf-8") as f:
            manual_data = json.load(f)
        stock_code = manual_data.get("stock_code", "MANUAL")
        print(f"[INFO] 加载手动数据: {args.manual}")
    else:
        stock_code = args.stock

    # LLM配置提示
    llm_info = f"LLM: {args.llm}"
    if args.llm != "heuristic":
        llm_info += f" (模型: {args.model or '默认'})"
        if args.llm == "ollama":
            llm_info += " — 确保Ollama已启动且有对应模型"

    print("=" * 60)
    print(f"  财务造假识别 Agent v2.0")
    print(f"  三维度: 失真扫描 + 会计勾稽 + 商业逻辑")
    print(f"  {llm_info}")
    print("=" * 60)
    print()

    # 创建Agent并执行
    if args.quiet:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            agent = FraudDetectionAgent(
                data_source=args.source,
                tushare_token=args.token,
                llm_backend=args.llm,
                llm_model=args.model,
                llm_api_key=args.apikey,
                llm_api_base=args.apibase,
            )
            report = agent.analyze(stock_code, years=args.years, manual_data=manual_data, industry=args.industry, skip_realtime=args.skip_realtime)
    else:
        agent = FraudDetectionAgent(
            data_source=args.source,
            tushare_token=args.token,
            llm_backend=args.llm,
            llm_model=args.model,
            llm_api_key=args.apikey,
            llm_api_base=args.apibase,
        )
        report = agent.analyze(stock_code, years=args.years, manual_data=manual_data, industry=args.industry, skip_realtime=args.skip_realtime)

    # 生成报告
    print()
    print("=" * 60)
    print("  生成报告...")
    print("=" * 60)

    generator = ReportGenerator(report, output_dir=args.output)
    text_path = generator.save_text_report()
    html_path = generator.save_html_report()

    chart_path = None
    if not args.no_viz:
        try:
            chart_path = generator.generate_visualization()
        except Exception as e:
            print(f"[WARN] 可视化生成失败: {e}")

    # 终端摘要
    rp = report.risk_profile
    if rp:
        print()
        print("=" * 60)
        print(f"  分析完成 — {report.company_name} ({report.stock_code})")
        print(f"  风险等级: {rp.risk_band} — {rp.risk_band_cn}")
        print(f"  综合得分: {rp.normalized_score}/100 (原始: {rp.total_score}/200)")
        print(f"  FAIL: {rp.fail_count} | WARN: {rp.warn_count} | PASS: {rp.pass_count}")
        print(f"  三维度: 规则{rp.level_10_count+rp.level_8_count+rp.level_5_count}项 "
              f"+ 勾稽{rp.crosscheck_fail_count}项 + 商业逻辑{rp.bizlogic_fail_count}项")
        if rp.red_flag_count:
            print(f"  红色警报: {rp.red_flag_count}个商业逻辑矛盾")
        print(f"  造假类型: {', '.join(rp.fraud_types[:5]) if rp.fraud_types else '无匹配'}")
        print(f"  置信度: {rp.confidence*100:.0f}%")
        rt = report.realtime_price
        if rt and "error" not in rt:
            sign = "+" if rt.get("change_pct", 0) >= 0 else ""
            print(f"  实时行情: {rt['price']:.2f}元 ({sign}{rt['change_pct']:.2f}%) [仅作参考]")
        if rp.llm_summary:
            print(f"  LLM分析: {rp.llm_summary[:120]}...")
        print("=" * 60)
        print(f"  文本报告: {text_path}")
        print(f"  HTML报告: {html_path}")
        if chart_path:
            print(f"  可视化:   {chart_path}")


if __name__ == "__main__":
    main()
