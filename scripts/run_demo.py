#!/usr/bin/env python3
"""
财务造假识别 Agent v2.0 — 演示脚本
支持 --ticker --industry --skip-llm 参数直接传入规则函数

用法:
  python scripts/run_demo.py --ticker 600519 --industry 白酒 --skip-llm
  python scripts/run_demo.py --ticker 000858 --industry 白酒
  python scripts/run_demo.py --manual ../sample_data.json --industry 医药制造
"""
import sys
import os
import argparse
import json

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_pipeline import FraudDetectionAgent
from report import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="财务造假识别 Agent v2.0 Demo")
    parser.add_argument("--ticker", type=str, default="", help="股票代码，如 600519")
    parser.add_argument("--industry", type=str, default="", help="行业覆盖 (如: 白酒, 科技, 医药制造, 新能源)")
    parser.add_argument("--skip-llm", action="store_true", default=True, help="跳过LLM，使用规则引擎")
    parser.add_argument("--llm", type=str, default="heuristic", help="LLM后端 heuristic/ollama/openai")
    parser.add_argument("--model", type=str, default="", help="LLM模型名")
    parser.add_argument("--manual", type=str, default="", help="手动数据JSON文件路径")
    parser.add_argument("--years", type=int, default=5, help="分析年数")
    parser.add_argument("--output", type=str, default="../output", help="输出目录")
    parser.add_argument("--source", type=str, default="akshare", help="数据源")

    args = parser.parse_args()

    if not args.ticker and not args.manual:
        parser.print_help()
        print("\n示例: python scripts/run_demo.py --ticker 600519 --industry 白酒 --skip-llm")
        sys.exit(1)

    # 加载手动数据（如果提供了）
    manual_data = None
    stock_code = args.ticker
    if args.manual:
        manual_path = args.manual
        if not os.path.isabs(manual_path):
            manual_path = os.path.join(os.getcwd(), manual_path)
        if os.path.exists(manual_path):
            with open(manual_path, "r", encoding="utf-8") as f:
                manual_data = json.load(f)
            stock_code = manual_data.get("stock_code", "MANUAL")
        else:
            print(f"[ERROR] 文件不存在: {manual_path}")
            sys.exit(1)

    # 确定LLM后端
    llm_backend = "heuristic" if args.skip_llm else args.llm

    print("=" * 60)
    print(f"  财务造假识别 Agent v2.0 Demo")
    print(f"  Ticker: {args.ticker or '手动数据'}")
    print(f"  Industry: {args.industry or '(自动检测)'}")
    print(f"  LLM: {llm_backend}")
    print("=" * 60)
    print()

    # 创建Agent
    agent = FraudDetectionAgent(
        data_source=args.source,
        llm_backend=llm_backend,
        llm_model=args.model,
    )

    # 执行分析
    report = agent.analyze(
        stock_code=stock_code,
        years=args.years,
        manual_data=manual_data,
        industry=args.industry,  # <-- 行业参数传入规则函数
    )

    # 生成报告
    output_dir = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
    generator = ReportGenerator(report, output_dir=output_dir)
    text_path = generator.save_text_report()

    try:
        chart_path = generator.generate_visualization()
    except Exception as e:
        chart_path = None
        print(f"[WARN] 可视化失败: {e}")

    # 终端摘要
    rp = report.risk_profile
    if rp:
        print()
        print("=" * 60)
        print(f"  结果 — {report.company_name} ({report.stock_code})")
        print(f"  行业: {report.industry}")
        print(f"  风险等级: {rp.risk_band} — {rp.risk_band_cn}")
        print(f"  得分: {rp.normalized_score}/100")
        print(f"  FAIL: {rp.fail_count} | WARN: {rp.warn_count} | PASS: {rp.pass_count}")
        print(f"  RED_FLAG: {rp.red_flag_count}")
        print(f"  造假类型: {', '.join(rp.fraud_types[:5]) if rp.fraud_types else '无'}")
        print(f"  置信度: {rp.confidence*100:.0f}%")
        print(f"  报告: {text_path}")
        if chart_path:
            print(f"  图表: {chart_path}")
        print("=" * 60)

        # 返回 exit code
        if rp.normalized_score >= 35:
            print(f"\n[WARN] 高风险警告: score={rp.normalized_score}")
        else:
            print(f"\n[OK] 低风险通过: score={rp.normalized_score}")


if __name__ == "__main__":
    main()
