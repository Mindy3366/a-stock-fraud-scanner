#!/usr/bin/env python3
"""
批量回测 — 验证财务造假识别系统的有效性。

读取 backtest/companies.csv，逐条调用 run_demo.py 分析，解析综合风险得分，
计算准确率/召回率/精确率/F1/AUC-ROC，自动搜索最优阈值并生成可视化。

用法:
  python scripts/batch_backtest.py                 # 全部80家
  python scripts/batch_backtest.py --limit 10      # 只跑前10家（冒烟测试）
  python scripts/batch_backtest.py --realtime      # 同时获取实时行情（默认跳过加速）
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent.parent       # 项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent            # scripts/
BACKTEST_DIR = BASE_DIR / "backtest"
RESULTS_DIR = BACKTEST_DIR / "results"
CSV_PATH = BACKTEST_DIR / "companies.csv"
RUN_DEMO = SCRIPT_DIR / "run_demo.py"
RAW_DIR = RESULTS_DIR / "raw_reports"


def load_companies(csv_path):
    """读取公司清单 CSV，返回 dict 列表。兼容 UTF-8 / GBK（Excel 另存）等编码。"""
    text = None
    for enc in ("utf-8-sig", "gbk", "utf-8", "gb18030"):
        try:
            with open(csv_path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    companies = []
    for row in csv.DictReader(text.splitlines()):
        companies.append({
            "code": (row.get("stock_code") or "").strip(),
            "name": (row.get("name") or "").strip(),
            "industry": (row.get("industry") or "").strip(),
            "is_fraud": int((row.get("is_fraud") or "0").strip()),
            "fraud_year": (row.get("fraud_year") or "").strip(),
            "source": (row.get("source") or "").strip(),
        })
    return companies


def parse_output(text):
    """从 run_demo.py 终端输出解析 综合风险得分 / 风险等级 / 置信度。"""
    result = {"score": None, "band": None, "confidence": None}
    m = re.search(r"得分:\s*(\d+)\s*/\s*100", text)
    if m:
        result["score"] = int(m.group(1))
    m = re.search(r"风险等级:\s*([A-Z]+)", text)
    if m:
        result["band"] = m.group(1)
    m = re.search(r"置信度:\s*(\d+)\s*%", text)
    if m:
        result["confidence"] = int(m.group(1))
    return result


def parse_fraud_window(fraud_year, expand=1):
    """从 fraud_year（如 '2016-2018'）解析起止年份，前后各扩 expand 年。"""
    nums = [int(x) for x in re.findall(r"(?:19|20)\d{2}", fraud_year or "")]
    if not nums:
        return None, None
    return min(nums) - expand, max(nums) + expand


def run_one(code, industry, years, realtime, viz, timeout=300, start_year=None, end_year=None):
    """调用 run_demo.py 分析单只股票，返回解析结果（失败时给默认值，不抛异常）。"""
    cmd = [sys.executable, str(RUN_DEMO), "--ticker", code,
           "--industry", industry, "--skip-llm", "--output", str(RAW_DIR)]
    if start_year is not None and end_year is not None:
        cmd += ["--start-year", str(start_year), "--end-year", str(end_year)]
    else:
        cmd += ["--years", str(years)]
    if not realtime:
        cmd.append("--skip-realtime")
    if not viz:
        cmd.append("--no-viz")

    cmd_str = " ".join(cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
        result = parse_output(proc.stdout or "")
        result["returncode"] = proc.returncode
        result["command"] = cmd_str
        result["stderr"] = (proc.stderr or "").strip()
        if result["score"] is None and proc.returncode != 0:
            result["error"] = f"returncode={proc.returncode}"
        return result
    except subprocess.TimeoutExpired:
        return {"score": None, "band": None, "confidence": None, "returncode": None,
                "command": cmd_str, "stderr": "", "error": "timeout"}
    except Exception as e:
        return {"score": None, "band": None, "confidence": None, "returncode": None,
                "command": cmd_str, "stderr": "", "error": str(e)}


def _plot_confusion_matrix(tn, fp, fn, tp, outdir):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["预测 正常", "预测 造假"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["实际 正常", "实际 造假"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]}\n({labels[i][j]})", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_title("混淆矩阵")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_score_distribution(y_true, scores, threshold, outdir):
    fraud_scores = scores[y_true == 1]
    normal_scores = scores[y_true == 0]
    bins = np.arange(0, 105, 5)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(fraud_scores, bins=bins, alpha=0.6, color="#ef4444", label="造假公司", edgecolor="white")
    ax.hist(normal_scores, bins=bins, alpha=0.6, color="#3b82f6", label="正常公司", edgecolor="white")
    ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1.5, label=f"最优阈值={threshold}")
    ax.set_xlabel("综合风险得分")
    ax.set_ylabel("公司数量")
    ax.set_title("风险得分分布（造假 vs 正常）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curve(y_true, scores, auc, outdir):
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    auc_label = f"{auc:.3f}" if np.isfinite(auc) else "N/A"
    ax.plot(fpr, tpr, color="#d32f2f", linewidth=2, label=f"ROC (AUC={auc_label})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.5, label="随机猜测")
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.set_title("ROC 曲线")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(outdir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="财务造假识别系统批量回测")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="公司清单CSV路径")
    parser.add_argument("--years", type=int, default=3, help="分析年数（默认3）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前N家（0=全部）")
    parser.add_argument("--realtime", action="store_true", help="获取实时行情（默认跳过以加速）")
    parser.add_argument("--viz", action="store_true", help="生成单票可视化PNG（默认跳过）")
    parser.add_argument("--timeout", type=int, default=300, help="单票超时秒数（默认300）")
    parser.add_argument("--debug", action="store_true", help="打印每家的命令/返回码/stderr")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    companies = load_companies(args.csv)
    if args.limit > 0:
        companies = companies[:args.limit]

    N = len(companies)
    y_true = []
    scores = []
    results = []

    t_start = time.time()
    for i, c in enumerate(companies, 1):
        # 造假公司按造假年份窗口定向抓取；正常公司用最新数据（验证"不误杀"）
        start_year, end_year = None, None
        if c["is_fraud"] == 1 and c["fraud_year"]:
            start_year, end_year = parse_fraud_window(c["fraud_year"])
        win = f"{start_year}-{end_year}" if start_year else f"近{args.years}年"
        print(f"[{i}/{N}] {c['name']} ({c['code']}) 窗口={win}", flush=True)

        try:
            r = run_one(c["code"], c["industry"], args.years, args.realtime, args.viz, args.timeout,
                        start_year, end_year)
        except Exception as e:
            r = {"score": None, "band": None, "confidence": None, "error": str(e),
                 "command": "", "stderr": ""}

        score = r.get("score") if r.get("score") is not None else 0
        status = "成功" if r.get("score") is not None else "失败"
        print(f"  cmd: {r.get('command') or '(无)'}", flush=True)
        print(f"  -> {status}: score={score} band={r.get('band')} conf={r.get('confidence')} err={r.get('error') or '-'}", flush=True)
        if args.debug and r.get("stderr"):
            print(f"  stderr: {r['stderr'][:300]}", flush=True)

        record = {
            "code": c["code"], "name": c["name"], "industry": c["industry"],
            "is_fraud": c["is_fraud"], "fraud_year": c["fraud_year"], "source": c["source"],
            "fetch_window": f"{start_year}-{end_year}" if start_year else "latest",
            "score": score, "band": r.get("band"), "confidence": r.get("confidence"),
            "error": r.get("error"),
        }
        results.append(record)
        y_true.append(c["is_fraud"])
        scores.append(score)

    y_true = np.array(y_true)
    scores = np.array(scores, dtype=float)

    # ---- 最优阈值搜索（30-90，步长5，F1最高）----
    best_threshold, best_f1 = 50, -1.0
    for th in range(30, 91, 5):
        y_pred = (scores >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, th

    y_pred = (scores >= best_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # ---- 终端打印 ----
    print()
    print("=" * 60)
    print(f"样本总数: {N}")
    print(f"  - 造假公司: {int((y_true == 1).sum())}")
    print(f"  - 正常公司: {int((y_true == 0).sum())}")
    print(f"判定阈值: {best_threshold}（最优）")
    print("-" * 60)
    print(f"准确率:  {acc:.3f}")
    print(f"召回率:  {recall:.3f}")
    print(f"精确率:  {precision:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC-ROC:  {auc:.3f}")
    print("-" * 60)
    print(f"混淆矩阵: TP/{tp} FP/{fp} TN/{tn} FN/{fn}")
    print("=" * 60)

    # ---- 漏报 / 误报 ----
    missed = [r for r in results if r["is_fraud"] == 1 and r["score"] < best_threshold]
    false_pos = [r for r in results if r["is_fraud"] == 0 and r["score"] >= best_threshold]
    print("\n漏报案例列表")
    if missed:
        for r in missed:
            print(f"  - {r['name']} ({r['code']}) 得分={r['score']}")
    else:
        print("  （无）")
    print("\n误报案例列表（前5）")
    if false_pos:
        for r in false_pos[:5]:
            print(f"  - {r['name']} ({r['code']}) 得分={r['score']}")
    else:
        print("  （无）")

    # ---- 标记预测结果 ----
    for r in results:
        r["predicted"] = int(r["score"] >= best_threshold)

    # ---- 保存 JSON ----
    with open(RESULTS_DIR / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": N,
            "fraud_count": int((y_true == 1).sum()),
            "normal_count": int((y_true == 0).sum()),
            "threshold": best_threshold,
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "auc_roc": (None if not np.isfinite(auc) else float(auc)),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        }, f, ensure_ascii=False, indent=2)

    # ---- 生成图表 ----
    _plot_confusion_matrix(tn, fp, fn, tp, RESULTS_DIR)
    _plot_score_distribution(y_true, scores, best_threshold, RESULTS_DIR)
    _plot_roc_curve(y_true, scores, auc, RESULTS_DIR)

    print(f"\n图表与结果已保存至: {RESULTS_DIR}")
    print(f"总耗时: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
