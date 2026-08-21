# A股财务造假鉴别智能体

基于 53 条规则、三层防御体系的 A 股财务造假鉴别系统，支持单票分析、实时行情、交互式 HTML 报告与批量回测验证。

## 核心能力

- **三层防御**：L10 造假扫描 + L8 利润操纵 + L5 盈余管理（53 条规则）
- **会计勾稽**：10 项跨报表校验
- **商业逻辑**：16 项行业对比与关联交易识别
- **LLM 增强**：heuristic / ollama / openai 三模架构
- **实时行情**：新浪财经实时股价/涨跌幅（涨红跌绿，仅作参考，不纳入评分）
- **交互式报告**：ECharts + Tailwind 生成专业级 HTML 报告
- **批量回测**：80 家公司（20 造假 + 60 正常）自动评估模型有效性

## 回测验证结果

对 80 家 A 股公司（20 家已处罚/退市造假公司 + 60 家行业龙头正常公司）批量回测：

| 指标 | 值 |
|---|---|
| 准确率 Accuracy | 0.850 |
| 召回率 Recall | 0.650 |
| 精确率 Precision | 0.722 |
| F1-Score | 0.684 |
| AUC-ROC | 0.859 |
| 最优判定阈值 | 30 |

混淆矩阵：TP 13 / FP 5 / TN 55 / FN 7

> 注：漏报主要集中在已退市公司（乐视网、康得新等），因 akshare 无法拉全其造假期历史数据；误报多集中在房地产等承压行业的正常公司。

## 安装依赖

```bash
pip install akshare numpy pandas scikit-learn matplotlib
```

## 使用方式

### 单票分析

```bash
# 最新 N 年数据
python scripts/run_demo.py --ticker 600519 --industry 白酒 --years 3 --skip-llm

# 指定年份区间（如回测造假期）
python scripts/run_demo.py --ticker 600518 --industry 医药生物 --start-year 2015 --end-year 2019 --skip-llm

# 主入口
python main.py --stock 600519 --no-viz
```

### 批量回测

```bash
python scripts/batch_backtest.py                # 全部 80 家
python scripts/batch_backtest.py --limit 10     # 冒烟测试前 10 家
python scripts/batch_backtest.py --debug        # 打印每家的命令/返回码/stderr
```

回测逻辑：造假公司按 `fraud_year` 前后各扩 1 年定向抓取造假期数据，正常公司用最新 3 年数据（验证「不误杀」）。

## 输出

- 文本报告：`output/fraud_report_{code}_{date}.txt`
- 交互式 HTML 报告：`output/fraud_report_{code}_{date}.html`
- 可视化图表：`output/fraud_chart_{code}_{date}.png`
- 回测结果：`backtest/results/`（混淆矩阵、分数分布、ROC 曲线、metrics_summary.json）

## 项目结构

```
fraud_detection_agent/
├── agent_pipeline.py        # 思维链管道
├── data_fetcher.py          # 数据采集（akshare / tushare / 手动）
├── rule_engine.py           # 53 条规则引擎
├── accounting_crosscheck.py # 会计勾稽验证
├── business_logic_verify.py # 商业逻辑验证
├── scoring.py               # 评分与风险分级
├── report.py                # 文本/HTML 报告生成
├── main.py                  # 主入口
├── scripts/
│   ├── run_demo.py          # 演示脚本
│   └── batch_backtest.py    # 批量回测
└── backtest/
    ├── companies.csv        # 回测样本（80 家）
    └── results/             # 回测输出
```

## 免责声明

本工具输出仅用于研究参考，不构成投资建议。
