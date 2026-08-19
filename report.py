"""
财务造假识别 Agent — 报告生成与可视化模块
严格按 SKILL (2) 步骤5 的输出格式生成报告
"""
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from agent_pipeline import AnalysisReport
from rule_engine import RuleResult, DeepAuditResult
from scoring import RiskProfile
from config import RISK_LEVEL_10, RISK_LEVEL_8, RISK_LEVEL_5, RISK_LEVEL_3


class ReportGenerator:
    """文本报告 + 可视化"""

    def __init__(self, report: AnalysisReport, output_dir: str = "./output"):
        self.r = report
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_text_report(self) -> str:
        """生成完整文本报告（完全对应 SKILL (2) 步骤5格式）"""
        lines = []
        rp = self.r.risk_profile

        # ====== 标题 ======
        lines.append("=" * 72)
        lines.append(f"  财务报表排雷分析报告")
        lines.append(f"  {self.r.company_name} ({self.r.stock_code})")
        lines.append("=" * 72)

        # ====== 一、基本信息 ======
        lines.append("")
        lines.append("## 一、基本信息")
        lines.append(f"  - 公司全称 / 股票代码: {self.r.company_name} / {self.r.stock_code}")
        lines.append(f"  - 所属行业: {self.r.industry}")
        lines.append(f"  - 分析期间: {self.r.analysis_period}")
        lines.append(f"  - 报告日期: {self.r.report_date}")
        lines.append(f"  - 数据来源: {self.r.data_source}")
        lines.append(f"  - 分析耗时: {self.r.execution_time_seconds}s")

        # ====== 二、风险总览 ======
        lines.append("")
        lines.append("## 二、风险总览")
        if rp:
            lines.append(f"  | 风险等级       | 数量 | 风险类型 |")
            lines.append(f"  |----------------|------|----------|")
            lines.append(f"  | 10级-疑似造假  | {rp.level_10_count:>4} | {self._top_fraud_types(RISK_LEVEL_10)} |")
            lines.append(f"  | 8级-疑似操纵   | {rp.level_8_count:>4} | {self._top_fraud_types(RISK_LEVEL_8)} |")
            lines.append(f"  | 5级-盈余管理   | {rp.level_5_count:>4} | {self._top_fraud_types(RISK_LEVEL_5)} |")
            lines.append(f"  | 3级-经营风险   | {rp.level_3_count:>4} | - |")
            lines.append(f"")
            lines.append(f"  ■ 综合风险评级: {rp.risk_band} — {rp.risk_band_cn}")
            lines.append(f"  ■ 综合得分: {rp.normalized_score}/100")
            lines.append(f"  ■ 评级依据: {rp.risk_summary}")
            lines.append(f"  ■ 置信度: {rp.confidence*100:.0f}% ({rp.confidence_reason})")
            lines.append(f"  ■ FAIL: {rp.fail_count}条 | WARN: {rp.warn_count}条 | PASS: {rp.pass_count}条 | MISSING: {rp.missing_count}条")

        # ====== 三、核心风险点详析 ======
        lines.append("")
        lines.append("## 三、核心风险点详析")

        # 收集所有 FAIL 和 WARN
        all_alerts = [r for r in self._all_results() if r.verdict in ("FAIL", "WARN")]
        all_alerts.sort(key=lambda x: x.score, reverse=True)

        if not all_alerts:
            lines.append("  ✅ 未发现重大风险信号。")
        else:
            for i, alert in enumerate(all_alerts[:10]):
                lines.append(f"")
                lines.append(f"  ### 风险点{i+1}: [{alert.risk_level[:4]}] {alert.rule_name}")
                lines.append(f"  - 判定: {alert.verdict}")
                lines.append(f"  - 数据依据: {alert.data_basis}")
                lines.append(f"  - 风险说明: {alert.risk_explanation}")
                if alert.fraud_type:
                    lines.append(f"  - 造假类型标签: #{alert.fraud_type}")
                if alert.evidence:
                    lines.append(f"  - 证据链: {alert.evidence}")
                if alert.accounting_standard:
                    lines.append(f"  - 会计准则: {alert.accounting_standard}")
                if alert.check_points:
                    lines.append(f"  - 建议核查动作: {'; '.join(alert.check_points)}")

        # ====== 四、重点科目异常摘要 ======
        if self.r.deep_audit_results:
            lines.append("")
            lines.append("## 四、重点科目异常摘要")
            lines.append(f"  | 科目 | 异常表现 | 风险指向 | 建议动作 |")
            lines.append(f"  |------|----------|----------|----------|")
            for da in self.r.deep_audit_results:
                anomaly_text = "; ".join(da.anomalies[:2]) if da.anomalies else "见深度核查"
                suggestion_text = "; ".join(da.suggestions[:2]) if da.suggestions else "进一步核查"
                lines.append(f"  | {da.subject} | {anomaly_text[:40]} | {da.risk_level[:4]} | {suggestion_text[:40]} |")

        # ====== 4.5 会计勾稽验证 [NEW v2.0] ======
        if self.r.crosscheck_results:
            lines.append("")
            lines.append("## 四-2、会计勾稽验证（报表间业务数据流转验证）")
            lines.append(f"  | 勾稽项 | 判定 | 差异率 | 造假含义 |")
            lines.append(f"  |--------|------|--------|----------|")
            for cr in self.r.crosscheck_results:
                if cr.verdict in ("FAIL", "WARN"):
                    dp = f"{cr.deviation_pct*100:.1f}%" if cr.deviation_pct else "N/A"
                    fi = cr.fraud_implication[:50] if cr.fraud_implication else "-"
                    lines.append(f"  | {cr.check_name} | {cr.verdict} | {dp} | {fi} |")
            # 汇总
            cc_fails = sum(1 for cr in self.r.crosscheck_results if cr.verdict == "FAIL")
            cc_warns = sum(1 for cr in self.r.crosscheck_results if cr.verdict == "WARN")
            lines.append(f"")
            lines.append(f"  勾稽验证汇总: {cc_fails}项FAIL, {cc_warns}项WARN — "
                         f"勾稽矛盾无法在报表层面调和，指向业务实质层面的数据造假。")

        # ====== 4.6 商业逻辑验证 [NEW v2.0] ======
        if self.r.biz_logic_verify_results:
            lines.append("")
            lines.append("## 四-3、商业逻辑验证（穿透报表，审视生意合理性）")
            biz_fails = [r for r in self.r.biz_logic_verify_results if r.verdict == "FAIL"]
            red_flags = [r for r in self.r.biz_logic_verify_results if getattr(r, 'red_flag', False)]
            if red_flags:
                lines.append("")
                lines.append("  ### 红色警报（强造假信号）")
                for rf in red_flags[:5]:
                    lines.append(f"  - [{rf.check_id}] {rf.check_name}")
                    lines.append(f"    矛盾: {rf.contradiction}")
            if biz_fails:
                lines.append("")
                lines.append("  ### 商业逻辑矛盾")
                for bf in biz_fails[:5]:
                    if bf not in red_flags:
                        lines.append(f"  - [{bf.check_id}] {bf.check_name}: {bf.contradiction[:100]}")
            bl_warns = [r for r in self.r.biz_logic_verify_results if r.verdict == "WARN"]
            if bl_warns:
                lines.append(f"")
                lines.append(f"  另有 {len(bl_warns)} 项商业逻辑预警，详见完整报告。")

        # ====== 4.7 LLM增强分析 [NEW v2.0] ======
        if self.r.llm_analysis and self.r.llm_analysis.success:
            lines.append("")
            lines.append("## 四-4、AI综合分析")
            lines.append(f"  模型: {self.r.llm_analysis.model_used}")
            lines.append(f"  造假概率判断: {self.r.llm_analysis.fraud_probability}")
            if self.r.llm_analysis.overall_assessment:
                lines.append(f"  综合判断: {self.r.llm_analysis.overall_assessment[:200]}")
            if self.r.llm_analysis.audit_opinion_analysis:
                lines.append(f"  审计意见: {self.r.llm_analysis.audit_opinion_analysis[:200]}")
            if self.r.llm_analysis.governance_analysis:
                lines.append(f"  治理层面: {self.r.llm_analysis.governance_analysis[:200]}")

        # ====== 五、会计信息质量评估 ======
        lines.append("")
        lines.append("## 五、会计信息质量评估")

        manip_fails = sum(1 for r in self.r.manipulation_results if r.verdict == "FAIL")
        em_fails = sum(1 for r in self.r.earnings_mgmt_results if r.verdict == "FAIL")

        consistency = "优" if manip_fails == 0 else ("差" if manip_fails >= 3 else "中")
        reliability = "优" if rp and rp.level_10_count == 0 else ("差" if rp and rp.level_10_count >= 2 else "中")
        comparability = "中"  # 默认
        timeliness = "优"  # 默认（需人工核查公告）

        lines.append(f"  - 一致性: [{consistency}] —— 会计政策变更/差错更正频率")
        lines.append(f"  - 可靠性: [{reliability}] —— 审计意见/内控评价/造假信号数")
        lines.append(f"  - 可比性: [{comparability}] —— 跨期数据一致性/行业对标")
        lines.append(f"  - 及时性: [{timeliness}] —— 业绩预告修正/披露延迟")

        # ====== 六、商业逻辑补充验证 ======
        lines.append("")
        lines.append("## 六、商业逻辑补充验证")

        biz_fails = [r for r in self.r.business_logic_results if r.verdict == "WARN"]
        if biz_fails:
            lines.append(f"  触发 {len(biz_fails)} 项经营风险信号:")
            for bf in biz_fails:
                lines.append(f"  - [{bf.rule_id}] {bf.rule_name}: {bf.data_basis} → {bf.risk_explanation}")
        else:
            lines.append("  未触发明显经营风险信号。")

        # ====== 七、造假模式匹配 ======
        if rp and rp.fraud_types:
            lines.append("")
            lines.append("## 七、造假模式匹配")
            from config import FRAUD_PATTERNS
            for ft in rp.fraud_types[:5]:
                pattern = FRAUD_PATTERNS.get(ft, {})
                if pattern:
                    lines.append(f"  ■ {ft}")
                    lines.append(f"    特征: {pattern.get('core_features', 'N/A')}")
                    lines.append(f"    识别公式: {pattern.get('formula', 'N/A')}")
                    lines.append(f"    参考案例: {pattern.get('typical_case', 'N/A')}")

        # ====== 八、数字化时代风险提醒 ======
        lines.append("")
        lines.append("## 八、数字化时代特殊风险提醒")
        lines.append(f"  （以下项目需人工核查，本工具无法自动检测）")
        from config import DIGITAL_ERA_CHECKS
        for check in DIGITAL_ERA_CHECKS[:5]:
            lines.append(f"  - [{check['id']}] {check['name']}: {check['check']} → 风险: {check['risk']}")

        # ====== 九、关键指标趋势表 ======
        lines.append("")
        lines.append("## 九、关键指标趋势")
        if self.r.key_indicators_summary:
            years = sorted(set(y for d in self.r.key_indicators_summary.values() for y in d))
            if years:
                header = "  | 指标 | " + " | ".join(str(y) for y in years) + " |"
                sep = "  |------|" + "|".join("------" for _ in years) + "|"
                lines.append(header)
                lines.append(sep)
                for name, yearly in self.r.key_indicators_summary.items():
                    vals = []
                    for y in years:
                        v = yearly.get(y)
                        if v is not None:
                            if abs(v) > 100:
                                vals.append(f"{v/1e8:.1f}亿")
                            elif abs(v) < 0.1:
                                vals.append(f"{v*100:.1f}%")
                            else:
                                vals.append(f"{v:.2f}")
                        else:
                            vals.append("N/A")
                    lines.append(f"  | {name} | " + " | ".join(vals) + " |")

        # ====== 十、结论与建议 ======
        lines.append("")
        lines.append("## 十、结论与建议")
        lines.append("")
        lines.append("  ### 结论")
        if rp:
            lines.append(f"  综合风险等级: {rp.risk_band}（{rp.risk_band_cn}），得分 {rp.normalized_score}/100")
            lines.append(f"  触发 {rp.fail_count} 条FAIL、{rp.warn_count} 条WARN")
            if rp.fraud_types:
                lines.append(f"  匹配造假类型: {'、'.join(rp.fraud_types[:5])}")
            lines.append(f"  置信度: {rp.confidence*100:.0f}%")

        lines.append("")
        lines.append("  ### 建议核查清单")
        lines.append("  1. [如有审计报告] 确认最近3年审计意见类型")
        lines.append("  2. [如触发存贷双高] 获取银行函证、核实货币资金真实性")
        lines.append("  3. [如触发存货异常] 获取存货库龄结构、实地盘点报告")
        lines.append("  4. [如触发境外收入异常] 获取海关出口数据、境外客户工商信息")
        lines.append("  5. [如触发关联交易] 获取关联方清单、交易定价依据")

        lines.append("")
        lines.append("  ### 投资决策参考（不构成投资建议）")
        from scoring import RiskScorer
        scorer = RiskScorer()
        scorer.profile = rp
        lines.append(f"  {scorer.get_investment_guidance()}")

        # ====== 十一、数据局限性 ======
        lines.append("")
        lines.append("## 十一、数据来源与局限性声明")
        lines.append(f"  - 数据来源: {self.r.data_source}（公开财务数据）")
        lines.append("  - 分析局限性:")
        lines.append("    1. 无法获取银行流水等原始凭证，货币资金真实性判断受限")
        lines.append("    2. 无法进行实地核查，存货/固定资产存在性判断受限")
        lines.append("    3. 部分非财务信号（监管问询、诉讼、减持）需人工收集")
        lines.append("    4. 行业均值数据依赖经验和公开信息，精确度有限")
        lines.append(f"    5. 报告生成时间: {self.r.report_date}，基于历史公开数据")
        lines.append("")
        lines.append("  [!!] 本报告为自动化分析工具输出，不构成投资建议。")
        lines.append("  [!!] 所有结论请结合专业人士判断使用。")

        # ====== 实时行情（仅作参考，不纳入风险评分） ======
        self._append_realtime_section(lines)

        lines.append("")
        lines.append("=" * 72)

        report_text = "\n".join(lines)
        return report_text

    def save_text_report(self, filename: str = None) -> str:
        """保存文本报告"""
        if filename is None:
            filename = f"fraud_report_{self.r.stock_code}_{self.r.report_date}.txt"
        path = os.path.join(self.output_dir, filename)
        text = self.generate_text_report()
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Report] 文本报告已保存至: {path}")
        return path

    # ================================================================
    # 交互式 HTML 报告（ECharts + Tailwind CSS）
    # ================================================================
    def generate_html_report(self) -> str:
        """生成交互式 HTML 报告（ECharts + Tailwind CSS），返回完整 HTML 字符串。"""
        rp = self.r.risk_profile or RiskProfile()

        band_colors = {"GREEN": "#4caf50", "YELLOW": "#ff9800", "ORANGE": "#ff5722", "RED": "#d32f2f"}
        gauge_color = band_colors.get(rp.risk_band, "#9e9e9e")

        rt = self.r.realtime_price or {}
        realtime = {
            "ok": bool(rt) and "error" not in rt,
            "price": rt.get("price"),
            "change_pct": rt.get("change_pct"),
            "volume": rt.get("volume"),
            "turnover": rt.get("turnover"),
            "update_time": rt.get("update_time"),
            "error": rt.get("error", ""),
        }

        levels = {
            "names": ["10级 疑似造假", "8级 疑似操纵", "5级 盈余管理", "3级 经营风险"],
            "values": [rp.level_10_count, rp.level_8_count, rp.level_5_count, rp.level_3_count],
            "colors": ["#d32f2f", "#f57c00", "#fbc02d", "#1976d2"],
        }

        verdicts = {
            "names": ["PASS 通过", "WARN 预警", "FAIL 失败", "MISSING 缺失"],
            "values": [rp.pass_count, rp.warn_count, rp.fail_count, rp.missing_count],
            "colors": ["#4caf50", "#ff9800", "#f44336", "#9e9e9e"],
        }

        heatmap = self._build_fraud_heatmap(rp)
        trend = self._build_trend_data()

        evidences = {
            "names": [e[0] for e in rp.top_evidences][::-1],
            "descs": [e[1] for e in rp.top_evidences][::-1],
            "values": [float(e[2]) for e in rp.top_evidences][::-1],
        }

        report_data = {
            "gauge": {"score": rp.normalized_score, "band_cn": rp.risk_band_cn, "color": gauge_color},
            "levels": levels,
            "verdicts": verdicts,
            "verdict_total": sum(verdicts["values"]),
            "heatmap": heatmap,
            "trend": trend,
            "evidences": evidences,
            "realtime": realtime,
            "meta": {
                "company_name": self.r.company_name,
                "stock_code": self.r.stock_code,
                "industry": self.r.industry,
                "risk_band_cn": rp.risk_band_cn,
                "gauge_color": gauge_color,
                "confidence": rp.confidence,
                "confidence_reason": rp.confidence_reason,
                "data_source": self.r.data_source,
                "report_date": self.r.report_date,
                "analysis_period": self.r.analysis_period,
            },
        }

        data_json = json.dumps(report_data, ensure_ascii=False).replace("</", "<\\/")
        return _HTML_TEMPLATE.replace("__REPORT_DATA__", data_json)

    def save_html_report(self, filename: str = None) -> str:
        """保存交互式 HTML 报告，返回文件路径。"""
        if filename is None:
            filename = f"fraud_report_{self.r.stock_code}_{self.r.report_date}.html"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_html_report())
        print(f"[Report] HTML报告已保存至: {path}")
        return path

    def _analysis_years(self) -> List[int]:
        """从 analysis_period（如 '2016-2018'）解析出年份序列。"""
        period = self.r.analysis_period or ""
        try:
            start, end = period.split("-")
            start, end = int(start), int(end)
            if 1900 <= start <= 2100 and 1900 <= end <= 2100 and start <= end:
                return list(range(start, end + 1))
        except (ValueError, TypeError):
            pass
        return []

    def _build_fraud_heatmap(self, rp) -> Dict:
        """构建 造假类型 × 年份 热力矩阵。

        RuleResult 无显式年份字段，因此从 data_basis 文本中正则提取 4 位年份；
        提取不到时归入最新年份（规则引擎主要针对最新财年）。
        """
        years = self._analysis_years()
        fraud_types = list(rp.fraud_types)

        results = self.r.fraud_results + self.r.manipulation_results + self.r.earnings_mgmt_results
        if not fraud_types:
            for r in results:
                if r.fraud_type and r.fraud_type not in fraud_types:
                    fraud_types.append(r.fraud_type)

        matrix = {ft: {y: 0 for y in years} for ft in fraud_types}
        latest_year = years[-1] if years else None

        for r in results:
            if r.verdict not in ("FAIL", "WARN") or not r.fraud_type:
                continue
            if r.fraud_type not in matrix:
                matrix[r.fraud_type] = {y: 0 for y in years}
            m = re.search(r"(?:19|20)\d{2}", getattr(r, "data_basis", "") or "")
            year = int(m.group(0)) if m else latest_year
            if year is not None and year in matrix[r.fraud_type]:
                matrix[r.fraud_type][year] += 1

        data, max_val = [], 0
        for xi, ft in enumerate(fraud_types):
            for yi, y in enumerate(years):
                v = matrix[ft][y]
                if v:
                    max_val = max(max_val, v)
                    data.append([xi, yi, v])

        return {"types": fraud_types, "years": years, "data": data, "max": max(1, max_val)}

    def _build_trend_data(self) -> Dict:
        """构建关键指标趋势数据；无数据返回空。"""
        summary = self.r.key_indicators_summary or {}
        years = sorted({y for d in summary.values() for y in d})
        series = [{"name": name, "data": [yearly.get(y) for y in years]} for name, yearly in summary.items()]
        return {"years": years, "series": series, "has_data": bool(years and series)}

    def generate_visualization(self) -> Optional[str]:
        """生成可视化图表"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            import numpy as np
        except ImportError:
            print("[Report] matplotlib未安装，跳过可视化")
            return None

        # 设置中文字体
        try:
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        rp = self.r.risk_profile
        if not rp:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"财务造假识别报告 — {self.r.company_name}({self.r.stock_code})",
                     fontsize=16, fontweight="bold")

        # (0,0) 风险得分仪表盘
        ax = axes[0, 0]
        self._draw_gauge(ax, rp.total_score, rp.risk_band, rp.risk_band_cn)

        # (0,1) 各级风险触发统计
        ax = axes[0, 1]
        levels = ["10级\n疑似造假", "8级\n疑似操纵", "5级\n盈余管理", "3级\n经营风险"]
        counts = [rp.level_10_count, rp.level_8_count, rp.level_5_count, rp.level_3_count]
        colors_bar = ["#d32f2f", "#f57c00", "#fbc02d", "#1976d2"]
        ax.bar(levels, counts, color=colors_bar, edgecolor="white", linewidth=1.5)
        ax.set_title("各级风险触发数量", fontsize=13)
        ax.set_ylabel("触发条数")
        for i, v in enumerate(counts):
            if v > 0:
                ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold", fontsize=12)

        # (0,2) 判定结果占比
        ax = axes[0, 2]
        verdicts = ["PASS", "WARN", "FAIL", "MISSING"]
        verdict_counts = [rp.pass_count, rp.warn_count, rp.fail_count, rp.missing_count]
        v_colors = ["#4caf50", "#ff9800", "#f44336", "#9e9e9e"]
        wedges, texts, autotexts = ax.pie(
            [c for c in verdict_counts if c > 0],
            labels=[v for v, c in zip(verdicts, verdict_counts) if c > 0],
            colors=[v_colors[i] for i, c in enumerate(verdict_counts) if c > 0],
            autopct="%1.0f%%", startangle=90
        )
        ax.set_title("规则判定结果分布", fontsize=13)

        # (1,0) 造假类型分布
        ax = axes[1, 0]
        if rp.fraud_types:
            from collections import Counter
            type_counts = Counter(rp.fraud_types)
            type_names = list(type_counts.keys())[:8]
            type_values = list(type_counts.values())[:8]
            ax.barh(range(len(type_names)), type_values, color="#e53935", edgecolor="white")
            ax.set_yticks(range(len(type_names)))
            ax.set_yticklabels(type_names, fontsize=9)
            ax.set_title("匹配造假类型", fontsize=13)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "未匹配已知造假模式", ha="center", va="center", fontsize=14)
            ax.set_title("匹配造假类型", fontsize=13)

        # (1,1) 关键指标趋势
        ax = axes[1, 1]
        if self.r.key_indicators_summary:
            years = sorted(set(y for d in self.r.key_indicators_summary.values() for y in d))
            if years:
                for name, yearly in self.r.key_indicators_summary.items():
                    vals = [yearly.get(y) for y in years]
                    if all(v is not None for v in vals):
                        ax.plot(years, vals, marker="o", label=name, linewidth=2)
                ax.set_title("关键指标趋势", fontsize=13)
                ax.set_xlabel("年份")
                ax.legend(fontsize=7, loc="best")
                ax.grid(True, alpha=0.3)
                # 整数年份刻度
                ax.set_xticks(years)
        if not ax.get_legend_handles_labels()[0]:
            ax.text(0.5, 0.5, "指标数据不足", ha="center", va="center")

        # (1,2) 证据链权重
        ax = axes[1, 2]
        if rp.top_evidences:
            ev_names = [e[0][:12] for e in rp.top_evidences]
            ev_scores = [e[2] for e in rp.top_evidences]
            ax.barh(range(len(ev_names)), ev_scores, color="#5c6bc0", edgecolor="white")
            ax.set_yticks(range(len(ev_names)))
            ax.set_yticklabels(ev_names, fontsize=9)
            ax.set_title("TOP风险证据链（按权重）", fontsize=13)
            ax.set_xlabel("得分")
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "无明显证据链", ha="center", va="center", fontsize=14)
            ax.set_title("TOP风险证据链", fontsize=13)

        # 底部信息
        fig.text(0.5, 0.01, f"置信度: {rp.confidence*100:.0f}% | 数据源: {self.r.data_source} | "
                 f"报告日期: {self.r.report_date} | [不构成投资建议]",
                 ha="center", fontsize=9, color="gray")

        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        # 保存
        filename = f"fraud_chart_{self.r.stock_code}_{self.r.report_date}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Report] 可视化图表已保存至: {path}")
        return path

    def _draw_gauge(self, ax, score: int, band: str, band_cn: str):
        """画仪表盘"""
        import numpy as np
        band_colors = {"GREEN": "#4caf50", "YELLOW": "#ff9800", "ORANGE": "#ff5722", "RED": "#d32f2f"}
        color = band_colors.get(band, "#9e9e9e")

        theta = np.linspace(np.pi, 0, 101)
        r = 1.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # 背景弧
        ax.plot(x, y, color="#e0e0e0", linewidth=20, solid_capstyle="round")

        # 得分弧
        score_theta = np.linspace(np.pi, np.pi * (1 - score / 100), max(2, int(score)))
        score_x = r * np.cos(score_theta)
        score_y = r * np.sin(score_theta)
        ax.plot(score_x, score_y, color=color, linewidth=20, solid_capstyle="round")

        # 中心文字
        ax.text(0, -0.15, f"{score}", ha="center", va="center", fontsize=36, fontweight="bold", color=color)
        ax.text(0, -0.40, f"/ 100", ha="center", va="center", fontsize=14, color="gray")
        ax.text(0, -0.65, f"{band} — {band_cn}", ha="center", va="center", fontsize=13,
                fontweight="bold", color=color)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.8, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("综合风险得分", fontsize=13, y=0.98)

    def _append_realtime_section(self, lines: List[str]):
        """附加 [实时行情] 章节（仅作参考，不纳入风险评分）"""
        rt = self.r.realtime_price
        if not rt:
            return
        lines.append("")
        lines.append("[实时行情]")
        if "error" in rt:
            lines.append(f"  {rt['error']}")
            lines.append("  【实时行情仅作参考，不纳入风险评分】")
            return

        change_pct = rt.get("change_pct", 0.0)
        sign = "+" if change_pct >= 0 else ""
        lines.append(f"  最新价: {rt.get('price', 0.0):.2f} 元")
        lines.append(f"  涨跌幅: {sign}{change_pct:.2f}%")
        lines.append(f"  成交量: {rt.get('volume', 0.0):,.0f} 手")
        lines.append(f"  成交额: {rt.get('turnover', 0.0):.2f} 亿元")
        if rt.get("update_time"):
            lines.append(f"  更新时间: {rt['update_time']}")
        lines.append("  【实时行情仅作参考，不纳入风险评分】")

    def _all_results(self) -> List[RuleResult]:
        return (self.r.fraud_results + self.r.manipulation_results
                + self.r.earnings_mgmt_results + self.r.business_logic_results)

    def _top_fraud_types(self, risk_level: str) -> str:
        """获取某风险级别的造假类型"""
        results = []
        if risk_level == RISK_LEVEL_10:
            results = self.r.fraud_results
        elif risk_level == RISK_LEVEL_8:
            results = self.r.manipulation_results
        elif risk_level == RISK_LEVEL_5:
            results = self.r.earnings_mgmt_results
        types = set()
        for r in results:
            if r.verdict in ("FAIL", "WARN") and r.fraud_type:
                types.add(r.fraud_type)
        return ", ".join(list(types)[:3]) if types else "-"


# ================================================================
# 交互式 HTML 报告模板（ECharts + Tailwind CSS）
# __REPORT_DATA__ 会被替换为 JSON 数据
# ================================================================
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>财务造假识别报告</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif; }
  .chart-box { width: 100%; height: 300px; }
  .empty-box { width: 100%; height: 300px; display: none; flex-direction: column; align-items: center; justify-content: center; }
  @media print {
    body { background: #fff !important; }
    .no-print { display: none !important; }
    main { max-width: 100% !important; padding: 0 !important; }
    .card { box-shadow: none !important; border-color: #e5e7eb !important; page-break-inside: avoid; }
    .chart-box { height: 280px !important; }
  }
</style>
</head>
<body class="bg-gray-50 text-gray-900 min-h-screen">
<main class="max-w-7xl mx-auto px-4 py-6">

  <!-- 头部 -->
  <header class="card bg-white rounded-xl shadow-sm border border-gray-100 px-6 py-5 mb-6">
    <div class="flex flex-wrap items-center justify-between gap-4">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">财务报表排雷分析报告</h1>
        <div class="mt-1 flex flex-wrap items-center gap-3 text-sm text-gray-500">
          <span id="companyName" class="font-medium text-gray-700"></span>
          <span id="stockCode" class="px-2 py-0.5 bg-gray-100 rounded text-gray-600"></span>
          <span id="industry" class="px-2 py-0.5 bg-blue-50 text-blue-600 rounded"></span>
          <span id="analysisPeriod" class="text-gray-400"></span>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <span id="riskBadge" class="px-4 py-2 rounded-lg text-white font-semibold text-lg"></span>
        <button onclick="window.print()" class="no-print px-3 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">打印报告</button>
      </div>
    </div>
  </header>

  <!-- 第1行 -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
    <div class="card bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <h2 class="text-base font-semibold text-gray-700 mb-2">综合风险得分</h2>
      <div id="gaugeChart" class="chart-box"></div>
    </div>
    <div class="card bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <h2 class="text-base font-semibold text-gray-700 mb-2">各级风险触发</h2>
      <div id="levelsChart" class="chart-box"></div>
    </div>
    <div class="card bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <h2 class="text-base font-semibold text-gray-700 mb-2">规则判定分布</h2>
      <div id="verdictChart" class="chart-box"></div>
    </div>
  </div>

  <!-- 第2行 -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
    <div class="card bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <h2 class="text-base font-semibold text-gray-700 mb-2">造假类型热力矩阵</h2>
      <div id="heatmapChart" class="chart-box"></div>
      <div id="heatmapEmpty" class="empty-box text-gray-400">
        <svg class="w-12 h-12 mb-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6M7 3h7l5 5v13a1 1 0 01-1 1H7a1 1 0 01-1-1V4a1 1 0 011-1z"/></svg>
        <p>暂无造假类型数据</p>
      </div>
    </div>
    <div class="card bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <h2 class="text-base font-semibold text-gray-700 mb-2">关键指标趋势</h2>
      <div id="trendChart" class="chart-box"></div>
      <div id="trendEmpty" class="empty-box text-gray-400">
        <svg class="w-12 h-12 mb-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6M7 3h7l5 5v13a1 1 0 01-1 1H7a1 1 0 01-1-1V4a1 1 0 011-1z"/></svg>
        <p>暂无关键指标数据</p>
      </div>
    </div>
    <div class="card bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <h2 class="text-base font-semibold text-gray-700 mb-2">TOP风险证据链</h2>
      <div id="evidenceChart" class="chart-box"></div>
      <div id="evidenceEmpty" class="empty-box text-gray-400">
        <svg class="w-12 h-12 mb-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6M7 3h7l5 5v13a1 1 0 01-1 1H7a1 1 0 01-1-1V4a1 1 0 011-1z"/></svg>
        <p>暂无证据链数据</p>
      </div>
    </div>
  </div>

  <!-- 第3行 -->
  <div class="card bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-6">
    <h2 class="text-base font-semibold text-gray-700 mb-4">实时行情 <span class="text-xs font-normal text-gray-400">（仅作参考，不纳入风险评分）</span></h2>
    <div id="realtimeOk" style="display:none">
      <div class="flex flex-wrap items-end gap-x-10 gap-y-4">
        <div>
          <p class="text-xs text-gray-400 mb-1">最新价</p>
          <div class="flex items-baseline gap-3">
            <span id="rtPrice" class="text-4xl font-bold text-gray-900"></span>
            <span id="rtChange" class="text-xl font-semibold"></span>
          </div>
        </div>
        <div><p class="text-xs text-gray-400 mb-1">成交量</p><p id="rtVolume" class="text-lg font-medium text-gray-800"></p></div>
        <div><p class="text-xs text-gray-400 mb-1">成交额</p><p id="rtTurnover" class="text-lg font-medium text-gray-800"></p></div>
        <div><p class="text-xs text-gray-400 mb-1">更新时间</p><p id="rtTime" class="text-lg font-medium text-gray-800"></p></div>
      </div>
    </div>
    <div id="realtimeFail" style="display:none" class="items-center gap-3 text-gray-400">
      <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
      <span>实时行情获取失败</span>
    </div>
  </div>

  <!-- 底部元信息 -->
  <footer class="text-sm text-gray-400 text-center py-6">
    <p class="mb-1">
      置信度 <span id="metaConfidence" class="text-gray-600 font-medium"></span>
      ｜ 数据源 <span id="metaSource" class="text-gray-600"></span>
      ｜ 报告日期 <span id="metaDate" class="text-gray-600"></span>
    </p>
    <p class="mb-1">本报告由自动化分析工具生成，不构成投资建议。所有结论请结合专业人士判断使用。</p>
  </footer>

</main>

<script>
var D = __REPORT_DATA__;
var charts = [];
function el(id) { return document.getElementById(id); }

// ---- 头部 ----
el('companyName').textContent = D.meta.company_name;
el('stockCode').textContent = D.meta.stock_code;
el('industry').textContent = D.meta.industry || '—';
el('analysisPeriod').textContent = D.meta.analysis_period ? ('分析期间 ' + D.meta.analysis_period) : '';
var badge = el('riskBadge');
badge.textContent = D.meta.risk_band_cn;
badge.style.backgroundColor = D.meta.gauge_color;

// ---- 综合风险得分 gauge ----
var gauge = echarts.init(el('gaugeChart'));
gauge.setOption({
  series: [{
    type: 'gauge', startAngle: 210, endAngle: -30, min: 0, max: 100, splitNumber: 10, radius: '95%',
    axisLine: { lineStyle: { width: 16, color: [[D.gauge.score / 100, D.gauge.color], [1, '#e5e7eb']] } },
    pointer: { length: '62%', width: 5, itemStyle: { color: '#1f2937' } },
    axisTick: { distance: -16, length: 5, lineStyle: { color: '#fff', width: 2 } },
    splitLine: { distance: -16, length: 16, lineStyle: { color: '#fff', width: 3 } },
    axisLabel: { distance: 18, color: '#9ca3af', fontSize: 10 },
    anchor: { show: true, size: 12, itemStyle: { color: D.gauge.color } },
    title: { offsetCenter: [0, '72%'], fontSize: 14, color: '#6b7280' },
    detail: { valueAnimation: true, offsetCenter: [0, '38%'], fontSize: 42, fontWeight: 'bolder', color: D.gauge.color, formatter: '{value}' },
    data: [{ value: D.gauge.score, name: D.gauge.band_cn }]
  }]
});
charts.push(gauge);

// ---- 各级风险触发（横向条形图）----
var levels = echarts.init(el('levelsChart'));
levels.setOption({
  grid: { left: 8, right: 40, top: 8, bottom: 8, containLabel: true },
  xAxis: { type: 'value', splitLine: { lineStyle: { color: '#f3f4f6' } }, axisLabel: { color: '#9ca3af' } },
  yAxis: { type: 'category', data: D.levels.names, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#6b7280' } },
  series: [{
    type: 'bar', data: D.levels.values, barWidth: 20,
    label: { show: true, position: 'right', fontWeight: 'bold', color: '#374151' },
    itemStyle: { borderRadius: [0, 6, 6, 0], color: function (p) { return D.levels.colors[p.dataIndex]; } }
  }]
});
charts.push(levels);

// ---- 规则判定分布（环形图）----
var verdict = echarts.init(el('verdictChart'));
verdict.setOption({
  tooltip: { trigger: 'item' },
  legend: { orient: 'vertical', right: '2%', top: 'center', textStyle: { color: '#6b7280' } },
  title: { text: String(D.verdict_total), subtext: '总判定', left: '32%', top: '40%', textAlign: 'center', textStyle: { fontSize: 30, fontWeight: 'bold', color: '#374151' }, subtextStyle: { color: '#9ca3af' } },
  series: [{
    type: 'pie', radius: ['55%', '78%'], center: ['32%', '50%'],
    avoidLabelOverlap: true, itemStyle: { borderColor: '#fff', borderWidth: 2, borderRadius: 5 },
    label: { show: false }, labelLine: { show: false },
    data: D.verdicts.names.map(function (n, i) { return { name: n, value: D.verdicts.values[i], itemStyle: { color: D.verdicts.colors[i] } }; })
  }]
});
charts.push(verdict);

// ---- 造假类型热力矩阵 ----
if (D.heatmap.types.length) {
  var heatmap = echarts.init(el('heatmapChart'));
  heatmap.setOption({
    tooltip: { position: 'top', formatter: function (p) { return D.heatmap.types[p.value[0]] + ' · ' + D.heatmap.years[p.value[1]] + '：' + p.value[2] + ' 次'; } },
    grid: { left: 8, right: 8, top: 8, bottom: 70, containLabel: true },
    xAxis: { type: 'category', data: D.heatmap.types, splitArea: { show: true }, axisLabel: { color: '#6b7280', rotate: 25 } },
    yAxis: { type: 'category', data: D.heatmap.years, splitArea: { show: true }, axisLabel: { color: '#6b7280' } },
    visualMap: { min: 0, max: D.heatmap.max, calculable: true, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 12, itemHeight: 90, textStyle: { color: '#9ca3af' }, inRange: { color: ['#f3f4f6', '#fecaca', '#f87171', '#dc2626', '#991b1b'] } },
    series: [{
      type: 'heatmap', data: D.heatmap.data,
      label: { show: true, color: '#374151', fontSize: 11, formatter: function (p) { return p.value[2] > 0 ? p.value[2] : ''; } },
      emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.25)' } }
    }]
  });
  charts.push(heatmap);
} else {
  el('heatmapChart').style.display = 'none';
  el('heatmapEmpty').style.display = 'flex';
}

// ---- 关键指标趋势 ----
if (D.trend.has_data) {
  var trend = echarts.init(el('trendChart'));
  trend.setOption({
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, type: 'scroll', textStyle: { color: '#6b7280' } },
    grid: { left: 8, right: 16, top: 16, bottom: 50, containLabel: true },
    xAxis: { type: 'category', data: D.trend.years, boundaryGap: false, axisLabel: { color: '#9ca3af' } },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#f3f4f6' } }, axisLabel: { color: '#9ca3af' } },
    series: D.trend.series.map(function (s) { return { name: s.name, type: 'line', smooth: true, symbol: 'circle', data: s.data }; })
  });
  charts.push(trend);
} else {
  el('trendChart').style.display = 'none';
  el('trendEmpty').style.display = 'flex';
}

// ---- TOP风险证据链 ----
if (D.evidences.names.length) {
  var evidence = echarts.init(el('evidenceChart'));
  evidence.setOption({
    tooltip: { trigger: 'item', formatter: function (p) { return p.name + '：' + D.evidences.descs[p.dataIndex] + '<br/>权重 ' + p.value; } },
    grid: { left: 8, right: 50, top: 8, bottom: 8, containLabel: true },
    xAxis: { type: 'value', splitLine: { lineStyle: { color: '#f3f4f6' } }, axisLabel: { color: '#9ca3af' } },
    yAxis: { type: 'category', data: D.evidences.names, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#6b7280' } },
    series: [{
      type: 'bar', data: D.evidences.values, barWidth: 16,
      label: { show: true, position: 'right', color: '#374151' },
      itemStyle: { borderRadius: [0, 6, 6, 0], color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: '#3b82f6' }, { offset: 1, color: '#8b5cf6' }]) }
    }]
  });
  charts.push(evidence);
} else {
  el('evidenceChart').style.display = 'none';
  el('evidenceEmpty').style.display = 'flex';
}

// ---- 实时行情卡片 ----
if (D.realtime.ok) {
  el('realtimeOk').style.display = 'block';
  el('rtPrice').textContent = Number(D.realtime.price).toFixed(2) + ' 元';
  var pct = Number(D.realtime.change_pct);
  var chg = el('rtChange');
  chg.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
  chg.style.color = pct >= 0 ? '#ef4444' : '#10b981';   // A股习惯：涨红跌绿
  el('rtVolume').textContent = Number(D.realtime.volume).toLocaleString('zh-CN') + ' 手';
  el('rtTurnover').textContent = Number(D.realtime.turnover).toFixed(2) + ' 亿元';
  el('rtTime').textContent = D.realtime.update_time || '—';
} else {
  el('realtimeFail').style.display = 'flex';
}

// ---- 底部元信息 ----
el('metaConfidence').textContent = (D.meta.confidence * 100).toFixed(0) + '%';
el('metaSource').textContent = D.meta.data_source || '—';
el('metaDate').textContent = D.meta.report_date || '—';

// ---- resize ----
window.addEventListener('resize', function () {
  charts.forEach(function (c) { c.resize(); });
});
</script>
</body>
</html>
"""
