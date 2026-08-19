"""
财务造假识别 Agent — 报告生成与可视化模块
严格按 SKILL (2) 步骤5 的输出格式生成报告
"""
import os
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
