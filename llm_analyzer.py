"""
LLM增强分析模块 — 解决定性分析和综合判断缺失的问题
支持双模式: LLM增强 / 规则回退(不再跳过AI判断，而是用启发式替代)

三种LLM后端:
  - ollama: 本地模型 (qwen3, deepseek-r1, llama3 等)
  - openai: OpenAI兼容API (GPT, DeepSeek API 等)
  - heuristic: 规则回退 (不依赖外部AI，输出结构化分析)
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from data_fetcher import FinancialData
from indicator_calc import IndicatorCalculator
from config import FRAUD_PATTERNS, ACCOUNTING_STANDARDS


@dataclass
class LLMAnalysisResult:
    """LLM分析结果"""
    # 综合判断
    overall_assessment: str = ""       # 整体评价
    fraud_probability: str = ""        # 造假概率判断
    key_concerns: List[str] = field(default_factory=list)

    # 定性分析
    audit_opinion_analysis: str = ""   # 审计意见解读
    business_logic_analysis: str = ""  # 商业逻辑推理
    governance_analysis: str = ""      # 治理层面分析
    fraud_pattern_analysis: str = ""   # 造假模式识别

    # 证据链综合
    evidence_chain_summary: str = ""   # 证据链综述

    # 建议
    investigation_suggestions: List[str] = field(default_factory=list)
    risk_narrative: str = ""           # 风险描述叙事

    # 元数据
    model_used: str = ""
    success: bool = False


class LLMAnalyzer:
    """
    LLM增强分析器
    - 支持Ollama本地模型
    - 支持OpenAI兼容API
    - 规则回退模式(当LLM不可用时)
    """

    def __init__(
        self,
        backend: str = "heuristic",  # ollama / openai / heuristic
        model: str = "",
        api_key: str = "",
        api_base: str = "",
    ):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    def analyze(
        self,
        data: FinancialData,
        indicators: IndicatorCalculator,
        rule_results: List,
        crosscheck_results: List,
        biz_logic_results: List,
    ) -> LLMAnalysisResult:
        """执行LLM增强分析"""

        if self.backend == "ollama":
            return self._analyze_ollama(data, indicators, rule_results, crosscheck_results, biz_logic_results)
        elif self.backend == "openai":
            return self._analyze_openai(data, indicators, rule_results, crosscheck_results, biz_logic_results)
        else:
            return self._analyze_heuristic(data, indicators, rule_results, crosscheck_results, biz_logic_results)

    # ================================================================
    # 规则回退模式 — 不依赖外部AI的结构化分析
    # ================================================================
    def _analyze_heuristic(
        self,
        data: FinancialData,
        indicators: IndicatorCalculator,
        rule_results: List,
        crosscheck_results: List,
        biz_logic_results: List,
    ) -> LLMAnalysisResult:
        """基于启发式规则的综合分析（不依赖LLM）"""
        result = LLMAnalysisResult(model_used="heuristic (规则引擎)", success=True)

        # ----- 收集所有关键信号 -----
        fail_signals = []
        warn_signals = []
        fraud_types = set()
        total_score = 0

        for r in rule_results:
            if hasattr(r, 'verdict') and r.verdict == "FAIL":
                fail_signals.append(r)
                total_score += getattr(r, 'score', 0)
                if hasattr(r, 'fraud_type') and r.fraud_type:
                    fraud_types.add(r.fraud_type)
            elif hasattr(r, 'verdict') and r.verdict == "WARN":
                warn_signals.append(r)
                total_score += getattr(r, 'score', 0)

        # 勾稽结果
        for r in crosscheck_results:
            if hasattr(r, 'verdict') and r.verdict == "FAIL":
                fail_signals.append(r)
                total_score += getattr(r, 'score', 0)
            elif hasattr(r, 'verdict') and r.verdict == "WARN":
                warn_signals.append(r)

        # 商业逻辑结果
        red_flags = [r for r in biz_logic_results if hasattr(r, 'red_flag') and r.red_flag]

        # ----- 综合判断 -----
        if total_score >= 60:
            result.overall_assessment = (
                "财务报表存在严重异常，多项关键指标触发高危信号。"
                "造假可能性极高，建议立即启动深度调查。"
            )
            result.fraud_probability = "极高 (90%+)"
        elif total_score >= 35:
            result.overall_assessment = (
                "财务报表存在多项重大异常，会计信息质量受到严重质疑。"
                "建议高度警惕，获取更多证据后再做出判断。"
            )
            result.fraud_probability = "高 (60%-90%)"
        elif total_score >= 16:
            result.overall_assessment = (
                "存在若干警示信号，虽未达到严重造假的标准，"
                "但需持续跟踪关键指标变化，审慎对待财务报表数据。"
            )
            result.fraud_probability = "中等 (30%-60%)"
        else:
            result.overall_assessment = (
                "基于现有数据，未发现重大财务造假信号。"
                "但限于分析工具无法获取原始凭证和银行流水，不排除隐蔽造假的可能。"
            )
            result.fraud_probability = "较低 (<30%)"

        # ----- 关键关注点 -----
        result.key_concerns = []
        for signal in fail_signals[:5]:
            name = getattr(signal, 'rule_name', '') or getattr(signal, 'check_name', '')
            evidence = getattr(signal, 'evidence', '') or getattr(signal, 'fraud_implication', '')
            if name:
                result.key_concerns.append(f"[{name}] {evidence}")

        # ----- 审计意见分析 -----
        audit_opinion = data.non_financial.get("audit_opinion", "")
        if not audit_opinion or audit_opinion == "":
            result.audit_opinion_analysis = "未获取到审计意见信息。正常企业应能提供标准无保留意见的审计报告。无法获取审计意见本身就是一个警示信号。"
        elif any(kw in str(audit_opinion) for kw in ["保留意见", "否定意见", "无法表示意见"]) and \
             "无保留" not in str(audit_opinion):
            result.audit_opinion_analysis = (
                f"审计意见为'{audit_opinion}'——非标准意见。这是最高级别的会计信息质量警告。"
                "审计师发现了重大错报或无法获取充分证据，意味着财务报表的可靠性已崩塌。"
                "在A股市场，非标意见往往伴随着后续的监管调查和股价暴跌。"
            )
        elif "强调事项" in str(audit_opinion) or "持续经营" in str(audit_opinion):
            result.audit_opinion_analysis = (
                f"审计意见为'{audit_opinion}'。虽为标准无保留，但带有强调事项段。"
                "审计师在提醒关注特定风险(如持续经营能力、重大诉讼等)。"
            )
        else:
            result.audit_opinion_analysis = (
                f"审计意见为'{audit_opinion}'。但需注意：标准无保留意见不代表报表没问题。"
                "审计师只能提供合理保证而非绝对保证，合谋造假的案例(A股)并不少见。"
            )

        # ----- 商业逻辑推理 -----
        if red_flags:
            result.business_logic_analysis = "以下商业逻辑矛盾无法用正常经营解释：\n"
            for rf in red_flags[:3]:
                name = getattr(rf, 'check_name', '')
                contradiction = getattr(rf, 'contradiction', '')
                result.business_logic_analysis += f"  - {name}: {contradiction}\n"
        else:
            result.business_logic_analysis = "未发现明显的商业逻辑矛盾。但需注意：数据层面的验证无法替代对业务实质的实地核查。"

        # ----- 治理层面分析 -----
        pledge = data.non_financial.get("pledge_ratio", 0)
        governance_issues = []
        if pledge > 0.5:
            governance_issues.append(f"大股东质押{pledge*100:.0f}%，存在资金占用动机")
        if data.non_financial.get("forecast_revisions_count", 0) >= 3:
            governance_issues.append("业绩预告频繁修正，内控存在缺陷")
        if data.non_financial.get("policy_changes_3yr", 0) >= 2:
            governance_issues.append("会计政策频繁变更")

        if governance_issues:
            result.governance_analysis = "公司治理层面存在以下警示信号：\n" + "\n".join(f"  - {g}" for g in governance_issues)
        else:
            result.governance_analysis = "基于已有信息，未发现重大治理缺陷。但内部治理的真实情况需实地尽调才能确认。"

        # ----- 造假模式识别 -----
        if fraud_types:
            result.fraud_pattern_analysis = "匹配到以下已知造假模式：\n"
            for ft in list(fraud_types)[:5]:
                pattern = FRAUD_PATTERNS.get(ft, {})
                if pattern:
                    result.fraud_pattern_analysis += (
                        f"  - {ft}: {pattern.get('core_features', '')} "
                        f"(参考案例: {pattern.get('typical_case', '')})\n"
                    )
            result.fraud_pattern_analysis += "这些模式在A股市场已有大量案例，需立即核查。"

        # ----- 证据链综述 -----
        if fail_signals:
            result.evidence_chain_summary = "核心证据链：\n"
            for i, signal in enumerate(fail_signals[:5], 1):
                name = getattr(signal, 'rule_name', '') or getattr(signal, 'check_name', '')
                evidence = getattr(signal, 'evidence', '') or getattr(signal, 'fraud_implication', '')
                result.evidence_chain_summary += f"  {i}. {name} → {evidence}\n"

            # 构建链条
            if len(fail_signals) >= 2:
                chains = []
                for s in fail_signals:
                    ft = getattr(s, 'fraud_type', '')
                    if '存贷双高' in ft or '资金占用' in ft:
                        chains.append("存贷双高→资金占用")
                    if '存货' in ft:
                        chains.append("存货异常→成本虚减/资产虚增")
                    if '减值' in ft:
                        chains.append("减值波动→利润调节/业绩洗澡")
                if chains:
                    result.evidence_chain_summary += f"\n造假链条推测: {' + '.join(chains[:3])}"
        else:
            result.evidence_chain_summary = "未发现完整造假证据链。"

        # ----- 调查建议 -----
        result.investigation_suggestions = [
            "获取最近3年完整审计报告及审计师沟通记录",
            "核对银行对账单与账面货币资金(银行函证)",
            "实地盘点存货，核查库龄结构和减值测试方法",
            "获取前5大客户的工商信息、合同及回款记录",
            "核查关联方资金往来的商业实质和定价公允性",
            "对比海关出口数据与境外收入确认",
            "检查在建工程进度和转固时点的合理性",
        ]

        # ----- 风险叙事(供报告使用) -----
        company_name = data.company_name or "目标公司"
        if total_score >= 60:
            result.risk_narrative = (
                f"{company_name}的财务报表呈现多处高危信号，"
                f"涉及{len(fraud_types)}种已知造假模式。"
                f"关键指标与商业逻辑严重背离，财务数据可信度极低。"
                f"建议在获取银行流水、存货盘点报告、客户回函等核心证据前，"
                f"将该公司列为高风险并暂停所有业务往来。"
            )
        elif total_score >= 35:
            result.risk_narrative = (
                f"{company_name}的财务数据存在多项异常，"
                f"虽然尚未形成完整的造假证据链，但已触发多条警示规则。"
                f"建议补充核查后重新评估。"
            )
        else:
            result.risk_narrative = (
                f"基于现有公开财务数据，{company_name}未触发重大风险信号。"
                f"但本分析仅基于财务指标量化判断，"
                f"不能替代对业务实质的实地尽调和原始凭证核查。"
            )

        return result

    # ================================================================
    # Ollama 本地模型
    # ================================================================
    def _analyze_ollama(
        self,
        data: FinancialData,
        indicators: IndicatorCalculator,
        rule_results: List,
        crosscheck_results: List,
        biz_logic_results: List,
    ) -> LLMAnalysisResult:
        """使用Ollama本地模型进行深度分析"""
        model = self.model or "qwen3:latest"

        # 构建结构化prompt
        prompt = self._build_analysis_prompt(data, indicators, rule_results, crosscheck_results, biz_logic_results)

        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            if response.status_code == 200:
                output = response.json().get("response", "")
                return self._parse_llm_output(output, model)
        except Exception as e:
            print(f"[LLM] Ollama调用失败: {e}，回退到规则引擎分析")
            return self._analyze_heuristic(data, indicators, rule_results, crosscheck_results, biz_logic_results)

    # ================================================================
    # OpenAI 兼容 API
    # ================================================================
    def _analyze_openai(
        self,
        data: FinancialData,
        indicators: IndicatorCalculator,
        rule_results: List,
        crosscheck_results: List,
        biz_logic_results: List,
    ) -> LLMAnalysisResult:
        """使用OpenAI兼容API进行深度分析"""
        model = self.model or "deepseek-chat"

        prompt = self._build_analysis_prompt(data, indicators, rule_results, crosscheck_results, biz_logic_results)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.api_base or "https://api.openai.com/v1")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是中国A股市场财务造假识别专家，精通中国会计准则(CAS)和证监会监管规则。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            output = response.choices[0].message.content
            return self._parse_llm_output(output, model)
        except Exception as e:
            print(f"[LLM] API调用失败: {e}，回退到规则引擎分析")
            return self._analyze_heuristic(data, indicators, rule_results, crosscheck_results, biz_logic_results)

    # ================================================================
    # Prompt构建
    # ================================================================
    def _build_analysis_prompt(
        self,
        data: FinancialData,
        indicators: IndicatorCalculator,
        rule_results: List,
        crosscheck_results: List,
        biz_logic_results: List,
    ) -> str:
        """构建分析prompt"""
        lines = []
        lines.append(f"## 公司信息")
        lines.append(f"- 名称: {data.company_name}")
        lines.append(f"- 代码: {data.stock_code}")
        lines.append(f"- 行业: {data.industry}")
        lines.append(f"- 分析期间: {data.years}")

        # 关键指标摘要
        lines.append("\n## 关键财务指标")
        latest = data.years[-1] if data.years else 0
        for name in ["毛利率", "经营现金流/净利润", "资产负债率", "ROE", "销售收现比", "存货周转率"]:
            vals = indicators.indicators.get(name, {})
            lines.append(f"- {name}: " + ", ".join(f"{y}: {vals.get(y, 'N/A')}" for y in data.years))

        # 规则扫描结果
        lines.append("\n## 规则扫描异常")
        for r in rule_results:
            if hasattr(r, 'verdict') and r.verdict in ("FAIL", "WARN"):
                name = getattr(r, 'rule_name', '')
                evidence = getattr(r, 'evidence', '')
                risk = getattr(r, 'risk_explanation', '')
                lines.append(f"- [{r.verdict}] {name}: {evidence} | {risk}")

        # 勾稽验证结果
        lines.append("\n## 会计勾稽验证")
        for r in crosscheck_results:
            if hasattr(r, 'verdict') and r.verdict in ("FAIL", "WARN"):
                lines.append(f"- [{r.verdict}] {r.check_name}: {r.fraud_implication}")

        # 商业逻辑验证
        lines.append("\n## 商业逻辑验证")
        for r in biz_logic_results:
            if hasattr(r, 'verdict') and r.verdict == "FAIL":
                lines.append(f"- [{r.verdict}] {r.check_name}: {r.contradiction}")

        # 分析要求
        lines.append("\n## 分析要求")
        lines.append("请以中国A股财务造假识别专家的身份，完成以下分析：")
        lines.append("1. 综合判断: 这家公司财务报表存在造假的可能性有多高？(给出百分比)")
        lines.append("2. 审计意见分析: 结合审计意见类型判断会计信息可靠性")
        lines.append("3. 商业逻辑推理: 哪些业务数据违背了正常的商业逻辑？")
        lines.append("4. 造假模式识别: 根据A股常见造假模式，该公司的特征最接近哪种？")
        lines.append("5. 证据链综述: 构建从数据异常到造假结论的完整推理链")
        lines.append("6. 调查建议: 列出现场检查应重点核查的项目")
        lines.append("请用中文回复，保持专业、客观。")

        return "\n".join(lines)

    # ================================================================
    # LLM输出解析
    # ================================================================
    def _parse_llm_output(self, output: str, model: str) -> LLMAnalysisResult:
        """解析LLM输出为结构化结果"""
        result = LLMAnalysisResult(model_used=model, success=True)

        # 尝试提取各部分
        result.overall_assessment = self._extract_section(output, ["综合判断", "整体评价", "1."])
        result.fraud_probability = self._extract_probability(output)
        result.audit_opinion_analysis = self._extract_section(output, ["审计意见分析", "2."])
        result.business_logic_analysis = self._extract_section(output, ["商业逻辑推理", "3."])
        result.fraud_pattern_analysis = self._extract_section(output, ["造假模式识别", "4."])
        result.evidence_chain_summary = self._extract_section(output, ["证据链综述", "5."])
        result.risk_narrative = output[:500]  # 前500字符作为摘要

        # 提取建议
        suggestion_section = self._extract_section(output, ["调查建议", "建议", "6."])
        if suggestion_section:
            result.investigation_suggestions = [
                s.strip("- ").strip()
                for s in suggestion_section.split("\n")
                if s.strip() and len(s.strip()) > 10
            ]

        return result

    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """从文本中提取特定章节"""
        lines = text.split("\n")
        capturing = False
        captured = []
        for line in lines:
            if any(kw in line for kw in keywords):
                capturing = True
                captured.append(line)
            elif capturing and line.strip() and any(kw in line for kw in ["#", "##", "综合", "审计", "商业", "造假", "证据", "调查", "建议"]):
                break
            elif capturing:
                captured.append(line)
        return "\n".join(captured) if captured else ""

    def _extract_probability(self, text: str) -> str:
        """提取造假概率"""
        for line in text.split("\n"):
            if any(kw in line for kw in ["概率", "%", "可能性"]):
                return line.strip()
        return ""
