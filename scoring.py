"""
财务造假识别 Agent — 评分与风险分级模块 v2.0
整合: 三级失真扫描 + 会计勾稽验证 + 商业逻辑验证 + LLM分析
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import math

from config import RISK_BANDS, RISK_LEVEL_10, RISK_LEVEL_8, RISK_LEVEL_5, RISK_LEVEL_3
from rule_engine import RuleResult


@dataclass
class RiskProfile:
    """风险画像 v2.0"""
    total_score: int = 0
    max_score: int = 200  # 扩展到200分制（100分三级扫描+100分勾稽/商业逻辑）
    normalized_score: int = 0  # 归一化到100分
    risk_band: str = "GREEN"
    risk_band_cn: str = "低风险"
    risk_summary: str = ""

    # 三级扫描统计
    level_10_count: int = 0
    level_8_count: int = 0
    level_5_count: int = 0
    level_3_count: int = 0

    # 勾稽验证统计
    crosscheck_fail_count: int = 0
    crosscheck_warn_count: int = 0
    crosscheck_score: int = 0

    # 商业逻辑统计
    bizlogic_fail_count: int = 0
    bizlogic_warn_count: int = 0
    bizlogic_score: int = 0
    red_flag_count: int = 0

    # 综合判定
    fail_count: int = 0
    warn_count: int = 0
    pass_count: int = 0
    missing_count: int = 0

    # 造假类型
    fraud_types: List[str] = field(default_factory=list)

    # 证据链
    top_evidences: List[Tuple[str, str, int]] = field(default_factory=list)

    # 勾稽异常
    crosscheck_fail_items: List[Tuple[str, str]] = field(default_factory=list)

    # 商业逻辑矛盾
    bizlogic_red_flags: List[Tuple[str, str]] = field(default_factory=list)

    # 置信度
    confidence: float = 0.0
    confidence_reason: str = ""

    # LLM分析摘要
    llm_summary: str = ""


class RiskScorer:
    """风险打分模型 v2.0 — 三维度加权评分"""

    def __init__(self):
        self.profile = RiskProfile()

    def compute(
        self,
        rule_results: List,
        crosscheck_results: Optional[List] = None,
        biz_logic_results: Optional[List] = None,
        llm_result: Any = None,
    ) -> RiskProfile:
        """综合计算风险得分"""

        # ========== 维度一：三级失真扫描 (最多100分) ==========
        rule_score = 0
        evidences = []

        for r in rule_results:
            if not hasattr(r, 'verdict'):
                continue

            if r.verdict == "FAIL":
                self.profile.fail_count += 1
                rule_score += getattr(r, 'score', 0)
                ev = getattr(r, 'evidence', '') or getattr(r, 'risk_explanation', '')
                if ev:
                    evidences.append((getattr(r, 'rule_name', r.rule_id), ev, getattr(r, 'score', 0)))
            elif r.verdict == "WARN":
                self.profile.warn_count += 1
                rule_score += getattr(r, 'score', 0)
                ev = getattr(r, 'evidence', '') or getattr(r, 'risk_explanation', '')
                if ev:
                    evidences.append((getattr(r, 'rule_name', r.rule_id), ev, getattr(r, 'score', 0)))
            elif r.verdict == "PASS":
                self.profile.pass_count += 1
            else:
                self.profile.missing_count += 1

            # 统计风险等级
            rl = getattr(r, 'risk_level', '')
            if r.verdict in ("FAIL", "WARN"):
                if rl == RISK_LEVEL_10:
                    self.profile.level_10_count += 1
                elif rl == RISK_LEVEL_8:
                    self.profile.level_8_count += 1
                elif rl == RISK_LEVEL_5:
                    self.profile.level_5_count += 1
                elif rl == RISK_LEVEL_3:
                    self.profile.level_3_count += 1

            # 造假类型
            ft = getattr(r, 'fraud_type', '')
            if ft and ft not in self.profile.fraud_types:
                self.profile.fraud_types.append(ft)

        # ========== 维度二：会计勾稽验证 (最多60分) ==========
        crosscheck_score = 0
        if crosscheck_results:
            for r in crosscheck_results:
                if not hasattr(r, 'verdict'):
                    continue
                if r.verdict == "FAIL":
                    self.profile.crosscheck_fail_count += 1
                    crosscheck_score += getattr(r, 'score', 0)
                    self.profile.fail_count += 1
                    ev = getattr(r, 'fraud_implication', '') or getattr(r, 'evidence', '')
                    if ev:
                        evidences.append((getattr(r, 'check_name', r.check_id), ev, getattr(r, 'score', 0)))
                    self.profile.crosscheck_fail_items.append(
                        (getattr(r, 'check_name', r.check_id), getattr(r, 'fraud_implication', ''))
                    )
                elif r.verdict == "WARN":
                    self.profile.crosscheck_warn_count += 1
                    crosscheck_score += getattr(r, 'score', 0)
                    self.profile.warn_count += 1
            self.profile.crosscheck_score = min(crosscheck_score, 60)

        # ========== 维度三：商业逻辑验证 (最多40分) ==========
        bizlogic_score = 0
        if biz_logic_results:
            for r in biz_logic_results:
                if not hasattr(r, 'verdict'):
                    continue
                if r.verdict == "FAIL":
                    self.profile.bizlogic_fail_count += 1
                    bizlogic_score += getattr(r, 'score', 0)
                    self.profile.fail_count += 1
                    ev = getattr(r, 'contradiction', '') or getattr(r, 'evidence', '')
                    if ev:
                        evidences.append((getattr(r, 'check_name', r.check_id), ev, getattr(r, 'score', 0)))
                elif r.verdict == "WARN":
                    self.profile.bizlogic_warn_count += 1
                    bizlogic_score += getattr(r, 'score', 0)
                    self.profile.warn_count += 1
                if getattr(r, 'red_flag', False):
                    self.profile.red_flag_count += 1
                    self.profile.bizlogic_red_flags.append(
                        (getattr(r, 'check_name', r.check_id), getattr(r, 'contradiction', ''))
                    )
            self.profile.bizlogic_score = min(bizlogic_score, 40)

        # ========== 综合得分 ==========
        # 三级扫描最多100分 + 勾稽最多60分 + 商业逻辑最多40分 = 200分
        # 归一化：勾稽和商业逻辑的打分已经包含权重，直接相加
        self.profile.total_score = min(rule_score, 100) + self.profile.crosscheck_score + self.profile.bizlogic_score
        self.profile.total_score = min(self.profile.total_score, 200)
        # 归一化到100分
        self.profile.normalized_score = min(round(self.profile.total_score / 2), 100)

        # ========== 风险等级（基于归一化分数） ==========
        for low, high, band, cn, summary in RISK_BANDS:
            if low <= self.profile.normalized_score <= high:
                self.profile.risk_band = band
                self.profile.risk_band_cn = cn
                self.profile.risk_summary = summary
                break

        # ========== 证据链排序 ==========
        evidences.sort(key=lambda x: x[2], reverse=True)
        self.profile.top_evidences = evidences[:5]

        # ========== 置信度 ==========
        self._compute_confidence(rule_results, crosscheck_results, biz_logic_results)

        # ========== LLM摘要 ==========
        if llm_result and hasattr(llm_result, 'risk_narrative'):
            self.profile.llm_summary = llm_result.risk_narrative

        return self.profile

    def _compute_confidence(self, rule_results, crosscheck_results, biz_logic_results):
        """置信度计算 v2.0 — 考虑三维度数据覆盖"""
        total_items = len(rule_results) + (len(crosscheck_results) if crosscheck_results else 0) + (len(biz_logic_results) if biz_logic_results else 0)
        if total_items == 0:
            self.profile.confidence = 0.0
            return

        # 数据覆盖率
        valid = self.profile.pass_count + self.profile.fail_count + self.profile.warn_count
        coverage = valid / max(total_items, 1)

        # 多维验证加分：三个维度都产生信号时置信度最高
        dimension_signals = 0
        if self.profile.level_10_count > 0 or self.profile.level_8_count > 0:
            dimension_signals += 1
        if self.profile.crosscheck_fail_count > 0:
            dimension_signals += 1
        if self.profile.red_flag_count > 0:
            dimension_signals += 1

        # 信号一致性
        if self.profile.fraud_types:
            type_concentration = 1.0 / max(len(set(self.profile.fraud_types)), 1)
        else:
            type_concentration = 0.5

        self.profile.confidence = round(
            min(1.0, coverage * 0.4 + (dimension_signals / 3) * 0.3 + type_concentration * 0.3),
            2
        )

        if coverage > 0.9 and dimension_signals >= 2:
            self.profile.confidence_reason = "高 — 多维度数据覆盖，多信号交叉验证"
        elif coverage > 0.7:
            self.profile.confidence_reason = "中 — 数据覆盖较好，部分维度有信号"
        elif coverage > 0.5:
            self.profile.confidence_reason = "中低 — 部分数据缺失，结论需补充信息验证"
        else:
            self.profile.confidence_reason = "低 — 数据缺失较多，结论仅供参考"

    def get_investment_guidance(self) -> str:
        band = self.profile.risk_band
        if band == "RED":
            return "【极高风险】建议回避，存在严重财务造假嫌疑，直至风险点全部消除并获取充分证据"
        elif band == "ORANGE":
            return "【高风险】强烈不建议基于当前公开信息做出决策，需获取额外证据后重新评估"
        elif band == "YELLOW":
            return "【中风险】谨慎观望，需持续跟踪关键指标变化"
        elif band == "GREEN":
            return "【低风险】可纳入进一步研究范围，但需持续监控"
        return "无法判断"
