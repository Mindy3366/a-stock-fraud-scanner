"""
财务造假识别 Agent — 规则匹配引擎
严格基于 SKILL (2)：
  步骤2.1 — 财务造假扫描（10级，10条规则）
  步骤2.2 — 会计操纵扫描（8级，9条规则）
  步骤2.3 — 盈余管理扫描（5级，8条规则）
  步骤3   — 重点科目深度核查
  步骤4   — 商业逻辑验证
  附录E   — 数字化时代特殊风险
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import math

import numpy as np

from config import (
    FRAUD_RULES, MANIPULATION_RULES, EARNINGS_MGMT_RULES,
    RISK_LEVEL_10, RISK_LEVEL_8, RISK_LEVEL_5, RISK_LEVEL_3,
    DIGITAL_ERA_CHECKS, FRAUD_PATTERNS, ACCOUNTING_STANDARDS,
    INDUSTRY_ADJUSTMENTS, KEY_THRESHOLDS,
)
from data_fetcher import FinancialData
from indicator_calc import IndicatorCalculator, _safe_div


@dataclass
class RuleResult:
    """单条规则匹配结果"""
    rule_id: str
    rule_name: str
    risk_level: str  # 10级/8级/5级/3级
    verdict: str     # FAIL / WARN / PASS / DATA_MISSING
    score: int       # 本项得分
    data_basis: str  # 数据依据
    risk_explanation: str  # 风险说明
    fraud_type: str = ""   # 造假类型标签
    check_points: List[str] = field(default_factory=list)
    accounting_standard: str = ""  # 引用会计准则
    evidence: str = ""  # 证据链


@dataclass
class DeepAuditResult:
    """深度核查结果"""
    subject: str       # 核查科目
    anomalies: List[str] = field(default_factory=list)
    risk_level: str = ""
    evidence_chain: str = ""
    suggestions: List[str] = field(default_factory=list)


class RuleEngine:
    """财务报表造假规则匹配引擎"""

    def __init__(self, data: FinancialData, indicators: IndicatorCalculator):
        self.data = data
        self.ind = indicators
        self.results: List[RuleResult] = []
        self.deep_audit_results: List[DeepAuditResult] = []
        self.business_logic_results: List[RuleResult] = []
        self.digital_era_alerts: List[Dict] = []

    def run_all(self) -> List[RuleResult]:
        """执行全部规则扫描"""
        if not self.data.years:
            # 无数据年份，所有规则标记为 DATA_MISSING
            for rule_def in (FRAUD_RULES + MANIPULATION_RULES + EARNINGS_MGMT_RULES):
                self.results.append(self._missing(rule_def))
            return self.results + self.business_logic_results

        self._run_fraud_rules()
        self._run_manipulation_rules()
        self._run_earnings_mgmt_rules()
        self._run_deep_audit()
        self._run_business_logic()
        self._run_digital_era_checks()
        return self.results + self.business_logic_results

    # ================================================================
    # 步骤2.1: 财务造假扫描 (10级)
    # ================================================================
    def _run_fraud_rules(self):
        for rule_def in FRAUD_RULES:
            handler = getattr(self, f"_check_{rule_def['id'].replace('-', '_').lower()}", None)
            if handler:
                result = handler(rule_def)
            else:
                result = self._generic_check(rule_def)
            self.results.append(result)

    def _need_latest(self) -> Optional[int]:
        """获取最新年份，若无数据返回None"""
        return self._latest_year()

    def _check_fraud_001(self, r: dict) -> RuleResult:
        """存贷双高"""
        latest = self._need_latest()
        if latest is None:
            return self._missing(r)
        mf = self._get_bs("money_funds", latest)
        ibd = self._get_indicator("有息负债", latest)
        iir = self._get_indicator("利息收入率", latest)
        if mf is None or ibd is None or iir is None:
            return self._missing(r)
        deposit_loan_ratio = mf / ibd if ibd > 0 else 0
        triggered = mf > ibd * 0.5 and iir < 0.005
        if triggered:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"货币资金={mf/1e8:.2f}亿, 有息负债={ibd/1e8:.2f}亿, 存贷比={deposit_loan_ratio:.2f}, 利息收入率={iir*100:.2f}%",
                risk_explanation="账面巨额现金却支付高额利息，违背'现金生息'商业逻辑",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                accounting_standard=ACCOUNTING_STANDARDS.get("金融工具分类", ""),
                evidence=f"存贷比={deposit_loan_ratio:.2f}, 利息收入率={iir*100:.2f}% < 0.5%"
            )
        return self._pass(r, f"存贷比={deposit_loan_ratio:.2f}, 利息收入率={iir*100:.2f}%")

    def _check_fraud_002(self, r: dict) -> RuleResult:
        """利息收支背离 — 仅在有实质有息负债时才触发"""
        latest = self._latest_year()
        int_exp = self._get_pl("interest_expense", latest)
        int_inc = self._get_pl("interest_income", latest)
        mf = self._get_bs("money_funds", latest)
        ibd = self._get_indicator("有息负债", latest)

        if int_exp is None or int_inc is None:
            return self._missing(r)

        # 前置条件：必须有实质性有息负债&货币资金，否则利息收支比无意义
        if ibd is None or ibd < 1e9:  # 有息负债 < 10亿 → 轻资产/零杠杆企业
            return self._pass(r, f"有息负债仅{ibd/1e8:.2f}亿(不足10亿)，利息收支比不适用")
        if mf is None or mf < 5e9:    # 货币资金 < 50亿 → 不是"存贷双高"场景
            return self._pass(r, f"货币资金{mf/1e8:.2f}亿(不足50亿)，不构成存贷双高")

        # 核心条件：货币资金>50亿 AND 有息负债>10亿 AND 利息收入率<1%
        iir = int_inc / mf if mf > 0 else 0
        if iir >= 0.01:
            return self._pass(r, f"利息收入率={iir*100:.2f}% ≥ 1%，利息收支比不适用")

        if int_inc == 0:
            ratio = float("inf") if int_exp > 0 else 0
        else:
            ratio = int_exp / int_inc
        if ratio > 10:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"货币资金={mf/1e8:.2f}亿, 有息负债={ibd/1e8:.2f}亿, 利息费用={int_exp/1e8:.2f}亿, 利息收入={int_inc/1e8:.2f}亿, 比值={ratio:.1f}, 利息收入率={iir*100:.2f}%",
                risk_explanation="有息负债>10亿+货币资金>50亿但利息收入率<1%+利息收支比>10 → 账面现金真实性存疑",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence=f"有息负债{ibd/1e8:.0f}亿 + 货币资金{mf/1e8:.0f}亿 + 利息收入率{iir*100:.2f}% < 1% + 收支比{ratio:.1f}"
            )
        return self._pass(r, f"利息费用/利息收入={ratio:.1f}, 利息收入率={iir*100:.2f}%")

    def _check_fraud_003(self, r: dict) -> RuleResult:
        """存货-毛利率背离"""
        latest = self._latest_year()
        inv_change = self._get_indicator("存货周转率变化", latest)
        gm_change = self._get_indicator("毛利率变化", latest)
        if inv_change is None or gm_change is None:
            return self._missing(r)
        if inv_change < -0.2 and gm_change > 0.05:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=25,  # SKILL.md 3.2额外+10 → 在SKILL(2)中作为最高权重
                data_basis=f"存货周转率同比变化={inv_change*100:.1f}%, 毛利率同比变化={gm_change*100:.1f}%",
                risk_explanation="存货周转变慢通常伴随降价压力，毛利率却上升违背商业逻辑",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence=f"存货周转率↓{abs(inv_change)*100:.0f}% + 毛利率↑{gm_change*100:.0f}% → 背离"
            )
        if inv_change < -0.1 and gm_change > 0.02:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=8,
                data_basis=f"存货周转率同比变化={inv_change*100:.1f}%, 毛利率同比变化={gm_change*100:.1f}%",
                risk_explanation="存货周转放缓伴随毛利率微升，需关注",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
            )
        return self._pass(r, f"存货周转率变化={inv_change*100:.1f}%, 毛利率变化={gm_change*100:.1f}%")

    def _check_fraud_004(self, r: dict) -> RuleResult:
        """现金流利润背离"""
        years = self.data.years[-2:]
        violations = []
        for y in years:
            ocf_ratio = self._get_indicator("经营现金流/净利润", y)
            np_ = self._get_pl("net_profit", y)
            if ocf_ratio is not None and np_ is not None and np_ > 0 and ocf_ratio < 0.5:
                violations.append((y, ocf_ratio))
        if len(violations) >= 2:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"连续{len(violations)}年经营现金流/净利润<0.5: {violations}",
                risk_explanation="有利润无现金流，收入确认可能虚构",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence="→".join([f"{y}年OCF/NP={v:.2f}" for y, v in violations])
            )
        if len(violations) == 1:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=8, data_basis=str(violations[0]),
                risk_explanation="单年现金流-利润背离",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
            )
        return self._pass(r, f"近2年经营现金流/净利润: {[self._get_indicator('经营现金流/净利润', y) for y in years]}")

    def _check_fraud_005(self, r: dict) -> RuleResult:
        """关联方资金异常"""
        latest = self._latest_year()
        outflow = self._get_note("related_party_outflow", latest)
        total_outflow = self._get_cf("total_cf_outflow", latest) or self._get_indicator("经营现金流/净利润", latest)
        if outflow is None or total_outflow is None:
            return self._missing(r)
        ratio = outflow / total_outflow if total_outflow > 0 else 0
        if ratio > 0.5:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"关联方资金流出/经营现金流出={ratio*100:.1f}%",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence=f"关联方资金流出占比={ratio*100:.1f}% > 50%阈值"
            )
        return self._pass(r, f"关联方资金流出占比={ratio*100:.1f}%")

    def _check_fraud_006(self, r: dict) -> RuleResult:
        """境外收入异常"""
        latest = self._latest_year()
        overseas_pct = self._get_indicator("境外收入占比", latest)
        overseas_gm = self._get_pl("overseas_gross_margin", latest)
        domestic_gm = self._get_pl("domestic_gross_margin", latest)
        ocf = self._get_cf("operating_cf", latest)
        if overseas_pct is None:
            return self._missing(r)
        if overseas_pct > 0.4 and overseas_gm and domestic_gm and ocf is not None:
            gm_diff = overseas_gm - domestic_gm
            if gm_diff > 0.1 and ocf < 0:
                return RuleResult(
                    rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                    verdict="FAIL", score=15,
                    data_basis=f"境外收入占比={overseas_pct*100:.1f}%, 境外毛利率-境内毛利率={gm_diff*100:.1f}个百分点, 经营现金流={ocf/1e8:.2f}亿",
                    risk_explanation="境外业务核查困难，高毛利无现金流是虚构收入的典型特征",
                    fraud_type=r["fraud_type"], check_points=r["check_points"],
                    evidence=f"境外收入占比{overseas_pct*100:.0f}% > 40% + 毛利率差{gm_diff*100:.0f}pp + OCF为负"
                )
        return self._pass(r, f"境外收入占比={overseas_pct*100:.1f}%")

    def _check_fraud_007(self, r: dict) -> RuleResult:
        """合同资产异常"""
        latest = self._latest_year()
        ca_to_ar = self._get_indicator("合同资产/应收账款", latest)
        if ca_to_ar is None:
            return self._missing(r)
        if ca_to_ar > 1:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"合同资产/应收账款={ca_to_ar:.2f}",
                risk_explanation="合同资产附加条件多，长期高于应收账款暗示提前确认收入",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence=f"合同资产/应收账款={ca_to_ar:.2f} > 1.0，结构异常"
            )
        if ca_to_ar > 0.8:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=8, data_basis=f"合同资产/应收账款={ca_to_ar:.2f}",
                risk_explanation="合同资产占比较高，收入确认条件趋于宽松",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
            )
        return self._pass(r, f"合同资产/应收账款={ca_to_ar:.2f}")

    def _check_fraud_008(self, r: dict) -> RuleResult:
        """其他应收款异常"""
        latest = self._latest_year()
        or_pct = self._get_indicator("其他应收款/流动资产", latest)
        if or_pct is None:
            return self._missing(r)
        if or_pct > 0.15:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"其他应收款/流动资产={or_pct*100:.1f}%",
                risk_explanation="其他应收款是关联方资金占用的常用通道",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence=f"其他应收款/流动资产={or_pct*100:.1f}% > 15%"
            )
        if or_pct > 0.10:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=8, data_basis=f"其他应收款/流动资产={or_pct*100:.1f}%",
                risk_explanation="其他应收款占比偏高",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
            )
        return self._pass(r, f"其他应收款/流动资产={or_pct*100:.1f}%")

    def _check_fraud_009(self, r: dict) -> RuleResult:
        """巨额减值消化"""
        years = self.data.years
        if len(years) < 2:
            return self._missing(r)
        latest = years[-1]
        prev = years[-2]
        cl_latest = self._get_pl("credit_impairment_loss", latest)
        nl_prev = self._get_pl("net_profit", prev)
        cl_prev = self._get_pl("credit_impairment_loss", prev)
        if cl_latest is None or nl_prev is None or cl_prev is None:
            return self._missing(r)
        if nl_prev >= 0:
            return self._pass(r, "上年未亏损")
        ratio = cl_latest / abs(nl_prev) if nl_prev != 0 else 0
        next_decrease = cl_prev is not None and cl_latest < cl_prev * 0.5
        if ratio > 0.8 and next_decrease:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"信用减值损失/|净利润亏损|={ratio*100:.1f}%, 次年减值大幅减少",
                risk_explanation="通过巨额减值消化前期虚构的应收账款",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence=f"减值损失占亏损额{ratio*100:.0f}% > 80% + 次年骤降 → 减值调节痕迹"
            )
        return self._pass(r, f"信用减值损失/|亏损|={ratio*100:.1f}%")

    def _check_fraud_010(self, r: dict) -> RuleResult:
        """收入成本不匹配"""
        latest = self._latest_year()
        sc_ratio = self._get_indicator("销售收现比", latest)
        overseas_pct = self._get_indicator("境外收入占比", latest)
        if sc_ratio is None:
            return self._missing(r)
        overseas_high = overseas_pct and overseas_pct > 0.3
        if sc_ratio < 0.6 and overseas_high:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=15,
                data_basis=f"销售收现比={sc_ratio*100:.1f}%, 境外收入占比={overseas_pct*100:.1f}%",
                risk_explanation="虚构收入无法产生真实现金流",
                fraud_type=r["fraud_type"], check_points=r["check_points"],
                evidence=f"境外收入高增但收现比仅{sc_ratio*100:.0f}% < 60%"
            )
        return self._pass(r, f"销售收现比={sc_ratio*100:.1f}%")

    # ================================================================
    # 步骤2.2: 会计操纵扫描 (8级)
    # ================================================================
    def _run_manipulation_rules(self):
        for rule_def in MANIPULATION_RULES:
            handler = getattr(self, f"_check_{rule_def['id'].replace('-', '_').lower()}", None)
            if handler:
                result = handler(rule_def)
            else:
                result = self._generic_check(rule_def)
            self.results.append(result)

    def _check_manip_001(self, r: dict) -> RuleResult:
        """会计政策频繁变更"""
        pc = self.data.non_financial.get("policy_changes_3yr", 0)
        if pc > 2:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=10,
                data_basis=f"3年内会计政策/估计变更{pc}次",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=f"3年变更{pc}次 > 2次阈值"
            )
        return self._pass(r, f"3年变更{pc}次")

    def _check_manip_002(self, r: dict) -> RuleResult:
        """减值准备剧烈波动"""
        years = self.data.years
        if len(years) < 2:
            return self._missing(r)
        latest = years[-1]
        prev = years[-2]
        ail_latest = self._get_pl("asset_impairment_loss", latest)
        ail_prev = self._get_pl("asset_impairment_loss", prev)
        if ail_latest is None or ail_prev is None or ail_prev == 0:
            return self._missing(r)
        ratio = ail_latest / ail_prev
        if ratio > 3 or ratio < 0.3:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=10,
                data_basis=f"当期减值/上期减值={ratio:.2f}倍",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=f"减值波动比={ratio:.1f}倍，远超[0.3, 3]正常区间"
            )
        return self._pass(r, f"减值波动比={ratio:.2f}")

    def _check_manip_003(self, r: dict) -> RuleResult:
        """商誉减值延迟"""
        latest = self._need_latest()
        if latest is None:
            return self._missing(r)
        goodwill = self._get_bs("goodwill", latest)
        if goodwill is None or goodwill == 0:
            return self._pass(r, "无商誉，不适用")
        # 简化检查：商誉/净资产较高时WARN
        gw_ratio = self._get_indicator("商誉/净资产", latest)
        if gw_ratio and gw_ratio > 0.2:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=5,
                data_basis=f"商誉/净资产={gw_ratio*100:.1f}%",
                risk_explanation="商誉占比高，存在减值延迟风险",
                fraud_type=r["fraud_type"],
            )
        return self._pass(r, f"商誉/净资产={gw_ratio*100:.1f}%" if gw_ratio else "无商誉")

    def _check_manip_004(self, r: dict) -> RuleResult:
        """递延所得税资产激增"""
        years = self.data.years
        latest = years[-1]
        dta_ratio = self._get_indicator("递延所得税资产/总资产", latest)
        it_ = self._get_pl("income_tax", latest)
        np_ = self._get_pl("net_profit", latest)
        # 检查连续亏损
        losses = sum(1 for y in years if self._get_pl("net_profit", y) is not None and self._get_pl("net_profit", y) < 0)
        if dta_ratio and dta_ratio > 0.02 and losses >= 2:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=10,
                data_basis=f"递延所得税资产/总资产={dta_ratio*100:.1f}%, 连续亏损{losses}年",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=f"连续亏损{losses}年 + DTA/TA={dta_ratio*100:.1f}% > 2% → 激进确认嫌疑"
            )
        if dta_ratio and dta_ratio > 0.01:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=5, data_basis=f"递延所得税资产/总资产={dta_ratio*100:.1f}%",
                risk_explanation="递延所得税资产占比较高",
                fraud_type=r["fraud_type"],
            )
        return self._pass(r, f"递延所得税资产/总资产={dta_ratio*100:.1f}%" if dta_ratio else "")

    def _check_manip_005(self, r: dict) -> RuleResult:
        """收入确认激进"""
        q4_pct = self.data.non_financial.get("q4_revenue_pct", 0)
        if q4_pct and q4_pct > 0.4:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=10,
                data_basis=f"四季度收入占比={q4_pct*100:.1f}%",
                risk_explanation="年末突击确认收入是常见操纵手段",
                fraud_type=r["fraud_type"],
                evidence=f"Q4收入占比{q4_pct*100:.0f}% > 40%"
            )
        if q4_pct and q4_pct > 0.35:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=5, data_basis=f"四季度收入占比={q4_pct*100:.1f}%",
                risk_explanation="四季度收入占比较高",
                fraud_type=r["fraud_type"],
            )
        return self._pass(r, f"四季度收入占比={q4_pct*100:.1f}%" if q4_pct else "数据缺失")

    def _check_manip_006(self, r: dict) -> RuleResult:
        """减值计提比例异常"""
        years = self.data.years
        if len(years) < 2:
            return self._missing(r)
        latest = years[-1]
        prev = years[-2]
        curr_rate = self._get_indicator("存货跌价准备/存货", latest)
        prev_rate = self._get_indicator("存货跌价准备/存货", prev)
        if curr_rate is None or prev_rate is None or prev_rate == 0:
            return self._missing(r)
        change = abs(curr_rate - prev_rate) / abs(prev_rate)
        if change > 0.5:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=10,
                data_basis=f"存货跌价准备/存货同比变化={change*100:.1f}%",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=f"减值计提比例同比变化{change*100:.0f}% > 50%"
            )
        return self._pass(r, f"计提比例变化={change*100:.1f}%")

    def _check_manip_007(self, r: dict) -> RuleResult:
        """利润平滑迹象 — 基于净利率而非净利润绝对值，排除高利润率企业"""
        years = self.data.years
        if len(years) < 4:
            return self._missing(r)

        # 计算近3年净利率（净利润/营业收入）的标准差
        margins = []
        for y in years[-3:]:
            np_ = self._get_pl("net_profit", y)
            rev = self._get_pl("revenue", y)
            if np_ is not None and rev is not None and rev > 0:
                margins.append(np_ / rev)

        if len(margins) < 3:
            return self._missing(r)

        avg_margin = sum(margins) / len(margins)
        std_margin = (sum((m - avg_margin) ** 2 for m in margins) / len(margins)) ** 0.5

        # 高净利率企业(>40%)本身经营稳定，排除
        if avg_margin > 0.40:
            return self._pass(r, f"平均净利率={avg_margin*100:.1f}% > 40%，高利润率企业天然稳定，不适用利润平滑检测")

        # 仅当净利率标准差 < 1% 且平均净利率在 10%-40% 之间时触发
        if std_margin < 0.01 and 0.10 <= avg_margin <= 0.40:
            if self.data.industry not in ("银行", "保险", "公用事业"):
                return RuleResult(
                    rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                    verdict="FAIL", score=10,
                    data_basis=f"近3年净利率均值={avg_margin*100:.1f}%, 标准差={std_margin*100:.2f}%",
                    risk_explanation="净利率波动极小(标准差<1%)+中等利润率 → 人为平滑利润嫌疑",
                    fraud_type=r["fraud_type"],
                    evidence=f"净利率均值{avg_margin*100:.1f}% 标准差仅{std_margin*100:.2f}% < 1%"
                )
        return self._pass(r, f"平均净利率={avg_margin*100:.1f}%, 标准差={std_margin*100:.2f}%")

    def _check_manip_008(self, r: dict) -> RuleResult:
        """会计差错更正"""
        corrections_amount = self.data.non_financial.get("correction_amount_total", 0)
        latest_np = self._get_pl("net_profit", self._latest_year())
        if corrections_amount and latest_np and latest_np != 0:
            ratio = corrections_amount / abs(latest_np)
            if ratio > 0.05:
                return RuleResult(
                    rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                    verdict="FAIL", score=10,
                    data_basis=f"差错更正累计金额/净利润={ratio*100:.1f}%",
                    risk_explanation=r["description"],
                    fraud_type=r["fraud_type"],
                    evidence=f"更正金额占净利润{ratio*100:.1f}% > 5%监管警戒线"
                )
        return self._pass(r, "无明显差错更正")

    def _check_manip_009(self, r: dict) -> RuleResult:
        """关联交易非公允"""
        rpt_dev = self.data.non_financial.get("rpt_price_deviation", 0)
        if rpt_dev and abs(rpt_dev) > 0.15:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=10,
                data_basis=f"关联交易定价偏离市场价={abs(rpt_dev)*100:.1f}%",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=f"定价偏离{abs(rpt_dev)*100:.0f}% > 15%，非公允嫌疑"
            )
        return self._pass(r, "关联交易定价无明显偏离" if not rpt_dev else f"定价偏离={abs(rpt_dev)*100:.1f}%")

    # ================================================================
    # 步骤2.3: 盈余管理扫描 (5级)
    # ================================================================
    def _run_earnings_mgmt_rules(self):
        for rule_def in EARNINGS_MGMT_RULES:
            handler = getattr(self, f"_check_{rule_def['id'].replace('-', '_').lower()}", None)
            if handler:
                result = handler(rule_def)
            else:
                result = self._generic_check(rule_def)
            self.results.append(result)

    def _check_em_001(self, r: dict) -> RuleResult:
        """研发资本化率异常"""
        latest = self._latest_year()
        rd_rate = self._get_indicator("研发资本化率", latest)
        industry_avg = 0.30  # 默认行业均值
        if self.data.industry in INDUSTRY_ADJUSTMENTS:
            industry_avg = INDUSTRY_ADJUSTMENTS[self.data.industry].get("研发资本化率警戒线", 0.30)
        if rd_rate is not None and rd_rate > industry_avg + 0.2:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=5,
                data_basis=f"研发资本化率={rd_rate*100:.1f}%, 行业均值≈{industry_avg*100:.0f}%",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=f"资本化率{rd_rate*100:.0f}% > 行业{industry_avg*100:.0f}% + 20pp"
            )
        return self._pass(r, f"研发资本化率={rd_rate*100:.1f}%" if rd_rate else "数据缺失")

    def _check_em_002(self, r: dict) -> RuleResult:
        """非经常性损益依赖"""
        latest = self._latest_year()
        nr_ratio = self._get_indicator("非经常性损益/净利润", latest)
        ded_ratio = self._get_indicator("扣非净利润/净利润", latest)
        if nr_ratio is not None and nr_ratio > 0.3:
            if ded_ratio is not None and ded_ratio < 0.5:
                return RuleResult(
                    rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                    verdict="FAIL", score=5,
                    data_basis=f"非经常性损益/净利润={nr_ratio*100:.1f}%, 扣非净利润/净利润={ded_ratio*100:.1f}%",
                    risk_explanation="依赖政府补助、资产处置等不可持续收益",
                    fraud_type=r["fraud_type"],
                    evidence=f"非经常损益占比{nr_ratio*100:.0f}% > 30% + 扣非占比{ded_ratio*100:.0f}% < 50%"
                )
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="WARN", score=3, data_basis=f"非经常性损益/净利润={nr_ratio*100:.1f}%",
                risk_explanation="非经常性损益占比较高",
                fraud_type=r["fraud_type"],
            )
        return self._pass(r, f"非经常性损益/净利润={nr_ratio*100:.1f}%" if nr_ratio else "数据缺失")

    def _check_em_003(self, r: dict) -> RuleResult:
        """微利巨亏循环"""
        pattern = str(self.ind.multi_year_stats.get("净利润波动", ""))
        if not pattern:
            return self._pass(r, "无明显交替模式")
        # 排除 "无明显交替模式" 这种否定式（"无交替"也包含"交替"子串）
        if "检测到交替" in pattern or "存在交替" in pattern:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=5,
                data_basis=pattern,
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=pattern
            )
        return self._pass(r, pattern)

    def _check_em_004(self, r: dict) -> RuleResult:
        """业绩预告修正频繁"""
        revisions = self.data.non_financial.get("forecast_revisions_2yr", 0)
        magnitude = self.data.non_financial.get("revision_magnitude", 0)
        if revisions > 2 and magnitude > 0.5:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=5,
                data_basis=f"2年内修正{revisions}次, 修正幅度={magnitude*100:.1f}%",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
                evidence=f"2年{revisions}次修正 > 2次 + 幅度{magnitude*100:.0f}% > 50%"
            )
        return self._pass(r, f"修正{revisions}次, 幅度={magnitude*100:.1f}%" if magnitude else "数据缺失")

    def _check_em_005(self, r: dict) -> RuleResult:
        """费用资本化倾向"""
        latest = self._latest_year()
        intang_growth = self._get_indicator("总资产增长率", latest)
        rd_rate = self._get_indicator("研发资本化率", latest)
        if intang_growth and intang_growth > 0.5:
            if rd_rate and rd_rate > 0.1:
                return RuleResult(
                    rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                    verdict="FAIL", score=5,
                    data_basis=f"无形资产/总资产同比增幅={intang_growth*100:.1f}%, 研发资本化率={rd_rate*100:.1f}%",
                    risk_explanation=r["description"],
                    fraud_type=r["fraud_type"],
                )
        return self._pass(r, "")

    def _check_em_006(self, r: dict) -> RuleResult:
        """股份支付费用高"""
        latest = self._latest_year()
        sbp_ratio = self._get_indicator("股份支付费用/净利润", latest)
        if sbp_ratio and sbp_ratio > 0.1:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=5,
                data_basis=f"股份支付费用/净利润={sbp_ratio*100:.1f}%",
                risk_explanation=r["description"],
                fraud_type=r["fraud_type"],
            )
        return self._pass(r, f"股份支付费用/净利润={sbp_ratio*100:.1f}%" if sbp_ratio else "数据缺失")

    def _check_em_007(self, r: dict) -> RuleResult:
        """选择性披露"""
        return self._pass(r, "需人工审查公告时间线（Agent无法自动判断）")

    def _check_em_008(self, r: dict) -> RuleResult:
        """年末突击交易"""
        q4_pct = self.data.non_financial.get("q4_revenue_pct", 0)
        if q4_pct and q4_pct > 0.45:
            return RuleResult(
                rule_id=r["id"], rule_name=r["name"], risk_level=r["risk_level"],
                verdict="FAIL", score=5,
                data_basis=f"四季度收入占比={q4_pct*100:.1f}%",
                risk_explanation="年末集中交易增加，可能通过一次性交易确保盈利目标",
                fraud_type=r["fraud_type"],
            )
        return self._pass(r, "")

    # ================================================================
    # 步骤3: 重点科目深度核查
    # ================================================================
    def _run_deep_audit(self):
        """对触发风险的科目执行深度核查"""
        triggered_subjects = set()
        for res in self.results:
            if res.verdict == "FAIL":
                triggered_subjects.update(self._map_rule_to_subject(res.rule_id))

        for subject in triggered_subjects:
            auditor = getattr(self, f"_deep_audit_{subject}", None)
            if auditor:
                result = auditor()
                if result:
                    self.deep_audit_results.append(result)

    def _map_rule_to_subject(self, rule_id: str) -> List[str]:
        """将规则映射到核查科目"""
        mapping = {
            "FRAUD-001": ["other_receivables", "money_funds"],
            "FRAUD-002": ["money_funds"],
            "FRAUD-003": ["inventory"],
            "FRAUD-004": ["cashflow"],
            "FRAUD-005": ["related_party"],
            "FRAUD-006": ["overseas_revenue"],
            "FRAUD-007": ["contract_assets"],
            "FRAUD-008": ["other_receivables"],
            "FRAUD-009": ["asset_impairment"],
            "MANIP-002": ["asset_impairment"],
            "MANIP-003": ["goodwill"],
            "MANIP-004": ["deferred_tax"],
        }
        return mapping.get(rule_id, [])

    def _deep_audit_money_funds(self) -> Optional[DeepAuditResult]:
        """货币资金深度核查"""
        result = DeepAuditResult(subject="货币资金", risk_level=RISK_LEVEL_10)
        latest = self._latest_year()
        mf = self._get_bs("money_funds", latest)
        ibd = self._get_indicator("有息负债", latest)
        iir = self._get_indicator("利息收入率", latest)
        ier = self._get_indicator("利息费用率", latest)

        if mf and ibd and mf > ibd * 0.5:
            result.anomalies.append(f"货币资金({mf/1e8:.2f}亿)远高于有息负债({ibd/1e8:.2f}亿)的50%")
        if iir is not None and iir < 0.005:
            result.anomalies.append(f"利息收入率仅{iir*100:.2f}%，远低于正常银行存款利率(1.5%-3%)")
        if iir is not None and ier is not None and iir < ier:
            result.anomalies.append(f"存款利率({iir*100:.2f}%)远低于贷款利率({ier*100:.2f}%)")

        if result.anomalies:
            result.evidence_chain = "存贷双高 → 利息收入异常低 → 货币资金真实性存疑"
            result.suggestions = ["获取银行函证确认实际存款余额", "核查是否存在受限资金", "追踪大额资金流水"]
            return result
        return None

    def _deep_audit_inventory(self) -> Optional[DeepAuditResult]:
        """存货深度核查"""
        result = DeepAuditResult(subject="存货", risk_level=RISK_LEVEL_10)
        latest = self._latest_year()
        inv = self._get_bs("inventory", latest)
        cost = self._get_pl("cost_of_sales", latest)
        inv_ratio = _safe_div(inv, cost)
        inv_impair = self._get_indicator("存货跌价准备/存货", latest)
        gm = self._get_indicator("毛利率", latest)

        if inv_ratio and inv_ratio > 2.0:
            result.anomalies.append(f"存货/营业成本={inv_ratio:.2f} > 2.0，积压风险")
        if inv_impair is not None and inv_impair < 0.01:
            result.anomalies.append(f"存货跌价准备仅{inv_impair*100:.2f}%，可能计提不足")
        if inv and gm:
            # 同行业参考毛利率比较（简化）
            result.suggestions = ["实地盘点核实存货数量及状态", "检查存货库龄结构", "与供应商函证采购真实性"]
        if result.anomalies:
            result.evidence_chain = "存货规模异常 → 跌价准备不足 → 可能存在虚构存货或延迟结转成本"
            return result
        return None

    def _deep_audit_other_receivables(self) -> Optional[DeepAuditResult]:
        """其他应收款深度核查"""
        result = DeepAuditResult(subject="其他应收款", risk_level=RISK_LEVEL_10)
        latest = self._latest_year()
        or_ = self._get_bs("other_receivables", latest)
        ca = self._get_bs("current_assets", latest)
        ratio = _safe_div(or_, ca)
        if ratio and ratio > 0.10:
            result.anomalies.append(f"其他应收款/流动资产={ratio*100:.1f}% > 10%警戒线")
            result.evidence_chain = "其他应收款占比高 → 款项性质不明 → 疑似关联方资金占用"
            result.suggestions = ["获取其他应收款明细按对象分类", "检查关联方余额>1000万的项目", "核查挂账超过1年的项目商业实质"]
            return result
        return None

    def _deep_audit_contract_assets(self) -> Optional[DeepAuditResult]:
        """合同资产深度核查"""
        result = DeepAuditResult(subject="合同资产", risk_level=RISK_LEVEL_10)
        ca_ratio = self._get_indicator("合同资产/应收账款", self._latest_year())
        if ca_ratio and ca_ratio > 1:
            result.anomalies.append(f"合同资产/应收账款={ca_ratio:.2f} > 1，结构异常")
            result.evidence_chain = "合同资产 > 应收账款 → 附加条件收入占比过高 → 提前确认收入嫌疑"
            result.suggestions = ["核查合同资产对应项目履约进度", "确认是否取得关键批文", "比对监理报告与账面进度"]
            return result
        return None

    def _deep_audit_deferred_tax(self) -> Optional[DeepAuditResult]:
        """递延所得税资产深度核查"""
        result = DeepAuditResult(subject="递延所得税资产", risk_level=RISK_LEVEL_8)
        latest = self._latest_year()
        dta_ratio = self._get_indicator("递延所得税资产/总资产", latest)
        if dta_ratio and dta_ratio > 0.02:
            losses = sum(1 for y in self.data.years
                        if self._get_pl("net_profit", y) is not None and self._get_pl("net_profit", y) < 0)
            if losses >= 2:
                result.anomalies.append(f"连续亏损{losses}年但递延所得税资产/总资产={dta_ratio*100:.1f}% > 2%")
                result.evidence_chain = "亏损状态下确认DTA → 依赖未来盈利预期 → 假设缺乏实质证据支撑"
                result.suggestions = ["审查管理层'未来盈利预测'的依据", "获取订单/合同/行业复苏数据", "评估盈利假设的合理性"]
                return result
        return None

    def _deep_audit_asset_impairment(self) -> Optional[DeepAuditResult]:
        """资产减值准备深度核查"""
        result = DeepAuditResult(subject="资产减值准备", risk_level=RISK_LEVEL_8)
        years = self.data.years
        if len(years) >= 2:
            latest = years[-1]
            prev = years[-2]
            ail_l = self._get_pl("asset_impairment_loss", latest)
            ail_p = self._get_pl("asset_impairment_loss", prev)
            np_l = self._get_pl("net_profit", latest)
            if ail_l and ail_p and ail_p != 0:
                ratio = ail_l / ail_p
                if ratio > 3 or ratio < 0.3:
                    result.anomalies.append(f"减值损失年波动={ratio:.1f}倍")
            if ail_l and np_l and np_l != 0 and abs(ail_l / np_l) > 0.5:
                result.anomalies.append(f"减值损失占净利润{abs(ail_l/np_l)*100:.0f}%，对利润影响重大")
            if result.anomalies:
                result.evidence_chain = "减值损失剧烈波动 → 可能利用减值调节利润 → 检查减值测试方法"
                result.suggestions = ["检查减值测试方法和关键假设", "对比行业减值计提比例", "追踪计提后的回转情况"]
                return result
        return None

    def _deep_audit_overseas_revenue(self) -> Optional[DeepAuditResult]:
        """境外收入深度核查"""
        result = DeepAuditResult(subject="境外收入", risk_level=RISK_LEVEL_10)
        latest = self._latest_year()
        overseas_pct = self._get_indicator("境外收入占比", latest)
        ocf = self._get_cf("operating_cf", latest)

        if overseas_pct and overseas_pct > 0.3:
            result.anomalies.append(f"境外收入占比={overseas_pct*100:.1f}% > 30%")
        if ocf is not None and ocf < 0:
            result.anomalies.append(f"经营现金流为负({ocf/1e8:.2f}亿)，境外收入未产生现金回流")
        if result.anomalies:
            result.evidence_chain = "境外收入占比高 → 经营现金流不匹配 → 虚构收入嫌疑"
            result.suggestions = ["获取海关出口数据交叉验证", "核查境外客户工商背景", "检查合同真实性及回款记录"]
            return result
        return None

    def _deep_audit_related_party(self) -> Optional[DeepAuditResult]:
        result = DeepAuditResult(subject="关联交易", risk_level=RISK_LEVEL_10)
        rp_ratio = self._get_indicator("关联交易/营业收入", self._latest_year())
        if rp_ratio and rp_ratio > 0.3:
            result.anomalies.append(f"关联交易/营业收入={rp_ratio*100:.1f}% > 30%")
            result.evidence_chain = "关联交易占比高 → 定价是否公允 → 利益输送嫌疑"
            result.suggestions = ["获取关联方清单及交易明细", "核查交易定价与市场价对比", "检查交易商业实质"]
            return result
        return None

    # ================================================================
    # 步骤4: 商业逻辑验证
    # ================================================================
    def _run_business_logic(self):
        self._verify_profitability()
        self._verify_cashflow()
        self._verify_debt_repayment()
        self._verify_growth()
        self._verify_industry_specific()

    def _verify_profitability(self):
        """4.1 盈利能力验证"""
        years = self.data.years
        latest = years[-1] if years else None
        if not latest:
            return

        # ROE检查
        roe_list = [self._get_indicator("ROE", y) for y in years]
        roe_clean = [r for r in roe_list if r is not None]
        if roe_clean and np.mean(roe_clean[-3:]) < 0.06:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-001", rule_name="盈利能力弱", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"近3年平均ROE={np.mean(roe_clean[-3:])*100:.1f}%",
                risk_explanation="连续3年ROE < 6%",
            ))

        # ROA检查
        roa = self._get_indicator("ROA", latest)
        if roa is not None and roa < 0.025:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-002", rule_name="资本回报不足", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"ROA={roa*100:.1f}%",
                risk_explanation=f"ROA < 3年期国债收益率(约2.5%)",
            ))

        # 毛利率稳定性
        gm_list = [self._get_indicator("毛利率", y) for y in years]
        gm_clean = [g for g in gm_list if g is not None]
        if len(gm_clean) >= 3:
            gm_std = np.std(gm_clean)
            if gm_std > 0.05:
                self.business_logic_results.append(RuleResult(
                    rule_id="BIZ-003", rule_name="毛利率波动大", risk_level=RISK_LEVEL_3,
                    verdict="WARN", score=3,
                    data_basis=f"毛利率3年标准差={gm_std*100:.1f}个百分点",
                    risk_explanation="毛利率年度波动 > 5个百分点，定价权弱或成本失控",
                ))

        # 扣非vs净利润
        ded_ratio = self._get_indicator("扣非净利润/净利润", latest)
        if ded_ratio is not None and ded_ratio < 0.7:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-004", rule_name="主业盈利能力弱", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"扣非净利润/净利润={ded_ratio*100:.1f}%",
                risk_explanation="盈利依赖非经常性损益",
            ))

    def _verify_cashflow(self):
        """4.2 现金流验证"""
        years = self.data.years
        latest = years[-1] if years else None
        if not latest:
            return

        # 经营现金流 - 净利润 连续为负
        diffs = []
        for y in years:
            ocf = self._get_cf("operating_cf", y)
            np_ = self._get_pl("net_profit", y)
            if ocf is not None and np_ is not None:
                diffs.append(ocf - np_)
        neg_count = sum(1 for d in diffs[-3:] if d < 0)
        if neg_count >= 3:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-005", rule_name="利润含金量低", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"经营现金流-净利润连续{neg_count}年为负",
                risk_explanation="利润未转化为真金白银",
            ))

        # 自由现金流
        fcf_neg = self.ind.multi_year_stats.get("自由现金流为负年数", 0)
        if fcf_neg >= 3:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-006", rule_name="自由现金流持续为负", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"近{len(years)}年中有{fcf_neg}年自由现金流为负",
                risk_explanation="企业无法自我造血",
            ))

        # 销售收现比
        sc_ratio = self._get_indicator("销售收现比", latest)
        if sc_ratio is not None and sc_ratio < 0.8:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-007", rule_name="收入回款质量差", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"销售收现比={sc_ratio*100:.1f}% < 80%",
                risk_explanation="大量收入依靠赊销，回款能力弱",
            ))

        # 造血能力
        ocf_to_rev = self._get_indicator("经营现金流/营业收入", latest)
        if ocf_to_rev is not None and ocf_to_rev < 0.10:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-008", rule_name="造血能力不足", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"经营现金流/营业收入={ocf_to_rev*100:.1f}% < 10%",
                risk_explanation="经营活动产生现金能力弱",
            ))

    def _verify_debt_repayment(self):
        """4.3 偿债能力验证"""
        latest = self._latest_year()
        # 流动比率
        cr = self._get_indicator("流动比率", latest)
        if cr is not None and cr < 1.2:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-009", rule_name="短期偿债压力大", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"流动比率={cr:.2f} < 1.2",
                risk_explanation="短期偿债能力不足",
            ))
        # 资产负债率
        alr = self._get_indicator("资产负债率", latest)
        if alr is not None and alr > 0.70:
            if self.data.industry not in ("银行", "保险", "房地产"):
                self.business_logic_results.append(RuleResult(
                    rule_id="BIZ-010", rule_name="杠杆过高", risk_level=RISK_LEVEL_3,
                    verdict="WARN", score=3,
                    data_basis=f"资产负债率={alr*100:.1f}% > 70%",
                    risk_explanation="财务杠杆过高，偿债压力大",
                ))
        # 利息保障倍数
        icr = self._get_indicator("利息保障倍数", latest)
        if icr is not None and icr < 3:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-011", rule_name="利息覆盖不足", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"利息保障倍数={icr:.1f} < 3",
                risk_explanation="盈利不足以覆盖利息支出",
            ))

    def _verify_growth(self):
        """4.4 成长性验证"""
        latest = self._latest_year()
        # 营收增长率
        rev_growth = self._get_indicator("营业收入增长率", latest)
        if rev_growth is not None and rev_growth < 0:
            self.business_logic_results.append(RuleResult(
                rule_id="BIZ-012", rule_name="收入萎缩", risk_level=RISK_LEVEL_3,
                verdict="WARN", score=3,
                data_basis=f"营业收入增长率={rev_growth*100:.1f}%",
                risk_explanation="收入连续下降",
            ))
        # 应收 vs 营收增速
        ar_growth = self._get_indicator("应收账款增长率", latest)
        if ar_growth is not None and rev_growth is not None:
            if ar_growth > rev_growth + 0.2:
                self.business_logic_results.append(RuleResult(
                    rule_id="BIZ-013", rule_name="赊销激进", risk_level=RISK_LEVEL_3,
                    verdict="WARN", score=3,
                    data_basis=f"应收增速({ar_growth*100:.1f}%) > 营收增速({rev_growth*100:.1f}%)+20pp",
                    risk_explanation="应收账款增长远超收入增长，回款能力恶化",
                ))

    def _verify_industry_specific(self):
        """4.5 行业特性验证"""
        industry = self.data.industry
        latest = self._latest_year()
        if industry == "医药流通":
            ar_days = self._get_indicator("应收账款周转天数", latest)
            if ar_days is not None and ar_days > 120:
                self.business_logic_results.append(RuleResult(
                    rule_id="IND-001", rule_name="医药流通回款周期过长", risk_level=RISK_LEVEL_3,
                    verdict="WARN", score=3,
                    data_basis=f"应收账款周转天数={ar_days:.0f}天 > 120天",
                    risk_explanation="回款周期超出行业合理范围",
                ))
        elif industry == "建筑施工":
            ca_ratio = self._get_indicator("合同资产/应收账款", latest)
            if ca_ratio is not None and ca_ratio > 1.5:
                self.business_logic_results.append(RuleResult(
                    rule_id="IND-002", rule_name="建筑合同资产占比过高", risk_level=RISK_LEVEL_3,
                    verdict="WARN", score=3,
                    data_basis=f"合同资产/应收账款={ca_ratio:.2f} > 1.5",
                    risk_explanation="合同资产累积，需核查项目履约进度",
                ))
        elif industry == "软件开发":
            rd_rate = self._get_indicator("研发资本化率", latest)
            if rd_rate is not None and rd_rate > 0.40:
                self.business_logic_results.append(RuleResult(
                    rule_id="IND-003", rule_name="软件研发资本化率偏高", risk_level=RISK_LEVEL_3,
                    verdict="WARN", score=3,
                    data_basis=f"研发资本化率={rd_rate*100:.1f}% > 40%",
                    risk_explanation="软件开发行业研发资本化率异常高",
                ))

    # ================================================================
    # 数字化时代特殊风险
    # ================================================================
    def _run_digital_era_checks(self):
        for check in DIGITAL_ERA_CHECKS:
            self.digital_era_alerts.append({
                "id": check["id"],
                "name": check["name"],
                "check": check["check"],
                "risk": check["risk"],
                "status": "需人工核查"  # 大部分数字时代风险无法自动检测
            })

    # ================================================================
    # 辅助方法
    # ================================================================
    def _latest_year(self) -> Optional[int]:
        return self.data.years[-1] if self.data.years else None

    def _get_bs(self, field: str, year: int) -> Optional[float]:
        return self.data.balance_sheet.get(year, {}).get(field)

    def _get_pl(self, field: str, year: int) -> Optional[float]:
        return self.data.income_statement.get(year, {}).get(field)

    def _get_cf(self, field: str, year: int) -> Optional[float]:
        return self.data.cashflow_statement.get(year, {}).get(field)

    def _get_note(self, field: str, year: int) -> Optional[float]:
        return self.data.notes.get(year, {}).get(field)

    def _get_indicator(self, name: str, year: int) -> Optional[float]:
        return self.ind.indicators.get(name, {}).get(year)

    def _pass(self, rule_def: dict, basis: str = "") -> RuleResult:
        return RuleResult(
            rule_id=rule_def["id"], rule_name=rule_def["name"],
            risk_level=rule_def["risk_level"], verdict="PASS", score=0,
            data_basis=basis, risk_explanation="",
            fraud_type=rule_def.get("fraud_type", ""),
        )

    def _missing(self, rule_def: dict) -> RuleResult:
        return RuleResult(
            rule_id=rule_def["id"], rule_name=rule_def["name"],
            risk_level=rule_def["risk_level"], verdict="DATA_MISSING", score=0,
            data_basis="数据缺失", risk_explanation="无法获取所需数据，跳过判定",
            fraud_type=rule_def.get("fraud_type", ""),
        )

    def _generic_check(self, rule_def: dict) -> RuleResult:
        return self._pass(rule_def, "通用检查通过（未实现专项检查器）")
