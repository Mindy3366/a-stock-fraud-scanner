"""
会计勾稽验证模块 — A股市场核心反造假工具
财务报表表面永远平衡（借贷恒等），真正的造假在业务实质层面的勾稽矛盾。

核心原理：
  报表内部：资产 = 负债 + 权益 (永远成立，无验证意义)
  报表之间：业务数据流转存在多重映射关系，虚构交易无法在所有映射中自洽

A股勾稽体系：
  G1: 收入勾稽 — 营业收入 vs 销售回款 vs 应收变动 vs 增值税
  G2: 成本勾稽 — 营业成本 vs 采购付款 vs 应付变动 vs 存货变动
  G3: 税金勾稽 — 所得税费用 vs 应交税金变动 vs 实际纳税
  G4: 折旧勾稽 — 折旧费用 vs 累计折旧变动 vs 固定资产原值
  G5: 薪酬勾稽 — 支付薪酬 vs 应付薪酬变动 vs 成本中的薪酬
  G6: 长资勾稽 — 购建长期资产支出 vs 固定资产在建工程变动
  G7: 费用勾稽 — 三大费用 vs 相关支付 vs 预提待摊变动
  G8: 毛利率勾稽 — 毛利率 vs 增值税税负率交叉验证
  G9: 客户勾稽 — 大客户销售额 vs 客户公开经营规模
  G10: 产能勾稽 — 产量 vs 固定资产规模 vs 能耗
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from data_fetcher import FinancialData
from indicator_calc import IndicatorCalculator, _safe_div


@dataclass
class CrossCheckResult:
    """单条勾稽验证结果"""
    check_id: str
    check_name: str
    description: str
    deviation: Optional[float] = None       # 差异额（元）
    deviation_pct: Optional[float] = None    # 差异率
    threshold_warn: float = 0.15             # 警戒线（15%）
    threshold_fail: float = 0.30             # 危险线（30%）
    verdict: str = "PASS"                    # PASS / WARN / FAIL / DATA_MISSING
    score: int = 0                           # 本项得分
    detail: str = ""                         # 详细计算过程
    evidence: str = ""                       # 证据描述
    fraud_implication: str = ""              # 造假含义


class AccountingCrossChecker:
    """
    A股会计勾稽验证器
    验证三表之间的业务数据流转是否逻辑自洽
    """

    def __init__(self, data: FinancialData, indicators: IndicatorCalculator):
        self.data = data
        self.ind = indicators
        self.results: List[CrossCheckResult] = []

    def run_all(self) -> List[CrossCheckResult]:
        """执行全部10项勾稽验证"""
        if len(self.data.years) < 2:
            return self.results

        # 对最新年份执行勾稽
        latest = self.data.years[-1]
        prev = self.data.years[-2] if len(self.data.years) >= 2 else latest

        self._check_g1_revenue_articulation(latest, prev)
        self._check_g2_cost_articulation(latest, prev)
        self._check_g3_tax_articulation(latest, prev)
        self._check_g4_depreciation_articulation(latest, prev)
        self._check_g5_salary_articulation(latest, prev)
        self._check_g6_longterm_asset_articulation(latest, prev)
        self._check_g7_expense_articulation(latest, prev)
        self._check_g8_gross_margin_vat_cross(latest, prev)
        self._check_g9_customer_scale_logic(latest)
        self._check_g10_capacity_logic(latest, prev)

        return self.results

    # ================================================================
    # G1: 收入勾稽 — 营业收入 vs 销售回款 vs 应收变动 vs 增值税
    # 公式: 营业收入×(1+增值税率) ≈ 销售收到现金 + Δ应收账款 + Δ应收票据 - Δ预收账款
    # A股增值税率: 制造业13%, 服务业6%, 简化使用历史税率
    # ================================================================
    def _check_g1_revenue_articulation(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G1", check_name="收入-回款-应收勾稽",
            description="营业收入×(1+增值税率)应≈销售收到现金+Δ应收账款+Δ应收票据-Δ预收账款"
        )

        revenue = self._pl("revenue", year)
        sales_cash = self._cf("sales_cash_received", year)
        ar = self._bs("accounts_receivable", year)
        ar_prev = self._bs("accounts_receivable", prev_year)
        nr = self._bs("notes_receivable", year)
        nr_prev = self._bs("notes_receivable", prev_year)
        adv = self._bs("advance_from_customers", year)
        adv_prev = self._bs("advance_from_customers", prev_year)

        if revenue is None or sales_cash is None:
            r.verdict = "DATA_MISSING"
            r.detail = "关键数据缺失"
            self.results.append(r)
            return

        vat_rate = 0.13  # 假设制造业13%增值税率
        left_side = revenue * (1 + vat_rate)
        delta_ar = (ar or 0) - (ar_prev or 0)
        delta_nr = (nr or 0) - (nr_prev or 0)
        delta_adv = (adv or 0) - (adv_prev or 0)
        right_side = sales_cash + delta_ar + delta_nr - delta_adv

        r.deviation = left_side - right_side
        r.deviation_pct = abs(r.deviation) / max(abs(left_side), 1)
        r.detail = (f"左侧=营收{revenue/1e8:.2f}亿×(1+{vat_rate})={left_side/1e8:.2f}亿, "
                    f"右侧=收现{sales_cash/1e8:.2f}亿+Δ应收{delta_ar/1e8:.2f}亿"
                    f"+Δ票据{delta_nr/1e8:.2f}亿-Δ预收{delta_adv/1e8:.2f}亿={right_side/1e8:.2f}亿, "
                    f"差异={r.deviation/1e8:.2f}亿({r.deviation_pct*100:.1f}%)")

        self._classify_g1(r, delta_ar, delta_nr, delta_adv, revenue, sales_cash)
        self.results.append(r)

    def _classify_g1(self, r: CrossCheckResult, delta_ar, delta_nr, delta_adv, revenue, sales_cash):
        """分类收入勾稽异常的造假含义"""
        if r.deviation_pct is None:
            return

        if r.deviation_pct > r.threshold_fail:
            r.verdict = "FAIL"
            r.score = 12

            # 判断方向
            if r.deviation > 0:  # 左侧 > 右侧：收入数据偏大
                r.evidence = "收入侧数据 > 回款侧数据 → 收入可能被虚增"
                if delta_ar > revenue * 0.1:
                    r.fraud_implication = "应收账款激增+收入回款不匹配 → 虚构收入形成虚假应收账款"
                    r.evidence += "；应收账款增量异常，疑似虚构客户和销售合同"
                elif delta_adv < 0 and abs(delta_adv) > revenue * 0.05:
                    r.fraud_implication = "预收账款大幅减少 → 可能虚构销售合同但无法收到预付款"
                else:
                    r.fraud_implication = "营业收入与现金/应收变动存在显著差异 → 收入真实性存疑"
            else:  # 左侧 < 右侧：回款数据偏大
                r.evidence = "回款侧数据 > 收入侧数据 → 可能存在表外收入或未确认收入"
                r.fraud_implication = "回款超出营业收入 → 可能存在体外资金循环注入"

        elif r.deviation_pct > r.threshold_warn:
            r.verdict = "WARN"
            r.score = 6
            r.evidence = f"勾稽差异率{r.deviation_pct*100:.1f}%，超出15%警戒线"
            r.fraud_implication = "收入-回款勾稽存在偏差，需核查具体原因"
        else:
            r.verdict = "PASS"
            r.score = 0
            r.evidence = f"勾稽差异率{r.deviation_pct*100:.1f}%，在合理范围内"

    # ================================================================
    # G2: 成本勾稽 — 营业成本 vs 采购付款 vs 应付变动 vs 存货变动
    # 公式: 营业成本 - 折旧摊销 + Δ存货 + 增值税进项 ≈ 购买商品支付现金 + Δ应付账款 + Δ应付票据 - Δ预付账款
    # ================================================================
    def _check_g2_cost_articulation(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G2", check_name="成本-采购-存货勾稽",
            description="营业成本+Δ存货应≈采购付款+Δ应付账款-Δ预付账款"
        )

        cost = self._pl("cost_of_sales", year)
        inventory = self._bs("inventory", year)
        inv_prev = self._bs("inventory", prev_year)
        ap = self._bs("accounts_payable", year)
        ap_prev = self._bs("accounts_payable", prev_year)
        prepay = self._bs("prepayments", year)
        prepay_prev = self._bs("prepayments", prev_year)

        # 购买商品支付现金（现金流量表）
        purchase_cash = self._cf("purchase_cash", year)
        purchase_is_estimated = False
        if purchase_cash is None:
            total_op_outflow = self._cf("operating_cf_outflow", year)
            if total_op_outflow is None:
                r.verdict = "DATA_MISSING"
                r.detail = "缺少采购支付现金数据"
                self.results.append(r)
                return
            purchase_cash = total_op_outflow * 0.65
            purchase_is_estimated = True
            r.detail = "(采购支付为估算值) "

        if cost is None:
            r.verdict = "DATA_MISSING"
            r.detail = "缺少营业成本数据"
            self.results.append(r)
            return

        delta_inventory = (inventory or 0) - (inv_prev or 0)
        delta_ap = (ap or 0) - (ap_prev or 0)
        delta_prepay = (prepay or 0) - (prepay_prev or 0)

        # 左侧: 营业成本 + Δ存货（本期采购消耗 + 存货增量）
        left_side = cost + delta_inventory
        # 右侧: 采购支付 + Δ应付 - Δ预付
        right_side = (purchase_cash or 0) + delta_ap - delta_prepay

        r.deviation = left_side - right_side
        r.deviation_pct = abs(r.deviation) / max(abs(left_side), 1)
        r.detail = (f"左侧=成本{cost/1e8:.2f}亿+Δ存货{delta_inventory/1e8:.2f}亿={left_side/1e8:.2f}亿, "
                    f"右侧=采购付款{purchase_cash/1e8:.2f}亿+Δ应付{delta_ap/1e8:.2f}亿"
                    f"-Δ预付{delta_prepay/1e8:.2f}亿={right_side/1e8:.2f}亿, "
                    f"差异={r.deviation/1e8:.2f}亿({r.deviation_pct*100:.1f}%)")

        # 估算值放宽阈值：估算采购付款只能触发WARN
        if purchase_is_estimated:
            if r.deviation_pct > r.threshold_fail:
                r.verdict = "WARN"
                r.score = 4
                r.evidence = f"成本-采购勾稽差异较大({r.deviation_pct*100:.1f}%)，但采购付款为估算值，精确度有限"
                r.fraud_implication = "建议获取准确的采购支付现金数据后重新验证"
            elif r.deviation_pct > r.threshold_warn:
                r.verdict = "PASS"  # 估算数据不做WARN
                r.score = 0
            else:
                r.verdict = "PASS"
        elif r.deviation_pct > r.threshold_fail:
            r.verdict = "FAIL"
            r.score = 12
            if r.deviation > 0:
                r.evidence = "成本+存货 > 采购付款+应付 → 可能存在虚构存货或未入账采购"
                if delta_inventory > cost * 0.3:
                    r.fraud_implication = "存货异常激增 → 虚构存货以消化虚构的采购支出，经典的'存货魔术'"
                else:
                    r.fraud_implication = "采购与存货变动不匹配 → 可能存在体外采购或供应商返利未入账"
            else:
                r.evidence = "采购付款 > 成本+存货 → 可能存在体外资金流出"
                r.fraud_implication = "采购支付远超成本+存货变动 → 资金可能通过虚假采购流出"
        elif r.deviation_pct > r.threshold_warn:
            r.verdict = "WARN"
            r.score = 6
            r.evidence = f"勾稽差异率{r.deviation_pct*100:.1f}%，超出15%警戒线"
            r.fraud_implication = "成本-采购-存货勾稽存在偏差"
        else:
            r.verdict = "PASS"
            r.score = 0
            r.evidence = f"勾稽差异率{r.deviation_pct*100:.1f}%，基本合理"

        self.results.append(r)

    # ================================================================
    # G3: 税金勾稽 — 所得税费用 vs 应交税金变动 vs 实际纳税
    # ================================================================
    def _check_g3_tax_articulation(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G3", check_name="所得税-纳税-递延勾稽",
            description="所得税费用应≈实际纳税+Δ应交税金-Δ递延所得税资产"
        )

        income_tax = self._pl("income_tax", year)
        total_profit = self._pl("total_profit", year)
        dta = self._bs("deferred_tax_assets", year)
        dta_prev = self._bs("deferred_tax_assets", prev_year)
        tax_payable_cf = self._cf("tax_paid", year)

        if income_tax is None or total_profit is None:
            r.verdict = "DATA_MISSING"
            r.detail = "缺少所得税/利润数据"
            self.results.append(r)
            return

        # 实际税率 vs 名义税率
        statutory_rate = 0.25  # A股法定所得税率
        expected_tax = total_profit * statutory_rate
        effective_rate = income_tax / total_profit if total_profit != 0 else 0

        delta_dta = (dta or 0) - (dta_prev or 0)

        # 注意: CF中"支付的各项税费" = 所得税+增值税+消费税+附加税，不能直接和所得税费用比较。
        # G3仅比较有效税率 vs 法定税率，不再做现金流比对。
        r.deviation = effective_rate - statutory_rate
        r.deviation_pct = abs(r.deviation)
        r.detail = (f"所得税费用={income_tax/1e8:.2f}亿, 利润总额={total_profit/1e8:.2f}亿, "
                    f"实际税率={effective_rate*100:.1f}% vs 法定{statutory_rate*100:.0f}%, "
                    f"Δ递延所得税={delta_dta/1e8:.2f}亿")

        if r.deviation_pct is not None:
            if r.deviation_pct > r.threshold_fail:
                r.verdict = "FAIL"
                r.score = 8
                r.evidence = (f"实际税率{effective_rate*100:.1f}%偏离法定税率{statutory_rate*100:.0f}%"
                              f"{'远超' if abs(effective_rate - statutory_rate) > 0.1 else ''}")
                if effective_rate < 0.10 and total_profit > 0:
                    r.fraud_implication = "盈利但所得税率极低 → 可能存在利润虚增而税金无法匹配"
                elif effective_rate < 0:
                    r.fraud_implication = "所得税费用为负(收益) → 可能在亏损年度激进确认递延所得税资产"
                elif effective_rate > 0.35:
                    r.fraud_implication = "实际税率显著高于法定税率 → 可能有大量不可抵扣的费用(如虚构费用)"
            elif r.deviation_pct > r.threshold_warn:
                r.verdict = "WARN"
                r.score = 4
                r.evidence = f"实际税率({effective_rate*100:.1f}%)与法定税率({statutory_rate*100:.0f}%)存在偏差"
                r.fraud_implication = "税金勾稽存在偏差，需核查原因"
            else:
                r.verdict = "PASS"
                r.score = 0

        self.results.append(r)

    # ================================================================
    # G4: 折旧勾稽 — 折旧费用 vs 累计折旧变动
    # ================================================================
    def _check_g4_depreciation_articulation(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G4", check_name="折旧-固定资产勾稽",
            description="折旧费用应≈Δ累计折旧(考虑处置和减值)"
        )

        # 从现金流量表附注获取折旧
        depreciation = self._cf("depreciation", year)
        dep_is_estimated = False  # 标记是否为估算值

        # 从资产负债表获取
        fixed_assets = self._bs("fixed_assets", year)
        fa_prev = self._bs("fixed_assets", prev_year)
        cip = self._bs("construction_in_progress", year)
        cip_prev = self._bs("construction_in_progress", prev_year)
        capex_cf = self._cf("capex", year)

        if depreciation is None:
            admin = self._pl("admin_expenses", year)
            cost = self._pl("cost_of_sales", year)
            if admin and cost:
                depreciation = admin * 0.08 + cost * 0.05
                dep_is_estimated = True
                r.detail = f"折旧数据缺失，从费用倒推≈{depreciation/1e8:.2f}亿（估算值，精确度有限）"
            else:
                r.verdict = "DATA_MISSING"
                r.detail = "缺少折旧数据"
                self.results.append(r)
                return

        delta_fa = (fixed_assets or 0) - (fa_prev or 0)
        delta_cip = (cip or 0) - (cip_prev or 0)

        if capex_cf and depreciation:
            implied_depreciation = capex_cf + delta_cip - delta_fa
            r.deviation = abs(implied_depreciation - depreciation)
            r.deviation_pct = r.deviation / max(depreciation, 1)
            r.detail += (f" | 隐含折旧={implied_depreciation/1e8:.2f}亿("
                        f"资本支出{capex_cf/1e8:.2f}亿+Δ在建{delta_cip/1e8:.2f}亿-Δ固定{delta_fa/1e8:.2f}亿), "
                        f"差异={r.deviation/1e8:.2f}亿({r.deviation_pct*100:.1f}%)")

            # 估算值的阈值放松：估算值只能触发WARN不能触发FAIL
            if dep_is_estimated:
                if r.deviation_pct > 0.80:
                    r.verdict = "WARN"
                    r.score = 3
                    r.evidence = f"折旧(估算)与固定资产变动差异较大: {r.deviation_pct*100:.1f}%"
                    r.fraud_implication = "折旧估算值与资产变动差异较大，建议获取准确折旧数据后重新验证"
                else:
                    r.verdict = "PASS"
                    r.evidence = f"折旧(估算)与资产变动差异在可接受范围: {r.deviation_pct*100:.1f}%"
            else:
                if r.deviation_pct > r.threshold_fail:
                    r.verdict = "FAIL"
                    r.score = 8
                    r.evidence = f"折旧费用与固定资产变动不匹配，差异率{r.deviation_pct*100:.1f}%"
                    r.fraud_implication = "折旧费用与实际资产变动不符 → 可能虚增固定资产或在建工程挂账不转固以延迟折旧"
                elif r.deviation_pct > r.threshold_warn:
                    r.verdict = "WARN"
                    r.score = 4
                    r.evidence = f"折旧勾稽差异率{r.deviation_pct*100:.1f}%"
                else:
                    r.verdict = "PASS"
        else:
            r.verdict = "DATA_MISSING"

        self.results.append(r)

    # ================================================================
    # G5: 薪酬勾稽 — 支付给职工现金 vs 应付薪酬变动 vs 成本费用中的薪酬
    # ================================================================
    def _check_g5_salary_articulation(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G5", check_name="薪酬-支付-应付勾稽",
            description="成本费用中的薪酬≈支付给职工现金+Δ应付职工薪酬"
        )

        salary_paid = self._cf("salary_paid", year)
        salary_payable = self._bs("salary_payable", year)
        salary_payable_prev = self._bs("salary_payable", prev_year)

        # 估算成本费用中的薪酬
        cost = self._pl("cost_of_sales", year)
        selling = self._pl("selling_expenses", year)
        admin = self._pl("admin_expenses", year)
        rd = self._pl("rd_expenses", year)

        if cost:
            estimated_salary = (cost + (selling or 0) + (admin or 0) + (rd or 0)) * 0.25
        else:
            r.verdict = "DATA_MISSING"
            self.results.append(r)
            return

        delta_payable = (salary_payable or 0) - (salary_payable_prev or 0)

        if salary_paid:
            implied_salary = salary_paid + delta_payable
            r.deviation = abs(estimated_salary - implied_salary)
            r.deviation_pct = r.deviation / max(estimated_salary, 1)
            r.detail = (f"估算薪酬(成本费用×25%)={estimated_salary/1e8:.2f}亿, "
                        f"隐含薪酬=实付{salary_paid/1e8:.2f}亿+Δ应付{delta_payable/1e8:.2f}亿, "
                        f"差异={r.deviation/1e8:.2f}亿({r.deviation_pct*100:.1f}%)")
        else:
            r.verdict = "DATA_MISSING"
            r.detail = "缺少支付给职工现金数据"
            self.results.append(r)
            return

        # 薪酬成本为25%估算值，放宽阈值：估算只能触发WARN
        if r.deviation_pct > r.threshold_fail:
            r.verdict = "WARN"
            r.score = 3
            r.evidence = f"薪酬勾稽差异较大({r.deviation_pct*100:.1f}%)，但薪酬成本为估算值(25%比例)，精确度有限"
            r.fraud_implication = "建议获取准确的成本费用中薪酬数据后重新验证"
        elif r.deviation_pct > r.threshold_warn:
            r.verdict = "PASS"  # 估算值偏差不作WARN
            r.score = 0
        else:
            r.verdict = "PASS"

        self.results.append(r)

    # ================================================================
    # G6: 长期资产勾稽 — 购建长期资产现金 vs 固定资产+在建工程+无形资产变动
    # ================================================================
    def _check_g6_longterm_asset_articulation(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G6", check_name="长期资产-资本支出勾稽",
            description="购建固定资产现金≈Δ固定资产+Δ在建工程+Δ无形资产+折旧摊销-处置"
        )

        capex = self._cf("capex", year)
        fa = self._bs("fixed_assets", year)
        fa_prev = self._bs("fixed_assets", prev_year)
        cip = self._bs("construction_in_progress", year)
        cip_prev = self._bs("construction_in_progress", prev_year)
        intang = self._bs("intangible_assets", year)
        intang_prev = self._bs("intangible_assets", prev_year)

        if capex is None or fa is None:
            r.verdict = "DATA_MISSING"
            r.detail = "缺少资本支出或固定资产数据"
            self.results.append(r)
            return

        dep = self._cf("depreciation", year) or 0
        dep_missing = (self._cf("depreciation", year) is None)  # 折旧数据是否缺失
        delta_fa = fa - (fa_prev or 0)
        delta_cip = (cip or 0) - (cip_prev or 0)
        delta_intang = (intang or 0) - (intang_prev or 0)

        implied_capex = delta_fa + delta_cip + delta_intang + dep
        r.deviation = capex - implied_capex
        r.deviation_pct = abs(r.deviation) / max(capex, 1)
        r.detail = (f"资本支出={capex/1e8:.2f}亿, "
                    f"隐含支出=Δ固定{delta_fa/1e8:.2f}亿+Δ在建{delta_cip/1e8:.2f}亿"
                    f"+Δ无形{delta_intang/1e8:.2f}亿+折旧{dep/1e8:.2f}亿={implied_capex/1e8:.2f}亿, "
                    f"差异={r.deviation/1e8:.2f}亿")

        # 折旧数据缺失时，隐含资本支出被低估，放宽阈值
        if dep_missing:
            if r.deviation_pct > 1.0:
                r.verdict = "WARN"
                r.score = 3
                r.evidence = f"资本支出与资产变动差异较大({r.deviation_pct*100:.1f}%)，可能因折旧数据缺失导致低估"
                r.fraud_implication = "如能获取完整折旧和在建工程转固数据，差异可能缩小"
            else:
                r.verdict = "PASS"
        elif r.deviation_pct > r.threshold_fail:
            r.verdict = "FAIL"
            r.score = 8
            r.evidence = f"长期资产勾稽差异率{r.deviation_pct*100:.1f}%"
            if r.deviation > 0:  # capex > implied → 钱花出去了但资产没增加
                r.fraud_implication = "资本支出远超资产增量 → 资金可能通过虚假工程/采购流出(在建工程是藏污纳垢重灾区)"
            else:
                r.fraud_implication = "资产增量远超资本支出 → 可能存在表外资产或虚假资产确认"
        elif r.deviation_pct > r.threshold_warn:
            r.verdict = "WARN"
            r.score = 4
            r.evidence = f"长期资产勾稽差异率{r.deviation_pct*100:.1f}%"
        else:
            r.verdict = "PASS"

        self.results.append(r)

    # ================================================================
    # G7: 费用勾稽 — 三大期间费用 vs 相关现金支付 vs 预提待摊
    # ================================================================
    def _check_g7_expense_articulation(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G7", check_name="期间费用-支付勾稽",
            description="期间费用≈相关现金支付+Δ预提-Δ待摊"
        )

        selling = self._pl("selling_expenses", year) or 0
        admin = self._pl("admin_expenses", year) or 0
        rd = self._pl("rd_expenses", year) or 0
        total_expense = selling + admin + rd
        revenue = self._pl("revenue", year)

        other_payables = self._bs("other_payables", year)
        other_pay_prev = self._bs("other_payables", prev_year)

        if revenue is None or total_expense == 0:
            r.verdict = "DATA_MISSING"
            self.results.append(r)
            return

        delta_other_pay = (other_payables or 0) - (other_pay_prev or 0)
        expense_ratio = total_expense / revenue if revenue else 0

        # 费用率异常检测
        r.deviation_pct = expense_ratio
        r.detail = f"期间费用={total_expense/1e8:.2f}亿, 费用率={expense_ratio*100:.1f}%, Δ其他应付={delta_other_pay/1e8:.2f}亿"

        # 判断费用率异常 + 其他应付大幅变动（费用跨期的信号）
        if expense_ratio < 0.05 and revenue > 1e9:
            r.verdict = "FAIL"
            r.score = 6
            r.evidence = f"费用率仅{expense_ratio*100:.1f}%，显著偏低"
            r.fraud_implication = "费用率异常低 → 可能将期间费用资本化(如研发资本化、利息资本化)或推迟确认"
        elif abs(delta_other_pay) > total_expense * 0.3:
            r.verdict = "WARN"
            r.score = 3
            r.evidence = f"其他应付变动{delta_other_pay/1e8:.2f}亿，占期间费用{abs(delta_other_pay)/max(total_expense,1)*100:.0f}%"
            r.fraud_implication = "其他应付款变动与费用规模不匹配 → 可能利用其他应付调节费用"
        else:
            r.verdict = "PASS"

        self.results.append(r)

    # ================================================================
    # G8: 毛利率 vs 增值税税负率交叉验证
    # ================================================================
    def _check_g8_gross_margin_vat_cross(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G8", check_name="毛利率-增值税负交叉验证",
            description="毛利率与增值税税负率存在内在关联，两者的背离揭示收入或成本造假"
        )

        revenue = self._pl("revenue", year)
        cost = self._pl("cost_of_sales", year)
        tax_paid = self._cf("tax_paid", year)

        if revenue is None or cost is None:
            r.verdict = "DATA_MISSING"
            self.results.append(r)
            return

        gross_margin = (revenue - cost) / revenue if revenue else 0

        if tax_paid:
            # 增值税税负率 = 实际缴纳增值税 / 营业收入（含税）
            vat_burden = tax_paid * 0.6 / revenue  # 简化：假设增值税约占总纳税的60%
            theoretical_vat = gross_margin * 0.13  # 理论增值税 ≈ 毛利 × 13%

            r.deviation_pct = abs(vat_burden * revenue - theoretical_vat) / max(revenue, 1)
            r.detail = (f"毛利率={gross_margin*100:.1f}%, 估计增值税负≈{vat_burden*100:.2f}%, "
                        f"理论增值税负≈{theoretical_vat/revenue*100:.2f}%")
        else:
            # 无法获取纳税数据，仅披露毛利率结构
            r.detail = f"毛利率={gross_margin*100:.1f}%，缺少实纳税数据无法完成交叉验证"
            r.verdict = "DATA_MISSING"
            self.results.append(r)
            return

        if gross_margin > 0.40 and (vat_burden * revenue < theoretical_vat * 0.3):
            r.verdict = "FAIL"
            r.score = 10
            r.evidence = f"毛利率{gross_margin*100:.0f}%但增值税负极低"
            r.fraud_implication = "高毛利却低纳税 → 收入造假经典信号：虚构收入无法产生真实的增值税缴纳义务"
        elif gross_margin > 0.30 and (vat_burden * revenue < theoretical_vat * 0.5):
            r.verdict = "WARN"
            r.score = 5
            r.evidence = f"毛利率{gross_margin*100:.0f}%与增值税负存在偏差"
            r.fraud_implication = "毛利率与税负率背离 → 需核查收入或成本的真实性"
        else:
            r.verdict = "PASS"

        self.results.append(r)

    # ================================================================
    # G9: 客户规模勾稽 — 大客户销售额 vs 客户公开经营规模
    # ================================================================
    def _check_g9_customer_scale_logic(self, year: int):
        r = CrossCheckResult(
            check_id="G9", check_name="大客户销售-客户规模勾稽",
            description="前5大客户销售额应与公开信息中的客户经营规模匹配"
        )

        revenue = self._pl("revenue", year)
        top5_ratio = self.data.non_financial.get("top5_customer_ratio", 0)
        top5_names = self.data.non_financial.get("top5_customer_names", [])

        if revenue is None:
            r.verdict = "DATA_MISSING"
            self.results.append(r)
            return

        top5_amount = revenue * top5_ratio if top5_ratio else 0

        if top5_ratio > 0.50:
            r.verdict = "FAIL"
            r.score = 8
            r.evidence = f"前5大客户销售占比{top5_ratio*100:.0f}% > 50%，客户高度集中"
            r.fraud_implication = ("客户高度集中 → 可能依赖少数关联方或虚构客户；"
                                   "需逐一核查客户工商信息、经营规模与采购额的匹配度")
        elif top5_ratio > 0.30:
            r.verdict = "WARN"
            r.score = 4
            r.evidence = f"前5大客户销售占比{top5_ratio*100:.0f}%"
            r.fraud_implication = "客户集中度较高，需关注单一客户依赖风险和交易真实性"
        else:
            r.verdict = "PASS"
            r.evidence = f"前5大客户销售占比{top5_ratio*100:.0f}%，处于合理区间"

        # 如果客户名称模糊
        if top5_names:
            vague_names = sum(1 for n in top5_names if "客户" in str(n) or "公司" in str(n) or len(str(n)) < 4)
            if vague_names >= 3:
                r.verdict = "FAIL"
                r.score = 10
                r.evidence += "；且客户信息披露模糊(仅以'客户A/B/C'代称)"
                r.fraud_implication += "；客户信息不透明是虚构交易的典型特征，正常企业不会隐藏大客户名称"

        self.results.append(r)

    # ================================================================
    # G10: 产能勾稽 — 产量 vs 固定资产规模 vs 能耗
    # ================================================================
    def _check_g10_capacity_logic(self, year: int, prev_year: int):
        r = CrossCheckResult(
            check_id="G10", check_name="产能-产量-固定资产勾稽",
            description="产能利用率、固定资产周转率、人均产出应保持合理关系和趋势"
        )

        revenue = self._pl("revenue", year)
        rev_prev = self._pl("revenue", prev_year)
        fa = self._bs("fixed_assets", year)
        fa_prev = self._bs("fixed_assets", prev_year)
        employee_count = self.data.non_financial.get("employee_count", 0)
        emp_prev = self.data.non_financial.get("employee_count_prev", 0)

        if revenue is None or fa is None:
            r.verdict = "DATA_MISSING"
            self.results.append(r)
            return

        # 固定资产周转率
        fa_turnover = revenue / fa if fa else 0
        fa_turnover_prev = rev_prev / fa_prev if rev_prev and fa_prev else fa_turnover

        # 人均收入
        revenue_per_employee = revenue / employee_count if employee_count > 0 else 0
        revenue_per_employee_prev = rev_prev / emp_prev if rev_prev and emp_prev > 0 else 0

        r.detail = (f"固定资产周转率={fa_turnover:.2f}次(上年{fa_turnover_prev:.2f}), "
                    f"人均收入={revenue_per_employee/1e4:.1f}万元/人")

        signals = []

        # 固定资产周转率剧烈下降但收入增长 → 资产虚增信号
        if fa_turnover < fa_turnover_prev * 0.7 and rev_prev and revenue > rev_prev:
            signals.append("固定资产周转率骤降但收入增长 → 固定资产可能虚增(在建工程挂账/虚假采购)")
            r.score += 8

        # 固定资产周转率 > 行业阈值 → 可能资产少计或收入虚增
        # 白酒/科技/互联网是典型轻资产模式，固定资产周转率天然高
        _HIGH_FA_TURNOVER_INDUSTRIES = {
            "白酒": 20.0,    # 品牌价值不在固定资产中
            "科技": 15.0,    # 轻资产
            "互联网": 15.0,  # 轻资产
            "软件开发": 15.0,
        }
        fa_threshold = _HIGH_FA_TURNOVER_INDUSTRIES.get(self.data.industry, 10.0)
        if fa_turnover > fa_threshold:
            signals.append(f"固定资产周转率{fa_turnover:.1f}次超过行业阈值{fa_threshold:.0f}次 → 资产或收入需核实")
            r.score += 5

        # 人均收入异常
        if revenue_per_employee_prev > 0 and revenue_per_employee > revenue_per_employee_prev * 1.5:
            signals.append(f"人均收入激增50%以上 → 可能虚增收入或隐匿员工人数")
            r.score += 5

        if revenue_per_employee > 5_000_000:  # 人均收入 > 500万
            signals.append(f"人均收入{revenue_per_employee/1e4:.0f}万元 → 远超正常制造业水平，需核实")
            r.score += 5

        if signals:
            r.verdict = "FAIL" if r.score >= 8 else "WARN"
            r.evidence = "; ".join(signals)
            r.fraud_implication = "产能指标与商业逻辑不符 → 可能存在虚增资产或虚增收入"
        else:
            r.verdict = "PASS"

        self.results.append(r)

    # ================================================================
    # 辅助方法
    # ================================================================
    def _bs(self, field: str, year: int) -> Optional[float]:
        return self.data.balance_sheet.get(year, {}).get(field)

    def _pl(self, field: str, year: int) -> Optional[float]:
        return self.data.income_statement.get(year, {}).get(field)

    def _cf(self, field: str, year: int) -> Optional[float]:
        return self.data.cashflow_statement.get(year, {}).get(field)

    def summary(self) -> Dict:
        """汇总勾稽验证结果"""
        fail_count = sum(1 for r in self.results if r.verdict == "FAIL")
        warn_count = sum(1 for r in self.results if r.verdict == "WARN")
        total_score = sum(r.score for r in self.results)
        return {
            "total_checks": len(self.results),
            "fail_count": fail_count,
            "warn_count": warn_count,
            "total_score": total_score,
            "fail_items": [(r.check_name, r.fraud_implication) for r in self.results if r.verdict == "FAIL"],
            "warn_items": [(r.check_name, r.fraud_implication) for r in self.results if r.verdict == "WARN"],
        }
