"""
商业逻辑验证模块 — 穿透报表数字，验证业务实质
财务报表永远平衡（借贷恒等），真正的造假矛盾在：
  - 高毛利 vs 低研发: 声称技术领先但研发投入极低
  - 高增长 vs 低薪酬: 收入激增但员工人数/薪酬不匹配
  - 高利润 vs 低纳税: 赚大钱却不交税
  - 大客户 vs 小公司: 大客户是注册资本10万的小微企业
  - 产能满载 vs 低能耗: 满产满销但电费水费不涨
  - 行业衰退 vs 逆势增长: 全行业下滑唯独你高增长
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from data_fetcher import FinancialData
from indicator_calc import IndicatorCalculator, _safe_div


@dataclass
class BizLogicResult:
    """商业逻辑验证结果"""
    check_id: str
    check_name: str
    category: str           # 分类: 盈利能力/资产质量/增长质量/治理信号
    verdict: str = "PASS"
    score: int = 0
    detail: str = ""
    evidence: str = ""      # 商业逻辑矛盾点
    contradiction: str = "" # 核心矛盾描述
    red_flag: bool = False  # 是否为强造假信号


class BusinessLogicVerifier:
    """
    商业逻辑验证器
    核心思路: 不查账本，只看生意是否讲得通
    """

    def __init__(self, data: FinancialData, indicators: IndicatorCalculator):
        self.data = data
        self.ind = indicators
        self.results: List[BizLogicResult] = []

    def run_all(self) -> List[BizLogicResult]:
        if len(self.data.years) < 2:
            return self.results

        latest = self.data.years[-1]
        prev = self.data.years[-2] if len(self.data.years) >= 2 else latest

        # 五维度商业逻辑验证
        self._verify_profitability_logic(latest, prev)
        self._verify_asset_quality_logic(latest, prev)
        self._verify_growth_quality_logic(latest, prev)
        self._verify_governance_signals(latest)
        self._verify_industry_logic(latest, prev)
        self._verify_competitor_benchmark(latest)

        return self.results

    # ================================================================
    # 维度1: 盈利能力商业逻辑
    # ================================================================
    def _verify_profitability_logic(self, year: int, prev: int):
        """盈利能力的商业逻辑验证"""
        revenue = self._pl("revenue", year)
        cost = self._pl("cost_of_sales", year)
        rd = self._pl("rd_expenses", year)
        np_ = self._pl("net_profit", year)
        total_profit = self._pl("total_profit", year)
        income_tax = self._pl("income_tax", year)
        non_recurring = self._pl("non_recurring_pl", year)

        if not revenue:
            return

        gross_margin = (revenue - cost) / revenue if revenue and cost else 0
        rd_ratio = rd / revenue if rd and revenue else 0

        # ----- B1: 高毛利 vs 低研发 -----
        # 白酒/消费品/金融行业: 品牌/牌照壁垒而非技术壁垒，高毛利+低研发是正常商业模式
        _B1_EXCLUDED_INDUSTRIES = {"白酒", "银行", "保险", "房地产"}
        r = BizLogicResult(
            check_id="B1", check_name="毛利率vs研发投入匹配",
            category="盈利能力",
            detail=f"毛利率={gross_margin*100:.1f}%, 研发/营收={rd_ratio*100:.1f}%"
        )
        if self.data.industry in _B1_EXCLUDED_INDUSTRIES:
            r.verdict = "PASS"
            r.evidence = f"{self.data.industry}行业高毛利来自品牌/牌照壁垒而非研发投入，不适用此规则"
        elif gross_margin > 0.40 and rd_ratio < 0.02:
            r.verdict = "FAIL"
            r.score = 10
            r.red_flag = True
            r.evidence = f"毛利率{gross_margin*100:.0f}%但研发投入仅{rd_ratio*100:.1f}%"
            r.contradiction = ("高毛利率意味着强定价权/技术壁垒，但极低的研发投入无法支撑技术壁垒。"
                               "要么毛利率造假，要么技术壁垒不存在(毛利率无法持续)。"
                               "A股典型: 毛利率50%+但研发费率<2%的'高科技'企业。")
        elif gross_margin > 0.30 and rd_ratio < 0.01:
            r.verdict = "WARN"
            r.score = 5
            r.evidence = f"毛利率{gross_margin*100:.0f}%与研发投入{rd_ratio*100:.1f}%存在不匹配"
            r.contradiction = "毛利率与研发投入不匹配，技术壁垒支撑存疑"
        self.results.append(r)

        # ----- B2: 高利润 vs 低纳税 -----
        if total_profit and total_profit > 0:
            effective_rate = income_tax / total_profit if income_tax else 0
            r = BizLogicResult(
                check_id="B2", check_name="利润-纳税匹配",
                category="盈利能力",
                detail=f"利润总额={total_profit/1e8:.2f}亿, 所得税={income_tax/1e8:.2f}亿, 有效税率={effective_rate*100:.1f}%"
            )
            if effective_rate < 0.10 and total_profit > 1e8:
                r.verdict = "FAIL"
                r.score = 10
                r.red_flag = True
                r.evidence = f"利润{total_profit/1e8:.1f}亿但有效税率仅{effective_rate*100:.1f}%"
                r.contradiction = ("大额利润但极低税负 → 要么利润造假(虚构的利润不需要交税)，"
                                   "要么存在大量不可持续税收优惠。A股法定税率25%，"
                                   "高新技术企业15%，有效税率<10%需要强有力的解释。")
            elif effective_rate < 0.15 and total_profit > 5e8:
                r.verdict = "WARN"
                r.score = 5
                r.evidence = f"有效税率{effective_rate*100:.1f}%偏低"
                r.contradiction = "有效税率显著低于法定水平，需核查税收优惠依据"
            self.results.append(r)

        # ----- B3: 非经常性损益依赖度 -----
        if np_ and np_ > 0 and non_recurring:
            nr_ratio = non_recurring / np_
            r = BizLogicResult(
                check_id="B3", check_name="盈利持续性",
                category="盈利能力",
                detail=f"非经常性损益/净利润={nr_ratio*100:.1f}%"
            )
            if nr_ratio > 0.5:
                r.verdict = "FAIL"
                r.score = 8
                r.evidence = f"净利润中非经常性损益占{nr_ratio*100:.0f}%"
                r.contradiction = "过半利润来自非经常项目 → 主业无法造血，靠卖资产/政府补助/投资收益维持盈利"
            elif nr_ratio > 0.3:
                r.verdict = "WARN"
                r.score = 4
                r.evidence = f"非经常性损益占比{nr_ratio*100:.0f}%"
            self.results.append(r)

    # ================================================================
    # 维度2: 资产质量商业逻辑
    # ================================================================
    def _verify_asset_quality_logic(self, year: int, prev: int):
        """资产质量的商业逻辑验证"""
        revenue = self._pl("revenue", year)
        rev_prev = self._pl("revenue", prev)
        ar = self._bs("accounts_receivable", year)
        ar_prev = self._bs("accounts_receivable", prev)
        inventory = self._bs("inventory", year)
        inv_prev = self._bs("inventory", prev)
        goodwill = self._bs("goodwill", year)
        net_equity = self._bs("net_equity", year)

        # ----- B4: 应收增速 vs 营收增速 -----
        if revenue and rev_prev and ar and ar_prev and rev_prev > 0:
            rev_growth = (revenue - rev_prev) / rev_prev
            ar_growth = (ar - ar_prev) / ar_prev if ar_prev > 0 else 0

            r = BizLogicResult(
                check_id="B4", check_name="应收账款vs收入增速匹配",
                category="资产质量",
                detail=f"营收增速={rev_growth*100:.1f}%, 应收增速={ar_growth*100:.1f}%"
            )
            if ar_growth > rev_growth + 0.30:
                r.verdict = "FAIL"
                r.score = 10
                r.red_flag = True
                r.evidence = f"应收增速{ar_growth*100:.0f}%远超营收增速{rev_growth*100:.0f}%"
                r.contradiction = ("应收账款增长远超收入增长 → 要么放宽信用政策(客户质量恶化)，"
                                   "要么虚构收入(无法收回的虚假应收账款)。"
                                   "A股经典: 收入增20%但应收增80%。")
            elif ar_growth > rev_growth + 0.15:
                r.verdict = "WARN"
                r.score = 5
                r.evidence = f"应收增速({ar_growth*100:.0f}%)快于营收增速({rev_growth*100:.0f}%)"
                r.contradiction = "赊销政策趋于激进"
            else:
                r.verdict = "PASS"
            self.results.append(r)

        # ----- B5: 存货积压 vs 收入增长 -----
        if inventory and inv_prev and inv_prev > 0:
            inv_growth = (inventory - inv_prev) / inv_prev
            rev_growth = (revenue - rev_prev) / rev_prev if revenue and rev_prev and rev_prev > 0 else 0

            r = BizLogicResult(
                check_id="B5", check_name="存货增长vs收入增长匹配",
                category="资产质量",
                detail=f"存货增速={inv_growth*100:.1f}%, 营收增速={rev_growth*100:.1f}%"
            )
            if inv_growth > 0.50 and rev_growth < 0.10:
                r.verdict = "FAIL"
                r.score = 12
                r.red_flag = True
                r.evidence = f"存货激增{inv_growth*100:.0f}%但收入仅增{rev_growth*100:.0f}%"
                r.contradiction = ("存货暴增收入停滞 → 产品卖不出去还在大量生产？"
                                   "经典存货魔术：虚构存货消化虚增的采购款，"
                                   "或隐瞒存货减值。A股案例: 獐子岛扇贝、康美药业中药材。")
            elif inv_growth > 0.30 and rev_growth < 0.05:
                r.verdict = "WARN"
                r.score = 6
                r.evidence = f"存货增长{inv_growth*100:.0f}%与收入增速不匹配"
                r.contradiction = "存货增长速度远超收入，存在积压或虚构风险"
            else:
                r.verdict = "PASS"
            self.results.append(r)

        # ----- B5b: 存货/营业成本 — 分行业阈值 -----
        inv_to_cost = self._get_indicator("存货/营业成本", year)
        if inv_to_cost is not None:
            # 分行业危险线：白酒基酒需多年陈化，房地产需要开发周期
            _INDUSTRY_INV_COST_THRESHOLD = {
                "白酒": 8.0,    # 基酒陈化3-5年，存货周转极慢
                "房地产": 3.0,  # 开发周期2-3年
                "科技": 1.5,    # 技术迭代快，存货应低
                "新能源": 2.0,  # 原材料+产成品
                "医药制造": 2.5, # 中药材+产成品
                "生猪养殖": 3.0, # 活体资产
                "建筑施工": 1.5, # 按进度确认
            }
            threshold = _INDUSTRY_INV_COST_THRESHOLD.get(self.data.industry, 2.0)
            r = BizLogicResult(
                check_id="B5b", check_name="存货/营业成本行业对标",
                category="资产质量",
                detail=f"存货/营业成本={inv_to_cost:.1f}, 行业阈值={threshold:.1f} (行业: {self.data.industry})"
            )
            if inv_to_cost > threshold:
                r.verdict = "FAIL"
                r.score = 10
                r.red_flag = True
                r.evidence = f"存货/营业成本={inv_to_cost:.1f} > 行业阈值{threshold:.1f}"
                r.contradiction = (f"存货/营业成本={inv_to_cost:.1f}超出{self.data.industry}行业阈值{threshold:.1f}。"
                                   "存货规模相对营业成本过高，存在积压、虚构或减值不足风险。")
            elif inv_to_cost > threshold * 0.7:
                r.verdict = "WARN"
                r.score = 5
                r.evidence = f"存货/营业成本={inv_to_cost:.1f}接近行业阈值{threshold:.1f}"
                r.contradiction = "存货水平偏高，需关注"
            else:
                r.verdict = "PASS"
            self.results.append(r)

        # ----- B6: 商誉占比 -----
        if goodwill and net_equity and goodwill > 0:
            gw_ratio = goodwill / net_equity
            r = BizLogicResult(
                check_id="B6", check_name="商誉泡沫风险",
                category="资产质量",
                detail=f"商誉/净资产={gw_ratio*100:.1f}%"
            )
            if gw_ratio > 0.50:
                r.verdict = "FAIL"
                r.score = 8
                r.red_flag = True
                r.evidence = f"商誉占净资产{gw_ratio*100:.0f}%"
                r.contradiction = ("商誉超过净资产一半 → 企业靠收购撑大，一旦减值将直接击穿净资产。"
                                   "A股2018-2019商誉减值潮: 大量公司一次性计提数十亿商誉减值。")
            elif gw_ratio > 0.30:
                r.verdict = "WARN"
                r.score = 4
                r.evidence = f"商誉占比{gw_ratio*100:.0f}%"
                r.contradiction = "商誉占比偏高，重点关注收购标的业绩承诺完成情况"
            else:
                r.verdict = "PASS"
            self.results.append(r)

    # ================================================================
    # 维度3: 增长质量商业逻辑
    # ================================================================
    def _verify_growth_quality_logic(self, year: int, prev: int):
        """增长质量的商业逻辑验证"""
        revenue = self._pl("revenue", year)
        rev_prev = self._pl("revenue", prev)
        ocf = self._cf("operating_cf", year)
        np_ = self._pl("net_profit", year)
        employees = self.data.non_financial.get("employee_count", 0)
        emp_prev = self.data.non_financial.get("employee_count_prev", 0)

        if not revenue or not rev_prev or rev_prev <= 0:
            return

        rev_growth = (revenue - rev_prev) / rev_prev

        # ----- B7: 高增长 vs 无现金流 -----
        r = BizLogicResult(
            check_id="B7", check_name="增长-现金流匹配",
            category="增长质量",
            detail=f"营收增速={rev_growth*100:.1f}%, 经营现金流={'正' if ocf and ocf > 0 else '负'}"
        )
        if rev_growth > 0.20 and ocf and ocf < 0:
            r.verdict = "FAIL"
            r.score = 12
            r.red_flag = True
            r.evidence = f"收入增长{rev_growth*100:.0f}%但经营现金流为负({ocf/1e8:.2f}亿)"
            r.contradiction = ("高增长企业却无现金流入 → 增长质量极差，"
                               "所有'增长'都变成了应收账款而非现金。"
                               "要么收入造假，要么客户根本不付款(坏账风险)。")
        elif rev_growth > 0.15 and ocf and np_ and ocf < np_ * 0.3:
            r.verdict = "WARN"
            r.score = 6
            r.evidence = f"收入增长{rev_growth*100:.0f}%但现金流仅覆盖利润{ocf/max(np_,1)*100:.0f}%"
            r.contradiction = "增长质量不佳，利润未有效转化为现金"
        else:
            r.verdict = "PASS"
        self.results.append(r)

        # ----- B8: 收入增长 vs 员工人数 -----
        if employees > 0 and emp_prev > 0:
            emp_change = (employees - emp_prev) / emp_prev
            r = BizLogicResult(
                check_id="B8", check_name="收入增长vs员工人数匹配",
                category="增长质量",
                detail=f"营收增速={rev_growth*100:.1f}%, 员工增速={emp_change*100:.1f}%"
            )
            if rev_growth > 0.30 and emp_change < -0.10:
                r.verdict = "FAIL"
                r.score = 8
                r.red_flag = True
                r.evidence = f"收入增长{rev_growth*100:.0f}%但员工减少{abs(emp_change)*100:.0f}%"
                r.contradiction = ("收入大增但员工大减 → 违背基本生产规律。"
                                   "除非是重大自动化替代(需有对应资本支出)，否则收入真实性存疑。")
            elif rev_growth > 0.20 and emp_change < 0:
                r.verdict = "WARN"
                r.score = 4
                r.evidence = f"收入增{rev_growth*100:.0f}%但员工减少"
                r.contradiction = "收入与人力投入趋势背离"
            else:
                r.verdict = "PASS"
            self.results.append(r)

        # ----- B9: 人均薪酬合理性 -----
        salary_paid = self._cf("salary_paid", year)
        if employees > 0 and salary_paid:
            avg_salary = salary_paid / employees
            r = BizLogicResult(
                check_id="B9", check_name="人均薪酬合理性",
                category="增长质量",
                detail=f"人均薪酬={avg_salary/1e4:.1f}万元/年"
            )
            industry = self.data.industry
            # 行业人均薪酬参考
            industry_avg = {
                "科技": 250000, "互联网": 300000, "软件开发": 280000,
                "银行": 350000, "保险": 300000,
                "白酒": 180000, "医药制造": 150000, "医药流通": 140000,
                "房地产": 200000, "建筑施工": 120000,
                "新能源": 160000, "生猪养殖": 90000,
                "制造业": 120000,
            }.get(industry, 120000)

            if avg_salary < industry_avg * 0.5:
                r.verdict = "FAIL"
                r.score = 6
                r.evidence = f"人均薪酬{avg_salary/1e4:.0f}万远低于行业{industry_avg/1e4:.0f}万"
                r.contradiction = "人均薪酬极低 → 要么员工人数虚报，要么大量使用劳务派遣(表外用工)，要么财务报表薪酬数据不实"
            elif avg_salary < industry_avg * 0.7:
                r.verdict = "WARN"
                r.score = 3
                r.evidence = f"人均薪酬({avg_salary/1e4:.0f}万)低于行业平均"
            else:
                r.verdict = "PASS"
            self.results.append(r)

    # ================================================================
    # 维度4: 治理信号
    # ================================================================
    def _verify_governance_signals(self, year: int):
        """公司治理商业逻辑"""
        pledge_ratio = self.data.non_financial.get("pledge_ratio", 0)
        audit_opinion = self.data.non_financial.get("audit_opinion", "")
        policy_changes = self.data.non_financial.get("policy_changes_3yr", 0)
        corrections_count = self.data.non_financial.get("accounting_corrections_count", 0)

        # ----- B10: 大股东高比例质押 -----
        if pledge_ratio:
            r = BizLogicResult(
                check_id="B10", check_name="大股东质押风险",
                category="治理信号",
                detail=f"大股东质押比例={pledge_ratio*100:.1f}%"
            )
            if pledge_ratio > 0.80:
                r.verdict = "FAIL"
                r.score = 10
                r.red_flag = True
                r.evidence = f"大股东质押比例{pledge_ratio*100:.0f}%，几乎全部质押"
                r.contradiction = ("大股东几乎清仓式质押 → 变相套现。"
                                   "高质押 + 股价下跌 = 爆仓风险 = 大股东挪用上市公司资金补仓的强烈动机。"
                                   "A股无数案例: 质押爆仓 → 资金占用 → ST → 退市。")
            elif pledge_ratio > 0.50:
                r.verdict = "WARN"
                r.score = 5
                r.evidence = f"质押比例{pledge_ratio*100:.0f}%，超过50%"
                r.contradiction = "大股东资金紧张，存在占用上市公司资金动机"
            else:
                r.verdict = "PASS"
            self.results.append(r)

        # ----- B11: 审计意见 -----
        if audit_opinion:
            r = BizLogicResult(
                check_id="B11", check_name="审计意见类型",
                category="治理信号",
                detail=f"审计意见: {audit_opinion}"
            )
            # 注意: "标准无保留意见"包含"保留"但子串匹配会误判
            # "保留意见"(否定式) ≠ "无保留意见"(肯定式)
            audit_str = str(audit_opinion)
            is_qualified = ("保留意见" in audit_str and "无保留" not in audit_str) or \
                           any(kw in audit_str for kw in ["否定意见", "无法表示意见", "非标", "拒绝表示"])
            if is_qualified:
                r.verdict = "FAIL"
                r.score = 15
                r.red_flag = True
                r.evidence = f"审计意见类型: {audit_opinion}"
                r.contradiction = "非标准审计意见 → 审计师不愿或不敢出具标准意见，报表可信度崩塌"
            elif "强调事项" in audit_str or "持续经营" in audit_str:
                r.verdict = "WARN"
                r.score = 8
                r.evidence = f"审计意见: {audit_opinion}"
                r.contradiction = "带强调事项段 → 虽非否定，但审计师提醒关注特定风险"
            else:
                r.verdict = "PASS"
            self.results.append(r)

        # ----- B12: 会计政策/差错频率 -----
        if policy_changes or corrections_count:
            r = BizLogicResult(
                check_id="B12", check_name="会计信息稳定性",
                category="治理信号",
                detail=f"3年政策变更{policy_changes}次, 差错更正{corrections_count}次"
            )
            if policy_changes >= 3 or corrections_count >= 3:
                r.verdict = "FAIL"
                r.score = 10
                r.evidence = f"频繁变更会计政策({policy_changes}次)或差错更正({corrections_count}次)"
                r.contradiction = "频繁变更会计政策或更正差错 → 财务核算基础薄弱，或故意掩盖问题"
            elif policy_changes >= 2 or corrections_count >= 2:
                r.verdict = "WARN"
                r.score = 5
                r.evidence = f"会计政策变更{policy_changes}次, 差错更正{corrections_count}次"
            else:
                r.verdict = "PASS"
            self.results.append(r)

    # ================================================================
    # 维度5: 行业逻辑验证
    # ================================================================
    def _verify_industry_logic(self, year: int, prev: int):
        """行业特有商业逻辑"""
        industry = self.data.industry
        revenue = self._pl("revenue", year)
        rev_prev = self._pl("revenue", prev)
        gross_margin = self._get_indicator("毛利率", year)
        gm_prev = self._get_indicator("毛利率", prev)

        # 行业共性: 逆周期高增长
        if revenue and rev_prev and rev_prev > 0:
            rev_growth = (revenue - rev_prev) / rev_prev

            # 判断是否处于下行行业
            cyclical = ["生猪养殖", "新能源", "电池", "房地产", "钢铁", "煤炭"]
            if industry in cyclical:
                r = BizLogicResult(
                    check_id="B13", check_name="周期性行业增长合理性",
                    category="行业逻辑",
                    detail=f"行业: {industry}(周期性), 营收增速={rev_growth*100:.1f}%"
                )
                if rev_growth > 0.30:
                    r.verdict = "WARN"
                    r.score = 5
                    r.evidence = f"周期性行业({industry})逆势增长{rev_growth*100:.0f}%"
                    r.contradiction = ("周期性行业高增长需结合行业周期位置判断。"
                                       "全行业下行时的逆势高增长需要强有力的商业解释。")
                else:
                    r.verdict = "PASS"
                self.results.append(r)

            # 白酒行业特定逻辑
            if industry == "白酒":
                adv = self._bs("advance_from_customers", year)
                adv_prev = self._bs("advance_from_customers", prev)
                if adv and adv_prev:
                    adv_change = (adv - adv_prev) / adv_prev if adv_prev > 0 else 0
                    r = BizLogicResult(
                        check_id="B14", check_name="白酒预收款趋势",
                        category="行业逻辑",
                        detail=f"预收账款变动={adv_change*100:.1f}%"
                    )
                    if adv_change < -0.30 and rev_growth > 0:
                        r.verdict = "WARN"
                        r.score = 6
                        r.evidence = f"预收款大降{abs(adv_change)*100:.0f}%但收入仍增长"
                        r.contradiction = "白酒行业预收款是业绩领先指标，预收款大降意味着渠道库存积压，未来收入承压"
                    else:
                        r.verdict = "PASS"
                    self.results.append(r)

        # 行业毛利率稳定性
        if industry == "白酒" and gross_margin and gm_prev:
            gm_change = abs(gross_margin - gm_prev)
            r = BizLogicResult(
                check_id="B15", check_name="毛利率稳定性(品牌消费品)",
                category="行业逻辑",
                detail=f"毛利率变动={gm_change*100:.1f}个百分点"
            )
            if gm_change > 0.05:
                r.verdict = "WARN"
                r.score = 5
                r.evidence = f"品牌白酒毛利率波动{gm_change*100:.1f}个百分点"
                r.contradiction = "品牌白酒毛利率应高度稳定，异常波动需核查"
            else:
                r.verdict = "PASS"
            self.results.append(r)

    # ================================================================
    # 维度6: 行业对标
    # ================================================================
    def _verify_competitor_benchmark(self, year: int):
        """与行业基准值的比较验证"""
        gross_margin = self._get_indicator("毛利率", year)
        ar_days = self._get_indicator("应收账款周转天数", year)

        if gross_margin is None:
            return

        # 行业毛利率基准
        industry_gm_benchmark = {
            "白酒": (0.60, 0.85), "银行": (0.35, 0.50), "保险": (0.15, 0.30),
            "医药制造": (0.40, 0.75), "医药流通": (0.08, 0.20),
            "科技": (0.25, 0.55), "互联网": (0.30, 0.65),
            "新能源": (0.15, 0.35), "生猪养殖": (0.05, 0.40),
            "房地产": (0.20, 0.40), "建筑施工": (0.08, 0.18),
            "软件开发": (0.40, 0.80), "制造业": (0.15, 0.35),
        }

        benchmark = industry_gm_benchmark.get(self.data.industry)
        if benchmark:
            low, high = benchmark
            r = BizLogicResult(
                check_id="B16", check_name="毛利率行业对标",
                category="行业逻辑",
                detail=f"毛利率={gross_margin*100:.1f}%, 行业区间=[{low*100:.0f}%,{high*100:.0f}%]"
            )
            if gross_margin > high * 1.3:
                r.verdict = "FAIL"
                r.score = 8
                r.red_flag = True
                r.evidence = f"毛利率{gross_margin*100:.0f}%远超行业上限{high*100:.0f}%"
                r.contradiction = ("毛利率远超行业天花板 → 除非有独一无二的技术/品牌壁垒，"
                                   "否则大概率是收入虚增或成本虚减。"
                                   "客观规律: 超额利润必然引来竞争，长期的超高毛利率不可持续。")
            elif gross_margin > high * 1.1:
                r.verdict = "WARN"
                r.score = 4
                r.evidence = f"毛利率{gross_margin*100:.0f}%高于行业上限{high*100:.0f}%"
                r.contradiction = "毛利率略高于行业上限，需确认是否有可持续的竞争优势支撑"
            else:
                r.verdict = "PASS"
            self.results.append(r)

    # ---- 辅助方法 ----
    def _bs(self, field: str, year: int) -> Optional[float]:
        return self.data.balance_sheet.get(year, {}).get(field)

    def _pl(self, field: str, year: int) -> Optional[float]:
        return self.data.income_statement.get(year, {}).get(field)

    def _cf(self, field: str, year: int) -> Optional[float]:
        return self.data.cashflow_statement.get(year, {}).get(field)

    def _get_indicator(self, name: str, year: int) -> Optional[float]:
        return self.ind.indicators.get(name, {}).get(year)

    def summary(self) -> Dict:
        fail_count = sum(1 for r in self.results if r.verdict == "FAIL")
        warn_count = sum(1 for r in self.results if r.verdict == "WARN")
        red_flags = [r for r in self.results if r.red_flag]
        return {
            "total_checks": len(self.results),
            "fail_count": fail_count,
            "warn_count": warn_count,
            "total_score": sum(r.score for r in self.results),
            "red_flags": [(r.check_name, r.contradiction) for r in red_flags],
        }
