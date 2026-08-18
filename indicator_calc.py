"""
财务造假识别 Agent — 指标计算模块
基于 SKILL (2) 步骤1.3 的衍生指标清单 + 步骤3/4 的分析需求
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from data_fetcher import FinancialData


class IndicatorCalculator:
    """计算所有财务衍生指标"""

    def __init__(self, data: FinancialData):
        self.data = data
        self.indicators: Dict[str, Dict[int, float]] = {}  # {indicator_name: {year: value}}
        self.multi_year_stats: Dict[str, float] = {}  # 多年汇总统计

    def compute_all(self) -> Dict[str, Dict[int, float]]:
        """计算全部衍生指标"""
        self._compute_balance_sheet_indicators()
        self._compute_income_indicators()
        self._compute_cashflow_indicators()
        self._compute_hybrid_indicators()
        self._compute_growth_rates()
        self._compute_multi_year_stats()
        return self.indicators

    # ---- 资产负债表衍生指标 ----
    def _compute_balance_sheet_indicators(self):
        for year in self.data.years:
            d = self.data.balance_sheet.get(year, {})
            self._set(year, "有息负债",
                      d.get("short_term_borrowings", 0)
                      + d.get("long_term_borrowings", 0)
                      + d.get("bonds_payable", 0))
            self._set(year, "资产负债率",
                      _safe_div(d.get("total_liabilities"), d.get("total_assets")))
            self._set(year, "流动比率",
                      _safe_div(d.get("current_assets"), d.get("current_liabilities")))
            self._set(year, "其他应收款/流动资产",
                      _safe_div(d.get("other_receivables"), d.get("current_assets")))
            self._set(year, "递延所得税资产/总资产",
                      _safe_div(d.get("deferred_tax_assets"), d.get("total_assets")))
            self._set(year, "合同资产/应收账款",
                      _safe_div(d.get("contract_assets"), d.get("accounts_receivable")))
            self._set(year, "商誉/净资产",
                      _safe_div(d.get("goodwill"), d.get("net_equity")))
            self._set(year, "应收账款/总资产",
                      _safe_div(d.get("accounts_receivable"), d.get("total_assets")))
            self._set(year, "存货/营业成本",
                      _safe_div(d.get("inventory"), self.data.income_statement.get(year, {}).get("cost_of_sales")))
            self._set(year, "存贷比",
                      _safe_div(d.get("money_funds"), self._get_indicator(year, "有息负债")))

    # ---- 利润表衍生指标 ----
    def _compute_income_indicators(self):
        for year in self.data.years:
            pl = self.data.income_statement.get(year, {})
            revenue = pl.get("revenue")
            cost = pl.get("cost_of_sales")
            np_ = pl.get("net_profit")

            self._set(year, "毛利率", _safe_div(revenue - cost, revenue) if revenue and cost else None)
            self._set(year, "费用率", _safe_div(
                (pl.get("selling_expenses", 0) + pl.get("admin_expenses", 0)
                 + pl.get("rd_expenses", 0) + pl.get("finance_expenses", 0)), revenue))
            self._set(year, "非经常性损益/净利润", _safe_div(pl.get("non_recurring_pl"), np_))
            self._set(year, "资产减值损失/净利润", _safe_div(pl.get("asset_impairment_loss"), np_))
            self._set(year, "信用减值损失/净利润绝对值",
                      _safe_div(pl.get("credit_impairment_loss"), abs(np_)) if np_ and np_ != 0 else None)
            self._set(year, "利息收入率", _safe_div(pl.get("interest_income"), self.data.balance_sheet.get(year, {}).get("money_funds")))
            self._set(year, "利息费用率", _safe_div(pl.get("interest_expense"), self._get_indicator(year, "有息负债")))
            self._set(year, "利息费用/利息收入", _safe_div(pl.get("interest_expense"), pl.get("interest_income")))
            self._set(year, "扣非净利润/净利润", _safe_div(pl.get("deducted_net_profit"), np_))
            self._set(year, "股份支付费用/净利润", _safe_div(pl.get("share_based_payment"), np_))
            self._set(year, "核心利润率", _safe_div(pl.get("deducted_net_profit"), revenue))
            self._set(year, "利息保障倍数", _safe_div(
                (pl.get("total_profit", 0) + pl.get("interest_expense", 0)),
                pl.get("interest_expense")))

    # ---- 现金流量表衍生指标 ----
    def _compute_cashflow_indicators(self):
        for year in self.data.years:
            cf = self.data.cashflow_statement.get(year, {})
            pl = self.data.income_statement.get(year, {})
            revenue = pl.get("revenue")

            self._set(year, "经营现金流/净利润",
                      _safe_div(cf.get("operating_cf"), pl.get("net_profit")))
            self._set(year, "销售收现比",
                      _safe_div(cf.get("sales_cash_received"), revenue))
            self._set(year, "自由现金流",
                      (cf.get("operating_cf") or 0) - (cf.get("capex") or 0))
            self._set(year, "经营现金流/营业收入",
                      _safe_div(cf.get("operating_cf"), revenue))

    # ---- 混合/交叉指标 ----
    def _compute_hybrid_indicators(self):
        for year in self.data.years:
            bs = self.data.balance_sheet.get(year, {})
            pl = self.data.income_statement.get(year, {})
            cf = self.data.cashflow_statement.get(year, {})

            # 存货周转率 = 营业成本 / 平均存货
            inventory = bs.get("inventory")
            prev_inventory = self.data.balance_sheet.get(year - 1, {}).get("inventory")
            cost = pl.get("cost_of_sales")
            avg_inventory = (inventory + prev_inventory) / 2 if inventory and prev_inventory else inventory
            self._set(year, "存货周转率", _safe_div(cost, avg_inventory))

            # 应收账款周转天数
            ar = bs.get("accounts_receivable")
            prev_ar = self.data.balance_sheet.get(year - 1, {}).get("accounts_receivable")
            revenue_ = pl.get("revenue")
            avg_ar = (ar + prev_ar) / 2 if ar and prev_ar else ar
            ar_turnover = _safe_div(revenue_, avg_ar)
            self._set(year, "应收账款周转天数", 365 / ar_turnover if ar_turnover and ar_turnover > 0 else None)

            # ROE
            equity = bs.get("net_equity")
            prev_equity = self.data.balance_sheet.get(year - 1, {}).get("net_equity")
            avg_equity = (equity + prev_equity) / 2 if equity and prev_equity else equity
            self._set(year, "ROE", _safe_div(pl.get("net_profit"), avg_equity))

            # ROA
            ta = bs.get("total_assets")
            prev_ta = self.data.balance_sheet.get(year - 1, {}).get("total_assets")
            avg_ta = (ta + prev_ta) / 2 if ta and prev_ta else ta
            self._set(year, "ROA", _safe_div(pl.get("net_profit"), avg_ta))

            # 存货跌价准备/存货余额
            self._set(year, "存货跌价准备/存货",
                      _safe_div(pl.get("inventory_impairment"), inventory))

            # 境外收入占比
            overseas = pl.get("overseas_revenue")
            self._set(year, "境外收入占比", _safe_div(overseas, revenue_))

            # 关联交易占比
            self._set(year, "关联交易/营业收入",
                      _safe_div(pl.get("related_party_revenue"), revenue_))

            # 研发资本化率
            self._set(year, "研发资本化率",
                      _safe_div(pl.get("rd_capitalized"), pl.get("rd_total")))

            # 现金流结构类型
            op = cf.get("operating_cf") or 0
            inv = cf.get("investing_cf") or 0
            fin = cf.get("financing_cf") or 0
            if op > 0 and inv < 0 and fin < 0:
                cf_type = "理想型(经营+、投资-、筹资-)"
            elif op < 0 and inv < 0 and fin > 0:
                cf_type = "风险型(经营-、投资-、筹资+)"
            elif op > 0 and inv > 0 and fin < 0:
                cf_type = "衰退型(经营+、投资+、筹资-)"
            elif op < 0 and inv < 0 and fin < 0:
                cf_type = "全面收缩型(经营-、投资-、筹资-)"
            else:
                cf_type = "混合型"
            self._set(year, "现金流结构类型", cf_type)

    # ---- 同比变化率 ----
    def _compute_growth_rates(self):
        """计算同比增长率"""
        growth_pairs = [
            ("营业收入增长率", "revenue"),
            ("应收账款增长率", "accounts_receivable"),
            ("存货增长率", "inventory"),
            ("总资产增长率", "total_assets"),
            ("毛利率变化", "毛利率"),
            ("存货周转率变化", "存货周转率"),
        ]
        for name, indicator in growth_pairs:
            for i, year in enumerate(self.data.years):
                if i == 0:
                    continue
                prev_year = self.data.years[i - 1]
                curr = self._get_any(indicator, year)
                prev = self._get_any(indicator, prev_year)
                if curr is not None and prev is not None and prev != 0:
                    self._set(year, name, (curr - prev) / abs(prev))

    # ---- 多年汇总统计 ----
    def _compute_multi_year_stats(self):
        """计算多年汇总统计"""
        years = self.data.years
        if len(years) < 2:
            return

        # 近5年（或全部）经营现金流总和 / 净利润总和
        ocf_sum = sum(self._get_indicator(y, "经营现金流/净利润") or 0 for y in years)
        np_sum = sum(self.data.income_statement.get(y, {}).get("net_profit") or 0 for y in years)
        self.multi_year_stats["5年经营现金流总和/5年净利润总和"] = _safe_div(ocf_sum, np_sum)

        # 自由现金流连续为负年数
        fcf_negative_count = sum(
            1 for y in years if self._get_indicator(y, "自由现金流") is not None
            and self._get_indicator(y, "自由现金流") < 0
        )
        self.multi_year_stats["自由现金流为负年数"] = fcf_negative_count

        # 经营现金流连续为负年数
        ocf_list = [self.data.cashflow_statement.get(y, {}).get("operating_cf") for y in years]
        ocf_negative_streak = 0
        max_streak = 0
        for o in ocf_list:
            if o is not None and o < 0:
                ocf_negative_streak += 1
                max_streak = max(max_streak, ocf_negative_streak)
            else:
                ocf_negative_streak = 0
        self.multi_year_stats["经营现金流连续为负最大年数"] = max_streak

        # 理想型现金流占比
        ideal_count = sum(
            1 for y in years
            if isinstance(self._get_indicator(y, "现金流结构类型"), str)
            and "理想型" in str(self._get_indicator(y, "现金流结构类型"))
        )
        self.multi_year_stats["理想型现金流年数占比"] = ideal_count / len(years) if years else 0

        # 风险型现金流年数
        risk_count = sum(
            1 for y in years
            if isinstance(self._get_indicator(y, "现金流结构类型"), str)
            and "风险型" in str(self._get_indicator(y, "现金流结构类型"))
        )
        self.multi_year_stats["风险型现金流年数"] = risk_count

        # 净利润波动模式检测
        np_list = [self.data.income_statement.get(y, {}).get("net_profit") for y in years]
        np_list_clean = [n for n in np_list if n is not None]
        if len(np_list_clean) >= 4:
            self.multi_year_stats["净利润波动"] = _detect_alternating_pattern(np_list_clean)

    # ---- 辅助方法 ----
    def _set(self, year, name, value):
        if value is None:
            return
        if name not in self.indicators:
            self.indicators[name] = {}
        if isinstance(value, str):
            self.indicators[name][year] = value
        else:
            self.indicators[name][year] = round(value, 6)

    def _get_indicator(self, year, name):
        return self.indicators.get(name, {}).get(year)

    def _get_any(self, name, year):
        """从 indicator 或三表中获取值"""
        val = self._get_indicator(year, name)
        if val is not None:
            return val
        return self.data.get_field(name, year)


def _safe_div(a, b) -> Optional[float]:
    """安全除法"""
    if a is None or b is None or b == 0:
        return None
    return a / b


def _detect_alternating_pattern(np_list: List[float]) -> str:
    """检测微利-巨亏交替模式"""
    signs = [1 if n >= 0 else -1 for n in np_list]
    changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    if changes >= len(signs) - 2 and changes > 0:
        # 检查盈亏规模
        small_threshold = 100_000_000  # 1亿
        large_threshold = 500_000_000  # 5亿
        has_small = any(0 < abs(n) < small_threshold for n in np_list if n is not None)
        has_large = any(abs(n) > large_threshold for n in np_list if n is not None)
        if has_small or has_large:
            return f"检测到交替模式({changes}次变号)，可能存在微利-巨亏交替"
        return f"有{changes}次盈亏交替，但规模不符合典型交替特征"
    return "无明显交替模式"
