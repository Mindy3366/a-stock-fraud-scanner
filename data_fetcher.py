"""
财务造假识别 Agent — 数据采集模块
支持三种数据源：akshare 自动拉取 / tushare API / 手动输入
"""
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import DATA_FIELD_MAP


@dataclass
class FinancialData:
    """标准化财务数据结构"""
    stock_code: str = ""
    company_name: str = ""
    industry: str = "制造业"
    years: List[int] = field(default_factory=list)

    # 资产负债表（每年一个dict，key=年份）
    balance_sheet: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # 利润表
    income_statement: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # 现金流量表
    cashflow_statement: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # 附注/非财务信息
    notes: Dict[int, Dict[str, object]] = field(default_factory=dict)

    # 非财务信号
    non_financial: Dict[str, object] = field(default_factory=dict)

    # 季度数据（可选，用于深层核查）
    quarterly_data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def get_field(self, field: str, year: int) -> Optional[float]:
        """从三表中查找字段值"""
        for stmt in [self.income_statement, self.balance_sheet, self.cashflow_statement, self.notes]:
            if year in stmt and field in stmt[year]:
                val = stmt[year][field]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return float(val)
        return None

    def get_multi_year(self, field: str) -> Dict[int, float]:
        """获取字段多年数据"""
        result = {}
        for year in self.years:
            val = self.get_field(field, year)
            if val is not None:
                result[year] = val
        return result


class DataFetcher:
    """数据获取器 —— 支持 akshare / tushare / 手动输入"""

    def __init__(self, source: str = "akshare", tushare_token: str = ""):
        self.source = source
        self.tushare_token = tushare_token
        self._ts_api = None

    def fetch(self, stock_code: str, years: int = 5, **kwargs) -> FinancialData:
        """
        主入口：根据股票代码拉取数据
        - stock_code: 如 '600519'（上交所）或 '000858'（深交所）
        - years: 拉取近N年数据
        """
        if self.source == "akshare":
            return self._fetch_akshare(stock_code, years)
        elif self.source == "tushare":
            return self._fetch_tushare(stock_code, years)
        elif self.source == "manual":
            return self._load_manual(kwargs.get("data"))
        else:
            raise ValueError(f"不支持的数据源: {self.source}")

    # ================================================================
    # AKShare 数据源 — 新浪财经接口
    # ================================================================
    def _fetch_akshare(self, stock_code: str, years: int) -> FinancialData:
        """
        使用 akshare + 新浪财经接口获取A股财报数据。
        之前使用的东方财富接口 (stock_*_by_report_em) 已失效，改用新浪。
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("请安装 akshare: pip install akshare")

        warnings.filterwarnings("ignore")

        # 获取公司名称 / 行业
        company_name = self._get_company_name_sina(stock_code)
        industry = _guess_industry(stock_code)

        data = FinancialData(stock_code=stock_code, company_name=company_name, industry=industry)

        from datetime import datetime
        current_year = datetime.now().year
        target_years = list(range(current_year - years, current_year + 1))

        # 拉取新浪财经三表
        for stmt_cn, symbol in [
            ("资产负债表", "资产负债表"),
            ("利润表",     "利润表"),
            ("现金流量表", "现金流量表"),
        ]:
            try:
                df = ak.stock_financial_report_sina(stock=stock_code, symbol=symbol)
                if df is not None and not df.empty:
                    self._parse_sina_statement(df, stmt_cn, data, target_years)
                else:
                    print(f"[WARN] 新浪财经{symbol}返回空数据")
            except Exception as e:
                print(f"[WARN] 新浪财经{symbol}获取失败: {e}")

        # 拉取质押比例（东方财富）
        try:
            pledge = ak.stock_gpzy_pledge_ratio_em()
            pledge_row = pledge[pledge["股票代码"] == stock_code]
            data.non_financial["pledge_ratio"] = float(pledge_row["质押比例"].values[0]) if len(pledge_row) > 0 else 0.0
        except Exception:
            data.non_financial["pledge_ratio"] = 0.0

        # 去重排序
        data.years = sorted(set(
            y for y in target_years
            if y in data.balance_sheet or y in data.income_statement or y in data.cashflow_statement
        ))
        return data

    def _get_company_name_sina(self, stock_code: str) -> str:
        """从新浪接口获取公司名称（失败则返回股票代码）"""
        try:
            import akshare as ak
            # 新浪的 stock_info 或者直接从报表里推断
            df = ak.stock_financial_report_sina(stock=stock_code, symbol='资产负债表')
            if df is not None and not df.empty:
                # 报表本身不返回股票名称，用行业猜测表映射
                return _get_stock_name(stock_code)
        except Exception:
            pass
        return stock_code

    # ================================================================
    # 新浪财经表解析 — 行=报告期，列=科目
    # ================================================================
    def _parse_sina_statement(self, df: pd.DataFrame, stmt_cn: str,
                               data: FinancialData, target_years: List[int]):
        """
        解析新浪财经三表。
        DataFrame 格式: 第一列='日期'（如20251231），其余列为各科目中文名。
        仅取年度报告（日期以'1231'结尾的行）。
        """
        # 第一列是报告期
        date_col = df.columns[0]
        df = df.copy()

        # 只保留年度报告行（日期以 1231 结尾）
        df[date_col] = df[date_col].astype(str)
        df = df[df[date_col].str.endswith("1231")].copy()

        if df.empty:
            print(f"[WARN] 新浪财报{stmt_cn}无年度数据")
            return

        # 选择 col_map
        if stmt_cn == "资产负债表":
            col_map = _SINA_BALANCE_MAP
            store = data.balance_sheet
        elif stmt_cn == "利润表":
            col_map = _SINA_INCOME_MAP
            store = data.income_statement
        else:
            col_map = _SINA_CASHFLOW_MAP
            store = data.cashflow_statement

        # 遍历每一行（每一年度）
        for _, row in df.iterrows():
            try:
                year = int(str(row[date_col])[:4])
            except (ValueError, IndexError):
                continue
            if year not in target_years:
                continue

            if year not in store:
                store[year] = {}

            # 遍历所有列，匹配科目名
            for col_name, val in row.items():
                if col_name == date_col or pd.isna(val):
                    continue
                dest = col_map.get(str(col_name))
                if dest:
                    try:
                        v = float(val)
                        if not np.isnan(v):
                            store[year][dest] = v
                    except (ValueError, TypeError):
                        pass

    # ================================================================
    # Tushare 数据源
    # ================================================================
    def _fetch_tushare(self, stock_code: str, years: int) -> FinancialData:
        """使用 tushare 获取数据"""
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("请安装 tushare: pip install tushare")

        if not self.tushare_token:
            raise ValueError("使用 tushare 需提供 token")

        ts.set_token(self.tushare_token)
        pro = ts.pro_api()

        from datetime import datetime
        current_year = datetime.now().year
        target_years = list(range(current_year - years, current_year))

        # 公司信息
        try:
            company = pro.stock_basic(ts_code=f"{stock_code}.{'SH' if stock_code.startswith('6') else 'SZ'}")
            company_name = company["name"].values[0] if len(company) > 0 else stock_code
        except Exception:
            company_name = stock_code

        data = FinancialData(stock_code=stock_code, company_name=company_name)

        # 资产负债表
        try:
            bs = pro.balancesheet(ts_code=f"{stock_code}.{'SH' if stock_code.startswith('6') else 'SZ'}",
                                  start_date=f"{target_years[0]-1}0101", end_date=f"{target_years[-1]}1231",
                                  fields="end_date,total_assets,total_liabilities,current_assets,"
                                         "current_liabilities,money_cap,accounts_receiv,inventories,"
                                         "fix_assets,intan_assets,goodwill,defer_tax_assets,"
                                         "oth_receiv,oth_pay,contract_assets,st_borrow,lt_borrow,"
                                         "bonds_payable,total_hldr_eqy_exc_min_int")
            bs_map = {
                "money_cap": "money_funds", "accounts_receiv": "accounts_receivable",
                "inventories": "inventory", "fix_assets": "fixed_assets",
                "intan_assets": "intangible_assets", "goodwill": "goodwill",
                "defer_tax_assets": "deferred_tax_assets", "oth_receiv": "other_receivables",
                "oth_pay": "other_payables", "contract_assets": "contract_assets",
                "st_borrow": "short_term_borrowings", "lt_borrow": "long_term_borrowings",
                "bonds_payable": "bonds_payable", "current_assets": "current_assets",
                "total_assets": "total_assets", "current_liabilities": "current_liabilities",
                "total_liabilities": "total_liabilities",
                "total_hldr_eqy_exc_min_int": "net_equity",
            }
            self._tushare_parse(bs, data.balance_sheet, bs_map, target_years)
        except Exception as e:
            print(f"[WARN] Tushare资产负债表获取失败: {e}")

        # 利润表
        try:
            pl = pro.income(ts_code=f"{stock_code}.{'SH' if stock_code.startswith('6') else 'SZ'}",
                            start_date=f"{target_years[0]-1}0101", end_date=f"{target_years[-1]}1231",
                            fields="end_date,revenue,oper_cost,sell_exp,admin_exp,rad_exp_sum,"
                                   "fin_exp,assets_impair_loss,credit_impair_loss,n_income,"
                                   "deducted_profit,income_tax,total_profit,int_income,int_exp")
            pl_map = {
                "revenue": "revenue", "oper_cost": "cost_of_sales",
                "sell_exp": "selling_expenses", "admin_exp": "admin_expenses",
                "rad_exp_sum": "rd_expenses", "fin_exp": "finance_expenses",
                "assets_impair_loss": "asset_impairment_loss",
                "credit_impair_loss": "credit_impairment_loss",
                "n_income": "net_profit", "deducted_profit": "deducted_net_profit",
                "income_tax": "income_tax", "total_profit": "total_profit",
                "int_income": "interest_income", "int_exp": "interest_expense",
            }
            self._tushare_parse(pl, data.income_statement, pl_map, target_years)
        except Exception as e:
            print(f"[WARN] Tushare利润表获取失败: {e}")

        # 现金流量表
        try:
            cf = pro.cashflow(ts_code=f"{stock_code}.{'SH' if stock_code.startswith('6') else 'SZ'}",
                              start_date=f"{target_years[0]-1}0101", end_date=f"{target_years[-1]}1231",
                              fields="end_date,n_cashflow_act,n_cashflow_inv_act,n_cashflow_fin_act,"
                                     "c_recp_sg_rs,stot_cash_out_inv_act")
            cf_map = {
                "n_cashflow_act": "operating_cf", "n_cashflow_inv_act": "investing_cf",
                "n_cashflow_fin_act": "financing_cf", "c_recp_sg_rs": "sales_cash_received",
            }
            self._tushare_parse(cf, data.cashflow_statement, cf_map, target_years)
        except Exception as e:
            print(f"[WARN] Tushare现金流量表获取失败: {e}")

        data.years = target_years
        return data

    def _tushare_parse(self, df: pd.DataFrame, store: dict, col_map: dict, target_years: List[int]):
        """Tushare数据解析"""
        for _, row in df.iterrows():
            year = int(str(row["end_date"])[:4])
            if year not in target_years:
                continue
            if year not in store:
                store[year] = {}
            for src, dest in col_map.items():
                if src in row and pd.notna(row[src]):
                    store[year][dest] = float(row[src])

    # ================================================================
    # 手动输入
    # ================================================================
    def _load_manual(self, data_dict: Optional[dict]) -> FinancialData:
        """从手动输入的字典加载数据"""
        if data_dict is None:
            raise ValueError("手动模式需提供 data 参数")

        fd = FinancialData()
        fd.stock_code = data_dict.get("stock_code", "MANUAL")
        fd.company_name = data_dict.get("company_name", "手动输入")
        fd.industry = data_dict.get("industry", "制造业")
        fd.years = [int(y) for y in data_dict.get("years", [])]

        # 加载三表数据（JSON key 为字符串年份，需转为 int）
        for stmt_name in ["balance_sheet", "income_statement", "cashflow_statement"]:
            src = data_dict.get(stmt_name, {})
            target = getattr(fd, stmt_name)
            for year_str, fields in src.items():
                year = int(year_str)
                target[year] = {str(k): float(v) for k, v in fields.items()}

        # 加载附注
        notes = data_dict.get("notes", {})
        for year_str, fields in notes.items():
            year = int(year_str)
            fd.notes[year] = fields

        # 加载非财务信号
        fd.non_financial = data_dict.get("non_financial", {})

        # 从数据中提取 years（如果未显式指定）
        if not fd.years:
            all_years = set()
            for stmt in [fd.balance_sheet, fd.income_statement, fd.cashflow_statement]:
                all_years.update(stmt.keys())
            fd.years = sorted(all_years)

        return fd


def _guess_industry(stock_code: str) -> str:
    """根据股票代码猜测行业（简版）"""
    industry_map = {
        "000858": "白酒", "000568": "白酒", "000799": "白酒",
        "600519": "白酒",
        "601939": "银行", "601398": "银行", "600036": "银行", "000001": "银行",
        "601318": "保险", "601601": "保险",
        "000002": "房地产", "600048": "房地产",
        "600276": "医药流通", "000538": "医药制造",
        "002714": "生猪养殖", "000876": "生猪养殖",
        "300750": "新能源", "002594": "新能源",
        "688981": "科技", "002415": "科技",
    }
    return industry_map.get(stock_code, "制造业")


_STOCK_NAME_MAP = {
    "600519": "贵州茅台", "000858": "五粮液", "000568": "泸州老窖",
    "600276": "恒瑞医药", "000538": "云南白药",
    "601398": "工商银行", "601939": "建设银行", "600036": "招商银行",
    "000002": "万科A", "000001": "平安银行",
    "601318": "中国平安", "300750": "宁德时代",
    "002714": "牧原股份", "002594": "比亚迪",
    "688981": "中芯国际", "002415": "海康威视",
}


def _get_stock_name(stock_code: str) -> str:
    return _STOCK_NAME_MAP.get(stock_code, stock_code)


# ================================================================
# 新浪财经 → 内部字段映射表
# ================================================================
_SINA_BALANCE_MAP = {
    "货币资金":         "money_funds",
    "应收票据及应收账款": "accounts_receivable",  # 新准则合并列
    "应收账款":         "accounts_receivable",
    "存货":             "inventory",
    "固定资产":         "fixed_assets",          # 固定资产净额
    "固定资产原值":     "fixed_assets",          # fallback：用原值
    "无形资产":         "intangible_assets",
    "商誉":             "goodwill",
    "递延所得税资产":   "deferred_tax_assets",
    "其他应收款":       "other_receivables",
    "其他应收款(合计)": "other_receivables",
    "其他应付款":       "other_payables",
    "其他应付款合计":   "other_payables",
    "合同资产":         "contract_assets",
    "合同负债":         "advance_from_customers",  # 新准则代替预收账款
    "预收账款":         "advance_from_customers",
    "预收款项":         "advance_from_customers",
    "短期借款":         "short_term_borrowings",
    "长期借款":         "long_term_borrowings",
    "应付债券":         "bonds_payable",
    "流动资产合计":     "current_assets",
    "资产总计":         "total_assets",
    "流动负债合计":     "current_liabilities",
    "负债合计":         "total_liabilities",
    "归属于母公司股东权益合计": "net_equity",
    "所有者权益(或股东权益)合计": "net_equity",
    "应收票据":         "notes_receivable",
    "应付票据":         "notes_payable",
    "应付票据及应付账款": "accounts_payable",
    "应付账款":         "accounts_payable",
    "预付账款":         "prepayments",
    "预付款项":         "prepayments",
    "在建工程合计":     "construction_in_progress",
    "在建工程":         "construction_in_progress",
    "应付职工薪酬":     "salary_payable",
}

_SINA_INCOME_MAP = {
    "一、营业总收入":   "revenue",
    "营业总收入":       "revenue",
    "营业收入":         "revenue",
    "二、营业总成本":   "cost_of_sales",
    "营业总成本":       "cost_of_sales",
    "营业成本":         "cost_of_sales",
    "研发费用":         "rd_expenses",
    "销售费用":         "selling_expenses",
    "管理费用":         "admin_expenses",
    "财务费用":         "finance_expenses",
    "利息收入":         "interest_income",
    "利息费用":         "interest_expense",
    "利息支出":         "interest_expense",
    "投资收益":         "investment_income",
    "资产减值损失":     "asset_impairment_loss",
    "信用减值损失":     "credit_impairment_loss",
    "五、净利润":       "net_profit",
    "净利润":           "net_profit",
    "归属于母公司所有者的净利润": "net_profit",
    "扣除非经常性损益后的净利润": "deducted_net_profit",
    "所得税费用":       "income_tax",
    "四、利润总额":     "total_profit",
    "利润总额":         "total_profit",
}

_SINA_CASHFLOW_MAP = {
    "经营活动产生的现金流量净额":       "operating_cf",
    "经营活动产生的现金流量净额小计":   "operating_cf",
    "投资活动产生的现金流量净额":       "investing_cf",
    "筹资活动产生的现金流量净额":       "financing_cf",
    "销售商品、提供劳务收到的现金":     "sales_cash_received",
    "购建固定资产、无形资产和其他长期资产支付的现金": "capex",
    "支付给职工以及为职工支付的现金":   "salary_paid",
    "支付的各项税费":                   "tax_paid",
    "经营活动现金流出小计":             "operating_cf_outflow",
}
