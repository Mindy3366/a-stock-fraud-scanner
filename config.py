"""
财务造假识别 Agent — 配置模块
严格基于 SKILL (2) 定义的：三级失真扫描、重点科目深度核查、商业逻辑验证、
行业特性调整、关键阈值速查、常见造假模式、数字化时代特殊风险。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ============================================================
# 三级风险定义
# ============================================================
RISK_LEVEL_10 = "10级-疑似造假"
RISK_LEVEL_8 = "8级-疑似操纵"
RISK_LEVEL_5 = "5级-盈余管理"
RISK_LEVEL_3 = "3级-经营风险"

# 风险等级得分映射
RISK_SCORE_MAP = {
    RISK_LEVEL_10: 15,  # 每触发一条 +15分
    RISK_LEVEL_8: 10,   # 每触发一条 +10分
    RISK_LEVEL_5: 5,    # 每触发一条 +5分
    RISK_LEVEL_3: 3,    # 每触发一条 +3分
}

# 综合风险等级区间
RISK_BANDS = [
    (0, 15, "GREEN", "低风险", "财务报表质量良好，未发现重大异常信号"),
    (16, 35, "YELLOW", "中风险", "存在若干警示信号，需进一步关注"),
    (36, 60, "ORANGE", "高风险", "存在多项重大异常，建议谨慎"),
    (61, 100, "RED", "极高风险", "强烈建议排除，存在严重造假嫌疑"),
]

# ============================================================
# 步骤2.1：财务造假扫描规则（10级风险）
# ============================================================
FRAUD_RULES = [
    {
        "id": "FRAUD-001",
        "name": "存贷双高",
        "risk_level": RISK_LEVEL_10,
        "description": "货币资金 > 有息负债 × 50%，且利息收入率 < 0.5%，账面巨额现金却支付高额利息",
        "formula": "money_funds > interest_bearing_debt * 0.5 and interest_income_rate < 0.005",
        "thresholds": {"interest_income_rate": 0.005, "deposit_loan_ratio": 0.5},
        "fraud_type": "存贷双高型",
        "check_points": ["银行函证", "资金流水", "关联方资金往来"],
    },
    {
        "id": "FRAUD-002",
        "name": "利息收支背离",
        "risk_level": RISK_LEVEL_10,
        "description": "利息费用 / 利息收入 > 10，利息费用为利息收入10倍以上，账面现金真实性存疑",
        "formula": "interest_expense / max(interest_income, 1) > 10",
        "thresholds": {"interest_expense_income_ratio": 10},
        "fraud_type": "存贷双高型",
        "check_points": ["银行函证", "利息收入明细", "货币资金明细"],
    },
    {
        "id": "FRAUD-003",
        "name": "存货-毛利率背离",
        "risk_level": RISK_LEVEL_10,
        "description": "存货周转率同比下降 > 20%，同时毛利率同比上升 > 5个百分点，违背商业逻辑",
        "formula": "inventory_turnover_yoy < -0.2 and gross_margin_yoy > 0.05",
        "thresholds": {"inv_turnover_change": -0.2, "gross_margin_change": 0.05},
        "fraud_type": "存货魔术型",
        "check_points": ["实地盘点", "供应商函证", "成本结转测试"],
    },
    {
        "id": "FRAUD-004",
        "name": "现金流利润背离",
        "risk_level": RISK_LEVEL_10,
        "description": "连续2年经营现金流/净利润比 < 0.5，且净利润为正，有利润无现金流",
        "formula": "连续2年 ocf_to_np < 0.5 and net_profit > 0",
        "thresholds": {"ocf_to_np": 0.5, "consecutive_years": 2},
        "fraud_type": "收入跨期型",
        "check_points": ["收入确认时点", "应收账款回款", "客户验收单"],
    },
    {
        "id": "FRAUD-005",
        "name": "关联方资金异常",
        "risk_level": RISK_LEVEL_10,
        "description": "关联方资金流出 / 经营活动现金流出 > 50%，资金大量流向关联方",
        "formula": "related_party_outflow / operating_cf_outflow > 0.5",
        "thresholds": {"related_party_outflow_ratio": 0.5},
        "fraud_type": "关联交易型",
        "check_points": ["关联方清单", "交易定价公允性", "资金流水追踪"],
    },
    {
        "id": "FRAUD-006",
        "name": "境外收入异常",
        "risk_level": RISK_LEVEL_10,
        "description": "境外收入占比 > 40%，且境外毛利率 > 境内毛利率 + 10个百分点，且经营现金流持续为负",
        "formula": "overseas_rev_pct > 0.4 and (overseas_gm - domestic_gm) > 0.1 and operating_cf < 0",
        "thresholds": {"overseas_rev_pct": 0.4, "gm_diff": 0.1},
        "fraud_type": "境外虚构型",
        "check_points": ["海关出口数据", "境外客户背景", "合同真实性"],
    },
    {
        "id": "FRAUD-007",
        "name": "合同资产异常",
        "risk_level": RISK_LEVEL_10,
        "description": "合同资产/应收账款 > 1，且减值准备计提比例 < 30%，暗示提前确认收入",
        "formula": "contract_asset_to_ar > 1 and contract_asset_impair_rate < 0.3",
        "thresholds": {"contract_asset_to_ar": 1.0, "impair_rate": 0.3},
        "fraud_type": "合同资产型",
        "check_points": ["项目批文", "监理报告", "履约进度依据"],
    },
    {
        "id": "FRAUD-008",
        "name": "其他应收款异常",
        "risk_level": RISK_LEVEL_10,
        "description": "其他应收款/流动资产 > 15%，且账龄1年以内 > 80%，且对象含关联方",
        "formula": "other_receivables_pct > 0.15 and within_1yr_pct > 0.8",
        "thresholds": {"other_receivables_pct": 0.15, "within_1yr_pct": 0.8},
        "fraud_type": "资金占用型",
        "check_points": ["款项性质", "挂账原因", "还款计划"],
    },
    {
        "id": "FRAUD-009",
        "name": "巨额减值消化",
        "risk_level": RISK_LEVEL_10,
        "description": "信用减值损失占净利润亏损额 > 80%，且次年减值大幅减少",
        "formula": "credit_impair_loss / abs(net_loss) > 0.8 and next_year_impair_decreases_significantly",
        "thresholds": {"impair_to_loss": 0.8},
        "fraud_type": "减值调节型",
        "check_points": ["减值测试方法", "假设参数", "行业对比"],
    },
    {
        "id": "FRAUD-010",
        "name": "收入成本不匹配",
        "risk_level": RISK_LEVEL_10,
        "description": "境外收入激增但销售收现比 < 60%，且无合理商业解释",
        "formula": "overseas_rev_growth_high and sales_cash_ratio < 0.6",
        "thresholds": {"sales_cash_ratio": 0.6},
        "fraud_type": "境外虚构型",
        "check_points": ["海关数据", "回款记录", "第三方验证"],
    },
]

# ============================================================
# 步骤2.2：会计操纵扫描规则（8级风险）
# ============================================================
MANIPULATION_RULES = [
    {
        "id": "MANIP-001",
        "name": "会计政策频繁变更",
        "risk_level": RISK_LEVEL_8,
        "description": "3年内会计政策/估计变更 > 2次",
        "formula": "policy_changes_3yr > 2",
        "thresholds": {"policy_changes": 2},
        "fraud_type": "差错更造型",
    },
    {
        "id": "MANIP-002",
        "name": "减值准备剧烈波动",
        "risk_level": RISK_LEVEL_8,
        "description": "当年资产减值损失 / 上年资产减值损失 > 3倍或 < 0.3倍",
        "formula": "asset_impair_ratio > 3 or asset_impair_ratio < 0.3",
        "thresholds": {"impair_ratio_upper": 3.0, "impair_ratio_lower": 0.3},
        "fraud_type": "减值调节型",
    },
    {
        "id": "MANIP-003",
        "name": "商誉减值延迟",
        "risk_level": RISK_LEVEL_8,
        "description": "收购标的连续2年未完成业绩承诺，但商誉未减值或减值比例 < 30%",
        "formula": "acquisition_underperform_2yr and (goodwill_impaired == False or goodwill_impair_rate < 0.3)",
        "thresholds": {"underperform_years": 2, "impair_rate": 0.3},
        "fraud_type": "商誉洗澡型",
    },
    {
        "id": "MANIP-004",
        "name": "递延所得税资产激增",
        "risk_level": RISK_LEVEL_8,
        "description": "连续亏损下，递延所得税资产/总资产 > 2%，且所得税费用为负",
        "formula": "consecutive_loss and dta_to_ta > 0.02 and income_tax < 0",
        "thresholds": {"dta_to_ta": 0.02},
        "fraud_type": "递延所得税型",
    },
    {
        "id": "MANIP-005",
        "name": "收入确认激进",
        "risk_level": RISK_LEVEL_8,
        "description": "四季度收入占全年 > 40%，且四季度毛利率显著高于前三季度",
        "formula": "q4_rev_pct > 0.4 and q4_gm > avg_q1q3_gm",
        "thresholds": {"q4_rev_pct": 0.4},
        "fraud_type": "收入跨期型",
    },
    {
        "id": "MANIP-006",
        "name": "减值计提比例异常",
        "risk_level": RISK_LEVEL_8,
        "description": "存货跌价准备/存货余额比例同比变化 > 50%",
        "formula": "abs(inventory_impair_rate_change) > 0.5",
        "thresholds": {"inv_impair_rate_change": 0.5},
        "fraud_type": "减值调节型",
    },
    {
        "id": "MANIP-007",
        "name": "利润平滑迹象",
        "risk_level": RISK_LEVEL_8,
        "description": "连续4个季度净利润波动幅度 < 5%（非金融/公用事业）",
        "formula": "quarterly_np_volatility < 0.05",
        "thresholds": {"np_volatility": 0.05},
        "fraud_type": "利润平滑型",
    },
    {
        "id": "MANIP-008",
        "name": "会计差错更正",
        "risk_level": RISK_LEVEL_8,
        "description": "3年内会计差错更正涉及金额 / 当期净利润 > 5%",
        "formula": "correction_amount / net_profit > 0.05",
        "thresholds": {"correction_to_np": 0.05},
        "fraud_type": "差错更造型",
    },
    {
        "id": "MANIP-009",
        "name": "关联交易非公允",
        "risk_level": RISK_LEVEL_8,
        "description": "关联交易定价较市场均价偏离 > 15%，且拒绝披露可比第三方数据",
        "formula": "rpt_price_deviation > 0.15",
        "thresholds": {"price_deviation": 0.15},
        "fraud_type": "关联交易型",
    },
]

# ============================================================
# 步骤2.3：盈余管理扫描规则（5级风险）
# ============================================================
EARNINGS_MGMT_RULES = [
    {
        "id": "EM-001",
        "name": "研发资本化率异常",
        "risk_level": RISK_LEVEL_5,
        "description": "研发支出资本化率 > 同行业均值 + 20个百分点",
        "formula": "rd_capitalization_rate > industry_avg + 0.2",
        "thresholds": {"rd_cap_rate_diff": 0.2},
        "fraud_type": "费用资本化型",
    },
    {
        "id": "EM-002",
        "name": "非经常性损益依赖",
        "risk_level": RISK_LEVEL_5,
        "description": "非经常性损益/净利润 > 30%，且扣非净利润 < 净利润 × 50%",
        "formula": "non_recurring_to_np > 0.3 and deducted_np < np * 0.5",
        "thresholds": {"non_recurring_ratio": 0.3, "deducted_np_ratio": 0.5},
        "fraud_type": "利润平滑型",
    },
    {
        "id": "EM-003",
        "name": "微利巨亏循环",
        "risk_level": RISK_LEVEL_5,
        "description": "近5年净利润呈现亏-盈-亏-盈交替，且盈亏绝对值均 < 1亿或 > 5亿",
        "formula": "alternating_profit_loss_pattern",
        "thresholds": {"small_profit": 100_000_000, "large_loss": 500_000_000},
        "fraud_type": "利润平滑型",
    },
    {
        "id": "EM-004",
        "name": "业绩预告修正频繁",
        "risk_level": RISK_LEVEL_5,
        "description": "2年内业绩预告修正 > 2次，且修正幅度 > 50%",
        "formula": "forecast_revisions_2yr > 2 and revision_magnitude > 0.5",
        "thresholds": {"revision_count": 2, "revision_magnitude": 0.5},
        "fraud_type": "预告修正型",
    },
    {
        "id": "EM-005",
        "name": "费用资本化倾向",
        "risk_level": RISK_LEVEL_5,
        "description": "无形资产/总资产比同比增幅 > 50%，且研发费用资本化率同步上升",
        "formula": "intangible_growth > 0.5 and rd_cap_rate_increasing",
        "thresholds": {"intangible_growth": 0.5},
        "fraud_type": "费用资本化型",
    },
    {
        "id": "EM-006",
        "name": "股份支付费用高",
        "risk_level": RISK_LEVEL_5,
        "description": "股份支付费用 / 净利润 > 10%",
        "formula": "share_based_payment / net_profit > 0.1",
        "thresholds": {"sbp_to_np": 0.1},
        "fraud_type": "人力资本型",
    },
    {
        "id": "EM-007",
        "name": "选择性披露",
        "risk_level": RISK_LEVEL_5,
        "description": "利好信息集中发布，利空信息延迟披露",
        "formula": "selective_disclosure_pattern",
        "thresholds": {},
        "fraud_type": "预告修正型",
    },
    {
        "id": "EM-008",
        "name": "年末突击交易",
        "risk_level": RISK_LEVEL_5,
        "description": "年末集中发生资产处置、债务重组、政府补助确认",
        "formula": "year_end_rush_transactions",
        "thresholds": {},
        "fraud_type": "利润平滑型",
    },
]

# ============================================================
# 全部规则汇总
# ============================================================
ALL_RULES = FRAUD_RULES + MANIPULATION_RULES + EARNINGS_MGMT_RULES

# ============================================================
# 关键阈值速查表（附录A）
# ============================================================
KEY_THRESHOLDS = {
    # (指标, 警戒线, 危险线)
    "存贷比": (0.5, 0.2, "money_funds / interest_bearing_debt"),
    "利息收入率": (0.005, 0.002, "interest_income / avg_money_funds"),
    "利息费用/利息收入": (5, 10, "interest_expense / interest_income"),
    "经营现金流/净利润": (0.5, 0, "operating_cf / net_profit"),
    "其他应收款/流动资产": (0.10, 0.20, "other_receivables / current_assets"),
    "递延所得税资产/总资产": (0.01, 0.02, "dta / total_assets"),
    "合同资产/应收账款": (0.8, 1.2, "contract_assets / accounts_receivable"),
    "境外收入占比": (0.30, 0.50, "overseas_revenue / total_revenue"),
    "非经常性损益/净利润": (0.20, 0.50, "non_recurring / net_profit"),
    "商誉/净资产": (0.20, 0.40, "goodwill / net_equity"),
    "资产负债率(非金融)": (0.60, 0.80, "total_liabilities / total_assets"),
    "流动比率": (1.5, 1.2, "current_assets / current_liabilities"),
    "利息保障倍数": (5, 3, "(profit_total + interest_expense) / interest_expense"),
    "销售收现比": (0.90, 0.80, "sales_cash_received / revenue"),
    "存货/营业成本": (1.5, 2.0, "inventory / cost_of_sales"),
    "应收账款周转天数": (90, 120, "365 / (revenue / avg_ar)"),
    "存货跌价准备/存货余额": (0.02, 0.01, "inv_impair / inventory"),
    "信用减值损失/净利润亏损额": (0.50, 0.80, "credit_impair / abs(net_loss)"),
    "四季度收入占比": (0.35, 0.40, "q4_revenue / annual_revenue"),
    "股权质押比例": (0.50, 0.70, "pledged_shares / total_holdings"),
    "前5大客户占比": (0.50, 0.70, "top5_customer_rev / total_revenue"),
    "研发资本化率": (0.30, 0.50, "capitalized_rd / total_rd"),
    "股份支付费用/净利润": (0.10, 0.20, "sbp / net_profit"),
}

# ============================================================
# 行业特性调整系数（附录B）
# ============================================================
INDUSTRY_ADJUSTMENTS = {
    "房地产": {
        "资产负债率警戒线": 0.70,
        "notes": "存货周转需结合开发周期；预收账款/营业收入比值下降可能暗示销售放缓"
    },
    "银行": {
        "notes": "不适用存贷比、利息收入率；关注不良贷款率(>2%标记)、拨备覆盖率(<150%标记)"
    },
    "保险": {
        "notes": "不适用存贷比、利息收入率；关注偿付能力充足率"
    },
    "医药流通": {
        "应收账款周转天数警戒线": 120,
        "notes": "存贷双高需结合行业资金密集特性判断；上游付款期vs下游回款期是行业共性"
    },
    "建筑施工": {
        "合同资产/应收账款警戒线": 1.5,
        "notes": "收入确认依赖进度核验，关注项目履约进度"
    },
    "生猪养殖": {
        "notes": "存货规模需结合存栏量判断，不适用常规存货周转率；毛利率波动大属猪周期正常现象"
    },
    "软件开发": {
        "研发资本化率警戒线": 0.40,
        "notes": "关注收入确认方法（时段法/时点法）"
    },
    "新能源": {
        "notes": "技术迭代快，需检查产能利用率和技术路线；存货跌价准备需考虑原材料价格波动"
    },
    "电池": {
        "notes": "同新能源，固定资产减值风险高，关注技术路线迭代"
    },
    "白酒": {
        "notes": "预收账款减少可能暗示渠道库存积压；品牌白酒毛利率应稳定，异常波动需核查"
    },
    "科技": {
        "notes": "人力资本密集，股份支付费用高属正常；数字化投入费用化处理方式需关注"
    },
    "互联网": {
        "notes": "同科技，人力资本密集；关注股份支付和研发支出会计处理"
    },
    "重资产制造业": {
        "notes": "存贷双高需结合扩张期判断；关注产能利用率"
    },
}

# ============================================================
# 常见造假模式识别卡（附录D）
# ============================================================
FRAUD_PATTERNS = {
    "存贷双高型": {
        "core_features": "账面现金充裕却高额负债",
        "formula": "货币资金 > 有息负债×50% 且 利息收入率 < 0.5%",
        "typical_case": "康美药业",
        "check_points": ["银行函证", "资金流水", "关联方资金往来"],
    },
    "存货魔术型": {
        "core_features": "存货规模异常且减值计提不足",
        "formula": "存货/营业成本 > 2 且 存货跌价准备/存货 < 1%",
        "typical_case": "康美药业",
        "check_points": ["实地盘点", "供应商函证", "成本结转测试"],
    },
    "关联交易型": {
        "core_features": "资金通过关联方循环",
        "formula": "关联方资金流出/经营现金流出 > 50%",
        "typical_case": "康美药业",
        "check_points": ["关联方清单", "交易定价", "资金流水"],
    },
    "境外虚构型": {
        "core_features": "境外收入激增但无现金回流",
        "formula": "境外收入占比 > 40% 且 经营现金流持续为负",
        "typical_case": "山河智能",
        "check_points": ["海关数据", "境外客户背景", "合同真实性"],
    },
    "合同资产型": {
        "core_features": "合同资产高于应收账款",
        "formula": "合同资产/应收账款 > 1 且 未取得关键批文",
        "typical_case": "ST中程",
        "check_points": ["项目批文", "监理报告", "履约进度依据"],
    },
    "减值调节型": {
        "core_features": "减值准备剧烈波动平滑利润",
        "formula": "当年减值/上年减值 > 3倍 或 < 0.3倍",
        "typical_case": "拓维信息",
        "check_points": ["减值测试方法", "假设参数", "行业对比"],
    },
    "递延所得税型": {
        "core_features": "亏损状态下确认大额递延所得税资产",
        "formula": "连续亏损且 递延所得税资产/总资产 > 2%",
        "typical_case": "东旭光电",
        "check_points": ["未来盈利预测依据", "行业景气度"],
    },
    "资金占用型": {
        "core_features": "其他应收款异常指向关联方",
        "formula": "其他应收款/流动资产 > 15% 且 关联方占比高",
        "typical_case": "东方海洋",
        "check_points": ["款项性质", "挂账原因", "还款计划"],
    },
    "预告修正型": {
        "core_features": "业绩预告频繁大幅修正",
        "formula": "2年内修正 > 2次 且 幅度 > 50%",
        "typical_case": "贝因美",
        "check_points": ["修正原因", "内控有效性", "核算基础"],
    },
    "差错更造型": {
        "core_features": "频繁会计差错更正",
        "formula": "3年内更正金额/净利润 > 5%",
        "typical_case": "ST星源",
        "check_points": ["差错性质", "是否系统性", "内控缺陷"],
    },
    "商誉洗澡型": {
        "core_features": "收购后延迟减值集中计提",
        "formula": "标的连续亏损但商誉未减值，后集中计提",
        "typical_case": "瑞康医药",
        "check_points": ["业绩承诺完成情况", "减值测试假设"],
    },
    "金融资产型": {
        "core_features": "多层嵌套金融资产风险隐蔽",
        "formula": "金融资产/总资产 > 20% 且 底层资产不透明",
        "typical_case": "横店东磁",
        "check_points": ["产品结构", "底层资产", "风险传导路径"],
    },
    "费用资本化型": {
        "core_features": "将应费用化支出资本化",
        "formula": "研发资本化率 > 同行均值 + 20%",
        "typical_case": "部分科技企业",
        "check_points": ["资本化条件", "研究阶段/开发阶段划分"],
    },
    "收入跨期型": {
        "core_features": "年末突击确认收入",
        "formula": "四季度收入占比 > 40% 且 毛利率异常高",
        "typical_case": "部分制造企业",
        "check_points": ["截止性测试", "期后退货", "客户验收单"],
    },
    "利润平滑型": {
        "core_features": "非经常性损益调节盈亏",
        "formula": "非经常性损益/净利润 > 30% 且 微利巨亏循环",
        "typical_case": "拓维信息",
        "check_points": ["非经常性损益构成", "可持续性", "发生频率"],
    },
}

# ============================================================
# 数字化时代特殊风险检查清单（附录E）
# ============================================================
DIGITAL_ERA_CHECKS = [
    {
        "id": "DIGI-001",
        "name": "金融资产多层嵌套",
        "check": "存在信托→私募基金→底层资产等多层结构",
        "risk": "风险传导路径不透明",
    },
    {
        "id": "DIGI-002",
        "name": "非标理财",
        "check": "投资于非标理财产品",
        "risk": "流动性差，估值困难，违约风险高",
    },
    {
        "id": "DIGI-003",
        "name": "公允价值变动",
        "check": "公允价值变动损益/净利润 > 20%",
        "risk": "利润波动大，缺乏现金流支撑",
    },
    {
        "id": "DIGI-004",
        "name": "技术迭代-固资减值",
        "check": "技术路线升级导致产能利用率 < 60%",
        "risk": "旧技术资产面临淘汰，需计提减值",
    },
    {
        "id": "DIGI-005",
        "name": "技术迭代-存货跌价",
        "check": "基于旧技术标准的库存产品价值缩水",
        "risk": "新技术替代导致库存贬值",
    },
    {
        "id": "DIGI-006",
        "name": "技术迭代-无形资产减值",
        "check": "前期布局的矿产资源/技术专利因市场变化减值",
        "risk": "产业链布局需承担迭代成本",
    },
    {
        "id": "DIGI-007",
        "name": "数字化投入费用化分散",
        "check": "数字化投入分散计入研发费用、管理费用、销售费用",
        "risk": "无法作为独立核心资产列示，资产负债表失真",
    },
    {
        "id": "DIGI-008",
        "name": "ROA失真",
        "check": "数字化投入费用化导致ROA虚高或虚低",
        "risk": "分母未包含数字化资产，指标失真",
    },
    {
        "id": "DIGI-009",
        "name": "股份支付费用扭曲",
        "check": "股份支付费用/净利润 > 10%",
        "risk": "对人才的投资被当作当期成本，利润扭曲",
    },
    {
        "id": "DIGI-010",
        "name": "人才流失",
        "check": "核心技术人员离职且未披露",
        "risk": "人力资本价值难以在报表反映",
    },
]

# ============================================================
# 会计准则引用速查（附录C）
# ============================================================
ACCOUNTING_STANDARDS = {
    "收入提前确认": "《企业会计准则第14号——收入》第四条、第五条",
    "收入确认方法变更": "《企业会计准则第14号——收入》第八条、第九条",
    "资产减值测试": "《企业会计准则第8号——资产减值》第五条、第六条",
    "关联方披露": "《企业会计准则第36号——关联方披露》第二条、第十条",
    "政府补助": "《企业会计准则第16号——政府补助》第六条、第七条",
    "股份支付": "《企业会计准则第11号——股份支付》第四条、第六条",
    "无形资产确认": "《企业会计准则第6号——无形资产》第三条、第四条",
    "金融工具分类": "《企业会计准则第22号——金融工具确认和计量》第十六条、第十七条",
    "存货计量": "《企业会计准则第1号——存货》第十五条、第十六条",
    "所得税": "《企业会计准则第18号——所得税》第十五条、第二十条",
    "会计政策变更": "《企业会计准则第28号——会计政策、会计估计变更和差错更正》第四条、第五条",
    "会计估计变更": "《企业会计准则第28号——会计政策、会计估计变更和差错更正》第八条、第九条",
    "差错更正": "《企业会计准则第28号——会计政策、会计估计变更和差错更正》第十二条、第十四条",
    "租赁": "《企业会计准则第21号——租赁》第四条、第十四条",
    "合并报表": "《企业会计准则第33号——合并财务报表》第七条、第二十条",
}

# ============================================================
# 数据列映射（输入数据的标准字段名）
# ============================================================
DATA_FIELD_MAP = {
    # 资产负债表
    "money_funds": ["货币资金", "money_funds", "cash_and_equivalents"],
    "accounts_receivable": ["应收账款", "accounts_receivable", "trade_receivables"],
    "inventory": ["存货", "inventory", "inventories"],
    "fixed_assets": ["固定资产", "fixed_assets", "ppe"],
    "intangible_assets": ["无形资产", "intangible_assets"],
    "goodwill": ["商誉", "goodwill"],
    "deferred_tax_assets": ["递延所得税资产", "deferred_tax_assets", "dta"],
    "other_receivables": ["其他应收款", "other_receivables"],
    "other_payables": ["其他应付款", "other_payables"],
    "contract_assets": ["合同资产", "contract_assets"],
    "short_term_borrowings": ["短期借款", "short_term_borrowings"],
    "long_term_borrowings": ["长期借款", "long_term_borrowings"],
    "bonds_payable": ["应付债券", "bonds_payable"],
    "current_assets": ["流动资产", "current_assets"],
    "total_assets": ["总资产", "total_assets", "total_assets"],
    "current_liabilities": ["流动负债", "current_liabilities"],
    "total_liabilities": ["总负债", "total_liabilities", "total_liabilities"],
    "net_equity": ["净资产", "net_equity", "shareholders_equity"],
    # 利润表
    "revenue": ["营业收入", "revenue", "operating_revenue"],
    "cost_of_sales": ["营业成本", "cost_of_sales", "operating_cost"],
    "selling_expenses": ["销售费用", "selling_expenses"],
    "admin_expenses": ["管理费用", "admin_expenses"],
    "rd_expenses": ["研发费用", "rd_expenses", "research_expenses"],
    "finance_expenses": ["财务费用", "finance_expenses"],
    "asset_impairment_loss": ["资产减值损失", "asset_impairment_loss"],
    "credit_impairment_loss": ["信用减值损失", "credit_impairment_loss"],
    "net_profit": ["净利润", "net_profit", "net_income"],
    "deducted_net_profit": ["扣非净利润", "deducted_net_profit"],
    "income_tax": ["所得税费用", "income_tax", "income_tax_expense"],
    "non_recurring_pl": ["非经常性损益", "non_recurring_pl"],
    "interest_income": ["利息收入", "interest_income"],
    "interest_expense": ["利息费用", "interest_expense"],
    "total_profit": ["利润总额", "total_profit", "profit_before_tax"],
    # 现金流量表
    "operating_cf": ["经营活动现金流净额", "operating_cf", "operating_cash_flow"],
    "investing_cf": ["投资活动现金流净额", "investing_cf", "investing_cash_flow"],
    "financing_cf": ["筹资活动现金流净额", "financing_cf", "financing_cash_flow"],
    "sales_cash_received": ["销售商品提供劳务收到的现金", "sales_cash_received"],
    "capex": ["购建固定资产无形资产和其他长期资产支付的现金", "capex"],
    # 附注
    "overseas_revenue": ["境外收入", "overseas_revenue"],
    "overseas_gross_margin": ["境外毛利率", "overseas_gross_margin"],
    "domestic_gross_margin": ["境内毛利率", "domestic_gross_margin"],
    "related_party_revenue": ["关联交易金额", "related_party_revenue"],
    "share_based_payment": ["股份支付费用", "share_based_payment"],
    "rd_capitalized": ["研发支出资本化金额", "rd_capitalized"],
    "rd_total": ["研发支出总额", "rd_total"],
    "inventory_impairment": ["存货跌价准备", "inventory_impairment"],
    # 非财务
    "audit_opinion": ["审计意见类型", "audit_opinion"],
    "pledge_ratio": ["大股东质押比例", "pledge_ratio"],
    "policy_changes": ["会计政策变更次数", "policy_changes"],
    "accounting_corrections": ["会计差错更正次数", "accounting_corrections"],
    "forecast_revisions": ["业绩预告修正次数", "forecast_revisions"],
    "management_reduction": ["董监高减持", "management_reduction"],
    "litigation_amount": ["诉讼涉及金额", "litigation_amount"],
    "regulatory_inquiries": ["监管问询次数", "regulatory_inquiries"],
}
