"""
财务造假识别 Agent v2.0 — 思维链管道
升级要点:
  1. 新增会计勾稽验证层（10项勾稽规则）
  2. 新增商业逻辑验证层（16项商业逻辑规则）
  3. 新增LLM增强分析层（LLM + 规则回退双模）
  4. 三维度加权评分体系

执行流程:
  STEP 1: 数据收集与预处理
  STEP 2: 衍生指标计算
  STEP 3: 三级失真扫描 (27条规则)
  STEP 4: 会计勾稽验证 (10条勾稽)       ← NEW
  STEP 5: 商业逻辑验证 (16条商业逻辑)    ← NEW
  STEP 6: 重点科目深度核查
  STEP 7: LLM增强分析                   ← NEW
  STEP 8: 综合风险评分与评级
  STEP 9: 报告输出
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from data_fetcher import DataFetcher, FinancialData, fetch_realtime_price
from indicator_calc import IndicatorCalculator
from rule_engine import RuleEngine, RuleResult, DeepAuditResult
from accounting_crosscheck import AccountingCrossChecker, CrossCheckResult
from business_logic_verify import BusinessLogicVerifier, BizLogicResult
from llm_analyzer import LLMAnalyzer, LLMAnalysisResult
from scoring import RiskScorer, RiskProfile


@dataclass
class AnalysisReport:
    """完整分析报告 v2.0"""
    company_name: str = ""
    stock_code: str = ""
    industry: str = ""
    analysis_period: str = ""
    report_date: str = ""
    data_source: str = ""

    # 风险总览
    risk_profile: Optional[RiskProfile] = None

    # === 各层分析结果 ===
    # 三级失真扫描
    fraud_results: List[RuleResult] = field(default_factory=list)
    manipulation_results: List[RuleResult] = field(default_factory=list)
    earnings_mgmt_results: List[RuleResult] = field(default_factory=list)
    business_logic_results: List[RuleResult] = field(default_factory=list)

    # 会计勾稽验证 ← NEW
    crosscheck_results: List[CrossCheckResult] = field(default_factory=list)

    # 商业逻辑验证 ← NEW
    biz_logic_verify_results: List[BizLogicResult] = field(default_factory=list)

    # 深度核查
    deep_audit_results: List[DeepAuditResult] = field(default_factory=list)

    # LLM分析 ← NEW
    llm_analysis: Optional[LLMAnalysisResult] = None

    # 数字化风险
    digital_era_alerts: List[Dict] = field(default_factory=list)

    # 关键指标
    key_indicators_summary: Dict[str, Dict[int, float]] = field(default_factory=dict)

    # 实时行情（仅作参考，不纳入风险评分）
    realtime_price: Optional[Dict] = None

    # 执行元数据
    execution_time_seconds: float = 0.0
    execution_steps: List[str] = field(default_factory=list)
    version: str = "2.0.0"


class FraudDetectionAgent:
    """财务造假识别 Agent v2.0 — 三维度思维链"""

    def __init__(
        self,
        data_source: str = "akshare",
        tushare_token: str = "",
        llm_backend: str = "heuristic",
        llm_model: str = "",
        llm_api_key: str = "",
        llm_api_base: str = "",
    ):
        self.data_source = data_source
        self.tushare_token = tushare_token
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.data: Optional[FinancialData] = None
        self.indicators: Optional[IndicatorCalculator] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.crosschecker: Optional[AccountingCrossChecker] = None
        self.biz_verifier: Optional[BusinessLogicVerifier] = None
        self.report = AnalysisReport()

    def analyze(self, stock_code: str, years: int = 5, manual_data: Optional[dict] = None,
                industry: str = "") -> AnalysisReport:
        """主入口"""
        start_time = time.time()

        # ---- STEP 1: 数据收集 ----
        self._log("STEP 1: 数据收集与预处理")
        fetcher = DataFetcher(source="manual" if manual_data else self.data_source,
                              tushare_token=self.tushare_token)

        # 同时拉取财务数据与实时行情：两者独立，实时行情异常不影响财务风险分析
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            if manual_data:
                fin_future = ex.submit(fetcher.fetch, stock_code, years, data=manual_data)
            else:
                fin_future = ex.submit(fetcher.fetch, stock_code, years)
            rt_future = ex.submit(fetch_realtime_price, stock_code)
            self.data = fin_future.result()
            try:
                self.report.realtime_price = rt_future.result()
            except Exception as e:
                self._log(f"  实时行情获取异常: {e}")
                self.report.realtime_price = None

        rt = self.report.realtime_price
        if rt and "error" in rt:
            self._log(f"  实时行情: {rt['error']}")
        elif rt:
            self._log(f"  实时行情: 最新价 {rt['price']:.2f} 元, 涨跌幅 {rt['change_pct']:+.2f}%")

        self.report.company_name = self.data.company_name
        self.report.stock_code = stock_code
        self.report.industry = self.data.industry
        self.report.analysis_period = f"{min(self.data.years)}-{max(self.data.years)}" if self.data.years else "N/A"
        self.report.data_source = "manual" if manual_data else self.data_source
        self.report.report_date = datetime.now().strftime("%Y-%m-%d")

        # 允许 CLI --industry 覆盖自动检测的行业
        if industry:
            self.data.industry = industry
            self.report.industry = industry

        self._log(f"  获取到 {len(self.data.years)} 年数据: {self.data.years}, 行业: {self.data.industry}")

        # ---- STEP 2: 指标计算 ----
        self._log("STEP 2: 衍生指标计算")
        self.indicators = IndicatorCalculator(self.data)
        self.indicators.compute_all()
        self._log(f"  计算完成 {len(self.indicators.indicators)} 项衍生指标")

        # ---- STEP 3: 三级失真扫描 ----
        self._log("STEP 3: 三级失真扫描 (10级造假 + 8级操纵 + 5级盈余管理)")
        self.rule_engine = RuleEngine(self.data, self.indicators)
        all_rule_results = self.rule_engine.run_all()

        from config import RISK_LEVEL_10, RISK_LEVEL_8, RISK_LEVEL_5, RISK_LEVEL_3
        self.report.fraud_results = [r for r in all_rule_results if r.risk_level == RISK_LEVEL_10]
        self.report.manipulation_results = [r for r in all_rule_results if r.risk_level == RISK_LEVEL_8]
        self.report.earnings_mgmt_results = [r for r in all_rule_results if r.risk_level == RISK_LEVEL_5]
        self.report.business_logic_results = [r for r in all_rule_results if r.risk_level == RISK_LEVEL_3]

        self._log(f"  造假扫描(10级): FAIL={self._cv(self.report.fraud_results, 'FAIL')}, "
                  f"WARN={self._cv(self.report.fraud_results, 'WARN')}, "
                  f"PASS={self._cv(self.report.fraud_results, 'PASS')}")
        self._log(f"  操纵扫描(8级): FAIL={self._cv(self.report.manipulation_results, 'FAIL')}, "
                  f"WARN={self._cv(self.report.manipulation_results, 'WARN')}")
        self._log(f"  盈余管理(5级): FAIL={self._cv(self.report.earnings_mgmt_results, 'FAIL')}, "
                  f"WARN={self._cv(self.report.earnings_mgmt_results, 'WARN')}")

        # ---- STEP 4: 会计勾稽验证 [NEW] ----
        self._log("STEP 4: 会计勾稽验证 (10项勾稽规则)")
        self.crosschecker = AccountingCrossChecker(self.data, self.indicators)
        self.report.crosscheck_results = self.crosschecker.run_all()
        cc_summary = self.crosschecker.summary()
        self._log(f"  勾稽验证: FAIL={cc_summary['fail_count']}, WARN={cc_summary['warn_count']}, "
                  f"得分={cc_summary['total_score']}")
        for item in cc_summary.get("fail_items", []):
            self._log(f"    [FAIL] {item[0]}: {item[1][:60]}...")

        # ---- STEP 5: 商业逻辑验证 [NEW] ----
        self._log("STEP 5: 商业逻辑验证 (16项商业逻辑规则)")
        self.biz_verifier = BusinessLogicVerifier(self.data, self.indicators)
        self.report.biz_logic_verify_results = self.biz_verifier.run_all()
        bl_summary = self.biz_verifier.summary()
        self._log(f"  商业逻辑: FAIL={bl_summary['fail_count']}, WARN={bl_summary['warn_count']}, "
                  f"RED_FLAG={len(bl_summary.get('red_flags', []))}, 得分={bl_summary['total_score']}")
        for item in bl_summary.get("red_flags", []):
            self._log(f"    [RED_FLAG] {item[0]}: {item[1][:60]}...")

        # ---- STEP 6: 深度核查 + 数字化风险 ----
        self._log("STEP 6: 重点科目深度核查")
        self.report.deep_audit_results = self.rule_engine.deep_audit_results
        self.report.digital_era_alerts = self.rule_engine.digital_era_alerts

        # ---- STEP 7: LLM增强分析 [NEW] ----
        self._log(f"STEP 7: LLM增强分析 (backend={self.llm_backend})")
        llm_analyzer = LLMAnalyzer(
            backend=self.llm_backend,
            model=self.llm_model,
            api_key=self.llm_api_key,
            api_base=self.llm_api_base,
        )
        self.report.llm_analysis = llm_analyzer.analyze(
            self.data, self.indicators,
            all_rule_results, self.report.crosscheck_results, self.report.biz_logic_verify_results,
        )
        if self.report.llm_analysis.success:
            self._log(f"  模型: {self.report.llm_analysis.model_used}")
            self._log(f"  造假概率: {self.report.llm_analysis.fraud_probability}")

        # ---- STEP 8: 综合评分 [UPGRADED] ----
        self._log("STEP 8: 三维度综合评分")
        scorer = RiskScorer()
        self.report.risk_profile = scorer.compute(
            rule_results=all_rule_results,
            crosscheck_results=self.report.crosscheck_results,
            biz_logic_results=self.report.biz_logic_verify_results,
            llm_result=self.report.llm_analysis,
        )
        rp = self.report.risk_profile
        self._log(f"  原始得分: {rp.total_score}/200 → 归一化: {rp.normalized_score}/100")
        self._log(f"  风险等级: {rp.risk_band} — {rp.risk_band_cn}")
        self._log(f"  三维度: 规则扫描{rp.level_10_count+rp.level_8_count+rp.level_5_count}项 + "
                  f"勾稽{rp.crosscheck_fail_count}项 + 商业逻辑{rp.bizlogic_fail_count}项(含{rp.red_flag_count}个红色警报)")
        self._log(f"  置信度: {rp.confidence*100:.0f}% ({rp.confidence_reason})")

        # ---- 完成 ----
        self.report.execution_time_seconds = round(time.time() - start_time, 2)
        self._log(f"分析完成，总耗时 {self.report.execution_time_seconds}s")
        return self.report

    def _cv(self, results, verdict):
        return sum(1 for r in results if r.verdict == verdict)

    def _log(self, msg: str):
        self.report.execution_steps.append(msg)
        print(f"[Agent] {msg}")


def analyze_stock(stock_code: str, source: str = "akshare",
                  tushare_token: str = "", years: int = 5,
                  llm: str = "heuristic") -> AnalysisReport:
    """快捷分析函数"""
    agent = FraudDetectionAgent(data_source=source, tushare_token=tushare_token, llm_backend=llm)
    return agent.analyze(stock_code, years=years)
