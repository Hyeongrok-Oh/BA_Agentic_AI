from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
from enum import Enum

class MetricEnum(str, Enum):
    # Revenue Metrics
    REVENUE = "Revenue"
    REVENUE_GROWTH_RATE = "Revenue Growth Rate"
    REVENUE_TREND = "Revenue Trend"
    MONTHLY_REVENUE_TREND = "Monthly Revenue Trend"
    MONTHLY_REVENUE = "Monthly Revenue"
    REVENUE_SHARE = "Revenue Share"
    REVENUE_COMPARISON = "Revenue Comparison"
    REVENUE_TARGET_ACHIEVEMENT = "Revenue Target Achievement"
    REVENUE_YOY = "Revenue YoY"
    REVENUE_BREAKDOWN = "Revenue Breakdown"
    REVENUE_GROWTH_COMPARISON = "Revenue Growth Comparison"
    REVENUE_GROWTH_FORECAST = "Revenue Growth Forecast"
    AVERAGE_DAILY_REVENUE = "Average Daily Revenue"
    TRANSACTION_VALUE = "Transaction Value"
    REVENUE_SHARE_BESTBUY = "Revenue Share (Best Buy)"
    
    # Sales Quantity Metrics
    SALES_QUANTITY = "Sales Quantity"
    SALES_QUANTITY_GROWTH_RATE = "Sales Quantity Growth Rate"
    SALES_QUANTITY_TREND = "Sales Quantity Trend"
    SALES_QUANTITY_COMPARISON = "Sales Quantity Comparison"
    QUARTERLY_SALES_QUANTITY = "Quarterly Sales Quantity"
    
    # Profit Metrics
    OPERATING_PROFIT = "Operating Profit"
    OPERATING_MARGIN = "Operating Margin"
    OPERATING_PROFIT_GROWTH_RATE = "Operating Profit Growth Rate"
    OPERATING_PROFIT_TREND = "Operating Profit Trend"
    OPERATING_PROFIT_TARGET_ACHIEVEMENT = "Operating Profit Target Achievement"
    OPERATING_PROFIT_YOY = "Operating Profit YoY"
    OPERATING_PROFIT_RANKING = "Operating Profit Ranking"
    OPERATING_MARGIN_YOY = "Operating Margin YoY"
    
    # Cost Metrics
    TOTAL_COST = "Total Cost"
    COST_RATIO = "Cost Ratio"
    LOGISTICS_COST = "Logistics Cost"
    MATERIAL_COST = "Material Cost"
    TARIFF = "Tariff"
    PRICE_PROTECTION = "Price Protection"
    MDF = "MDF"
    UNIT_COST = "Unit Cost"
    
    # Channel / Market Metrics
    CHANNEL_REVENUE_SHARE = "Channel Revenue Share"
    CHANNEL_PERFORMANCE_ANALYSIS = "Channel Performance Analysis"
    MARKET_SHARE = "Market Share"
    MARKET_SHARE_COMPARISON = "Market Share Comparison"
    MARKET_RANKING = "Market Ranking"
    COMBINED_MARKET_SHARE = "Combined Market Share"
    
    # Analysis & Reports
    PROFITABILITY_ANALYSIS_REPORT = "Profitability Analysis Report"
    PROFITABILITY_IMPACT = "Profitability Impact"
    PROFITABILITY_COMPARISON_REPORT = "Profitability Comparison Report"
    PROFITABILITY_ROOT_CAUSE_ANALYSIS = "Profitability Root Cause Analysis"
    PROFITABILITY_CRISIS_DIAGNOSIS = "Profitability Crisis Diagnosis"
    PERFORMANCE_ANALYSIS_REPORT = "Performance Analysis Report"
    TARGET_ACHIEVEMENT_ANALYSIS = "Target Achievement Analysis"
    SALES_TREND_ANALYSIS = "Sales Trend Analysis"
    CUSTOMER_PERFORMANCE_COMPARISON = "Customer Performance Comparison"
    MARKET_POSITION_ANALYSIS = "Market Position Analysis"
    SIZE_MIX_TREND_ANALYSIS = "Size Mix Trend Analysis"
    PORTFOLIO_STRATEGY_ANALYSIS = "Portfolio Strategy Analysis"
    MARKET_SHARE_STRATEGY_ANALYSIS = "Market Share Strategy Analysis"
    PRICE_ELASTICITY_ANALYSIS = "Price Elasticity Analysis"
    PARTNERSHIP_STRATEGY_ANALYSIS = "Partnership Strategy Analysis"
    MARKET_ENVIRONMENT_IMPACT_ANALYSIS = "Market Environment Impact Analysis"
    SKU_CONTRIBUTION_ANALYSIS = "SKU Contribution Analysis"
    MARKET_ENTRY_STRATEGY_ANALYSIS = "Market Entry Strategy Analysis"
    LOGISTICS_RISK_ANALYSIS = "Logistics Risk Analysis"
    COMPETITIVE_PRESSURE_ANALYSIS = "Competitive Pressure Analysis"
    CURRENCY_IMPACT_ANALYSIS = "Currency Impact Analysis"
    COST_INCREASE_ANALYSIS = "Cost Increase Analysis"
    ANOMALY_DETECTION_REPORT = "Anomaly Detection Report"
    OPERATING_MARGIN_COMPARISON_ANALYSIS = "Operating Margin Comparison Analysis"
    REGIONAL_MARGIN_GAP_ANALYSIS = "Regional Margin Gap Analysis"
    PRICE_DISCOUNT_IMPACT_ANALYSIS = "Price Discount Impact Analysis"
    LOGISTICS_COST_IMPACT_ANALYSIS = "Logistics Cost Impact Analysis"
    ANNUAL_REVIEW_FOR_2025_PLANNING = "Annual Review for 2025 Planning"
    
    # Other Metrics (Not in DB but requested)
    ASP = "ASP"
    NPS = "NPS"
    BRAND_AWARENESS = "Brand Awareness"
    AD_EFFECTIVENESS = "Ad Effectiveness"
    CAC = "CAC"
    DEFECT_RATE = "Defect Rate"
    RETURN_RATE = "Return Rate"
    SUPPLIER_CONTRACT_TERMS = "Supplier Contract Terms"
    HEADCOUNT = "Headcount"
    LABOR_COST = "Labor Cost"
    MARKETING_BUDGET = "Marketing Budget"
    RND_INVESTMENT = "R&D Investment"
    REGULATORY_COMPLIANCE_COST = "Regulatory Compliance Cost"
    LAUNCH_ROADMAP = "Launch Roadmap"
    PRICING_STRATEGY = "Pricing Strategy"
    FORECAST_2025 = "2025 Forecast"

class Period(BaseModel):
    year: Optional[int] = Field(None, description="Year (e.g., 2024)")
    quarter: Optional[Union[int, List[int]]] = Field(None, description="Quarter (1-4) or list of quarters for multi-quarter periods")
    month: Optional[Union[int, List[int]]] = Field(None, description="Month number (1-12) or list of months (e.g., [1, 2, 3] for Q1, [10] for October)")
    day: Optional[int] = Field(None, description="Day of the month")

class ExtractedEntities(BaseModel):
    company: Optional[str] = Field(
        None, 
        description="Company code. ONLY extract if explicitly mentioned. If user says 'LG전자', 'LG', '엘지' → 'LGE'. If NOT mentioned, set to null and ask clarifying_question."
    )
    product: Optional[Union[str, List[str]]] = Field(
        None, 
        description="Product name(s). Use single string for one product (e.g., 'OLED'), or list for multiple products (e.g., ['OLED', 'QNED'])"
    )
    region: Optional[Union[str, List[str]]] = Field(
        None, 
        description="Region name(s). Use single string for one region (e.g., 'North America'), or list for multiple regions (e.g., ['North America', 'Europe'])"
    )
    period: Optional[Period] = Field(None, description="Time period details")
    requested_metrics: Optional[List[MetricEnum]] = Field(None, description="Metrics that exist in the database taxonomy")
    unmapped_metrics: Optional[List[str]] = Field(None, description="Metrics requested by user but NOT in the taxonomy")


# ============================================================
# NEW: Hierarchical Intent Classification Enums
# ============================================================

class SubIntentEnum(str, Enum):
    """Sub-Intent 분류"""
    # ========================================
    # Report Generation Sub-Intents
    # ========================================
    DEFINED_REPORT = "Defined Report"   # 정의된 보고서 양식
    NEW_REPORT = "New Report"           # 사용자 정의 보고서 (양식 없음)
    
    # ========================================
    # Data QA Sub-Intents (Data Source Routing)
    # Based on RAG (Retrieval-Augmented Generation) Best Practices
    # ========================================
    INTERNAL_DATA = "Internal Data"     # 내부 DB에서 조회 가능 (매출, 영업이익 등)
    EXTERNAL_DATA = "External Data"     # 웹서치 API로 조회 → 내부 DB 저장 (경쟁사 정보, 시장 트렌드 등)
    HYBRID_DATA = "Hybrid Data"         # 내부 + 외부 데이터 조합 필요 (비교 분석 등)
    
    # ========================================
    # Out-of-Scope Sub-Intents
    # (Non-Business removed - handled by Guardrail Layer)
    # ========================================
    DATA_UNAVAILABLE = "Data Unavailable"  # 데이터 없음
    
    # ========================================
    # NEW: Ambiguous Sub-Intents
    # ========================================
    AMBIGUOUS_CLARIFICATION = "Ambiguous Clarification" # 의도 모호 (보고서 vs 데이터 조회)
    
    # ========================================
    # NEW: Analysis Mode (Depth of Analysis)
    # ========================================
    # This acts as the dimension for "Simple Retrieval" vs "Deep Analysis"

class AnalysisMode(str, Enum):
    """분석 깊이 (Depth) 분류"""
    DESCRIPTIVE = "Descriptive"   # 단순 조회 (Fact-based) - "What", "How much"
    DIAGNOSTIC = "Diagnostic"     # 심층 분석 (Reasoning-based) - "Why", "Cause", "Impact"


class DetailTypeEnum(str, Enum):
    """Detail Type 분류 (3단계)"""
    # ========================================
    # Defined Report Detail Types
    # ========================================
    PRE_CLOSING = "Pre-closing"         # 잠정 데이터 (현재 월/분기)
    POST_CLOSING = "Post-closing"       # 확정 데이터 (과거 기간)
    EXTERNAL_EVENT = "External Event"   # 외부 요인 (환율, 물류비, 관세, 원자재)
    
    # ========================================
    # Data Unavailable Detail Types
    # ========================================
    # 1) 필수 슬롯 누락 (조회 불가) - 회사 또는 기간 정보 없음
    REQUIRED_SLOT_MISSING = "Required Slot Missing"  # company 또는 period 누락
    
    # 2) 메트릭 미지원 - 요청한 지표가 시스템에서 지원하지 않음
    METRIC_UNAVAILABLE = "Metric Unavailable"
    
    # 3) 기간 범위 초과 - 데이터는 있지만 요청 기간에 해당 데이터 없음 (예: 1990년 요청 시)
    DATE_OUT_OF_RANGE = "Date Out of Range"


class IntentResult(BaseModel):
    thinking: str = Field(..., description="Step-by-step reasoning process for intent and entity extraction")
    intent: Literal["Report Generation", "Data QA", "Out-of-Scope", "Ambiguous"] = Field(..., description="Primary intent of the user")
    sub_intent: Optional[SubIntentEnum] = Field(None, description="Sub-intent classification (Data Source routing)")
    analysis_mode: AnalysisMode = Field(..., description="Analysis Depth: 'Descriptive' for simple retrieval, 'Diagnostic' for causal analysis/reasoning.")
    detail_type: Optional[DetailTypeEnum] = Field(None, description="Detail type for 3-level classification")
    context_continuity: Literal["continue", "partial_change", "new_topic"] = Field(..., description="Context continuity status")
    extracted_entities: Optional[ExtractedEntities] = Field(None, description="Entities extracted from the query")
    changed_entities: Optional[List[str]] = Field(None, description="List of entity types that changed from previous turn")
    clarifying_question: Optional[str] = Field(None, description="Question to ask user if essential info is missing")
    response_message: Optional[str] = Field(None, description="Contextual response message for the user")
    recommended_questions: Optional[List[str]] = Field(None, description="List of recommended follow-up questions based on context")


