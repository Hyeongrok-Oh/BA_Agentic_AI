# Database Schema Definition for LLM Prompt
# Derived from google_10/sqllite/schema_normalized_3nf.sql and README_DATABASE.md

DB_SCHEMA_PROMPT = """
You have access to a SQLite database 'lge_he_erp.db' which simulates the ERP system of LG Electronics Home Entertainment (HE) Division.
The database follows a 3NF/Snowflake normalized schema similar to SAP ERP.

## Database Schema

### 1. Master Data Tables

**TBL_MD_PRODUCT** (Product Catalog)
- PRODUCT_ID (PK): Unique product identifier (e.g., OLED65G4PUA)
- MODEL_NAME: Display name (e.g., "65" OLED evo G4")
- SERIES: Product series (G4, C4, B4, QNED80, UQ75)
- PANEL_TYPE: Technology (OLED, QNED, LCD)
- SCREEN_SIZE: Size in inches (55, 65, 75, 77, 83)
- LAUNCH_YEAR: Year introduced (2023, 2024, 2025)
- MFG_PLANT: Manufacturing location (MEX, POL, IDN)

**TBL_ORG_SUBSIDIARY** (Sales Subsidiaries)
- SUBSIDIARY_ID (PK): Subsidiary code (LGEUS, LGEKR, LGEUK, etc.)
- REGION: Geographic region (NA, KR, EU)
- CURRENCY: Local currency (USD, KRW, GBP, EUR, CAD)

**TBL_ORG_CUSTOMER** (Customers/Retailers)
- CUSTOMER_ID (PK): Unique customer identifier
- CUST_NAME: Customer name (Best Buy, Costco, Amazon, etc.)
- SUBSIDIARY_ID (FK): Managing subsidiary
- CHANNEL_TYPE: Channel (B2B, RETAIL, ONLINE)

### 2. Transaction Tables (Sales Order Process)

**TBL_TX_SALES_HEADER** (Sales Order Header)
- ORDER_NO (PK): Unique order number (SO-YYYY-NNNNNN)
- DOC_DATE: Transaction date (YYYY-MM-DD)
- CUSTOMER_ID (FK): Customer reference
- SUBSIDIARY_ID: Selling subsidiary
- TOTAL_NET_VALUE: Total order value
- CURRENCY: Transaction currency

**TBL_TX_SALES_ITEM** (Sales Order Line Items)
- ORDER_NO (FK): Reference to header
- ITEM_NO: Line item number (10, 20, 30...)
- PRODUCT_ID (FK): Product reference
- ORDER_QTY: Quantity ordered
- NET_VALUE: Line item total value

### 3. Pricing & Cost Details (Critical for Profitability Analysis)

**TBL_TX_PRICE_CONDITION** (Revenue Components - The "Why" behind Revenue)
*Stores each price component as a separate row (SAP style).*
- ORDER_NO, ITEM_NO (FK): Reference to line item
- COND_TYPE: Condition type code
  - 'PR00': Base list price (+)
  - 'K007': Volume discount (-)
  - 'ZPRO': Price protection / Price down compensation (-)
  - 'ZMDF': Marketing Development Fund (MDF, 판매장려금) (-)
- COND_VALUE: Amount (positive or negative)
- CURRENCY: Currency code

**TBL_TX_COST_DETAIL** (Cost Components - The "Why" behind Profit)
*Granular cost breakdown.*
- ORDER_NO, ITEM_NO (FK): Reference to line item
- COST_TYPE: Cost category
  - 'MAT': Material cost (panel + IC)
  - 'LOG': Logistics cost (freight, shipping)
  - 'TAR': Tariff and customs duties
  - 'OH': Manufacturing overhead
- COST_AMOUNT: Cost per unit
- CURRENCY: Currency code

## Key Relationships for Analysis
1. **Profitability**: Revenue (from Price Conditions) - Total Cost (from Cost Details)
2. **Revenue**: Sum of COND_VALUE in TBL_TX_PRICE_CONDITION
3. **Total Cost**: Sum of COST_AMOUNT * ORDER_QTY in TBL_TX_COST_DETAIL

## Embedded Business Context (Crisis Scenario)
- **Period**: Q4 2024 (Oct-Dec)
- **Region**: North America (LGEUS, LGECA)
- **Product**: Large OLED TVs (65"+)
- **Issue**: Profit margin collapse due to:
  1. **Logistics Cost Spike**: 'LOG' cost increased 4.5x (Red Sea crisis).
  2. **Price Protection Surge**: 'ZPRO' frequency and amount increased significantly.

## Supported Analysis Capabilities
1. **Derived Metrics**: YoY (Year-over-Year), MoM (Month-over-Month), Growth Rate, Average Price (ASP).
2. **Aggregations**: Total Revenue, Total Profit, Profit Margin (%), Sales Mix.
3. **Rankings**: Top selling products, Best performing regions.
4. **Specific Cost/Price Components**: You CAN analyze specific components like 'Logistics Cost' (LOG), 'Marketing Fund' (ZMDF), 'Price Protection' (ZPRO).

## Data Availability Period
- **Available Date Range**: 2023-01-01 to 2024-12-31
- Data outside this range (e.g., 1990, 2025 forecasts) is NOT available.
- If user requests data outside this range, respond with clarifying_question asking for a valid period.

## Defined Reports (Standard Reporting Templates)
The following are the ONLY defined report titles. If a user asks for these exact titles (or very similar), classify as 'Defined Report'.
1. **분기 실적 보고서** (Quarterly Performance Report)
2. **반기 실적 보고서** (Half-yearly Performance Report)
3. **연간 사업 계획서** (Annual Business Plan)
4. **수익성 분석 보고서** (Profitability Analysis Report)
"""

# Available data information for UI components
AVAILABLE_DATA_INFO = {
    "date_range": {
        "start": "2023-01-01",
        "end": "2024-12-31",
        "display": "2023년 ~ 2024년"
    },
    "company": "LG전자 HE사업부",
    "regions": ["북미 (North America)", "유럽 (Europe)", "한국 (Korea)"],
    "products": ["OLED TV", "QNED TV", "LCD TV"],
    "metrics": {
        "sales": ["매출액", "판매량", "평균판매가(ASP)"],
        "profit": ["영업이익", "영업이익률", "원가"],
        "cost_detail": ["물류비", "관세", "가격보호", "판촉비(MDF)"]
    },
    "service_description": """
이 서비스는 **LG전자 HE(Home Entertainment) 사업부**의 데이터 분석 에이전트입니다.

**제공 기능:**
1. 📊 **보고서 생성**: 수익성 분석, 전략 분석, 리스크 분석 등 종합 보고서
2. 📈 **데이터 QA**: 매출, 판매량, 영업이익 등 특정 지표 조회
""",
    "sample_questions": {
        "Report Generation": [
            "2024년 3분기 북미 OLED TV 수익성 분석 보고서 만들어줘",
            "2024년 하반기 물류비 증가 원인 분석해줘"
        ],
        "Data QA": [
            "2024년 3분기 북미 매출액 알려줘",
            "Best Buy 대상 2024년 거래액은 얼마야?"
        ]
    }
}
