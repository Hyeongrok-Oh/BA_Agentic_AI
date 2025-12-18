# Executive Summary - LG HE ERP Database

## Project Overview

**Objective**: Create a sophisticated SQLite database with realistic synthetic data to demonstrate AI Agent capabilities in diagnosing complex profitability issues in a normalized ERP environment.

**Database**: `lge_he_erp.db`
**Schema Type**: 3NF/Snowflake (SAP ERP-style)
**Period**: 2023-01-01 to 2025-12-31 (3 years, 1,096 days)
**Business Context**: LG Electronics Home Entertainment Division sales and profitability data

## Database Specifications

### Data Volume
| Component | Count | Description |
|-----------|-------|-------------|
| Products | 12 | LG TV models (OLED, QNED, LCD) |
| Subsidiaries | 6 | Regional organizations (NA, EU, Asia) |
| Customers | 13 | Major retailers (Best Buy, Amazon, Costco, etc.) |
| Sales Orders | 10,269 | Daily transactions over 3 years |
| Line Items | 10,269 | Order line items |
| Price Conditions | 23,797 | Pricing components (base, discount, protection) |
| Cost Details | 41,076 | Cost breakdown (material, logistics, tariff, overhead) |

### Schema Architecture

**Traditional ERP Design (3NF)**:
```
Master Data Layer:
├── TBL_MD_PRODUCT          (Product catalog)
├── TBL_ORG_SUBSIDIARY      (Regional organizations)
└── TBL_ORG_CUSTOMER        (Retailers)

Transaction Layer:
├── TBL_TX_SALES_HEADER     (Order headers)
├── TBL_TX_SALES_ITEM       (Line items)
├── TBL_TX_PRICE_CONDITION  (Revenue components)
└── TBL_TX_COST_DETAIL      (Cost components)
```

**Key Design Principle**: Revenue and costs are not stored as flat columns, but as **normalized rows** with type codes (like SAP). This forces AI agents to perform complex JOINs and aggregations.

## Embedded Business Scenario: Q4 2024 Profitability Crisis

### Crisis Summary

**WHEN**: October - December 2024 (Q4)
**WHERE**: North America only (LGEUS, LGECA subsidiaries)
**WHAT**: Large OLED TVs (65 inches and above)

### Root Causes (Implemented in Data)

#### 1. Logistics Cost Spike (350% increase)
Simulates Red Sea shipping crisis + emergency air freight

| Product | Normal Logistics | Crisis Logistics | Increase |
|---------|-----------------|------------------|----------|
| 65" OLED | $80/unit | $360/unit | 350% |
| 77" OLED | $120/unit | $540/unit | 350% |
| 83" OLED | $120/unit | $540/unit | 350% |

**Implementation**: `COST_TYPE = 'LOG'` multiplied by 4.5x for affected segment

#### 2. Price Protection Surge
Simulates aggressive retailer compensation for mid-season price drops

| Period | Frequency | Average Amount |
|--------|-----------|----------------|
| Normal | 10% of orders | -$30 to -$80/unit |
| Crisis | 75% of orders | -$150 to -$350/unit |

**Implementation**: `COND_TYPE = 'ZPRO'` frequency and amount increased for affected segment

### Financial Impact

#### Quarterly Comparison (LGEUS)
```
Quarter    Orders  Units   Revenue      Op. Profit   Margin %   Status
Q1 2024    302     1,636   $3,038,194   $839,808     27.6%      ✓ Normal
Q2 2024    328     1,924   $3,386,022   $933,560     27.6%      ✓ Normal
Q3 2024    250     1,472   $2,666,152   $724,150     27.2%      ✓ Normal
Q4 2024    495     4,552   $8,682,727   $964,676     11.1%      ⚠ CRISIS
Q1 2025    275     1,536   $2,683,303   $741,956     27.6%      ✓ Recovery
```

**Key Observations**:
1. **Revenue Paradox**: Q4 revenue is 3.3x higher than Q3 (pushing volume to meet targets)
2. **Margin Collapse**: Profit margin dropped from 27% to 11% (-59% decline)
3. **Volume vs. Value**: Despite 3x more units sold, profit only increased 33%
4. **Isolation**: Only North America affected; EU and Asia maintain normal margins
5. **Recovery**: Q1 2025 shows immediate return to normal profitability

#### Monthly Trend (LGEUS 2024)
```
Month       Revenue      Logistics Cost   Operating Profit   Margin %
2024-01     $882,093     $40,380          $243,670          27.6%
2024-02     $1,042,615   $44,880          $292,450          28.1%
2024-03     $1,107,487   $49,100          $304,008          27.5%
...         ...          ...              ...               ...
2024-09     $1,056,499   $47,600          $287,521          27.2%
2024-10     $2,056,057   $355,220         $206,050          10.0%  ⚠
2024-11     $3,517,676   $588,020         $403,563          11.5%  ⚠
2024-12     $3,108,994   $508,460         $355,063          11.4%  ⚠
2025-01     $777,392     $36,680          $212,534          27.3%
```

**Logistics Cost Spike Evidence**:
- September 2024: $47,600 total logistics cost
- October 2024: $355,220 total logistics cost (7.5x increase)
- January 2025: $36,680 total logistics cost (back to normal)

## AI Agent Testing Framework

### Diagnostic Questions (Increasing Difficulty)

#### Level 1: Pattern Recognition
**Question**: "What was the profitability trend for North America in 2024?"

**Expected Output**:
- Identify Q4 as anomaly period
- Note margin compression from ~27% to ~11%
- Recognize revenue increase but disproportionate profit

#### Level 2: Component Analysis
**Question**: "Why did profit margin drop in Q4 2024?"

**Expected Output**:
- Analyze revenue components (TBL_TX_PRICE_CONDITION)
- Analyze cost components (TBL_TX_COST_DETAIL)
- Identify both logistics cost spike and price protection surge
- Quantify contribution of each factor

#### Level 3: Segmentation
**Question**: "Which products and regions were affected by the Q4 2024 crisis?"

**Expected Output**:
- Region: North America only (LGEUS, LGECA)
- Products: OLED 65"+ only (not QNED or LCD)
- Evidence: Compare margins across segments

#### Level 4: Root Cause Diagnosis
**Question**: "What specific cost drivers caused the Q4 2024 margin compression for large OLED TVs?"

**Expected Output**:
- Logistics cost (LOG) increased 350% for OLED 65"+
- Price protection (ZPRO) frequency increased from 10% to 75%
- Quantify per-unit impact
- Calculate total financial impact

#### Level 5: Business Interpretation
**Question**: "Explain the business context behind the Q4 2024 profitability crisis"

**Expected Output**:
- Logistics: Likely supply chain disruption (Red Sea crisis, air freight)
- Price protection: Competitive pricing pressure during holiday season
- Strategic trade-off: Volume targets vs. margin preservation
- Recovery plan: Q1 2025 shows return to normal operations

## Sample Data Example

### Crisis Order Analysis: SO-2024-005646

**Order Details**:
- Date: 2024-10-01 (First day of crisis)
- Product: OLED65C4PUA (65" OLED Premium)
- Customer: LGEUS (North America)
- Quantity: 11 units

**Revenue Breakdown** (per unit):
```
PR00 (Base Price):          +$1,999.00
K007 (Volume Discount):       -$111.18  (5.6%)
ZPRO (Price Protection):      -$204.12  (10.2%)  ← Crisis
───────────────────────────────────────
Net Revenue:                 $1,683.70
```

**Cost Breakdown** (per unit):
```
MAT (Material Cost):         $1,099.45  (54.9%)
LOG (Logistics):               $360.00  (18.0%)  ← Crisis (normal: $80)
TAR (Tariff):                   $49.98  (2.5%)
OH  (Overhead):                $100.00  (5.0%)
───────────────────────────────────────
Total Cost:                  $1,609.43
```

**Margin Analysis**:
```
Net Revenue:                 $1,683.70
Total Cost:                  $1,609.43
Operating Profit:               $74.27
Margin %:                         4.4%   ← Normal: 27%
```

## Key Features for AI Agents

### 1. Complex Joins Required
No single table contains complete information. Agents must JOIN 4-6 tables for profitability analysis.

### 2. Condition-Based Logic
- Revenue = SUM of multiple COND_TYPE rows
- Cost = SUM of multiple COST_TYPE rows
- Agents must understand ERP data modeling

### 3. Time-Series Analysis
- Daily granularity
- Seasonal patterns
- Anomaly detection required

### 4. Multi-Dimensional Segmentation
- Product (OLED vs. QNED vs. LCD)
- Region (NA vs. EU vs. Asia)
- Size (55" vs. 65" vs. 77" vs. 83")
- Time period (Q1-Q4, monthly)

### 5. Root Cause Attribution
- Not just "what happened" but "why"
- Quantify contribution of each factor
- Business context interpretation

## Files Delivered

| File | Purpose |
|------|---------|
| `lge_he_erp.db` | Main database file (6.6 MB) |
| `generate_data.py` | Python script to regenerate database (23 KB) |
| `schema_normalized_3nf.sql` | DDL schema definition (8.7 KB) |
| `README_DATABASE.md` | Comprehensive documentation (11 KB) |
| `QUICK_START.md` | Quick reference guide (8.4 KB) |
| `EXECUTIVE_SUMMARY.md` | This file - executive overview |

## Usage

### Regenerate Database
```bash
python generate_data.py
```

### Query Database
```bash
sqlite3 lge_he_erp.db
```

### Sample Query (Monthly Profit)
```sql
SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    SUM(i.NET_VALUE) as REVENUE,
    SUM(c.COST_AMOUNT * i.ORDER_QTY) as TOTAL_COST,
    SUM(i.NET_VALUE) - SUM(c.COST_AMOUNT * i.ORDER_QTY) as PROFIT
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO
    AND i.ITEM_NO = c.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
GROUP BY strftime('%Y-%m', h.DOC_DATE);
```

## Success Criteria

An AI agent successfully demonstrates profitability diagnosis capability if it can:

1. ✓ Identify the Q4 2024 anomaly period without being told
2. ✓ Recognize that revenue increased while margin decreased
3. ✓ JOIN multiple normalized tables to calculate profit
4. ✓ Segment analysis by product, region, and time
5. ✓ Identify logistics cost (LOG) as primary cost driver
6. ✓ Identify price protection (ZPRO) as secondary revenue driver
7. ✓ Quantify the 350% logistics cost increase
8. ✓ Explain the business context (supply chain + competitive pricing)
9. ✓ Compare crisis period to normal periods
10. ✓ Validate recovery in Q1 2025

## Conclusion

This database provides a realistic, complex environment for testing AI agents' ability to:
- Navigate normalized (3NF) ERP schemas
- Perform multi-table JOINs and aggregations
- Detect time-series anomalies
- Conduct root cause analysis
- Interpret business context
- Communicate findings clearly

The embedded Q4 2024 crisis is **not obvious** from simple queries. Only through systematic analysis of cost components, pricing conditions, product segments, and regional patterns can an agent successfully diagnose the true drivers of margin compression.

---

**Generated**: 2025-11-29
**Database Version**: 1.0
**Schema**: 3NF/Snowflake (7 tables)
**Record Count**: 84,425 total records
**File Size**: 6.6 MB
