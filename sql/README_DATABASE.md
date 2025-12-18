# LG Electronics HE Division - ERP Database Documentation

## Overview
This is a sophisticated SQLite database (`lge_he_erp.db`) containing realistic synthetic data for LG Electronics Home Entertainment Division. The database follows a **3NF/Snowflake normalized schema** similar to SAP ERP systems, designed to demonstrate AI Agent capabilities in diagnosing complex profitability issues.

## Database Specifications

- **Period**: 2023-01-01 to 2025-12-31 (3 years)
- **Total Orders**: ~10,000+ sales orders
- **Schema Type**: 3NF Normalized (Snowflake)
- **Transaction Granularity**: Daily
- **Database File**: `lge_he_erp.db`

## Schema Structure

### Master Data Tables

#### 1. TBL_MD_PRODUCT
Product catalog with 12 LG TV models across different tiers:
- **OLED Flagship**: G4 Series (65", 77", 83")
- **OLED Premium**: C4 Series (55", 65", 77")
- **OLED Entry**: B4 Series (55", 65")
- **QNED Premium**: QNED80 Series (65", 75")
- **LCD Entry**: UQ75 Series (55", 65")

Columns:
- `PRODUCT_ID`: Unique product identifier (e.g., OLED65G4PUA)
- `MODEL_NAME`: Display name
- `SERIES`: Product series (G4, C4, B4, QNED80, UQ75)
- `PANEL_TYPE`: Technology (OLED, QNED, LCD)
- `SCREEN_SIZE`: Size in inches (55, 65, 75, 77, 83)
- `LAUNCH_YEAR`: Year introduced
- `MFG_PLANT`: Manufacturing location (MEX, POL, IDN)

#### 2. TBL_ORG_SUBSIDIARY
6 regional subsidiaries:
- **North America**: LGEUS (USD), LGECA (CAD)
- **Europe**: LGEUK (GBP), LGEDE (EUR), LGEFR (EUR)
- **Asia**: LGEKR (KRW)

Columns:
- `SUBSIDIARY_ID`: Subsidiary code
- `REGION`: Geographic region (NA, EU, KR)
- `CURRENCY`: Local currency

#### 3. TBL_ORG_CUSTOMER
13 major retailers across regions:
- **US**: Best Buy, Costco, Amazon, Target, Walmart
- **Canada**: Best Buy Canada
- **Korea**: Himart, Electro Mart
- **Europe**: John Lewis, Currys, MediaMarkt, Saturn, Fnac

Columns:
- `CUSTOMER_ID`: Unique customer identifier
- `CUST_NAME`: Customer/retailer name
- `SUBSIDIARY_ID`: Associated subsidiary
- `CHANNEL_TYPE`: Channel (RETAIL, ONLINE, B2B)

### Transaction Tables (SAP ERP Style)

#### 4. TBL_TX_SALES_HEADER
Sales order header (document level):
- `ORDER_NO`: Unique order number (SO-YYYY-NNNNNN)
- `DOC_DATE`: Transaction date
- `CUSTOMER_ID`: Reference to customer
- `SUBSIDIARY_ID`: Selling subsidiary
- `TOTAL_NET_VALUE`: Total order value
- `CURRENCY`: Transaction currency

#### 5. TBL_TX_SALES_ITEM
Sales order line items:
- `ORDER_NO`: Reference to header
- `ITEM_NO`: Line item number (10, 20, 30...)
- `PRODUCT_ID`: Product sold
- `ORDER_QTY`: Quantity ordered
- `NET_VALUE`: Line item total value

#### 6. TBL_TX_PRICE_CONDITION
Pricing components (the "why" behind revenue):
This is **critical** for root cause analysis. Instead of flat columns, each price component is a separate row.

Condition Types:
- **PR00**: Base list price (positive)
- **K007**: Volume discount (negative, 5-15%)
- **ZPRO**: Price protection (negative, crisis-driven)
- **ZMDF**: Marketing development fund (negative, occasional)

Columns:
- `ORDER_NO, ITEM_NO`: Reference to line item
- `COND_TYPE`: Condition type code
- `COND_VALUE`: Amount (positive or negative)
- `CURRENCY`: Currency code

#### 7. TBL_TX_COST_DETAIL
Cost components (the "why" behind profit):
Granular cost breakdown for margin analysis.

Cost Types:
- **MAT**: Material cost (panel + IC components)
- **LOG**: Logistics cost (freight, shipping)
- **TAR**: Tariff and customs duties
- **OH**: Manufacturing overhead

Columns:
- `ORDER_NO, ITEM_NO`: Reference to line item
- `COST_TYPE`: Cost category
- `COST_AMOUNT`: Cost per unit
- `CURRENCY`: Currency code

## Embedded Business Crisis: Q4 2024

### Crisis Specification

**Period**: October 2024 - December 2024 (Q4)

**Affected Segment**:
- **Region**: North America only (LGEUS, LGECA)
- **Products**: Large OLED TVs (65 inches and above)
  - OLED65G4PUA, OLED77G4PUA, OLED83G4PUA
  - OLED65C4PUA, OLED77C4PUA
  - OLED65B4PUA

### Crisis Triggers Implemented

#### 1. Logistics Cost Spike (350% increase)
**Root Cause**: Red Sea shipping crisis + shift to air freight

Normal Logistics Costs:
- 65" OLED: $80/unit
- 77" OLED: $120/unit
- 83" OLED: $120/unit

Crisis Logistics Costs (Q4 2024):
- 65" OLED: $360/unit (4.5x multiplier)
- 77" OLED: $540/unit (4.5x multiplier)
- 83" OLED: $540/unit (4.5x multiplier)

#### 2. Price Protection Surge
**Root Cause**: Aggressive retail compensation for mid-season price drops

Normal Period:
- Frequency: 10% of orders
- Amount: -$30 to -$80 per unit

Crisis Period (Q4 2024):
- Frequency: 75% of orders
- Amount: -$150 to -$350 per unit

### Observable Impact

#### Profit Margin Collapse
```
                   Revenue    Operating Profit    Margin %
Q3 2024 Average    $888,717   $241,383           27.2%
Q4 2024 Average    $2,894,242 $321,559           11.3%   <== CRISIS
Q1 2025 Average    $894,434   $247,384           27.6%   <== RECOVERY
```

#### Key Observations
1. **Revenue paradox**: Q4 2024 revenue is 3.3x higher than Q3 (pushing volume)
2. **Margin crash**: Profit margin dropped from 27% to 11% (58% decline)
3. **Absolute profit**: Despite higher revenue, profit only increased marginally
4. **Regional isolation**: Only North America affected; EU and Asia remain normal
5. **Product specificity**: Only large OLED TVs affected; QNED and LCD unaffected

## Usage Examples

### Example 1: Identify Crisis Period
```sql
SELECT
    strftime('%Y-%m', DOC_DATE) as MONTH,
    COUNT(*) as ORDERS,
    SUM(TOTAL_NET_VALUE) as REVENUE
FROM TBL_TX_SALES_HEADER
WHERE SUBSIDIARY_ID = 'LGEUS'
GROUP BY strftime('%Y-%m', DOC_DATE)
ORDER BY MONTH;
```

### Example 2: Analyze Logistics Cost Spike
```sql
SELECT
    h.DOC_DATE,
    p.PANEL_TYPE,
    p.SCREEN_SIZE,
    c.COST_TYPE,
    AVG(c.COST_AMOUNT) as AVG_COST
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_MD_PRODUCT p ON i.PRODUCT_ID = p.PRODUCT_ID
JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND c.COST_TYPE = 'LOG'
    AND p.PANEL_TYPE = 'OLED'
    AND p.SCREEN_SIZE >= 65
    AND h.DOC_DATE BETWEEN '2024-06-01' AND '2024-12-31'
GROUP BY strftime('%Y-%m', h.DOC_DATE), p.PANEL_TYPE, p.SCREEN_SIZE, c.COST_TYPE
ORDER BY h.DOC_DATE;
```

### Example 3: Price Protection Analysis
```sql
SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    COUNT(DISTINCT CASE WHEN pc.COND_TYPE = 'ZPRO' THEN h.ORDER_NO END) as ORDERS_WITH_PRICE_PROT,
    COUNT(DISTINCT h.ORDER_NO) as TOTAL_ORDERS,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN pc.COND_TYPE = 'ZPRO' THEN h.ORDER_NO END) /
          COUNT(DISTINCT h.ORDER_NO), 1) as PRICE_PROT_RATE_PCT,
    ROUND(AVG(CASE WHEN pc.COND_TYPE = 'ZPRO' THEN ABS(pc.COND_VALUE) END), 2) as AVG_PRICE_PROT_AMT
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
LEFT JOIN TBL_TX_PRICE_CONDITION pc ON i.ORDER_NO = pc.ORDER_NO AND i.ITEM_NO = pc.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND h.DOC_DATE BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY strftime('%Y-%m', h.DOC_DATE)
ORDER BY MONTH;
```

### Example 4: Profit Margin Decomposition
```sql
WITH item_level_profit AS (
    SELECT
        h.ORDER_NO,
        h.DOC_DATE,
        h.SUBSIDIARY_ID,
        p.PRODUCT_ID,
        p.PANEL_TYPE,
        p.SCREEN_SIZE,
        i.ORDER_QTY,
        i.NET_VALUE,
        SUM(CASE WHEN c.COST_TYPE = 'MAT' THEN c.COST_AMOUNT * i.ORDER_QTY END) as MAT_COST,
        SUM(CASE WHEN c.COST_TYPE = 'LOG' THEN c.COST_AMOUNT * i.ORDER_QTY END) as LOG_COST,
        SUM(CASE WHEN c.COST_TYPE = 'TAR' THEN c.COST_AMOUNT * i.ORDER_QTY END) as TAR_COST,
        SUM(CASE WHEN c.COST_TYPE = 'OH' THEN c.COST_AMOUNT * i.ORDER_QTY END) as OH_COST
    FROM TBL_TX_SALES_HEADER h
    JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
    JOIN TBL_MD_PRODUCT p ON i.PRODUCT_ID = p.PRODUCT_ID
    JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
    GROUP BY h.ORDER_NO, h.DOC_DATE, h.SUBSIDIARY_ID, p.PRODUCT_ID, i.ORDER_QTY, i.NET_VALUE
)
SELECT
    strftime('%Y-%m', DOC_DATE) as MONTH,
    PANEL_TYPE,
    COUNT(*) as ORDERS,
    ROUND(SUM(NET_VALUE), 0) as TOTAL_REVENUE,
    ROUND(SUM(MAT_COST), 0) as TOTAL_MAT_COST,
    ROUND(SUM(LOG_COST), 0) as TOTAL_LOG_COST,
    ROUND(SUM(MAT_COST + LOG_COST + TAR_COST + OH_COST), 0) as TOTAL_COST,
    ROUND(SUM(NET_VALUE - MAT_COST - LOG_COST - TAR_COST - OH_COST), 0) as OPERATING_PROFIT,
    ROUND(100.0 * SUM(NET_VALUE - MAT_COST - LOG_COST - TAR_COST - OH_COST) / SUM(NET_VALUE), 2) as MARGIN_PCT
FROM item_level_profit
WHERE SUBSIDIARY_ID = 'LGEUS'
    AND DOC_DATE BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY strftime('%Y-%m', DOC_DATE), PANEL_TYPE
ORDER BY MONTH, PANEL_TYPE;
```

## Data Generation Script

The `generate_data.py` script includes:

1. **Schema initialization**: Creates normalized 3NF tables with proper foreign keys
2. **Master data population**: 12 products, 6 subsidiaries, 13 customers
3. **Transaction generation**:
   - Daily sales orders (2023-2025)
   - Seasonal variation (holiday peaks, summer lows)
   - Realistic quantity and pricing patterns
4. **Crisis logic implementation**:
   - Date-based crisis detection (Q4 2024)
   - Product/region targeting (Large OLED + North America)
   - Cost multiplier application (4.5x logistics)
   - Price protection frequency surge (10% → 75%)
5. **Verification queries**:
   - Monthly profit analysis for LGEUS
   - Logistics cost spike validation
   - Crisis period highlighting

## Running the Generator

```bash
python generate_data.py
```

Expected output:
- Database file: `lge_he_erp.db`
- ~10,000 sales orders
- ~24,000 pricing conditions
- ~41,000 cost detail records
- Verification report showing Q4 2024 crisis

## Key Learning Objectives

This database demonstrates:

1. **Normalized schema complexity**: AI agents must JOIN multiple tables to answer business questions
2. **Root cause analysis**: Crisis requires analyzing both revenue (pricing conditions) and cost (cost details) separately
3. **Time-series pattern detection**: Identifying anomalies in monthly trends
4. **Multi-dimensional segmentation**: Crisis affects specific product-region combinations
5. **Condition-based pricing**: Understanding that revenue is sum of multiple condition types
6. **Cost decomposition**: Breaking down total cost into components to find the culprit

## Technical Notes

- **Foreign keys enabled**: Ensures referential integrity
- **Indexes created**: Optimizes query performance on date, product, and subsidiary fields
- **Deterministic pricing**: Base prices are consistent, not random
- **Realistic ratios**: Material costs, logistics, and margins reflect industry standards
- **Currency handling**: Each subsidiary has local currency (though amounts in USD for simplicity)

## Contact & Support

For questions about the database structure or data generation logic, refer to the inline comments in `generate_data.py`.
