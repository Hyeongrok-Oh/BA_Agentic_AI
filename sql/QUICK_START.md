# Quick Start Guide - LG HE ERP Database

## Files Overview

| File | Purpose |
|------|---------|
| `lge_he_erp.db` | SQLite database with 3 years of synthetic data (2023-2025) |
| `generate_data.py` | Python script to regenerate the database |
| `schema_normalized_3nf.sql` | DDL schema definition (3NF/Snowflake) |
| `README_DATABASE.md` | Comprehensive documentation |
| `QUICK_START.md` | This file - quick reference |

## Database Quick Stats

- **Period**: 2023-01-01 to 2025-12-31
- **Orders**: ~10,000+ sales transactions
- **Products**: 12 LG TV models (OLED, QNED, LCD)
- **Regions**: 6 subsidiaries (US, Canada, Korea, UK, Germany, France)
- **Customers**: 13 major retailers

## Schema Overview (7 Tables)

### Master Data (3 tables)
1. **TBL_MD_PRODUCT** - Product catalog (12 products)
2. **TBL_ORG_SUBSIDIARY** - Regional organizations (6 subsidiaries)
3. **TBL_ORG_CUSTOMER** - Retailers (13 customers)

### Transactions (4 tables)
4. **TBL_TX_SALES_HEADER** - Order headers (~10,000 orders)
5. **TBL_TX_SALES_ITEM** - Order line items (~10,000 items)
6. **TBL_TX_PRICE_CONDITION** - Pricing components (~24,000 rows)
7. **TBL_TX_COST_DETAIL** - Cost breakdown (~41,000 rows)

## The Crisis Story

### What Happened?
**Q4 2024 (Oct-Dec)** profitability crisis for **Large OLED TVs (65"+)** in **North America**

### Root Causes (Embedded in Data)

#### 1. Logistics Cost Spike (350% increase)
| Product Size | Normal Cost | Crisis Cost | Multiplier |
|--------------|-------------|-------------|------------|
| 65" OLED     | $80/unit    | $360/unit   | 4.5x       |
| 77" OLED     | $120/unit   | $540/unit   | 4.5x       |
| 83" OLED     | $120/unit   | $540/unit   | 4.5x       |

**Reason**: Red Sea shipping crisis + air freight usage

#### 2. Price Protection Surge
| Period | Frequency | Average Amount |
|--------|-----------|----------------|
| Normal | 10% of orders | -$30 to -$80 |
| Crisis | 75% of orders | -$150 to -$350 |

**Reason**: Aggressive retailer compensation for mid-season price drops

### Impact on Profit Margin

```
Quarter     Revenue       Op. Profit    Margin %
Q3 2024     $2,666,152    $724,150      27.2%    ← Normal
Q4 2024     $8,682,727    $964,676      11.1%    ← CRISIS
Q1 2025     $2,683,303    $741,956      27.6%    ← Recovery
```

**Key Insight**: Revenue nearly tripled in Q4, but margin collapsed by 59%

## Sample Data - Crisis Order

**Order**: SO-2024-005646 (Oct 1, 2024)
**Product**: OLED65C4PUA (65" OLED)
**Region**: LGEUS (North America)
**Quantity**: 11 units

### Pricing Breakdown (per unit)
```
PR00 (Base Price)          +$1,999.00
K007 (Volume Discount)       -$111.18
ZPRO (Price Protection)      -$204.12  ← Crisis driver
──────────────────────────────────────
Net Price per Unit         $1,683.70
```

### Cost Breakdown (per unit)
```
MAT (Material)              $1,099.45
LOG (Logistics)               $360.00  ← Crisis driver (normal: $80)
TAR (Tariff)                   $49.98
OH  (Overhead)                $100.00
──────────────────────────────────────
Total Cost per Unit         $1,609.43
```

### Margin Analysis
```
Revenue per Unit:           $1,683.70
Cost per Unit:              $1,609.43
Profit per Unit:               $74.27
Margin %:                       4.4%   ← Severely compressed
```

**Normal Margin**: ~27% → **Crisis Margin**: ~4%

## Quick SQL Queries

### 1. Monthly Profit Trend (LGEUS)
```sql
SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    COUNT(DISTINCT h.ORDER_NO) as ORDERS,
    SUM(i.ORDER_QTY) as UNITS,
    ROUND(SUM(i.NET_VALUE), 0) as REVENUE,
    ROUND(SUM(c.COST_AMOUNT * i.ORDER_QTY), 0) as TOTAL_COST,
    ROUND(SUM(i.NET_VALUE) - SUM(c.COST_AMOUNT * i.ORDER_QTY), 0) as PROFIT,
    ROUND(100.0 * (SUM(i.NET_VALUE) - SUM(c.COST_AMOUNT * i.ORDER_QTY)) / SUM(i.NET_VALUE), 2) as MARGIN_PCT
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
GROUP BY strftime('%Y-%m', h.DOC_DATE)
ORDER BY MONTH;
```

### 2. Logistics Cost Analysis (OLED 65"+)
```sql
SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    p.SCREEN_SIZE,
    COUNT(DISTINCT h.ORDER_NO) as ORDERS,
    ROUND(AVG(c.COST_AMOUNT), 2) as AVG_LOG_COST
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_MD_PRODUCT p ON i.PRODUCT_ID = p.PRODUCT_ID
JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND p.PANEL_TYPE = 'OLED'
    AND p.SCREEN_SIZE >= 65
    AND c.COST_TYPE = 'LOG'
    AND h.DOC_DATE BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY strftime('%Y-%m', h.DOC_DATE), p.SCREEN_SIZE
ORDER BY MONTH, p.SCREEN_SIZE;
```

### 3. Price Protection Frequency
```sql
SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    COUNT(DISTINCT h.ORDER_NO) as TOTAL_ORDERS,
    COUNT(DISTINCT CASE WHEN pc.COND_TYPE = 'ZPRO' THEN h.ORDER_NO END) as ORDERS_WITH_PRICE_PROT,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN pc.COND_TYPE = 'ZPRO' THEN h.ORDER_NO END) /
          COUNT(DISTINCT h.ORDER_NO), 1) as PRICE_PROT_RATE_PCT,
    ROUND(AVG(CASE WHEN pc.COND_TYPE = 'ZPRO' THEN ABS(pc.COND_VALUE) END), 2) as AVG_AMOUNT
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
LEFT JOIN TBL_TX_PRICE_CONDITION pc ON i.ORDER_NO = pc.ORDER_NO AND i.ITEM_NO = pc.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND h.DOC_DATE BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY strftime('%Y-%m', h.DOC_DATE)
ORDER BY MONTH;
```

### 4. Product Mix Analysis
```sql
SELECT
    p.PANEL_TYPE,
    p.SCREEN_SIZE,
    COUNT(DISTINCT h.ORDER_NO) as ORDERS,
    SUM(i.ORDER_QTY) as UNITS_SOLD,
    ROUND(SUM(i.NET_VALUE), 0) as REVENUE,
    ROUND(SUM(i.NET_VALUE) / SUM(i.ORDER_QTY), 2) as AVG_PRICE_PER_UNIT
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_MD_PRODUCT p ON i.PRODUCT_ID = p.PRODUCT_ID
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND h.DOC_DATE BETWEEN '2024-10-01' AND '2024-12-31'
GROUP BY p.PANEL_TYPE, p.SCREEN_SIZE
ORDER BY REVENUE DESC;
```

### 5. Cost Component Breakdown (Q4 2024)
```sql
SELECT
    c.COST_TYPE,
    SUM(c.COST_AMOUNT * i.ORDER_QTY) as TOTAL_COST,
    ROUND(100.0 * SUM(c.COST_AMOUNT * i.ORDER_QTY) /
          (SELECT SUM(c2.COST_AMOUNT * i2.ORDER_QTY)
           FROM TBL_TX_SALES_HEADER h2
           JOIN TBL_TX_SALES_ITEM i2 ON h2.ORDER_NO = i2.ORDER_NO
           JOIN TBL_TX_COST_DETAIL c2 ON i2.ORDER_NO = c2.ORDER_NO AND i2.ITEM_NO = c2.ITEM_NO
           WHERE h2.SUBSIDIARY_ID = 'LGEUS'
             AND h2.DOC_DATE BETWEEN '2024-10-01' AND '2024-12-31'), 2) as PCT_OF_TOTAL
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND h.DOC_DATE BETWEEN '2024-10-01' AND '2024-12-31'
GROUP BY c.COST_TYPE
ORDER BY TOTAL_COST DESC;
```

## AI Agent Testing Questions

Use these to test if your AI agent can diagnose the crisis:

1. **Basic**: "What was the profitability trend for LGEUS in 2024?"
   - Expected: Should identify Q4 as anomaly period

2. **Intermediate**: "Why did profit margin drop in Q4 2024 for North America?"
   - Expected: Should analyze both revenue and cost components

3. **Advanced**: "Which specific cost drivers caused the Q4 2024 margin compression for large OLED TVs?"
   - Expected: Should identify logistics cost spike (LOG) and price protection (ZPRO)

4. **Expert**: "Compare logistics costs for OLED 65"+ between Q3 and Q4 2024, and explain the business impact"
   - Expected: Should show 350% increase and calculate profit impact

## Regenerating the Database

If you need to regenerate the database with fresh data:

```bash
python generate_data.py
```

This will:
- Drop existing tables
- Create schema
- Generate ~10,000 orders
- Embed Q4 2024 crisis
- Run verification queries

## Technical Notes

- **Schema Type**: 3NF (Third Normal Form)
- **Foreign Keys**: Enabled for referential integrity
- **Indexes**: Created on frequently queried columns (date, product, subsidiary)
- **Data Quality**: Deterministic pricing (not random), realistic ratios
- **File Size**: ~15-20 MB database file

## Support

For detailed explanations, see [README_DATABASE.md](README_DATABASE.md)

For schema details, see [schema_normalized_3nf.sql](schema_normalized_3nf.sql)
