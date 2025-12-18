-- ============================================================================
-- LG Electronics HE Division - ERP Database Schema
-- Schema Type: 3NF/Snowflake (SAP ERP Style)
-- Database: SQLite
-- Purpose: AI Agent - Complex Profitability Diagnosis
-- ============================================================================
-- This schema represents a normalized ERP structure with separate tables for
-- Headers, Items, Pricing Conditions, and Cost Details - mirroring real SAP
-- systems. This forces AI agents to perform complex JOINs and understand
-- relational data models rather than working with simple flat files.
-- ============================================================================

-- ============================================================================
-- MASTER DATA TABLES
-- ============================================================================

-- (1) Product Master: TV product catalog with technical specifications
CREATE TABLE TBL_MD_PRODUCT (
    PRODUCT_ID TEXT PRIMARY KEY,        -- e.g., OLED65G4PUA
    MODEL_NAME TEXT,                    -- e.g., "65" OLED evo G4"
    SERIES TEXT,                        -- G4, C4, B4, QNED80, UQ75
    PANEL_TYPE TEXT,                    -- OLED, QNED, LCD
    SCREEN_SIZE INTEGER,                -- 55, 65, 75, 77, 83
    LAUNCH_YEAR INTEGER,                -- 2023, 2024, 2025
    MFG_PLANT TEXT                      -- MEX (Mexico), POL (Poland), IDN (Indonesia)
);

-- (2) Subsidiary Master: Regional sales organizations
CREATE TABLE TBL_ORG_SUBSIDIARY (
    SUBSIDIARY_ID TEXT PRIMARY KEY,     -- LGEUS, LGEKR, LGEUK, etc.
    REGION TEXT,                        -- NA, KR, EU
    CURRENCY TEXT                       -- USD, KRW, GBP, EUR, CAD
);

-- (3) Customer Master: Retailers and distributors
CREATE TABLE TBL_ORG_CUSTOMER (
    CUSTOMER_ID TEXT PRIMARY KEY,       -- CUST-BB-US, CUST-CS-US, etc.
    CUST_NAME TEXT,                     -- Best Buy, Costco, Amazon, etc.
    SUBSIDIARY_ID TEXT,                 -- Which subsidiary manages this customer
    CHANNEL_TYPE TEXT,                  -- B2B, RETAIL, ONLINE
    FOREIGN KEY(SUBSIDIARY_ID) REFERENCES TBL_ORG_SUBSIDIARY(SUBSIDIARY_ID)
);

-- ============================================================================
-- TRANSACTION TABLES (Sales Order Process)
-- ============================================================================

-- (4) Sales Header: Document-level information
-- This is the "Order" in SAP terms
CREATE TABLE TBL_TX_SALES_HEADER (
    ORDER_NO TEXT PRIMARY KEY,          -- SO-2024-000001
    DOC_DATE DATE,                      -- Transaction date (YYYY-MM-DD)
    CUSTOMER_ID TEXT,                   -- Who bought it
    SUBSIDIARY_ID TEXT,                 -- Which subsidiary sold it
    TOTAL_NET_VALUE REAL,               -- Total order value (sum of items)
    CURRENCY TEXT,                      -- Transaction currency
    FOREIGN KEY(CUSTOMER_ID) REFERENCES TBL_ORG_CUSTOMER(CUSTOMER_ID)
);

-- (5) Sales Item: Line-item level information
-- Each order can have multiple items (though in our data, typically 1)
CREATE TABLE TBL_TX_SALES_ITEM (
    ORDER_NO TEXT,                      -- Reference to header
    ITEM_NO INTEGER,                    -- Line number: 10, 20, 30...
    PRODUCT_ID TEXT,                    -- What was sold
    ORDER_QTY INTEGER,                  -- How many units
    NET_VALUE REAL,                     -- Line total (after all conditions)
    PRIMARY KEY (ORDER_NO, ITEM_NO),
    FOREIGN KEY(ORDER_NO) REFERENCES TBL_TX_SALES_HEADER(ORDER_NO),
    FOREIGN KEY(PRODUCT_ID) REFERENCES TBL_MD_PRODUCT(PRODUCT_ID)
);

-- ============================================================================
-- PRICING CONDITIONS (The "Why" behind Revenue)
-- ============================================================================
-- This is CRITICAL for understanding profitability issues.
-- Instead of storing "gross sales" and "discount" as columns, we store each
-- price component as a separate ROW. This is how SAP works.
--
-- Example: A $2000 TV with 10% discount and $200 price protection becomes:
--   Row 1: PR00 (Base Price)        +$2000
--   Row 2: K007 (Volume Discount)   -$200
--   Row 3: ZPRO (Price Protection)  -$200
--   Net Value = $2000 - $200 - $200 = $1600

CREATE TABLE TBL_TX_PRICE_CONDITION (
    ORDER_NO TEXT,
    ITEM_NO INTEGER,
    COND_TYPE TEXT,                     -- Condition type code
    COND_VALUE REAL,                    -- Amount (can be positive or negative)
    CURRENCY TEXT,
    PRIMARY KEY (ORDER_NO, ITEM_NO, COND_TYPE)
);

-- Common Condition Types in this database:
-- PR00: Base list price (always positive)
-- K007: Standard volume discount (negative)
-- ZPRO: Price protection / Price down compensation (negative) <== CRISIS DRIVER
-- ZMDF: Marketing Development Fund (negative)

-- ============================================================================
-- COST DETAILS (The "Why" behind Profit)
-- ============================================================================
-- Granular cost breakdown to enable root cause analysis.
-- Instead of storing "total_cost" as one column, we break it down by type.
-- This allows us to identify which cost component is causing margin erosion.

CREATE TABLE TBL_TX_COST_DETAIL (
    ORDER_NO TEXT,
    ITEM_NO INTEGER,
    COST_TYPE TEXT,                     -- Cost category
    COST_AMOUNT REAL,                   -- Cost per unit
    CURRENCY TEXT,
    PRIMARY KEY (ORDER_NO, ITEM_NO, COST_TYPE)
);

-- Cost Types in this database:
-- MAT: Material cost (panel + IC components)
-- LOG: Logistics cost (freight, shipping) <== CRISIS DRIVER (350% spike in Q4 2024)
-- TAR: Tariff and customs duties
-- OH:  Manufacturing overhead

-- ============================================================================
-- INDEXES (Performance Optimization)
-- ============================================================================
-- These indexes speed up common queries, especially date-based and
-- product/subsidiary-based aggregations

CREATE INDEX idx_sales_date ON TBL_TX_SALES_HEADER(DOC_DATE);
CREATE INDEX idx_sales_subsidiary ON TBL_TX_SALES_HEADER(SUBSIDIARY_ID);
CREATE INDEX idx_item_product ON TBL_TX_SALES_ITEM(PRODUCT_ID);

-- ============================================================================
-- DATA MODEL EXPLANATION
-- ============================================================================
--
-- To calculate profitability for a given period, an AI agent must:
--
-- 1. JOIN TBL_TX_SALES_HEADER and TBL_TX_SALES_ITEM to get order quantities
-- 2. JOIN TBL_TX_PRICE_CONDITION to reconstruct revenue components:
--    Revenue = SUM(COND_VALUE * ORDER_QTY) for all condition types
-- 3. JOIN TBL_TX_COST_DETAIL to calculate total costs:
--    Total Cost = SUM(COST_AMOUNT * ORDER_QTY) for all cost types
-- 4. JOIN TBL_MD_PRODUCT to segment by product attributes (panel type, size)
-- 5. JOIN TBL_ORG_SUBSIDIARY to segment by region
-- 6. GROUP BY time period (month, quarter) to identify trends
--
-- Example Profitability Query:
-- --------------------------
-- SELECT
--     strftime('%Y-%m', h.DOC_DATE) as MONTH,
--     p.PANEL_TYPE,
--     SUM(i.ORDER_QTY) as UNITS_SOLD,
--     SUM(i.NET_VALUE) as REVENUE,
--     SUM(cd.COST_AMOUNT * i.ORDER_QTY) as TOTAL_COST,
--     SUM(i.NET_VALUE) - SUM(cd.COST_AMOUNT * i.ORDER_QTY) as PROFIT
-- FROM TBL_TX_SALES_HEADER h
-- JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
-- JOIN TBL_MD_PRODUCT p ON i.PRODUCT_ID = p.PRODUCT_ID
-- JOIN TBL_TX_COST_DETAIL cd ON i.ORDER_NO = cd.ORDER_NO AND i.ITEM_NO = cd.ITEM_NO
-- WHERE h.SUBSIDIARY_ID = 'LGEUS'
-- GROUP BY strftime('%Y-%m', h.DOC_DATE), p.PANEL_TYPE
-- ORDER BY MONTH;
--
-- ============================================================================
-- EMBEDDED CRISIS (Q4 2024)
-- ============================================================================
--
-- The data generated by generate_data.py includes a profitability crisis:
--
-- WHEN:   October - December 2024 (Q4)
-- WHERE:  North America (LGEUS, LGECA)
-- WHAT:   Large OLED TVs (65 inches and above)
--
-- ROOT CAUSES:
-- 1. Logistics Cost Spike (350% increase)
--    - Normal LOG cost for 65" OLED: $80
--    - Crisis LOG cost for 65" OLED: $360
--    - Reason: Red Sea shipping disruption + air freight usage
--
-- 2. Price Protection Surge
--    - Normal: 10% of orders have ZPRO, avg -$50
--    - Crisis: 75% of orders have ZPRO, avg -$200
--    - Reason: Aggressive retailer compensation for mid-season price drops
--
-- EXPECTED OUTCOME:
-- - Revenue remains high (pushing volume to meet targets)
-- - Margin crashes from ~27% to ~11%
-- - Only visible when analyzing cost components separately
--
-- ============================================================================
-- END OF DDL
-- ============================================================================
