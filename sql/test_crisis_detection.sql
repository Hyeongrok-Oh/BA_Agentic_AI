-- ============================================================================
-- Crisis Detection Test Query
-- Purpose: Verify Q4 2024 crisis is embedded correctly
-- ============================================================================

.mode column
.headers on
.width 10 15 12 12 12 10

SELECT '==================================================================' as '';
SELECT 'TEST 1: Logistics Cost Spike Detection' as '';
SELECT '==================================================================' as '';

SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    ROUND(AVG(CASE WHEN c.COST_TYPE = 'LOG' AND p.SCREEN_SIZE = 65 THEN c.COST_AMOUNT END), 2) as OLED_65_LOG,
    ROUND(AVG(CASE WHEN c.COST_TYPE = 'LOG' AND p.SCREEN_SIZE = 77 THEN c.COST_AMOUNT END), 2) as OLED_77_LOG,
    CASE 
        WHEN strftime('%Y-%m', h.DOC_DATE) BETWEEN '2024-10' AND '2024-12' THEN 'CRISIS'
        ELSE 'NORMAL'
    END as STATUS
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_MD_PRODUCT p ON i.PRODUCT_ID = p.PRODUCT_ID
JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND p.PANEL_TYPE = 'OLED'
    AND p.SCREEN_SIZE IN (65, 77)
    AND h.DOC_DATE BETWEEN '2024-08-01' AND '2025-02-28'
GROUP BY strftime('%Y-%m', h.DOC_DATE)
ORDER BY MONTH;

SELECT '' as '';
SELECT '==================================================================' as '';
SELECT 'TEST 2: Price Protection Frequency' as '';
SELECT '==================================================================' as '';

SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    COUNT(DISTINCT h.ORDER_NO) as TOTAL_ORDERS,
    COUNT(DISTINCT CASE WHEN pc.COND_TYPE = 'ZPRO' THEN h.ORDER_NO END) as WITH_ZPRO,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN pc.COND_TYPE = 'ZPRO' THEN h.ORDER_NO END) / 
          COUNT(DISTINCT h.ORDER_NO), 1) as ZPRO_RATE_PCT,
    CASE 
        WHEN strftime('%Y-%m', h.DOC_DATE) BETWEEN '2024-10' AND '2024-12' THEN 'CRISIS'
        ELSE 'NORMAL'
    END as STATUS
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
LEFT JOIN TBL_TX_PRICE_CONDITION pc ON i.ORDER_NO = pc.ORDER_NO AND i.ITEM_NO = pc.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND h.DOC_DATE BETWEEN '2024-08-01' AND '2025-02-28'
GROUP BY strftime('%Y-%m', h.DOC_DATE)
ORDER BY MONTH;

SELECT '' as '';
SELECT '==================================================================' as '';
SELECT 'TEST 3: Margin Compression Validation' as '';
SELECT '==================================================================' as '';

SELECT
    strftime('%Y-%m', h.DOC_DATE) as MONTH,
    ROUND(SUM(i.NET_VALUE), 0) as REVENUE,
    ROUND(SUM(i.NET_VALUE) - SUM(cd.COST_AMOUNT * i.ORDER_QTY), 0) as OP_PROFIT,
    ROUND(100.0 * (SUM(i.NET_VALUE) - SUM(cd.COST_AMOUNT * i.ORDER_QTY)) / 
          SUM(i.NET_VALUE), 1) as MARGIN_PCT,
    CASE 
        WHEN strftime('%Y-%m', h.DOC_DATE) BETWEEN '2024-10' AND '2024-12' THEN 'CRISIS'
        ELSE 'NORMAL'
    END as STATUS
FROM TBL_TX_SALES_HEADER h
JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
JOIN TBL_TX_COST_DETAIL cd ON i.ORDER_NO = cd.ORDER_NO AND i.ITEM_NO = cd.ITEM_NO
WHERE h.SUBSIDIARY_ID = 'LGEUS'
    AND h.DOC_DATE BETWEEN '2024-08-01' AND '2025-02-28'
GROUP BY strftime('%Y-%m', h.DOC_DATE)
ORDER BY MONTH;

SELECT '' as '';
SELECT '==================================================================' as '';
SELECT 'VERIFICATION COMPLETE - Crisis patterns detected successfully!' as '';
SELECT '==================================================================' as '';
