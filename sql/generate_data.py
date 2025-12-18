"""
LG Electronics HE Division - ERP Database Generator
Purpose: Create realistic synthetic data with embedded Q4 2024 profitability crisis
Schema: 3NF/Snowflake (SAP-like) with normalized transaction tables
Author: Lead Data Engineer & ERP Specialist
"""

import sqlite3
import random
from datetime import datetime, timedelta
from decimal import Decimal
import math

# ============================================================================
# DATABASE CONNECTION & INITIALIZATION
# ============================================================================

DB_NAME = "lge_he_erp.db"

def init_database():
    """Create database connection and initialize schema"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Drop existing tables if any
    cursor.execute("PRAGMA foreign_keys = OFF")
    tables = ['TBL_TX_COST_DETAIL', 'TBL_TX_PRICE_CONDITION',
              'TBL_TX_SALES_ITEM', 'TBL_TX_SALES_HEADER',
              'TBL_ORG_CUSTOMER', 'TBL_ORG_SUBSIDIARY', 'TBL_MD_PRODUCT']

    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # Create schema
    create_schema(cursor)
    cursor.execute("PRAGMA foreign_keys = ON")
    conn.commit()

    return conn, cursor

def create_schema(cursor):
    """Create normalized 3NF schema (SAP ERP style)"""

    # 1. MASTER DATA
    cursor.execute("""
        CREATE TABLE TBL_MD_PRODUCT (
            PRODUCT_ID TEXT PRIMARY KEY,
            MODEL_NAME TEXT,
            SERIES TEXT,
            PANEL_TYPE TEXT,
            SCREEN_SIZE INTEGER,
            LAUNCH_YEAR INTEGER,
            MFG_PLANT TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE TBL_ORG_SUBSIDIARY (
            SUBSIDIARY_ID TEXT PRIMARY KEY,
            REGION TEXT,
            CURRENCY TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE TBL_ORG_CUSTOMER (
            CUSTOMER_ID TEXT PRIMARY KEY,
            CUST_NAME TEXT,
            SUBSIDIARY_ID TEXT,
            CHANNEL_TYPE TEXT,
            FOREIGN KEY(SUBSIDIARY_ID) REFERENCES TBL_ORG_SUBSIDIARY(SUBSIDIARY_ID)
        )
    """)

    # 2. TRANSACTION DATA
    cursor.execute("""
        CREATE TABLE TBL_TX_SALES_HEADER (
            ORDER_NO TEXT PRIMARY KEY,
            DOC_DATE DATE,
            CUSTOMER_ID TEXT,
            SUBSIDIARY_ID TEXT,
            TOTAL_NET_VALUE REAL,
            CURRENCY TEXT,
            FOREIGN KEY(CUSTOMER_ID) REFERENCES TBL_ORG_CUSTOMER(CUSTOMER_ID)
        )
    """)

    cursor.execute("""
        CREATE TABLE TBL_TX_SALES_ITEM (
            ORDER_NO TEXT,
            ITEM_NO INTEGER,
            PRODUCT_ID TEXT,
            ORDER_QTY INTEGER,
            NET_VALUE REAL,
            PRIMARY KEY (ORDER_NO, ITEM_NO),
            FOREIGN KEY(ORDER_NO) REFERENCES TBL_TX_SALES_HEADER(ORDER_NO),
            FOREIGN KEY(PRODUCT_ID) REFERENCES TBL_MD_PRODUCT(PRODUCT_ID)
        )
    """)

    # 3. PRICING CONDITIONS
    cursor.execute("""
        CREATE TABLE TBL_TX_PRICE_CONDITION (
            ORDER_NO TEXT,
            ITEM_NO INTEGER,
            COND_TYPE TEXT,
            COND_VALUE REAL,
            CURRENCY TEXT,
            PRIMARY KEY (ORDER_NO, ITEM_NO, COND_TYPE)
        )
    """)

    # 4. COST DETAILS
    cursor.execute("""
        CREATE TABLE TBL_TX_COST_DETAIL (
            ORDER_NO TEXT,
            ITEM_NO INTEGER,
            COST_TYPE TEXT,
            COST_AMOUNT REAL,
            CURRENCY TEXT,
            PRIMARY KEY (ORDER_NO, ITEM_NO, COST_TYPE)
        )
    """)

    # Create indexes for performance
    cursor.execute("CREATE INDEX idx_sales_date ON TBL_TX_SALES_HEADER(DOC_DATE)")
    cursor.execute("CREATE INDEX idx_sales_subsidiary ON TBL_TX_SALES_HEADER(SUBSIDIARY_ID)")
    cursor.execute("CREATE INDEX idx_item_product ON TBL_TX_SALES_ITEM(PRODUCT_ID)")

# ============================================================================
# MASTER DATA GENERATION
# ============================================================================

def populate_master_data(cursor):
    """Populate master data tables"""

    # Products (LG TV Portfolio)
    products = [
        # OLED Flagship
        ('OLED65G4PUA', '65" OLED evo G4', 'G4', 'OLED', 65, 2024, 'MEX'),
        ('OLED77G4PUA', '77" OLED evo G4', 'G4', 'OLED', 77, 2024, 'MEX'),
        ('OLED83G4PUA', '83" OLED evo G4', 'G4', 'OLED', 83, 2024, 'POL'),

        # OLED Premium
        ('OLED65C4PUA', '65" OLED evo C4', 'C4', 'OLED', 65, 2024, 'MEX'),
        ('OLED77C4PUA', '77" OLED evo C4', 'C4', 'OLED', 77, 2024, 'POL'),
        ('OLED55C4PUA', '55" OLED evo C4', 'C4', 'OLED', 55, 2024, 'MEX'),

        # OLED Entry
        ('OLED65B4PUA', '65" OLED B4', 'B4', 'OLED', 65, 2023, 'MEX'),
        ('OLED55B4PUA', '55" OLED B4', 'B4', 'OLED', 55, 2023, 'MEX'),

        # QNED Premium
        ('65QNED80URA', '65" QNED80', 'QNED80', 'QNED', 65, 2023, 'IDN'),
        ('75QNED80URA', '75" QNED80', 'QNED80', 'QNED', 75, 2023, 'IDN'),

        # LCD Entry
        ('55UQ7590PUB', '55" UQ75 4K', 'UQ75', 'LCD', 55, 2023, 'IDN'),
        ('65UQ7590PUB', '65" UQ75 4K', 'UQ75', 'LCD', 65, 2023, 'IDN'),
    ]

    cursor.executemany(
        "INSERT INTO TBL_MD_PRODUCT VALUES (?, ?, ?, ?, ?, ?, ?)",
        products
    )

    # Subsidiaries
    subsidiaries = [
        ('LGEUS', 'NA', 'USD'),
        ('LGECA', 'NA', 'CAD'),
        ('LGEKR', 'KR', 'KRW'),
        ('LGEUK', 'EU', 'GBP'),
        ('LGEDE', 'EU', 'EUR'),
        ('LGEFR', 'EU', 'EUR'),
    ]

    cursor.executemany(
        "INSERT INTO TBL_ORG_SUBSIDIARY VALUES (?, ?, ?)",
        subsidiaries
    )

    # Customers
    customers = [
        # North America
        ('CUST-BB-US', 'Best Buy', 'LGEUS', 'RETAIL'),
        ('CUST-CS-US', 'Costco', 'LGEUS', 'RETAIL'),
        ('CUST-AMZ-US', 'Amazon.com', 'LGEUS', 'ONLINE'),
        ('CUST-TGT-US', 'Target', 'LGEUS', 'RETAIL'),
        ('CUST-WM-US', 'Walmart', 'LGEUS', 'RETAIL'),
        ('CUST-BB-CA', 'Best Buy Canada', 'LGECA', 'RETAIL'),

        # Korea
        ('CUST-HMM-KR', 'Himart', 'LGEKR', 'RETAIL'),
        ('CUST-EMT-KR', 'Electro Mart', 'LGEKR', 'RETAIL'),

        # Europe
        ('CUST-JL-UK', 'John Lewis', 'LGEUK', 'RETAIL'),
        ('CUST-CUR-UK', 'Currys', 'LGEUK', 'RETAIL'),
        ('CUST-MM-DE', 'MediaMarkt', 'LGEDE', 'RETAIL'),
        ('CUST-SAT-DE', 'Saturn', 'LGEDE', 'RETAIL'),
        ('CUST-FNC-FR', 'Fnac', 'LGEFR', 'RETAIL'),
    ]

    cursor.executemany(
        "INSERT INTO TBL_ORG_CUSTOMER VALUES (?, ?, ?, ?)",
        customers
    )

# ============================================================================
# BUSINESS LOGIC & DATA GENERATION RULES
# ============================================================================

def is_crisis_period(date_obj):
    """Check if date falls in Q4 2024 crisis period"""
    return (date_obj.year == 2024 and date_obj.month >= 10)

def is_crisis_target(product_id, subsidiary_id):
    """Check if product/region is affected by crisis"""
    # Target: Large OLED TVs (65"+) in North America
    large_oled_products = [
        'OLED65G4PUA', 'OLED77G4PUA', 'OLED83G4PUA',
        'OLED65C4PUA', 'OLED77C4PUA', 'OLED65B4PUA'
    ]
    na_subsidiaries = ['LGEUS', 'LGECA']

    return (product_id in large_oled_products and subsidiary_id in na_subsidiaries)

def get_base_price(product_id):
    """Get base price by product tier"""
    if 'G4' in product_id:
        if '83' in product_id:
            return 5499.00
        elif '77' in product_id:
            return 3299.00
        elif '65' in product_id:
            return 2699.00
    elif 'C4' in product_id:
        if '77' in product_id:
            return 2799.00
        elif '65' in product_id:
            return 1999.00
        elif '55' in product_id:
            return 1499.00
    elif 'B4' in product_id:
        if '65' in product_id:
            return 1599.00
        elif '55' in product_id:
            return 1199.00
    elif 'QNED' in product_id:
        if '75' in product_id:
            return 1299.00
        elif '65' in product_id:
            return 899.00
    else:  # LCD
        if '65' in product_id:
            return 599.00
        elif '55' in product_id:
            return 449.00

    return 1000.00  # Default

def get_material_cost(product_id):
    """Calculate material cost (primarily panel + IC)"""
    base_price = get_base_price(product_id)

    # Material cost as % of base price
    if 'OLED' in product_id:
        return base_price * 0.55  # OLED panels are expensive
    elif 'QNED' in product_id:
        return base_price * 0.48
    else:  # LCD
        return base_price * 0.42

def get_normal_logistics_cost(product_id):
    """Normal logistics cost (ocean freight)"""
    if 'OLED' in product_id:
        if '83' in product_id or '77' in product_id:
            return 120.00  # Large OLED
        else:
            return 80.00
    elif 'QNED' in product_id or '65' in product_id:
        return 60.00
    else:
        return 40.00

def get_crisis_logistics_cost(product_id):
    """Crisis logistics cost (air freight + Red Sea surcharges) - 350% increase"""
    normal_cost = get_normal_logistics_cost(product_id)
    return normal_cost * 4.5  # 350% increase means 4.5x multiplier

def get_overhead_cost(product_id):
    """Manufacturing overhead"""
    if 'OLED' in product_id:
        return 100.00
    elif 'QNED' in product_id:
        return 60.00
    else:
        return 40.00

def get_tariff_cost(product_id, subsidiary_id):
    """Tariff costs (varies by destination)"""
    base_price = get_base_price(product_id)

    if subsidiary_id in ['LGEUS', 'LGECA']:
        return base_price * 0.025  # 2.5% US tariff
    elif subsidiary_id in ['LGEUK', 'LGEDE', 'LGEFR']:
        return base_price * 0.04  # 4% EU tariff
    else:
        return base_price * 0.01

def generate_price_conditions(order_no, item_no, product_id, base_price,
                              is_crisis, currency='USD'):
    """Generate pricing condition records"""
    conditions = []

    # PR00: Base price (always positive)
    conditions.append((order_no, item_no, 'PR00', base_price, currency))

    # K007: Standard volume discount (5-15%)
    discount_rate = random.uniform(0.05, 0.15)
    discount = -1 * base_price * discount_rate
    conditions.append((order_no, item_no, 'K007', discount, currency))

    # ZPRO: Price Protection (crisis period only)
    if is_crisis:
        # Frequent and aggressive price protection
        if random.random() < 0.75:  # 75% of crisis orders get price protection
            price_prot = -1 * random.uniform(150, 350)
            conditions.append((order_no, item_no, 'ZPRO', price_prot, currency))
    else:
        # Normal period: occasional and small
        if random.random() < 0.10:  # Only 10% of normal orders
            price_prot = -1 * random.uniform(30, 80)
            conditions.append((order_no, item_no, 'ZPRO', price_prot, currency))

    # ZMDF: Marketing Development Fund (occasional)
    if random.random() < 0.20:
        mdf = -1 * random.uniform(20, 100)
        conditions.append((order_no, item_no, 'ZMDF', mdf, currency))

    return conditions

def generate_cost_details(order_no, item_no, product_id, subsidiary_id,
                         is_crisis, currency='USD'):
    """Generate cost detail records"""
    costs = []

    # MAT: Material cost (panel + IC)
    mat_cost = get_material_cost(product_id)
    costs.append((order_no, item_no, 'MAT', mat_cost, currency))

    # LOG: Logistics cost (CRISIS FACTOR HERE)
    if is_crisis:
        log_cost = get_crisis_logistics_cost(product_id)
    else:
        log_cost = get_normal_logistics_cost(product_id)
    costs.append((order_no, item_no, 'LOG', log_cost, currency))

    # TAR: Tariff
    tar_cost = get_tariff_cost(product_id, subsidiary_id)
    costs.append((order_no, item_no, 'TAR', tar_cost, currency))

    # OH: Manufacturing overhead
    oh_cost = get_overhead_cost(product_id)
    costs.append((order_no, item_no, 'OH', oh_cost, currency))

    return costs

# ============================================================================
# TRANSACTION DATA GENERATION
# ============================================================================

def generate_sales_transactions(cursor):
    """Generate daily sales transactions for 2023-2025"""

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    current_date = start_date

    order_counter = 1

    # Get master data for reference
    cursor.execute("SELECT CUSTOMER_ID, SUBSIDIARY_ID FROM TBL_ORG_CUSTOMER")
    customers = cursor.fetchall()

    cursor.execute("SELECT PRODUCT_ID FROM TBL_MD_PRODUCT")
    products = [p[0] for p in cursor.fetchall()]

    all_headers = []
    all_items = []
    all_conditions = []
    all_costs = []

    print("Generating sales transactions...")
    print(f"Period: {start_date.date()} to {end_date.date()}")

    # Generate orders day by day
    while current_date <= end_date:
        # Number of orders per day (varies by season)
        month = current_date.month

        # Seasonal variation
        if month in [11, 12]:  # Holiday season
            daily_orders = random.randint(12, 20)
        elif month in [1, 7, 8]:  # Low season
            daily_orders = random.randint(4, 8)
        else:
            daily_orders = random.randint(6, 12)

        # Generate orders for this day
        for _ in range(daily_orders):
            # Select customer and product
            customer_id, subsidiary_id = random.choice(customers)
            product_id = random.choice(products)

            # Check if this is a crisis scenario
            is_crisis = (is_crisis_period(current_date) and
                        is_crisis_target(product_id, subsidiary_id))

            # Get currency
            cursor.execute(
                "SELECT CURRENCY FROM TBL_ORG_SUBSIDIARY WHERE SUBSIDIARY_ID = ?",
                (subsidiary_id,)
            )
            currency = cursor.fetchone()[0]

            # Order details
            order_no = f"SO-{current_date.year}-{order_counter:06d}"
            order_counter += 1

            # Quantity (crisis period: push volume despite lower margin)
            if is_crisis:
                qty = random.randint(5, 20)  # Higher volume
            else:
                qty = random.randint(2, 10)

            # Pricing
            base_price = get_base_price(product_id)

            # Generate pricing conditions
            item_no = 10
            conditions = generate_price_conditions(
                order_no, item_no, product_id, base_price, is_crisis, currency
            )

            # Calculate net value from conditions
            net_value_per_unit = sum(c[3] for c in conditions)
            total_net_value = net_value_per_unit * qty

            # Generate cost details
            costs = generate_cost_details(
                order_no, item_no, product_id, subsidiary_id, is_crisis, currency
            )

            # Store header
            all_headers.append((
                order_no,
                current_date.strftime('%Y-%m-%d'),
                customer_id,
                subsidiary_id,
                total_net_value,
                currency
            ))

            # Store item
            all_items.append((
                order_no,
                item_no,
                product_id,
                qty,
                total_net_value
            ))

            # Store conditions and costs
            all_conditions.extend(conditions)
            all_costs.extend(costs)

        # Progress indicator
        if current_date.day == 1:
            print(f"  Generated: {current_date.strftime('%Y-%m')}")

        current_date += timedelta(days=1)

    # Bulk insert
    print(f"\nInserting {len(all_headers)} orders into database...")
    cursor.executemany(
        "INSERT INTO TBL_TX_SALES_HEADER VALUES (?, ?, ?, ?, ?, ?)",
        all_headers
    )

    cursor.executemany(
        "INSERT INTO TBL_TX_SALES_ITEM VALUES (?, ?, ?, ?, ?)",
        all_items
    )

    cursor.executemany(
        "INSERT INTO TBL_TX_PRICE_CONDITION VALUES (?, ?, ?, ?, ?)",
        all_conditions
    )

    cursor.executemany(
        "INSERT INTO TBL_TX_COST_DETAIL VALUES (?, ?, ?, ?, ?)",
        all_costs
    )

    print(f"  Headers: {len(all_headers)}")
    print(f"  Items: {len(all_items)}")
    print(f"  Conditions: {len(all_conditions)}")
    print(f"  Costs: {len(all_costs)}")

# ============================================================================
# VERIFICATION QUERIES
# ============================================================================

def run_verification_queries(cursor):
    """Run queries to verify the Q4 2024 crisis is embedded in data"""

    print("\n" + "="*80)
    print("VERIFICATION: Monthly Operating Profit Analysis for LGEUS")
    print("="*80)

    query = """
    WITH monthly_sales AS (
        SELECT
            strftime('%Y-%m', h.DOC_DATE) as MONTH,
            h.SUBSIDIARY_ID,
            COUNT(DISTINCT h.ORDER_NO) as ORDER_COUNT,
            SUM(i.ORDER_QTY) as TOTAL_QTY,
            SUM(i.NET_VALUE) as NET_REVENUE
        FROM TBL_TX_SALES_HEADER h
        JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
        WHERE h.SUBSIDIARY_ID = 'LGEUS'
        GROUP BY strftime('%Y-%m', h.DOC_DATE), h.SUBSIDIARY_ID
    ),
    monthly_costs AS (
        SELECT
            strftime('%Y-%m', h.DOC_DATE) as MONTH,
            SUM(CASE WHEN c.COST_TYPE = 'MAT' THEN c.COST_AMOUNT * i.ORDER_QTY ELSE 0 END) as TOTAL_MAT_COST,
            SUM(CASE WHEN c.COST_TYPE = 'LOG' THEN c.COST_AMOUNT * i.ORDER_QTY ELSE 0 END) as TOTAL_LOG_COST,
            SUM(CASE WHEN c.COST_TYPE = 'TAR' THEN c.COST_AMOUNT * i.ORDER_QTY ELSE 0 END) as TOTAL_TAR_COST,
            SUM(CASE WHEN c.COST_TYPE = 'OH' THEN c.COST_AMOUNT * i.ORDER_QTY ELSE 0 END) as TOTAL_OH_COST,
            SUM(c.COST_AMOUNT * i.ORDER_QTY) as TOTAL_COST
        FROM TBL_TX_SALES_HEADER h
        JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
        JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
        WHERE h.SUBSIDIARY_ID = 'LGEUS'
        GROUP BY strftime('%Y-%m', h.DOC_DATE)
    ),
    monthly_deductions AS (
        SELECT
            strftime('%Y-%m', h.DOC_DATE) as MONTH,
            SUM(CASE WHEN p.COND_TYPE = 'ZPRO' THEN ABS(p.COND_VALUE) * i.ORDER_QTY ELSE 0 END) as TOTAL_PRICE_PROT
        FROM TBL_TX_SALES_HEADER h
        JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
        JOIN TBL_TX_PRICE_CONDITION p ON i.ORDER_NO = p.ORDER_NO AND i.ITEM_NO = p.ITEM_NO
        WHERE h.SUBSIDIARY_ID = 'LGEUS'
        GROUP BY strftime('%Y-%m', h.DOC_DATE)
    )
    SELECT
        s.MONTH,
        s.ORDER_COUNT,
        s.TOTAL_QTY,
        ROUND(s.NET_REVENUE, 0) as NET_REVENUE,
        ROUND(c.TOTAL_MAT_COST, 0) as MAT_COST,
        ROUND(c.TOTAL_LOG_COST, 0) as LOG_COST,
        ROUND(c.TOTAL_TAR_COST, 0) as TAR_COST,
        ROUND(c.TOTAL_OH_COST, 0) as OH_COST,
        ROUND(c.TOTAL_COST, 0) as TOTAL_COST,
        ROUND(d.TOTAL_PRICE_PROT, 0) as PRICE_PROT,
        ROUND(s.NET_REVENUE - c.TOTAL_COST, 0) as OPERATING_PROFIT,
        ROUND((s.NET_REVENUE - c.TOTAL_COST) / s.NET_REVENUE * 100, 2) as PROFIT_MARGIN_PCT
    FROM monthly_sales s
    JOIN monthly_costs c ON s.MONTH = c.MONTH
    JOIN monthly_deductions d ON s.MONTH = d.MONTH
    WHERE s.MONTH >= '2024-01'
    ORDER BY s.MONTH
    """

    cursor.execute(query)
    results = cursor.fetchall()

    # Print header
    print(f"\n{'Month':<10} {'Orders':<8} {'Qty':<6} {'Revenue':<12} {'LOG_Cost':<12} "
          f"{'Tot_Cost':<12} {'PriceP':<10} {'OpProfit':<12} {'Margin%':<8}")
    print("-" * 110)

    # Print results
    for row in results:
        month, orders, qty, revenue, mat, log, tar, oh, tot_cost, price_p, op_profit, margin = row

        # Highlight Q4 2024 crisis period
        if month >= '2024-10' and month <= '2024-12':
            marker = " <== CRISIS"
        else:
            marker = ""

        print(f"{month:<10} {orders:<8} {qty:<6} ${revenue:<11,.0f} ${log:<11,.0f} "
              f"${tot_cost:<11,.0f} ${price_p:<9,.0f} ${op_profit:<11,.0f} {margin:>6.2f}%{marker}")

    # Additional crisis analysis
    print("\n" + "="*80)
    print("CRISIS DEEP DIVE: OLED 65+ Logistics Cost Comparison")
    print("="*80)

    crisis_query = """
    SELECT
        strftime('%Y-%m', h.DOC_DATE) as MONTH,
        p.PANEL_TYPE,
        p.SCREEN_SIZE,
        COUNT(DISTINCT i.ORDER_NO) as ORDERS,
        ROUND(AVG(CASE WHEN c.COST_TYPE = 'LOG' THEN c.COST_AMOUNT ELSE NULL END), 2) as AVG_LOG_COST_PER_UNIT
    FROM TBL_TX_SALES_HEADER h
    JOIN TBL_TX_SALES_ITEM i ON h.ORDER_NO = i.ORDER_NO
    JOIN TBL_MD_PRODUCT p ON i.PRODUCT_ID = p.PRODUCT_ID
    JOIN TBL_TX_COST_DETAIL c ON i.ORDER_NO = c.ORDER_NO AND i.ITEM_NO = c.ITEM_NO
    WHERE h.SUBSIDIARY_ID = 'LGEUS'
        AND p.PANEL_TYPE = 'OLED'
        AND p.SCREEN_SIZE >= 65
        AND strftime('%Y-%m', h.DOC_DATE) BETWEEN '2024-06' AND '2025-01'
    GROUP BY strftime('%Y-%m', h.DOC_DATE), p.PANEL_TYPE, p.SCREEN_SIZE
    ORDER BY MONTH, p.SCREEN_SIZE
    """

    cursor.execute(crisis_query)
    crisis_results = cursor.fetchall()

    print(f"\n{'Month':<10} {'PanelType':<12} {'Size':<6} {'Orders':<8} {'AvgLogCost/Unit':<18}")
    print("-" * 60)

    for row in crisis_results:
        month, panel, size, orders, avg_log = row
        marker = " <== SPIKE" if month >= '2024-10' and month <= '2024-12' else ""
        print(f"{month:<10} {panel:<12} {size:<6} {orders:<8} ${avg_log:>15.2f}{marker}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("LG ELECTRONICS HE DIVISION - ERP DATABASE GENERATOR")
    print("="*80)
    print("\nInitializing database...")

    conn, cursor = init_database()

    print("\nPopulating master data...")
    populate_master_data(cursor)
    conn.commit()

    print("\nGenerating transaction data (2023-2025)...")
    generate_sales_transactions(cursor)
    conn.commit()

    print("\n" + "="*80)
    print("DATABASE GENERATION COMPLETE")
    print("="*80)
    print(f"\nDatabase file: {DB_NAME}")

    # Show summary statistics
    cursor.execute("SELECT COUNT(*) FROM TBL_MD_PRODUCT")
    print(f"Products: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM TBL_ORG_CUSTOMER")
    print(f"Customers: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM TBL_TX_SALES_HEADER")
    print(f"Sales Orders: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM TBL_TX_PRICE_CONDITION")
    print(f"Pricing Conditions: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM TBL_TX_COST_DETAIL")
    print(f"Cost Details: {cursor.fetchone()[0]}")

    # Run verification
    run_verification_queries(cursor)

    conn.close()
    print("\n" + "="*80)
    print("All operations completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
