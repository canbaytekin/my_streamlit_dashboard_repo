import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import psycopg2
import psycopg2.extras
import os
import sys
import json
from datetime import datetime
import time
import glob
import re
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

# Authentication functions
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def validate_login():
        # For Streamlit Cloud deployment - check if we're running in the cloud
        is_cloud = os.environ.get('STREAMLIT_SHARING', '') == 'true'
        
        # Get credentials from secrets with proper fallbacks
        if is_cloud:
            # In cloud deployment, use hardcoded credentials for now
            # IMPORTANT: Replace these with your actual credentials before deployment
            expected_username = "belara"
            expected_password = "password123"
        else:
            # In local development, use secrets
            expected_username = st.secrets.get("DASHBOARD_USERNAME", "admin")
            expected_password = st.secrets.get("DASHBOARD_PASSWORD", "password")
        
        # Debug info (remove in production)
        print(f"Is cloud deployment: {is_cloud}")
        print(f"Expected username: {expected_username}")
        print(f"Entered username: {st.session_state.get('username', '')}")
        
        # Check credentials
        return (
            st.session_state.get("username", "") == expected_username and 
            st.session_state.get("password", "") == expected_password
        )

    # Return True if the username + password is validated.
    if st.session_state.get("authenticated"):
        return True
    
    # Show login form - completely separate from the dashboard content
    st.markdown("## Dashboard Login")
    
    # Create login form
    with st.form("login_form"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        submitted = st.form_submit_button("Login")
    
    if submitted and validate_login():
        st.session_state["authenticated"] = True
        st.rerun()  # Rerun the app to show the dashboard
        return True
    elif submitted:
        st.error("❌ Invalid username or password")
        return False
    else:
        return False

# Language dictionary for translations
translations = {
    "English": {
        "page_title": "Warehouse Analytics Dashboard",
        "dashboard_title": "Warehouse Stock Deficit Dashboard",
        "dashboard_subtitle": "Monitor warehouses with stock deficits",
        "filters_header": "Filters",
        "settings_header": "Sales Target Settings",
        "sales_target_label": "Sales Target Multiplier",
        "sales_target_help": "Multiply last month's sales by this value to set the sales target",
        "save_settings": "Save Settings",
        "settings_saved": "Settings saved!",
        "impact_header": "Impact of Sales Target Multiplier: {0}",
        "total_products": "Total Unique Products",
        "products_needing_restock": "Products Needing Restock",
        "total_stock_deficit": "Total Stock Deficit",
        "percent_needing_restock": "% Products Needing Restock",
        "target_multiplier_info": "**Current Target Multiplier: {0}x**\n\nThis multiplier affects how aggressively you want to restock. Higher values mean:\n- More products will be flagged as needing restock\n- Larger quantities will be recommended for purchase\n- Less risk of stockouts, but potentially higher inventory costs\n\nAdjust the slider in the sidebar and observe how it affects your restocking needs.",
        "select_all_warehouses": "Select All Warehouses",
        "select_warehouses": "Select Warehouses",
        "select_all_brands": "Select All Brands",
        "select_brands": "Select Brands",
        "select_all_categories": "Select All Categories",
        "select_categories": "Select Categories",
        "select_all_subjects": "Select All Subjects",
        "select_subjects": "Select Subjects",
        "select_all_products": "Select All Products",
        "select_product_ids": "Select Product IDs",
        "select_all_sizes": "Select All Sizes",
        "select_sizes": "Select Sizes",
        "stock_deficit_overview": "Stock Deficit Overview",
        "no_warehouses_deficit": "No warehouses with stock deficits found in the current filtered data.",
        "warehouses_with_deficit": "Warehouses with Stock Deficits",
        "warehouse_name": "Warehouse Name",
        "products_with_deficit": "Products with Deficit",
        "total_deficit": "Total Deficit",
        "detailed_data_header": "Detailed Data - Products Needing Restock",
        "no_products_matching": "No products matching your filter criteria need restocking.",
        "download_button": "Download Product Restock Data",
        "database_query_header": "Database Query",
        "run_custom_query": "Run Custom PostgreSQL Query",
        "enter_sql_query": "Enter SQL Query",
        "execute_query": "Execute Query",
        "query_results": "Query Results",
        "download_query_results": "Download Query Results",
        "footer_text": "Simplified Warehouse Stock Deficit Dashboard | PostgreSQL Integration",
        "refresh_data": "Refresh Data",
        "data_refreshed": "Data refreshed successfully!",
        "last_refresh": "Last refresh: {0}",
        "cron_refresh": "Last automatic refresh: {0}",
        "sales_orders_tab": "Sales and Orders",
        "total_orders_label": "Total Orders",
        "total_sales_label": "Total Sales",
        "date_label": "Date",
        "count_label": "Count",
        "legend_title": "Metric",
        "sale_order_ratio_trend_header": "Sale/Order Ratio Trend",
        "sale_order_ratio_label": "Sale/Order Ratio",
        "ratio_label": "Ratio",
        "metric_label": "Metric",
        "ratio_data_missing": "Sale_Order_Ratio data not available to display trend.",
        "shipment_tab": "Shipment Information",
        "shipment_title": "Warehouse Shipment Analysis",
        "avg_processing_days": "Avg. Processing Days",
        "max_processing_days": "Max Processing Days",
        "shipment_count": "Shipment Count",
        "warehouse_performance": "Warehouse Performance",
        "processing_time": "Processing Time (Days)",
        "efficiency_rating": "Efficiency Rating",
        "restocking_priorities": "Restocking Priorities",
        "priority_score": "Priority Score",
        "product_id": "Product ID",
        "processing_speed": "Processing Speed",
        "no_shipment_data": "No shipment data available.",
        "efficiency_distribution": "Efficiency Rating Distribution",
        "rating": "Rating"
    },
    "Russian": {
        "page_title": "Панель аналитики склада",
        "dashboard_title": "Панель дефицита складских запасов",
        "dashboard_subtitle": "Мониторинг складов с дефицитом запасов",
        "filters_header": "Фильтры",
        "settings_header": "Настройки целевых продаж",
        "sales_target_label": "Множитель целевых продаж",
        "sales_target_help": "Умножьте продажи прошлого месяца на это значение, чтобы установить целевой показатель продаж",
        "save_settings": "Сохранить настройки",
        "settings_saved": "Настройки сохранены!",
        "impact_header": "Влияние множителя целевых продаж: {0}",
        "total_products": "Всего уникальных товаров",
        "products_needing_restock": "Товары, требующие пополнения",
        "total_stock_deficit": "Общий дефицит запасов",
        "percent_needing_restock": "% товаров, требующих пополнения",
        "target_multiplier_info": "**Текущий целевой множитель: {0}x**\n\nЭтот множитель влияет на то, насколько агрессивно вы хотите пополнять запасы. Более высокие значения означают:\n- Больше товаров будет отмечено как требующие пополнения\n- Большие количества будут рекомендованы для закупки\n- Меньший риск нехватки запасов, но потенциально более высокие затраты на хранение\n\nНастройте ползунок на боковой панели и наблюдайте, как это влияет на ваши потребности в пополнении запасов.",
        "select_all_warehouses": "Выбрать все склады",
        "select_warehouses": "Выбрать склады",
        "select_all_brands": "Выбрать все бренды",
        "select_brands": "Выбрать бренды",
        "select_all_categories": "Выбрать все категории",
        "select_categories": "Выбрать категории",
        "select_all_subjects": "Выбрать все предметы",
        "select_subjects": "Выбрать предметы",
        "select_all_products": "Выбрать все товары",
        "select_product_ids": "Выбрать ID товаров",
        "select_all_sizes": "Выбрать все размеры",
        "select_sizes": "Выбрать размеры",
        "stock_deficit_overview": "Обзор дефицита запасов",
        "no_warehouses_deficit": "В текущих отфильтрованных данных не найдено складов с дефицитом запасов.",
        "warehouses_with_deficit": "Склады с дефицитом запасов",
        "warehouse_name": "Название склада",
        "products_with_deficit": "Товары с дефицитом",
        "total_deficit": "Общий дефицит",
        "detailed_data_header": "Подробные данные - товары, требующие пополнения",
        "no_products_matching": "Нет товаров, соответствующих вашим критериям фильтра, требующих пополнения.",
        "download_button": "Скачать данные о пополнении товаров",
        "database_query_header": "Запрос к базе данных",
        "run_custom_query": "Выполнить пользовательский запрос PostgreSQL",
        "enter_sql_query": "Введите SQL-запрос",
        "execute_query": "Выполнить запрос",
        "query_results": "Результаты запроса",
        "download_query_results": "Скачать результаты запроса",
        "footer_text": "Упрощенная панель дефицита складских запасов | Интеграция PostgreSQL",
        "refresh_data": "Обновить данные",
        "data_refreshed": "Данные успешно обновлены!",
        "last_refresh": "Последнее обновление: {0}",
        "cron_refresh": "Последнее автоматическое обновление: {0}",
        "sales_orders_tab": "Продажи и заказы",
        "total_orders_label": "Общее количество заказов",
        "total_sales_label": "Общий объем продаж",
        "date_label": "Дата",
        "count_label": "Количество",
        "legend_title": "Показатель",
        "sale_order_ratio_trend_header": "Тренд отношения продаж к заказам",
        "sale_order_ratio_label": "Отношение продаж к заказам",
        "ratio_label": "Отношение",
        "metric_label": "Показатель",
        "ratio_data_missing": "Данные для расчета отношения продаж к заказам отсутствуют.",
        "shipment_tab": "Информация о доставке",
        "shipment_title": "Анализ доставки на склады",
        "avg_processing_days": "Сред. время обработки (дни)",
        "max_processing_days": "Макс. время обработки (дни)",
        "shipment_count": "Кол-во поставок",
        "warehouse_performance": "Производительность складов",
        "processing_time": "Время обработки (дни)",
        "efficiency_rating": "Рейтинг эффективности",
        "restocking_priorities": "Приоритеты пополнения",
        "priority_score": "Приоритетный балл",
        "product_id": "ID товара",
        "processing_speed": "Скорость обработки",
        "no_shipment_data": "Данные о поставках отсутствуют.",
        "efficiency_distribution": "Распределение рейтинга эффективности",
        "rating": "Рейтинг"
    }
}

# Set page configuration
st.set_page_config(page_title="Warehouse Analytics Dashboard", layout="wide")

# Path to configuration file - store in the same directory as the script
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_config.json')

# Check authentication before showing the dashboard
if not check_password():
    st.stop()  # Stop execution if not authenticated

# If we reach here, user is authenticated - show the dashboard

# Supabase connection parameters from Streamlit secrets
@st.cache_resource
def get_db_connection_params():
    try:
        # Print available secrets keys for debugging (remove in production)
        print("Available secrets keys:", list(st.secrets.keys()))
        
        return {
            "host": st.secrets["SUPABASE_HOST"],
            "database": st.secrets.get("SUPABASE_DATABASE", "postgres"),
            "user": st.secrets.get("SUPABASE_USER", "postgres"),
            "password": st.secrets["SUPABASE_PASSWORD"],
            "port": st.secrets.get("SUPABASE_PORT", "5432")
        }
    except KeyError as e:
        st.error(f"""
        ## Missing Streamlit Secrets! ({str(e)})
        This app requires database credentials in Streamlit secrets.
        
        ### For local development:
        1. Create a `.streamlit/secrets.toml` file in your project root
        2. Add your Supabase database credentials (see template in sidebar)
        
        ### For Streamlit Cloud deployment:
        Add your credentials to the Streamlit Cloud dashboard under "Secrets"
        """)
        st.stop()

# Load saved configuration if exists
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except:
            return {"sales_target_multiplier": 1.5, "language": "English"}
    return {"sales_target_multiplier": 1.5, "language": "English"}

# Save configuration
def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f)

# Load current config
config = load_config()

# Language selector in the top right
col1, col2 = st.columns([5, 1])
with col2:
    selected_language = st.selectbox(
        "Language / Язык",
        options=["English", "Russian"],
        index=0 if config.get("language", "English") == "English" else 1
    )
    if selected_language != config.get("language", "English"):
        config["language"] = selected_language
        save_config(config)
        st.rerun()

# Get translations for the selected language
t = translations[selected_language]

# Create dashboard title
with col1:
    st.title(t["dashboard_title"])
    st.markdown(t["dashboard_subtitle"])

# Create tabs
tab1_title = t["stock_deficit_overview"] # Or a more specific title for the first tab if needed
tab2_title = t["sales_orders_tab"]
tab3_title = t["shipment_tab"]
tab1, tab2, tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

with tab1:
    # Add refresh button at the top
    refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 2, 1])
    with refresh_col1:
        if st.button(t["refresh_data"]):
            # Clear all cached data
            st.cache_data.clear()
            try:
                # In standalone version, we'll just refresh the cached data
                st.session_state["last_refresh_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(t["data_refreshed"])
                time.sleep(1)  # Give user time to see the success message
                st.rerun()  # Rerun the app to show fresh data
            except Exception as e:
                st.error(f"Error refreshing data: {e}")

    # Show last refresh time if available
    with refresh_col3:
        if "last_refresh_time" in st.session_state:
            st.info(t["last_refresh"].format(st.session_state["last_refresh_time"]))
        else:
            st.session_state["last_refresh_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.info(t["last_refresh"].format(st.session_state["last_refresh_time"]))

    # Add auto-refresh option
    with refresh_col2:
        auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=False)
        if auto_refresh:
            # Add auto-refresh using HTML meta tag
            refresh_rate = 300  # 5 minutes in seconds
            st.markdown(f"""
                <meta http-equiv="refresh" content="{refresh_rate}">
            """, unsafe_allow_html=True)
            if "last_auto_refresh" not in st.session_state or \
               (datetime.now() - datetime.strptime(st.session_state["last_auto_refresh"], "%Y-%m-%d %H:%M:%S")).total_seconds() > refresh_rate:
                st.session_state["last_auto_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.cache_data.clear()

    # Create sidebar filters
    st.sidebar.header(t["filters_header"])

    # Add sales target multiplier slider
    st.sidebar.header(t["settings_header"])
    sales_target_multiplier = st.sidebar.slider(
        t["sales_target_label"],
        min_value=0.5,
        max_value=3.0,
        value=config.get("sales_target_multiplier", 1.5),
        step=0.1,
        help=t["sales_target_help"]
    )

    # Add save button for settings
    if st.sidebar.button(t["save_settings"]):
        config["sales_target_multiplier"] = sales_target_multiplier
        config["language"] = selected_language
        save_config(config)
        st.sidebar.success(t["settings_saved"])

    # Function to load data from Supabase (PostgreSQL)
    @st.cache_data(ttl=300)  # Cache expires after 5 minutes
    def load_data_from_supabase(sales_target_multiplier=1.5):
        try:
            # Get connection parameters from secrets
            db_params = get_db_connection_params()
            
            # Connect to the Supabase PostgreSQL database
            conn = psycopg2.connect(
                host=db_params["host"],
                database=db_params["database"],
                user=db_params["user"],
                password=db_params["password"],
                port=db_params["port"]
            )
            
            # Create the query with the parametric sales target multiplier
            query = f"""
            WITH current_month_sales AS (
                SELECT 
                    warehouse_name, nm_id, tech_size,
                    COUNT(*) AS current_month_sales,
                    SUM(price_with_disc) AS current_month_amount
                FROM belara_silver.sales
                WHERE EXTRACT(YEAR FROM last_change_date) = EXTRACT(YEAR FROM CURRENT_DATE) AND EXTRACT(MONTH FROM last_change_date) = EXTRACT(MONTH FROM CURRENT_DATE)
                GROUP BY warehouse_name, nm_id, tech_size
            ),
            last_month_sales AS (
                SELECT 
                    warehouse_name, nm_id, tech_size,
                    COUNT(*) AS last_month_sales,
                    SUM(price_with_disc) AS last_month_amount
                FROM belara_silver.sales
                WHERE 
                    (EXTRACT(MONTH FROM CURRENT_DATE) = 1 AND EXTRACT(YEAR FROM last_change_date) = EXTRACT(YEAR FROM CURRENT_DATE) - 1 AND EXTRACT(MONTH FROM last_change_date) = 12) OR
                    (EXTRACT(MONTH FROM CURRENT_DATE) > 1 AND EXTRACT(YEAR FROM last_change_date) = EXTRACT(YEAR FROM CURRENT_DATE) AND EXTRACT(MONTH FROM last_change_date) = EXTRACT(MONTH FROM CURRENT_DATE) - 1)
                GROUP BY warehouse_name, nm_id, tech_size
            ),
            current_stock AS (
                SELECT 
                    warehouse_name, nm_id, supplier_article, barcode, category, subject, brand, tech_size,
                    quantity AS current_stock, quantity_full AS total_stock,
                    in_way_to_client AS in_delivery, in_way_from_client AS in_return, price AS avg_price
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY warehouse_name, nm_id, tech_size ORDER BY last_change_date DESC) as rn
                    FROM belara_silver.warehouse
                ) ranked
                WHERE rn = 1
            )
            SELECT 
                cs.warehouse_name as "warehouseName", 
                cs.nm_id as "nmId", 
                cs.supplier_article as "supplierArticle", 
                cs.barcode, 
                cs.category, 
                cs.subject, 
                cs.brand, 
                cs.tech_size as "techSize",
                cs.current_stock, 
                cs.total_stock, 
                cs.in_delivery, 
                cs.in_return,
                COALESCE(cms.current_month_sales, 0) AS sales_this_month,
                COALESCE(cms.current_month_amount, 0) AS amount_this_month,
                COALESCE(lms.last_month_sales, 0) AS sales_last_month,
                COALESCE(lms.last_month_amount, 0) AS amount_last_month,
                COALESCE(lms.last_month_sales, 0) * {sales_target_multiplier} AS sales_target,
                CASE 
                    WHEN cs.current_stock + cs.in_return < (COALESCE(lms.last_month_sales, 0) * {sales_target_multiplier}) THEN TRUE
                    ELSE FALSE
                END AS needs_restock,
                CEIL((COALESCE(lms.last_month_sales, 0) * {sales_target_multiplier}) - (cs.current_stock + cs.in_return)) AS stock_deficit,
                (COALESCE(lms.last_month_sales, 0) * {sales_target_multiplier}) - (cs.current_stock + cs.in_return) AS sort_key
            FROM current_stock cs
            LEFT JOIN current_month_sales cms
                ON cs.warehouse_name = cms.warehouse_name AND cs.nm_id = cms.nm_id AND cs.tech_size = cms.tech_size
            LEFT JOIN last_month_sales lms
                ON cs.warehouse_name = lms.warehouse_name AND cs.nm_id = lms.nm_id AND cs.tech_size = lms.tech_size
            WHERE cs.current_stock IS NOT NULL
            ORDER BY 
                cs.warehouse_name,
                sort_key DESC
            """
            
            # Execute the query and load into DataFrame
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
            
            # Close the connection
            conn.close()
            
            return df
        except Exception as e:
            st.error(f"Error connecting to Supabase: {e}")
            # Return a sample dataframe with the correct columns as a fallback
            return pd.DataFrame(columns=[
                'warehouseName', 'nmId', 'techSize', 'brand', 'category', 
                'subject', 'current_stock', 'in_delivery', 'in_return', 
                'sales_last_month', 'sales_target', 'stock_deficit'
            ])

    # Function to load data from product_restock with filters (using Supabase)
    @st.cache_data(ttl=300)  # Cache expires after 5 minutes
    def load_product_restock_data(
        sales_target_multiplier=1.5, 
        warehouses=None, 
        brands=None, 
        categories=None, 
        subjects=None,
        products=None,
        sizes=None
    ):
        try:
            # Get connection parameters from secrets
            db_params = get_db_connection_params()
            
            # Connect to the Supabase PostgreSQL database
            conn = psycopg2.connect(
                host=db_params["host"],
                database=db_params["database"],
                user=db_params["user"],
                password=db_params["password"],
                port=db_params["port"]
            )
            
            # Build filter conditions
            filter_conditions = []
            
            if warehouses and len(warehouses) > 0:
                warehouse_list = ", ".join([f"'{w}'" for w in warehouses])
                filter_conditions.append(f"warehouse_name IN ({warehouse_list})")
                
            if brands and len(brands) > 0:
                brand_list = ", ".join([f"'{b}'" for b in brands])
                filter_conditions.append(f"brand IN ({brand_list})")
                
            if categories and len(categories) > 0:
                category_list = ", ".join([f"'{c}'" for c in categories])
                filter_conditions.append(f"category IN ({category_list})")
                
            if subjects and len(subjects) > 0:
                subject_list = ", ".join([f"'{s}'" for s in subjects])
                filter_conditions.append(f"subject IN ({subject_list})")
                
            if products and len(products) > 0:
                product_list = ", ".join([str(p) for p in products])
                filter_conditions.append(f"nm_id IN ({product_list})")
                
            if sizes and len(sizes) > 0:
                size_list = ", ".join([f"'{s}'" for s in sizes])
                filter_conditions.append(f"tech_size IN ({size_list})")
            
            # Combine all filter conditions
            where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"
            
            # Create the query
            query = f"""
            SELECT 
                warehouse_name AS "warehouseName",
                nm_id AS "nmId", 
                tech_size AS "techSize",
                brand,
                category,
                subject,
                current_stock,
                in_delivery,
                in_return,
                sales_last_month,
                sales_last_month * {sales_target_multiplier} AS sales_target,
                CEIL((sales_last_month * {sales_target_multiplier}) - (current_stock + in_return)) AS stock_deficit,
                CASE 
                    WHEN current_stock < (sales_last_month * {sales_target_multiplier}) THEN TRUE
                    ELSE FALSE
                END AS needs_restock
            FROM belara_gold_marts.product_restock
            WHERE {where_clause}
            ORDER BY warehouse_name, stock_deficit DESC
            """
            
            # Execute the query and load into DataFrame
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
            
            # Close the connection
            conn.close()
            
            return df
        except Exception as e:
            st.error(f"Error querying product_restock table: {e}")
            return pd.DataFrame(columns=[
                'warehouseName', 'nmId', 'techSize', 'brand', 'category', 
                'subject', 'current_stock', 'in_delivery', 'in_return', 
                'sales_last_month', 'sales_target', 'stock_deficit'
            ])

    # Load the data with the selected multiplier (using Supabase instead of DuckDB)
    df = load_data_from_supabase(sales_target_multiplier)

    # Check if data was loaded successfully
    if df.empty:
        st.error("No data was loaded from Supabase. Please check your database connection.")
        st.stop()

    # Add a metrics card to show the impact of the multiplier
    st.header(t["impact_header"].format(sales_target_multiplier))

    # Calculate metrics based on the current multiplier
    total_products = len(df['nmId'].unique())
    total_deficit_items = len(df[df['needs_restock'] == True]['nmId'].unique())
    total_deficit_quantity = df[df['needs_restock'] == True]['stock_deficit'].sum()
    deficit_percentage = (total_deficit_items / total_products * 100) if total_products > 0 else 0

    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t["total_products"], f"{total_products}")
    with col2:
        st.metric(t["products_needing_restock"], f"{total_deficit_items}")
    with col3:
        st.metric(t["total_stock_deficit"], f"{int(total_deficit_quantity)}")
    with col4:
        st.metric(t["percent_needing_restock"], f"{deficit_percentage:.1f}%")

    # Create a note about the multiplier impact
    st.info(t["target_multiplier_info"].format(sales_target_multiplier))

    # Create warehouse filter
    warehouses = sorted(df['warehouseName'].unique())
    select_all_warehouses = st.sidebar.checkbox(t["select_all_warehouses"], value=True)
    if select_all_warehouses:
        selected_warehouses = st.sidebar.multiselect(
            t["select_warehouses"],
            options=warehouses,
            default=warehouses
        )
    else:
        selected_warehouses = st.sidebar.multiselect(
            t["select_warehouses"],
            options=warehouses,
            default=[]
        )

    # Create brand filter
    brands = sorted(df['brand'].unique())
    select_all_brands = st.sidebar.checkbox(t["select_all_brands"], value=True)
    if select_all_brands:
        selected_brands = st.sidebar.multiselect(
            t["select_brands"],
            options=brands,
            default=brands
        )
    else:
        selected_brands = st.sidebar.multiselect(
            t["select_brands"],
            options=brands,
            default=[]
        )

    # Create category filter
    categories = sorted(df['category'].unique())
    select_all_categories = st.sidebar.checkbox(t["select_all_categories"], value=True)
    if select_all_categories:
        selected_categories = st.sidebar.multiselect(
            t["select_categories"],
            options=categories,
            default=categories
        )
    else:
        selected_categories = st.sidebar.multiselect(
            t["select_categories"],
            options=categories,
            default=[]
        )

    # Create subject filter
    subjects = sorted(df['subject'].unique())
    select_all_subjects = st.sidebar.checkbox(t["select_all_subjects"], value=True)
    if select_all_subjects:
        selected_subjects = st.sidebar.multiselect(
            t["select_subjects"],
            options=subjects,
            default=subjects
        )
    else:
        selected_subjects = st.sidebar.multiselect(
            t["select_subjects"],
            options=subjects,
            default=[]
        )

    # Create product filter
    products = sorted(df['nmId'].unique())
    select_all_products = st.sidebar.checkbox(t["select_all_products"], value=True)
    if select_all_products:
        selected_products = st.sidebar.multiselect(
            t["select_product_ids"],
            options=products,
            default=products
        )
    else:
        selected_products = st.sidebar.multiselect(
            t["select_product_ids"],
            options=products,
            default=products[:5] if len(products) > 5 else products
        )

    # Create size filter
    sizes = sorted(df['techSize'].unique())
    select_all_sizes = st.sidebar.checkbox(t["select_all_sizes"], value=True)
    if select_all_sizes:
        selected_sizes = st.sidebar.multiselect(
            t["select_sizes"],
            options=sizes,
            default=sizes
        )
    else:
        selected_sizes = st.sidebar.multiselect(
            t["select_sizes"],
            options=sizes,
            default=[]
        )

    # Filter the data based on selections
    filtered_df = df[
        (df['warehouseName'].isin(selected_warehouses)) &
        (df['brand'].isin(selected_brands)) &
        (df['category'].isin(selected_categories)) &
        (df['subject'].isin(selected_subjects)) &
        (df['nmId'].isin(selected_products)) &
        (df['techSize'].isin(selected_sizes))
    ]

    # Create a row for key metrics
    st.header(t["stock_deficit_overview"])

    # Find warehouses with stock deficits
    warehouses_with_deficit = filtered_df[filtered_df['needs_restock'] == True].copy()

    if warehouses_with_deficit.empty:
        st.info(t["no_warehouses_deficit"])
    else:
        # Group by warehouse and count products with deficit
        deficit_counts = warehouses_with_deficit.groupby('warehouseName').agg(
            deficit_product_count=('nmId', 'nunique'),
            total_deficit=('stock_deficit', 'sum')
        ).reset_index().sort_values('deficit_product_count', ascending=False)
        
        # Create table for warehouses with deficit
        st.subheader(t["warehouses_with_deficit"])
        
        # Use Plotly table for better visualization
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[t["warehouse_name"], t["products_with_deficit"], t["total_deficit"]],
                fill_color='grey',
                align='left',
                font=dict(color='black', size=14)
            ),
            cells=dict(
                values=[
                    deficit_counts['warehouseName'],
                    deficit_counts['deficit_product_count'],
                    deficit_counts['total_deficit'].astype(float).round(1)
                ],
                fill_color=[['grey' if i % 2 == 0 else 'white' for i in range(len(deficit_counts))]],
                align='left',
                font=dict(color='black', size=13)
            )
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Detailed data - replaced filtered_df with product_restock data
    st.header(t["detailed_data_header"])

    # Load product_restock data with filters
    product_restock_data = load_product_restock_data(
        sales_target_multiplier=sales_target_multiplier,
        warehouses=selected_warehouses,
        brands=selected_brands,
        categories=selected_categories,
        subjects=selected_subjects,
        products=selected_products,
        sizes=selected_sizes
    )

    if product_restock_data.empty:
        st.info(t["no_products_matching"])
    else:
        st.dataframe(
            product_restock_data.style.highlight_max(
                axis=0, 
                subset=['stock_deficit', 'sales_last_month', 'in_delivery']
            ), 
            use_container_width=True
        )

        # Optional - download filtered product_restock data
        csv = product_restock_data.to_csv(index=False)
        st.download_button(
            label=t["download_button"],
            data=csv,
            file_name="product_restock_filtered.csv",
            mime="text/csv",
        )

    # Add PostgreSQL direct query section
    st.header(t["database_query_header"])
    with st.expander(t["run_custom_query"], expanded=False):
        direct_query = st.text_area(
            t["enter_sql_query"],
            f"""
            WITH filtered_data AS (
                SELECT 
                    warehouse_name as "warehouseName",
                    nm_id as "nmId",
                    tech_size as "techSize",
                    current_stock,
                    sales_last_month,
                    sales_last_month * {sales_target_multiplier} as sales_target,
                    CASE 
                        WHEN current_stock < (sales_last_month * {sales_target_multiplier}) THEN TRUE
                        ELSE FALSE
                    END AS needs_restock,
                    CEIL((sales_last_month * {sales_target_multiplier}) - current_stock) AS stock_deficit
                FROM belara_gold_marts.product_restock
            )
            SELECT 
                "warehouseName",
                COUNT(*) as deficit_product_count,
                SUM(stock_deficit) as total_deficit
            FROM filtered_data
            WHERE needs_restock = TRUE
            GROUP BY "warehouseName"
            ORDER BY deficit_product_count DESC
            """,
            height=200
        )
        
        if st.button(t["execute_query"]):
            try:
                # Get connection parameters from secrets
                db_params = get_db_connection_params()
                
                conn = psycopg2.connect(
                    host=db_params["host"],
                    database=db_params["database"],
                    user=db_params["user"],
                    password=db_params["password"],
                    port=db_params["port"]
                )
                
                # Execute the query and load into DataFrame
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    cursor.execute(direct_query)
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    result_df = pd.DataFrame(data, columns=columns)
                
                conn.close()
                
                # Display the query results
                st.subheader(t["query_results"])
                
                # Show dataframe for interactive features
                st.dataframe(result_df, use_container_width=True)
                
                # Option to download query results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label=t["download_query_results"],
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Query execution error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(t["footer_text"])

    # Function to generate a secrets.toml template file
    def generate_secrets_template():
        """
        Generates a template for the .streamlit/secrets.toml file
        that needs to be created locally for development and
        added to Streamlit Cloud for deployment.
        """
        secrets_content = """# .streamlit/secrets.toml

# Supabase PostgreSQL Database Credentials
SUPABASE_HOST = "db.your-project-id.supabase.co"
SUPABASE_DATABASE = "postgres" 
SUPABASE_USER = "postgres"
SUPABASE_PASSWORD = "your-database-password"
SUPABASE_PORT = "5432"

# Add other secrets as needed
"""
        return secrets_content

    # Only show in development environment
    if st.sidebar.checkbox("Show Secrets Template", value=False):
        st.sidebar.code(generate_secrets_template(), language="toml")
        st.sidebar.warning(
            "⚠️ Create a .streamlit/secrets.toml file with your actual credentials.\n"
            "⚠️ Add .streamlit/secrets.toml to your .gitignore file to prevent exposing secrets."
        )

with tab2:
    st.header(t["sales_orders_tab"])

    @st.cache_data(ttl=300)  # Cache expires after 5 minutes
    def load_sales_orders_data():
        try:
            # Get connection parameters from secrets
            db_params = get_db_connection_params()
            
            conn = psycopg2.connect(
                host=db_params["host"],
                database=db_params["database"],
                user=db_params["user"],
                password=db_params["password"],
                port=db_params["port"]
            )
            
            query = """
                SELECT 
                    CAST(summary_date AS DATE) AS "Date",  -- Ensure Date is treated as date type
                    total_orders AS "Total Orders", 
                    total_sales AS "Total Sales",
                    sale_order_ratio AS "Sale_Order_Ratio"  -- Added Sale_Order_Ratio
                FROM belara_gold_marts.daily_sales_orders_summary
                ORDER BY summary_date DESC
            """
            
            # Execute the query and load into DataFrame
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
            
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error loading sales and orders data: {e}")
            return pd.DataFrame(columns=["Date", "Total Orders", "Total Sales", "Sale_Order_Ratio"])

    sales_orders_df = load_sales_orders_data()

    if sales_orders_df.empty:
        st.info("No sales or order data found.")
    else:
        # Ensure 'Date' is in datetime format for proper sorting and display
        sales_orders_df["Date"] = pd.to_datetime(sales_orders_df["Date"])
        sales_orders_df = sales_orders_df.sort_values(by="Date")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=sales_orders_df["Date"],
            y=sales_orders_df["Total Orders"],
            name=t.get("total_orders_label", "Total Orders"), # Assuming you'll add this to translations
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            x=sales_orders_df["Date"],
            y=sales_orders_df["Total Sales"],
            name=t.get("total_sales_label", "Total Sales"), # Assuming you'll add this to translations
            marker_color='darkblue'
        ))

        fig.update_layout(
            barmode='group', # Group bars for orders and sales side-by-side for each date
            xaxis_title=t.get("date_label", "Date"), # Assuming you'll add this to translations
            yaxis_title=t.get("count_label", "Count"), # Assuming you'll add this to translations
            legend_title_text=t.get("legend_title", "Metric") # Assuming you'll add this to translations
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Display the dataframe as well, if needed, or remove this line
        # st.dataframe(sales_orders_df, use_container_width=True) # Commented out as per potential previous discussion

        # Second chart: Sale_Order_Ratio over Date (Line Chart)
        if "Sale_Order_Ratio" in sales_orders_df.columns:
            st.subheader(t.get("sale_order_ratio_trend_header", "Sale/Order Ratio Trend")) # Placeholder for translation
            
            fig_ratio = go.Figure()

            fig_ratio.add_trace(go.Scatter(
                x=sales_orders_df["Date"],
                y=sales_orders_df["Sale_Order_Ratio"],
                mode='lines+markers',
                name=t.get("sale_order_ratio_label", "Sale/Order Ratio"), # Placeholder for translation
                marker_color='green'
            ))

            fig_ratio.update_layout(
                xaxis_title=t.get("date_label", "Date"), # Reusing existing translation
                yaxis_title=t.get("ratio_label", "Ratio"), # Placeholder for translation
                legend_title_text=t.get("metric_label", "Metric") # Placeholder for translation
            )
            
            st.plotly_chart(fig_ratio, use_container_width=True)
        else:
            st.info(t.get("ratio_data_missing", "Sale_Order_Ratio data not available to display trend."))

        # Display the dataframe with all columns including Sale_Order_Ratio, if needed for debugging or full view
        st.dataframe(sales_orders_df, use_container_width=True)

with tab3:
    st.header(t["shipment_title"])
    
    @st.cache_data(ttl=300)  # Cache expires after 5 minutes
    def load_warehouse_performance_data():
        try:
            # Get connection parameters from secrets
            db_params = get_db_connection_params()
            
            conn = psycopg2.connect(
                host=db_params["host"],
                database=db_params["database"],
                user=db_params["user"],
                password=db_params["password"],
                port=db_params["port"]
            )
            
            # Query for warehouse performance metrics
            query = """
                SELECT 
                    warehouse_name,
                    avg_days_to_accept,
                    min_days_to_accept,
                    max_days_to_accept,
                    shipment_count
                FROM belara_gold_marts.recent_warehouse_performance
                ORDER BY avg_days_to_accept ASC
            """
            
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                performance_df = pd.DataFrame(data, columns=columns)
            
            # Query for efficiency distribution
            query_efficiency = """
                SELECT 
                    processing_speed_rating as efficiency_rating,
                    COUNT(*) as warehouse_count
                FROM belara_gold_marts.restocking_efficiency
                GROUP BY processing_speed_rating
                ORDER BY CASE 
                    WHEN processing_speed_rating = 'Excellent' THEN 1
                    WHEN processing_speed_rating = 'Good' THEN 2
                    WHEN processing_speed_rating = 'Average' THEN 3
                    WHEN processing_speed_rating = 'Below Average' THEN 4
                    WHEN processing_speed_rating = 'Poor' THEN 5
                    ELSE 6
                END
            """
            
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query_efficiency)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                efficiency_df = pd.DataFrame(data, columns=columns)
            
            # Query for product shipping analysis with priority
            query_priorities = """
                SELECT 
                    nm_id as product_id,
                    processing_speed,
                    priority as priority_score,
                    warehouse_name
                FROM belara_gold_marts.product_shipping_analysis
                ORDER BY priority DESC
                LIMIT 15
            """
            
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query_priorities)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                priorities_df = pd.DataFrame(data, columns=columns)
            
            conn.close()
            return performance_df, efficiency_df, priorities_df
            
        except Exception as e:
            st.error(f"Error loading shipment data: {e}")
            return (pd.DataFrame(columns=["warehouse_name", "avg_days_to_accept", "min_days_to_accept", 
                                         "max_days_to_accept", "shipment_count"]),
                   pd.DataFrame(columns=["efficiency_rating", "warehouse_count"]),
                   pd.DataFrame(columns=["product_id", "processing_speed", "priority_score", "warehouse_name"]))

    # Load the shipment data
    performance_df, efficiency_df, priorities_df = load_warehouse_performance_data()
    
    if performance_df.empty:
        st.info(t["no_shipment_data"])
    else:
        # Visualization 1: Warehouse Performance Metrics
        st.subheader(t["warehouse_performance"])
        
        # Ensure numeric data types for visualization
        performance_df['avg_days_to_accept'] = pd.to_numeric(performance_df['avg_days_to_accept'])
        performance_df['max_days_to_accept'] = pd.to_numeric(performance_df['max_days_to_accept'])
        
        # Debug: display data types (only show if debug mode is enabled)
        debug_mode = False
        if debug_mode:
            st.write("Data types:", performance_df.dtypes)
        
        # Convert DataFrame to long format for Plotly Express
        try:
            performance_long_df = pd.melt(
                performance_df,
                id_vars=['warehouse_name'],
                value_vars=['avg_days_to_accept', 'max_days_to_accept'],
                var_name='metric',
                value_name='days'
            )
            
            # Map the metric names to their translated labels
            metric_mapping = {
                'avg_days_to_accept': t["avg_processing_days"],
                'max_days_to_accept': t["max_processing_days"]
            }
            performance_long_df['metric'] = performance_long_df['metric'].map(metric_mapping)
            
            # Create a bar chart for processing times by warehouse using the long format DataFrame
            fig1 = px.bar(
                performance_long_df,
                x="warehouse_name",
                y="days",
                color="metric",
                barmode="group",
                labels={
                    "warehouse_name": t["warehouse_name"],
                    "days": t["processing_time"],
                    "metric": t["metric_label"]
                },
                color_discrete_sequence=["#2C82C9", "#EF4836"],
                height=400
            )
            
            fig1.update_layout(
                xaxis_tickangle=-45,
                legend_title=t["metric_label"]
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating warehouse performance chart: {e}")
            # Fallback to simple table display if chart fails
            st.dataframe(performance_df, use_container_width=True)
        
        # Visualization 2: Efficiency Rating Distribution
        if not efficiency_df.empty:
            st.subheader(t["efficiency_distribution"])
            
            # Define colors based on rating
            colors = {
                'Excellent': '#27AE60',
                'Good': '#2ECC71',
                'Average': '#F1C40F',
                'Below Average': '#E67E22',
                'Poor': '#E74C3C'
            }
            
            # Create color list based on efficiency_rating
            color_list = [colors.get(rating, '#95A5A6') for rating in efficiency_df['efficiency_rating']]
            
            # Create a pie chart for efficiency rating distribution
            fig2 = px.pie(
                efficiency_df,
                values='warehouse_count',
                names='efficiency_rating',
                color='efficiency_rating',
                color_discrete_map={rating: colors.get(rating, '#95A5A6') for rating in efficiency_df['efficiency_rating']},
                title=t["efficiency_distribution"],
                labels={
                    'warehouse_count': t["shipment_count"],
                    'efficiency_rating': t["rating"]
                }
            )
            
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Visualization 3: Product Restocking Priorities
        if not priorities_df.empty:
            st.subheader(t["restocking_priorities"])
            
            # Create horizontal bar chart for top priority products
            fig3 = px.bar(
                priorities_df,
                y='product_id',
                x='priority_score',
                color='processing_speed',
                orientation='h',
                labels={
                    'product_id': t["product_id"],
                    'priority_score': t["priority_score"],
                    'processing_speed': t["processing_speed"],
                    'warehouse_name': t["warehouse_name"]
                },
                hover_data=['warehouse_name'],
                color_discrete_sequence=px.colors.sequential.Viridis,
                height=500
            )
            
            fig3.update_layout(
                xaxis_title=t["priority_score"],
                yaxis_title=t["product_id"],
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig3, use_container_width=True)