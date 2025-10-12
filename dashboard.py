import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import psycopg2.extras
import os
import sys
import json
from datetime import datetime
import time
from dotenv import load_dotenv

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
        "download_sales_data": "Download Sales Data",
        "download_shipment_performance_data": "Download Shipment Performance Data",
        "download_shipment_priorities_data": "Download Restocking Priorities Data",
        "footer_text": "Warehouse Stock Deficit Dashboard",
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
        "rating": "Rating",
        "warehouse_priority_distribution": "Warehouse Priority Distribution",
        "priority_product_distribution": "Priority Product Distribution by Warehouse",
        "unique_product_count": "Product Count",
        "hero_products_tab": "Hero Products List",
        "hero_products_title": "Hero Products Analysis",
        "hero_products_subtitle": "Top selling products by sales amount and quantity",
        "top_n_products_chart": "Top {0} Hero Products by Sales Amount",
        "hero_products_table": "Hero Products Data",
        "sales_amount": "Sales Amount",
        "sales_quantity": "Sales Quantity",
        "no_hero_products": "No hero products found for the selected filters.",
        "download_hero_products_data": "Download Hero Products Data"
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
        "download_sales_data": "Скачать данные о продажах",
        "download_shipment_performance_data": "Скачать данные о производительности доставки",
        "download_shipment_priorities_data": "Скачать данные о приоритетах пополнения",
        "footer_text": "Панель дефицита складских запасов",
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
        "rating": "Рейтинг",
        "warehouse_priority_distribution": "Распределение приоритетов склада",
        "priority_product_distribution": "Распределение товаров по приоритетам склада",
        "unique_product_count": "Количество товаров",
        "hero_products_tab": "Список продуктов-героев",
        "hero_products_title": "Анализ продуктов-героев",
        "hero_products_subtitle": "Самые продаваемые товары по сумме и количеству продаж",
        "top_n_products_chart": "Топ {0} продуктов-героев по сумме продаж",
        "hero_products_table": "Данные о продуктах-героях",
        "sales_amount": "Сумма продаж",
        "sales_quantity": "Количество продаж",
        "no_hero_products": "Продукты-герои для выбранных фильтров не найдены.",
        "download_hero_products_data": "Скачать данные о продуктах-героях"
    }
}

# Load environment variables
load_dotenv()

def get_environment():
    """
    Centralized function to determine the running environment.
    Returns:
        dict: Environment configuration with keys:
            - is_cloud: bool, True if running on Streamlit Cloud
            - has_https: bool, True if HTTPS is available
    """
    try:
        # Better cloud deployment detection - check multiple environment indicators
        is_cloud = (os.environ.get('STREAMLIT_SHARING', '') == 'true' or 
                   os.environ.get('IS_STREAMLIT_CLOUD', '') == 'true' or
                   os.environ.get('STREAMLIT_RUNTIME_ENVIRONMENT', '') == 'cloud')
        
        return {
            "is_cloud": is_cloud,
            "has_https": is_cloud  # Streamlit Cloud always has HTTPS
        }
    except Exception as e:
        st.error(f"Environment detection error: {e}")
        return {
            "is_cloud": False,
            "has_https": False
        }

# Store environment configuration once
ENVIRONMENT = get_environment()

# Authentication functions
def check_password():
    """
    Authenticate users using Streamlit's built-in security features.
    
    This function implements a secure authentication system using Streamlit's native 
    security features instead of custom password hashing:
    
    1. Credentials Management:
       - Username and password are stored securely in Streamlit's secrets management
       - Access credentials via st.secrets["DASHBOARD_USERNAME"] and st.secrets["DASHBOARD_PASSWORD"]
       - No need for additional password hashing as Streamlit handles security
    
    2. Session Management:
       - Uses Streamlit's session state to track authentication status
       - 30-minute session timeout for security
       - Session state is cleared on logout
    
    3. Security Features:
       - Tracks failed login attempts (maximum 3 attempts)
       - 5-minute account lockout after exceeding maximum attempts
       - Clear feedback messages for users
       - Automatic session expiry for inactive users
    
    Setup Instructions:
    1. Create `.streamlit/secrets.toml` in project root
    2. Add credentials:
       ```toml
       DASHBOARD_USERNAME = "admin"
       DASHBOARD_PASSWORD = "your-secure-password"
       ```
    3. For deployment, add these credentials in Streamlit Cloud dashboard under "Secrets"
    
    Returns:
        bool: True if user is authenticated, False otherwise
    """
    
    # Handle session persistence using URL parameters in a secure way
    params = st.query_params
    
    # Only use URL parameters if running on Cloud (HTTPS) or locally
    if params.get("authenticated") == "true":
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = True
            auth_time = params.get("auth_time", str(time.time()))
            st.session_state.authenticated_time = float(auth_time)
            
            # Add warning if HTTPS is not available
            if not ENVIRONMENT["has_https"]:
                st.warning("⚠️ For maximum security, deploy this dashboard on Streamlit Cloud where HTTPS is enabled.")
    
    # Initialize session state variables if they don't exist
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "last_attempt_time" not in st.session_state:
        st.session_state.last_attempt_time = 0
    if "authenticated_time" not in st.session_state:
        st.session_state.authenticated_time = 0
    
    # Check for session expiry (30 minutes)
    if st.session_state.get("authenticated"):
        if time.time() - st.session_state.authenticated_time > 1800:  # 30 minutes
            st.session_state.authenticated = False
            st.warning("Your session has expired. Please login again.")
            return False
        return True
    
    # Check for temporary lockout after 3 failed attempts
    if st.session_state.login_attempts >= 3:
        if time.time() - st.session_state.last_attempt_time < 300:  # 5 minutes lockout
            st.error("Too many failed attempts. Please try again in 5 minutes.")
            return False
        else:
            # Reset attempts after lockout period
            st.session_state.login_attempts = 0
    
    # Show login form
    st.markdown("## Dashboard Login")
    with st.form("login_form"):
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        submitted = st.form_submit_button("Login")
        if st.session_state.get("authenticated"):
            st.form_submit_button("Logout")
    
    if submitted:
        # Update last attempt time
        st.session_state.last_attempt_time = time.time()
        
        if (
            username == st.secrets["DASHBOARD_USERNAME"]
            and password == st.secrets["DASHBOARD_PASSWORD"]
        ):
            st.session_state["authenticated"] = True
            current_time = time.time()
            st.session_state.authenticated_time = current_time
            st.session_state.login_attempts = 0
            
            # Set URL parameters for session persistence
            st.query_params["authenticated"] = "true"
            st.query_params["auth_time"] = str(current_time)
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            remaining_attempts = 3 - st.session_state.login_attempts
            if remaining_attempts > 0:
                st.error(f"Invalid username or password. {remaining_attempts} attempts remaining.")
            else:
                st.error("Account temporarily locked. Please try again in 5 minutes.")
        return False
    else:
        st.stop()

def logout():
    """
    Logout the user by clearing the session state and URL parameters.
    
    This function handles the user logout process:
    1. Removes authentication status from session state and URL
    2. Clears login attempt tracking
    3. Removes session timeout tracking
    4. Forces page rerun to update UI
    
    The function is typically called when:
    - User clicks the logout button
    - Session expires (30-minute timeout)
    - Security violation is detected
    """
    if st.sidebar.button("Logout"):
        # Clear session state
        for key in ["authenticated", "authenticated_time", "login_attempts", "last_attempt_time"]:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear URL parameters
        st.query_params.clear()
        st.rerun()


# Set page configuration
st.set_page_config(
    page_title="Warehouse Analytics Dashboard",
    layout="wide",
    # Enable wider distribution on Streamlit Cloud
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': """
        # Warehouse Analytics Dashboard
        A secure, multi-language dashboard for warehouse analytics.
        
        - Secure authentication with session persistence
        - Supports both local development and Streamlit Cloud deployment
        - Multi-language support (English/Russian)
        """
    }
)

# Path to configuration file - store in the same directory as the script
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_config.json')

# Check authentication before showing the dashboard
if not check_password():
    st.stop()  # Stop execution if not authenticated

# If we reach here, user is authenticated - show the dashboard
logout()  # Add logout button to sidebar

# Supabase connection parameters from Streamlit secrets
@st.cache_resource
def get_db_connection_params():
    try:
        # In cloud environment, connection params are required
        if ENVIRONMENT["is_cloud"]:
            return {
                "host": st.secrets["SUPABASE_HOST"],
                "database": st.secrets["SUPABASE_DATABASE"],
                "user": st.secrets["SUPABASE_USER"],
                "password": st.secrets["SUPABASE_PASSWORD"],
                "port": st.secrets["SUPABASE_PORT"]
            }
        # In local environment, allow defaults for some params
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
tab4_title = t["hero_products_tab"]
tab1, tab2, tab3, tab4 = st.tabs([tab1_title, tab2_title, tab3_title, tab4_title])

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
                    # Add auto-refresh - using JavaScript in cloud for better control
            if ENVIRONMENT["is_cloud"]:
                st.markdown("""
                    <script>
                    function refreshPage() {
                        window.location.reload();
                    }
                    setTimeout(refreshPage, 300000);
                    </script>
                """, unsafe_allow_html=True)
            else:
                # Fallback to meta refresh for local development
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
    def load_data_from_supabase(sales_target_multiplier=1.5, start_date=None, end_date=None):
        try:
            db_params = get_db_connection_params()
            conn = psycopg2.connect(
                host=db_params["host"],
                database=db_params["database"],
                user=db_params["user"],
                password=db_params["password"],
                port=db_params["port"]
            )

            # Build sales filter for the selected time range
            sales_time_filter = ""
            if start_date and end_date:
                sales_time_filter = f"WHERE last_change_date >= '{start_date.strftime('%Y-%m-%d')}' AND last_change_date <= '{end_date.strftime('%Y-%m-%d')}'"
            else:
                sales_time_filter = ""

            # Query for sales in the selected period
            query = f"""
            WITH period_sales AS (
                SELECT 
                    warehouse_name, nm_id, tech_size,
                    COUNT(*) AS period_sales,
                    SUM(price_with_disc) AS period_amount
                FROM belara_silver.sales
                {sales_time_filter}
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
                COALESCE(ps.period_sales, 0) AS sales_in_period,
                COALESCE(ps.period_amount, 0) AS amount_in_period,
                COALESCE(ps.period_sales, 0) * {sales_target_multiplier} AS sales_target,
                CASE 
                    WHEN cs.current_stock + cs.in_return < (COALESCE(ps.period_sales, 0) * {sales_target_multiplier}) THEN TRUE
                    ELSE FALSE
                END AS needs_restock,
                CEIL((COALESCE(ps.period_sales, 0) * {sales_target_multiplier}) - (cs.current_stock + cs.in_return)) AS stock_deficit,
                (COALESCE(ps.period_sales, 0) * {sales_target_multiplier}) - (cs.current_stock + cs.in_return) AS sort_key
            FROM current_stock cs
            LEFT JOIN period_sales ps
                ON cs.warehouse_name = ps.warehouse_name AND cs.nm_id = ps.nm_id AND cs.tech_size = ps.tech_size
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

            conn.close()
            return df, query
        except Exception as e:
            st.error(f"Error connecting to Supabase: {e}")
            return pd.DataFrame(columns=[
                'warehouseName', 'nmId', 'techSize', 'brand', 'category', 
                'subject', 'current_stock', 'in_delivery', 'in_return', 
                'sales_in_period', 'sales_target', 'stock_deficit'
            ]), ""

    # Function to load data from product_restock with filters (using Supabase)
    @st.cache_data(ttl=300)  # Cache expires after 5 minutes
    def load_product_restock_data(
        sales_target_multiplier=1.5, 
        warehouses=None, 
        brands=None, 
        categories=None, 
        subjects=None,
        products=None,
        sizes=None,
        supplier_articles=None,
        start_date=None,
        end_date=None
    ):
        try:
            db_params = get_db_connection_params()
            conn = psycopg2.connect(
                host=db_params["host"],
                database=db_params["database"],
                user=db_params["user"],
                password=db_params["password"],
                port=db_params["port"]
            )

            filter_conditions = []

            if warehouses and len(warehouses) > 0:
                warehouse_list = ", ".join([f"'{w}'" for w in warehouses])
                filter_conditions.append(f"cs.warehouse_name IN ({warehouse_list})")
            if brands and len(brands) > 0:
                brand_list = ", ".join([f"'{b}'" for b in brands])
                filter_conditions.append(f"cs.brand IN ({brand_list})")
            if categories and len(categories) > 0:
                category_list = ", ".join([f"'{c}'" for c in categories])
                filter_conditions.append(f"cs.category IN ({category_list})")
            if subjects and len(subjects) > 0:
                subject_list = ", ".join([f"'{s}'" for s in subjects])
                filter_conditions.append(f"cs.subject IN ({subject_list})")
            if products and len(products) > 0:
                product_list = ", ".join([str(p) for p in products])
                filter_conditions.append(f"cs.nm_id IN ({product_list})")
            if sizes and len(sizes) > 0:
                size_list = ", ".join([f"'{s}'" for s in sizes])
                filter_conditions.append(f"cs.tech_size IN ({size_list})")
            if supplier_articles and len(supplier_articles) > 0:
                supplier_article_list = ", ".join([f"'{s}'" for s in supplier_articles])
                filter_conditions.append(f"cs.supplier_article IN ({supplier_article_list})")

            where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"

            # Dynamic sales period calculation
            sales_time_filter = ""
            params = []
            if start_date and end_date:
                sales_time_filter = "WHERE s.date >= %s AND s.date <= %s"
                params.extend([start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")])
            else:
                # Default to last month if not provided
                sales_time_filter = "WHERE s.date >= date_trunc('month', current_date) - interval '1 month' AND s.date < date_trunc('month', current_date)"

            query = f"""
            WITH current_stock AS (
                SELECT 
                    warehouse_name,
                    nm_id,
                    supplier_article,
                    tech_size,
                    brand,
                    category,
                    subject,
                    quantity AS current_stock,
                    in_way_to_client AS in_delivery,
                    in_way_from_client AS in_return
                FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY warehouse_name, nm_id, tech_size ORDER BY last_change_date DESC) AS rn
                    FROM belara_silver.warehouse
                ) ranked
                WHERE rn = 1
            ),
            period_sales AS (
                SELECT 
                    warehouse_name,
                    nm_id,
                    supplier_article,
                    tech_size,
                    COUNT(*) AS sales_in_period
                FROM belara_silver.sales s
                {sales_time_filter}
                GROUP BY warehouse_name, nm_id, supplier_article, tech_size
            )
            SELECT 
                cs.warehouse_name AS "warehouseName",
                cs.nm_id AS "nmId",
                cs.supplier_article AS "supplierArticle",
                cs.tech_size AS "techSize",
                cs.brand,
                cs.category,
                cs.subject,
                cs.current_stock,
                cs.in_delivery,
                cs.in_return,
                COALESCE(ps.sales_in_period, 0) AS sales_in_period,
                COALESCE(ps.sales_in_period, 0) * {sales_target_multiplier} AS sales_target,
                CEIL(GREATEST(0, (COALESCE(ps.sales_in_period, 0) * {sales_target_multiplier}) - (cs.current_stock + cs.in_return))) AS stock_deficit,
                CASE 
                    WHEN (cs.current_stock + cs.in_return) < (COALESCE(ps.sales_in_period, 0) * {sales_target_multiplier}) THEN TRUE
                    ELSE FALSE
                END AS needs_restock
            FROM current_stock cs
            LEFT JOIN period_sales ps
                ON cs.warehouse_name = ps.warehouse_name
                AND cs.nm_id = ps.nm_id
                AND cs.tech_size = ps.tech_size
                AND cs.supplier_article = ps.supplier_article
            WHERE {where_clause}
            ORDER BY cs.warehouse_name, stock_deficit DESC
            """

            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)

            conn.close()
            return df
        except Exception as e:
            st.error(f"Error querying dynamic product restock data: {e}")
            return pd.DataFrame(columns=[
                'warehouseName', 'nmId', 'supplierArticle', 'techSize', 'brand', 'category', 
                'subject', 'current_stock', 'in_delivery', 'in_return', 
                'sales_in_period', 'sales_target', 'stock_deficit'
            ])

    # Create time filter
    st.sidebar.header("Time Filter")
    time_filter_option = st.sidebar.radio(
        "Select time range",
        ("Last 1 day", "Last 7 days", "Last 30 days", "Custom Range", "All Time"),
        index=4,
        key="sidebar_time_filter"
    )

    start_date = None
    end_date = None

    if time_filter_option != "All Time":
        end_date = datetime.now()
        if time_filter_option == "Last 1 day":
            start_date = end_date - pd.Timedelta(days=1)
        elif time_filter_option == "Last 7 days":
            start_date = end_date - pd.Timedelta(days=7)
        elif time_filter_option == "Last 30 days":
            start_date = end_date - pd.Timedelta(days=30)
        elif time_filter_option == "Custom Range":
            start_date_input = st.sidebar.date_input("Start date", datetime.now() - pd.Timedelta(days=7), key="sidebar_time_filter_start")
            end_date_input = st.sidebar.date_input("End date", datetime.now(), key="sidebar_time_filter_end")
            start_date = datetime.combine(start_date_input, datetime.min.time())
            end_date = datetime.combine(end_date_input, datetime.max.time())
    
    # Load the data with the selected multiplier and time filter
    df, deficit_query = load_data_from_supabase(sales_target_multiplier, start_date, end_date)

    # Show the SQL query used for deficit calculation
    with st.expander("Show SQL query for Stock Deficit Overview"):
        st.code(deficit_query, language="sql")

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

    # Create supplier_article filter (above product filter)
    supplier_articles = sorted(df['supplierArticle'].dropna().unique())
    select_all_supplier_articles = st.sidebar.checkbox("Select All Supplier Articles", value=True)
    if select_all_supplier_articles:
        selected_supplier_articles = st.sidebar.multiselect(
            "Select Supplier Articles",
            options=supplier_articles,
            default=supplier_articles
        )
    else:
        selected_supplier_articles = st.sidebar.multiselect(
            "Select Supplier Articles",
            options=supplier_articles,
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
        (df['supplierArticle'].isin(selected_supplier_articles)) &
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
        
        # Display the data using a styled DataFrame
        st.dataframe(
            deficit_counts.style.highlight_max(
                axis=0, 
                subset=['deficit_product_count', 'total_deficit']
            ), 
            use_container_width=True
        )

    # Detailed data - replaced filtered_df with product_restock data
    st.header(t["detailed_data_header"])

    # Show the query for the deficit analysis graph
    # (already shown above in the expander)

    # Load product_restock data with filters and time filter
    product_restock_data = load_product_restock_data(
        sales_target_multiplier=sales_target_multiplier,
        warehouses=selected_warehouses,
        brands=selected_brands,
        categories=selected_categories,
        subjects=selected_subjects,
        products=selected_products,
        sizes=selected_sizes,
        supplier_articles=selected_supplier_articles,
        start_date=start_date,
        end_date=end_date
    )

    if product_restock_data.empty:
        st.info(t["no_products_matching"])
    else:
        # Sort by stock_deficit descending before displaying
        product_restock_data_sorted = product_restock_data.sort_values("stock_deficit", ascending=False)
        st.dataframe(
            product_restock_data_sorted.style.highlight_max(
                axis=0, 
                subset=['stock_deficit', 'sales_in_period', 'in_delivery']
            ), 
            use_container_width=True
        )

        # Optional - download filtered product_restock data
        csv = product_restock_data_sorted.to_csv(index=False)
        st.download_button(
            label=t["download_button"],
            data=csv,
            file_name="product_restock_filtered.csv",
            mime="text/csv",
        )


    # Footer

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

# Dashboard Authentication Credentials
DASHBOARD_USERNAME = "admin"
DASHBOARD_PASSWORD = "your-secure-password"

# Add other secrets as needed
"""
        return secrets_content

    # Only show template in local development environment
    if not ENVIRONMENT["is_cloud"] and st.sidebar.checkbox("Show Secrets Template", value=False):
        st.sidebar.code(generate_secrets_template(), language="toml")
        st.sidebar.warning(
            "⚠️ Create a .streamlit/secrets.toml file with your actual credentials.\n"
            "⚠️ Add .streamlit/secrets.toml to your .gitignore file to prevent exposing secrets."
        )

with tab2:
    st.header(t["sales_orders_tab"])

    @st.cache_data(ttl=300)  # Cache expires after 5 minutes
    def load_sales_orders_data(start_date=None, end_date=None):
        try:
            db_params = get_db_connection_params()
            conn = psycopg2.connect(
                host=db_params["host"],
                database=db_params["database"],
                user=db_params["user"],
                password=db_params["password"],
                port=db_params["port"]
            )
            date_filter = ""
            params = []
            if start_date and end_date:
                date_filter = "WHERE summary_date BETWEEN %s AND %s"
                params = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            query = f"""
                SELECT 
                    CAST(summary_date AS DATE) AS "Date",
                    total_orders AS "Total Orders", 
                    total_sales AS "Total Sales",
                    sale_order_ratio AS "Sale_Order_Ratio"
                FROM belara_gold_marts.daily_sales_orders_summary
                {date_filter}
                ORDER BY summary_date DESC
            """
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error loading sales and orders data: {e}")
            return pd.DataFrame(columns=["Date", "Total Orders", "Total Sales", "Sale_Order_Ratio"])

    sales_orders_df = load_sales_orders_data(start_date, end_date)
    if "Sale_Order_Ratio" in sales_orders_df.columns:
        sales_orders_df["Sale_Order_Ratio"] = sales_orders_df["Sale_Order_Ratio"].clip(upper=1)

    if sales_orders_df.empty:
        st.info("No sales or order data found.")
    else:
        # Ensure 'Date' is in datetime format for proper sorting and display
        sales_orders_df["Date"] = pd.to_datetime(sales_orders_df["Date"])
        sales_orders_df = sales_orders_df.sort_values(by="Date")

        # Combine daily and monthly views into a single chart with a toggle
        view_option = st.radio("Select View", ["Daily", "Monthly"], horizontal=True)

        if view_option == "Daily":
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=sales_orders_df["Date"],
                y=sales_orders_df["Total Orders"],
                name=t.get("total_orders_label", "Total Orders"),
                marker_color='lightblue'
            ))

            fig.add_trace(go.Bar(
                x=sales_orders_df["Date"],
                y=sales_orders_df["Total Sales"],
                name=t.get("total_sales_label", "Total Sales"),
                marker_color='darkblue'
            ))

            fig.update_layout(
                barmode='group',
                xaxis_title=t.get("date_label", "Date"),
                yaxis_title=t.get("count_label", "Count"),
                legend_title_text=t.get("legend_title", "Metric")
            )
        else:
            monthly_sales_orders_df = sales_orders_df.copy()
            monthly_sales_orders_df["Month"] = monthly_sales_orders_df["Date"].dt.to_period("M").dt.to_timestamp()
            monthly_summary = monthly_sales_orders_df.groupby("Month").agg(
                {"Total Orders": "sum", "Total Sales": "sum"}
            ).reset_index()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=monthly_summary["Month"],
                y=monthly_summary["Total Orders"],
                name=t.get("total_orders_label", "Total Orders"),
                marker_color='lightblue'
            ))

            fig.add_trace(go.Bar(
                x=monthly_summary["Month"],
                y=monthly_summary["Total Sales"],
                name=t.get("total_sales_label", "Total Sales"),
                marker_color='darkblue'
            ))

            fig.update_layout(
                barmode='group',
                xaxis_title=t.get("date_label", "Month"),
                yaxis_title=t.get("count_label", "Count"),
                legend_title_text=t.get("legend_title", "Metric")
            )

        st.plotly_chart(fig, use_container_width=True)

        # Display the dataframe as well, if needed, or remove this line
        # st.dataframe(sales_orders_df, use_container_width=True) # Commented out as per potential previous discussion

        # Second chart: Sale_Order_Ratio over Date (Line Chart)
        if "Sale_Order_Ratio" in sales_orders_df.columns:
            st.subheader(t.get("sale_order_ratio_trend_header", "Sale/Order Ratio Trend")) # Placeholder for translation
            
            if view_option == "Daily":
                fig_ratio = go.Figure()

                fig_ratio.add_trace(go.Scatter(
                    x=sales_orders_df["Date"],
                    y=sales_orders_df["Sale_Order_Ratio"],
                    mode='lines+markers',
                    name=t.get("sale_order_ratio_label", "Sale/Order Ratio"),
                    marker_color='green'
                ))

                fig_ratio.update_layout(
                    xaxis_title=t.get("date_label", "Date"),
                    yaxis_title=t.get("ratio_label", "Ratio"),
                    legend_title_text=t.get("metric_label", "Metric")
                )
            else:
                monthly_sales_orders_df = sales_orders_df.copy()
                monthly_sales_orders_df["Month"] = monthly_sales_orders_df["Date"].dt.to_period("M").dt.to_timestamp()
                monthly_summary = monthly_sales_orders_df.groupby("Month").agg(
                    {"Sale_Order_Ratio": "mean"}
                ).reset_index()

                fig_ratio = go.Figure()

                fig_ratio.add_trace(go.Scatter(
                    x=monthly_summary["Month"],
                    y=monthly_summary["Sale_Order_Ratio"],
                    mode='lines+markers',
                    name=t.get("sale_order_ratio_label", "Sale/Order Ratio"),
                    marker_color='green'
                ))

                fig_ratio.update_layout(
                    xaxis_title=t.get("date_label", "Month"),
                    yaxis_title=t.get("ratio_label", "Ratio"),
                    legend_title_text=t.get("metric_label", "Metric")
                )
            
            st.plotly_chart(fig_ratio, use_container_width=True)
        else:
            st.info(t.get("ratio_data_missing", "Sale_Order_Ratio data not available to display trend."))

        # Display the dataframe with all columns including Sale_Order_Ratio, if needed for debugging or full view
        st.dataframe(sales_orders_df.sort_values(by="Date", ascending=False), use_container_width=True)

        # Add download button for sales and orders data
        if not sales_orders_df.empty:
            csv_sales = sales_orders_df.to_csv(index=False)
            st.download_button(
                label=t.get("download_sales_data", "Download Sales Data"),
                data=csv_sales,
                file_name="sales_orders_data.csv",
                mime="text/csv",
            )

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
                    warehouse_name,
                    priority,
                    priority_score,
                    processing_speed,
                    unique_product_count
                FROM belara_gold_marts.warehouse_priority_summary
                ORDER BY warehouse_name, priority_score DESC
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
            
            # Pivot the data for the heatmap
            pivot_df = priorities_df.pivot_table(
                index='warehouse_name',
                columns='priority',
                values='unique_product_count',
                aggfunc='sum',
                fill_value=0
            )
            
            # Ensure consistent column order
            priority_order = ['Critical', 'High', 'Medium', 'Low']
            pivot_df = pivot_df.reindex(columns=priority_order, fill_value=0)
            
            # Create heatmap using plotly
            fig3 = px.imshow(
                pivot_df,
                labels=dict(
                    x=t.get("priority", "Priority"),
                    y=t["warehouse_name"],
                    color=t.get("unique_product_count", "Product Count")
                ),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="Viridis",
                aspect="auto",
                text_auto=True
            )
            
            fig3.update_layout(
                title=t.get("warehouse_priority_distribution", "Warehouse Priority Distribution"),
                xaxis_title=t.get("priority", "Priority"),
                yaxis_title=t["warehouse_name"],
                height=500
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Add a supplementary stacked bar chart to show distribution
            fig4 = px.bar(
                priorities_df,
                x="warehouse_name",
                y="unique_product_count",
                color="priority",
                category_orders={"priority": priority_order},
                labels={
                    "warehouse_name": t["warehouse_name"],
                    "unique_product_count": t.get("unique_product_count", "Product Count"),
                    "priority": t.get("priority", "Priority")
                },
                color_discrete_sequence=px.colors.sequential.Viridis_r,
                height=400
            )
            
            fig4.update_layout(
                title=t.get("priority_product_distribution", "Priority Product Distribution by Warehouse"),
                xaxis_title=t["warehouse_name"],
                yaxis_title=t.get("unique_product_count", "Product Count"),
                legend_title=t.get("priority", "Priority")
            )
            
            st.plotly_chart(fig4, use_container_width=True)

            # Download button for priorities data
            csv_priorities = priorities_df.to_csv(index=False)
            st.download_button(
                label=t.get("download_shipment_priorities_data", "Download Restocking Priorities Data"),
                data=csv_priorities,
                file_name="restocking_priorities_data.csv",
                mime="text/csv",
                key="download-priorities"
            )
        
        # Download button for performance data
        csv_performance = performance_df.to_csv(index=False)
        st.download_button(
            label=t.get("download_shipment_performance_data", "Download Shipment Performance Data"),
            data=csv_performance,
            file_name="shipment_performance_data.csv",
            mime="text/csv",
            key="download-performance"
        )

with tab4:
    st.header(t["hero_products_title"])
    st.markdown(t["hero_products_subtitle"])

    @st.cache_data(ttl=300)
    def load_hero_products_data(warehouses=None, brands=None, categories=None, subjects=None, start_date=None, end_date=None):
        try:
            db_params = get_db_connection_params()
            conn = psycopg2.connect(**db_params)
            filter_conditions = []
            if warehouses:
                warehouse_list = ", ".join([f"'{w.replace("'", "''")}'" for w in warehouses])
                filter_conditions.append(f"warehouse_name IN ({warehouse_list})")
            if brands:
                brand_list = ", ".join([f"'{b.replace("'", "''")}'" for b in brands])
                filter_conditions.append(f"brand IN ({brand_list})")
            if categories:
                category_list = ", ".join([f"'{c.replace("'", "''")}'" for c in categories])
                filter_conditions.append(f"category IN ({category_list})")
            if subjects:
                subject_list = ", ".join([f"'{s.replace("'", "''")}'" for s in subjects])
                filter_conditions.append(f"subject IN ({subject_list})")
            params = []
            if start_date and end_date:
                filter_conditions.append("last_change_date BETWEEN %s AND %s")
                params.extend([start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")])
            else:
                filter_conditions.append("last_change_date >= CURRENT_DATE - INTERVAL '90 days'")
            where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"
            query = f"""
                SELECT 
                    nm_id AS "nmId",
                    supplier_article AS "supplierArticle",
                    SUM(finished_price) AS "salesAmount",
                    COUNT(*) AS "salesQuantity"
                FROM belara_bronze.wildberries_sales
                WHERE {where_clause}
                GROUP BY nm_id, supplier_article
                ORDER BY "salesAmount" DESC
            """
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error loading hero products data: {e}")
            return pd.DataFrame(columns=["nmId", "supplierArticle", "salesAmount", "salesQuantity"])

    # Load hero products data using the filters from the sidebar and time filter
    hero_products_df = load_hero_products_data(
        warehouses=selected_warehouses,
        brands=selected_brands,
        categories=selected_categories,
        subjects=selected_subjects,
        start_date=start_date,
        end_date=end_date
    )

    if hero_products_df.empty:
        st.info(t["no_hero_products"])
    else:
        # Chart: Top 10 Hero Products (Bar: supplierArticle vs salesQuantity, salesAmount as BYN label and tooltip)
        st.subheader(t["top_n_products_chart"].format(10))
        top_10_hero = hero_products_df.head(10).copy()
        # Format salesAmount as Belarusian ruble (Br) string for label
        top_10_hero["salesAmountFormatted"] = top_10_hero["salesAmount"].apply(lambda x: f"Br {x:,.0f}")
        
        fig_hero = px.bar(
            top_10_hero,
            x="supplierArticle",
            y="salesQuantity",
            text="salesAmountFormatted",
            hover_data={
                "salesAmountFormatted": True,
                "salesAmount": False,
                "supplierArticle": False,
                "salesQuantity": False
            },
            labels={
                "supplierArticle": t["product_id"],
                "salesQuantity": t["sales_quantity"],
                "salesAmountFormatted": t["sales_amount"]
            },
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_hero.update_traces(texttemplate='%{text}', textposition='auto', textangle=0)
        fig_hero.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            xaxis_type='category',
            margin=dict(t=60)  # Add top margin to avoid label cutoff
        )
        st.plotly_chart(fig_hero, use_container_width=True)

        # Table: All Hero Products
        st.subheader(t["hero_products_table"])
        st.dataframe(hero_products_df, use_container_width=True)

        # Download button
        csv_hero = hero_products_df.to_csv(index=False)
        st.download_button(
            label=t["download_hero_products_data"],
            data=csv_hero,
            file_name="hero_products_data.csv",
            mime="text/csv",
            key="download-hero"
        )
