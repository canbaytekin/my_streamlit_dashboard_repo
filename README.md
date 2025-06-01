# Warehouse Analytics Dashboard

A comprehensive dashboard built with Streamlit to monitor warehouse stock deficits, analyze sales data, and make restocking recommendations based on configurable sales targets.

## Features

- Real-time monitoring of warehouse stock levels
- Stock deficit analysis and restocking recommendations
- Sales and order trend visualization
- Customizable sales target multiplier
- Multilingual support (English and Russian)
- Direct PostgreSQL query interface
- Data export capabilities

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/warehouse-dashboard.git
cd warehouse-dashboard
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your Supabase PostgreSQL credentials:
   - Create a `.streamlit/secrets.toml` file with your database credentials
   ```toml
   # .streamlit/secrets.toml
   SUPABASE_HOST = "your-project.supabase.co"
   SUPABASE_DATABASE = "postgres"
   SUPABASE_USER = "postgres"
   SUPABASE_PASSWORD = "your-password"
   SUPABASE_PORT = "5432"
   ```

## Running the Dashboard

```bash
streamlit run src/dashboard.py
```

## Configuration

The dashboard uses a configurable multiplier to determine restocking needs:

```
Restocking Target = Last Month's Sales × Sales Target Multiplier
```

You can adjust this multiplier using the slider in the dashboard's sidebar.

## Security

Database credentials are stored securely using Streamlit's secrets management. Make sure to add the `.streamlit/secrets.toml` file to your `.gitignore` to prevent accidentally exposing credentials.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 