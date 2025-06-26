#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create secrets.toml template if it doesn't exist
if [ ! -f .streamlit/secrets.toml ]; then
    echo "Creating .streamlit/secrets.toml template..."
    cat > .streamlit/secrets.toml << EOF
# Supabase PostgreSQL Database Credentials
SUPABASE_HOST = "db.your-project-id.supabase.co"
SUPABASE_DATABASE = "postgres" 
SUPABASE_USER = "postgres"
SUPABASE_PASSWORD = "your-database-password"
SUPABASE_PORT = "5432"

# Dashboard Authentication Credentials
DASHBOARD_USERNAME = "admin"
DASHBOARD_PASSWORD = "your-secure-password"
EOF
    echo "Please edit .streamlit/secrets.toml with your actual database credentials"
else
    echo ".streamlit/secrets.toml already exists, skipping..."
fi

echo ""
echo "Setup complete! You can now run the dashboard with:"
echo "source venv/bin/activate"
echo "streamlit run dashboard.py"