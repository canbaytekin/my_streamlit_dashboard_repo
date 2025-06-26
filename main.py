import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the path to the dashboard.py file in the root directory
    dashboard_path = os.path.join(current_dir, "dashboard.py")
    
    # Check if the file exists
    if not os.path.exists(dashboard_path):
        print(f"Error: {dashboard_path} not found")
        sys.exit(1)
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", dashboard_path, "--server.headless=true"]
    sys.exit(stcli.main()) 