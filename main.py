import os
import sys
import streamlit as st
import dotenv

# Add the Forcasting-tools-V3 directory to the path
v3_path = os.path.join(os.path.dirname(__file__), "Forcasting-tools-V3")
sys.path.insert(0, v3_path)

# Ensure the Forcasting-tools-V3/forecasting_tools is in the path
tools_path = os.path.join(v3_path, "forecasting_tools")
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)

# Load environment variables
dotenv.load_dotenv()

# Instead of importing directly, execute the app.py file's contents
# This approach avoids issues with directory names containing dashes
app_path = os.path.join(v3_path, "app.py")

# Get the current directory
current_dir = os.getcwd()

# Change to the V3 directory temporarily
os.chdir(v3_path)

# Execute the app.py file which will set up the Streamlit interface
exec(open(app_path).read())

# Restore original directory if needed for cleanup
os.chdir(current_dir)

# This file serves as a redirector to the V3 app
# When deploying to Streamlit Cloud, specify this file as the entry point 