import streamlit as st
import openai
from dotenv import load_dotenv
import os

def initialize_bot():
    """Initialize the forecasting bot with API keys from Streamlit secrets."""
    try:
        openai.api_key = st.secrets["api_keys"]["openai_api_key"]
        exa_api_key = st.secrets["api_keys"]["exa_api_key"]
        metaculus_api_key = st.secrets["api_keys"]["metaculus_api_key"]
        return True
    except Exception as e:
        st.error(f"Error initializing bot: {str(e)}")
        return False

def main():
    st.title("Forecasting Tools")
    
    if not initialize_bot():
        st.error("Please configure your API keys in Streamlit secrets.")
        return

    user_input = st.text_area("Enter your forecasting question:", height=100)
    
    if st.button("Generate Forecast"):
        if not user_input:
            st.warning("Please enter a question.")
            return
            
        try:
            with st.spinner("Generating forecast..."):
                # Basic response for now
                st.write("Forecast generation is being implemented...")
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

if __name__ == "__main__":
    main()
