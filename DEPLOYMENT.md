# Deploying to Streamlit Cloud

This document provides instructions for deploying the Forecasting Tools application to Streamlit Cloud.

## Configuration Files

The repository includes all necessary configuration files for Streamlit Cloud:
- `main.py` - The main entry point for the Streamlit application
- `requirements.txt` - All package dependencies
- `.streamlit/config.toml` - Streamlit configuration settings

## Deployment Steps

1. **Push the repository to GitHub**
   - Ensure all changes are committed and pushed to your GitHub repository

2. **Log in to Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account

3. **Create a New App**
   - Click "New app"
   - Select your repository
   - In the "Main file path" field, enter: `main.py`

4. **Configure Environment Variables**
   - Click "Advanced settings"
   - Add the following environment variables:
     - `OPENAI_API_KEY` - Your OpenAI API key
     - `PERPLEXITY_API_KEY` - Your Perplexity API key (if using)
     - `EXA_API_KEY` - Your Exa API key (if using)
     - Any other API keys your installation requires

5. **Deploy**
   - Click "Deploy"
   - Wait for the deployment to complete

## Troubleshooting

- If you see import errors, check that all dependencies are listed in `requirements.txt`
- If pages aren't displaying correctly, make sure the main.py file is properly set up
- For other issues, check the Streamlit Cloud logs

## Local Development

To run the application locally:

```bash
streamlit run main.py
```

This will use the same entry point as Streamlit Cloud. 