import logging
import re
import asyncio
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion, DateQuestion
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.model_interfaces.time_series_forecaster import TimeSeriesForecaster
from forecasting_tools.ai_models.model_interfaces.enhanced_llm_forecaster import EnhancedLLMForecaster

logger = logging.getLogger(__name__)


class TimeSeriesForecastInput(Jsonable, BaseModel):
    question: BinaryQuestion


class TimeSeriesForecastPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "📈 Time Series Forecast"
    URL_PATH: str = "/time-series-forecast"
    INPUT_TYPE = TimeSeriesForecastInput
    OUTPUT_TYPE = BinaryReport
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/forecast_page_examples.json"

    # Form input keys
    QUESTION_TEXT_BOX = "question_text_box_time_series"
    RESOLUTION_CRITERIA_BOX = "resolution_criteria_box_time_series"
    FINE_PRINT_BOX = "fine_print_box_time_series"
    BACKGROUND_INFO_BOX = "background_info_box_time_series"
    METACULUS_URL_INPUT = "metaculus_url_input_time_series"
    FETCH_BUTTON = "fetch_button_time_series"
    MODEL_TYPE = "model_type_time_series"
    DATA_UPLOAD = "data_upload_time_series"
    FORECAST_HORIZON = "forecast_horizon_time_series"
    
    @classmethod
    async def _display_intro_text(cls) -> None:
        st.write(
            "This page demonstrates time series forecasting for questions involving trends or events over time. "
            "Upload time series data or let the system automatically retrieve relevant data."
        )

    @classmethod
    async def _get_input(cls) -> TimeSeriesForecastInput | None:
        cls.__display_metaculus_url_input()
        
        with st.form("time_series_form"):
            st.subheader("Question Details")
            
            question_text = st.text_input(
                "Yes/No Binary Question", key=cls.QUESTION_TEXT_BOX
            )
            resolution_criteria = st.text_area(
                "Resolution Criteria (optional)",
                key=cls.RESOLUTION_CRITERIA_BOX,
            )
            fine_print = st.text_area(
                "Fine Print (optional)", key=cls.FINE_PRINT_BOX
            )
            background_info = st.text_area(
                "Background Info (optional)", key=cls.BACKGROUND_INFO_BOX
            )
            
            st.subheader("Time Series Configuration")
            
            # Create two columns for the inputs
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Model Type",
                    options=["auto", "arima", "prophet", "lstm"],
                    help="The model to use for time series forecasting",
                    key=cls.MODEL_TYPE
                )
                
                forecast_horizon = st.slider(
                    "Forecast Horizon (days)",
                    min_value=7,
                    max_value=365,
                    value=30,
                    help="Number of days to forecast into the future",
                    key=cls.FORECAST_HORIZON
                )
            
            with col2:
                st.markdown("**Data Upload (Optional)**")
                st.markdown("If no data is provided, the system will attempt to retrieve relevant data automatically.")
                st.markdown("Format: CSV with 'date' and 'value' columns")
                
                uploaded_file = st.file_uploader(
                    "Upload Time Series Data (CSV)",
                    type=["csv"],
                    key=cls.DATA_UPLOAD
                )
            
            submitted = st.form_submit_button("Run Time Series Forecast")

            if submitted:
                if not question_text:
                    st.error("Question Text is required.")
                    return None
                
                question = BinaryQuestion(
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria=resolution_criteria,
                    fine_print=fine_print,
                    page_url="",
                    api_json={},
                )
                
                # Store configuration in session state
                st.session_state['time_series_config'] = {
                    'model_type': model_type,
                    'forecast_horizon': forecast_horizon,
                    'has_uploaded_data': uploaded_file is not None
                }
                
                # Handle the uploaded file if provided
                if uploaded_file is not None:
                    # Save the uploaded file to a temporary location
                    try:
                        # Create data directory if it doesn't exist
                        data_dir = "forecasting_tools/data/time_series"
                        os.makedirs(data_dir, exist_ok=True)
                        
                        # Save the data with a unique name based on the question
                        question_hash = str(hash(question_text))
                        file_path = os.path.join(data_dir, f"q_{question_hash}.csv")
                        
                        # Read the file and save it
                        df = pd.read_csv(uploaded_file)
                        
                        # Ensure it has the right format (date and value columns)
                        if 'date' not in df.columns:
                            # Try to find a date column
                            date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                            if date_candidates:
                                df = df.rename(columns={date_candidates[0]: 'date'})
                            else:
                                # Assume first column is date
                                df = df.rename(columns={df.columns[0]: 'date'})
                        
                        # If value column is missing, assume the second numeric column is value
                        if 'value' not in df.columns:
                            # Find first numeric column that's not the date
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                df = df.rename(columns={numeric_cols[0]: 'value'})
                            else:
                                # Assume second column is value
                                df = df.rename(columns={df.columns[1]: 'value'})
                        
                        # Make sure date is in datetime format
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Save to csv
                        df.to_csv(file_path, index=False)
                        st.session_state['uploaded_data_path'] = file_path
                        
                    except Exception as e:
                        st.error(f"Error processing uploaded file: {e}")
                        return None
                
                return TimeSeriesForecastInput(question=question)
        return None

    @classmethod
    async def _run_tool(cls, input: TimeSeriesForecastInput) -> BinaryReport:
        with st.spinner("Setting up time series forecaster..."):
            # Get forecaster configuration from session state
            config = st.session_state.get('time_series_config', {})
            model_type = config.get('model_type', 'auto')
            forecast_horizon = config.get('forecast_horizon', 30)
            
            # Create a fallback forecaster for cases where time series isn't applicable
            fallback_forecaster = EnhancedLLMForecaster(model_name="openai/o1", temperature=0.2)
            
            # Initialize the time series forecaster
            time_series_forecaster = TimeSeriesForecaster(
                model_type=model_type,
                forecast_horizon=forecast_horizon,
                fallback_forecaster=fallback_forecaster
            )
            
            # Check if we have uploaded data to use
            if 'uploaded_data_path' in st.session_state:
                st.info(f"Using uploaded time series data")
            
            with st.spinner("Running time series forecast..."):
                # Use progress indicators
                progress_container = st.empty()
                progress_container.text("Checking if time series analysis is applicable...")
                
                # Check if time series analysis is applicable
                is_applicable = time_series_forecaster._is_time_series_applicable(input.question)
                
                if not is_applicable:
                    progress_container.warning("Time series analysis is not directly applicable to this question. Using enhanced LLM forecaster instead.")
                    st.info("For time series analysis to be applicable, the question should involve trends, rates, or thresholds over time.")
                else:
                    progress_container.text("Time series analysis is applicable! Getting forecast...")
                
                # Get the forecast result
                result = await time_series_forecaster.get_forecast_result(input.question)
                
                # Display the probability
                probability = result.probability
                progress_container.text(f"Time Series Forecast: {probability:.3f}")
                
                # Create the report
                report = BinaryReport(
                    question=input.question,
                    prediction=probability,
                    explanation=result.rationale,
                    other_notes=f"Confidence Interval: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]",
                    metadata=result.metadata
                )
                
                return report

    @classmethod
    async def _display_outputs(cls, outputs: list[BinaryReport]) -> None:
        ReportDisplayer.display_report_list(outputs)
        
        if not outputs:
            return
            
        report = outputs[0]
        
        # Check if we have time series metadata to visualize
        if hasattr(report, 'metadata') and report.metadata:
            metadata = report.metadata
            
            if 'forecast_data' in metadata and 'historical_data' in metadata:
                st.subheader("Time Series Visualization")
                
                # Extract data from metadata
                forecast_data = metadata['forecast_data']
                historical_data = metadata['historical_data']
                
                # Convert to pandas for plotting
                forecast_dates = pd.to_datetime(forecast_data['dates'])
                forecast_values = forecast_data['values']
                forecast_lower = forecast_data['lower_bound']
                forecast_upper = forecast_data['upper_bound']
                
                historical_dates = pd.to_datetime(historical_data['dates'])
                historical_values = historical_data['values']
                
                # Plot the time series
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Historical data
                ax.plot(historical_dates, historical_values, 'b.-', label='Historical Data')
                
                # Forecast
                ax.plot(forecast_dates, forecast_values, 'r.-', label='Forecast')
                
                # Confidence band
                ax.fill_between(forecast_dates, forecast_lower, forecast_upper, 
                               color='red', alpha=0.2, label='Confidence Interval')
                
                # Add vertical line separating history from forecast
                ax.axvline(x=historical_dates.iloc[-1], color='gray', linestyle='--')
                
                # Format the plot
                ax.set_title('Time Series Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Rotate date labels
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Add explanatory text
                st.info(
                    "The chart above shows historical data (blue) and the forecasted values (red). "
                    "The shaded red area represents the forecast confidence interval. "
                    "The dashed vertical line separates historical data from the forecast."
                )
                
                # Display model information
                if 'model_type' in metadata and 'model_summary' in metadata:
                    st.subheader(f"Model: {metadata['model_type'].upper()}")
                    
                    # Display model parameters in a nice format
                    st.json(metadata['model_summary'])
            
            # If time series wasn't applicable, show a message
            elif 'model_type' not in metadata:
                st.info(
                    "Time series analysis was not applicable to this question. "
                    "The forecast was generated using a standard forecasting model instead."
                )
                st.warning(
                    "For time series forecasting to be effective, questions should involve "
                    "trends, rates, thresholds, or events that evolve over time."
                )

    @classmethod
    def __display_metaculus_url_input(cls) -> None:
        with st.expander("Use an existing Metaculus Binary question"):
            st.write(
                "Enter a Metaculus question URL to autofill the form below."
            )

            metaculus_url = st.text_input(
                "Metaculus Question URL", key=cls.METACULUS_URL_INPUT
            )
            fetch_button = st.button("Fetch Question", key=cls.FETCH_BUTTON)

            if fetch_button and metaculus_url:
                with st.spinner("Fetching question details..."):
                    try:
                        question_id = cls.__extract_question_id(metaculus_url)
                        metaculus_question = (
                            MetaculusApi.get_question_by_post_id(question_id)
                        )
                        if isinstance(metaculus_question, BinaryQuestion):
                            cls.__autofill_form(metaculus_question)
                        else:
                            st.error(
                                "Only binary questions are supported at this time."
                            )
                    except Exception as e:
                        st.error(
                            f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                        )

    @classmethod
    def __extract_question_id(cls, url: str) -> int:
        match = re.search(r"/questions/(\d+)/", url)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
        )

    @classmethod
    def __autofill_form(cls, question: BinaryQuestion) -> None:
        st.session_state[cls.QUESTION_TEXT_BOX] = question.question_text
        st.session_state[cls.BACKGROUND_INFO_BOX] = (
            question.background_info or ""
        )
        st.session_state[cls.RESOLUTION_CRITERIA_BOX] = (
            question.resolution_criteria or ""
        )
        st.session_state[cls.FINE_PRINT_BOX] = question.fine_print or ""


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    TimeSeriesForecastPage.main() 