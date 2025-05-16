import logging
import re
import asyncio

import dotenv
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.model_interfaces.synthetic_forecaster import SyntheticForecaster
from forecasting_tools.ai_models.model_interfaces.enhanced_llm_forecaster import EnhancedLLMForecaster
from forecasting_tools.ai_models.model_interfaces.ensemble_forecaster import EnsembleForecaster
from forecasting_tools.ai_models.model_interfaces.expert_forecaster import ExpertForecaster
from forecasting_tools.ai_models.model_interfaces.historical_forecaster import HistoricalForecaster
from forecasting_tools.ai_models.model_interfaces.calibration_system import CalibratedForecaster
from forecasting_tools.ai_models.model_interfaces.dynamic_forecaster_selector import DynamicForecaster
from forecasting_tools.ai_models.model_interfaces.advanced_ensemble_methods import DynamicWeightingEnsemble, StackingEnsemble

logger = logging.getLogger(__name__)


class EnsembleForecastInput(Jsonable, BaseModel):
    question: BinaryQuestion


class EnsembleForecastPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🤝 Ensemble Forecast"
    URL_PATH: str = "/ensemble-forecast"
    INPUT_TYPE = EnsembleForecastInput
    OUTPUT_TYPE = BinaryReport
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/forecast_page_examples.json"

    # Form input keys
    QUESTION_TEXT_BOX = "question_text_box_ensemble"
    RESOLUTION_CRITERIA_BOX = "resolution_criteria_box_ensemble"
    FINE_PRINT_BOX = "fine_print_box_ensemble"
    BACKGROUND_INFO_BOX = "background_info_box_ensemble"
    METACULUS_URL_INPUT = "metaculus_url_input_ensemble"
    FETCH_BUTTON = "fetch_button_ensemble"
    
    # Model selection keys
    MODEL_SELECT = "model_select_ensemble"
    USE_LLM = "use_llm_ensemble"
    USE_ENHANCED_LLM = "use_enhanced_llm_ensemble"
    USE_SYNTHETIC = "use_synthetic_ensemble"
    USE_EXPERT = "use_expert_ensemble"
    USE_HISTORICAL = "use_historical_ensemble"
    USE_CALIBRATED = "use_calibrated_ensemble"
    USE_DYNAMIC = "use_dynamic_ensemble"
    
    LLM_WEIGHT = "llm_weight_ensemble"
    ENHANCED_LLM_WEIGHT = "enhanced_llm_weight_ensemble"
    SYNTHETIC_WEIGHT = "synthetic_weight_ensemble"
    EXPERT_WEIGHT = "expert_weight_ensemble"
    HISTORICAL_WEIGHT = "historical_weight_ensemble"
    CALIBRATED_WEIGHT = "calibrated_weight_ensemble"
    DYNAMIC_WEIGHT = "dynamic_weight_ensemble"
    
    ENSEMBLE_METHOD = "ensemble_method_ensemble"
    ADVANCED_OPTIONS = "advanced_options_ensemble"

    @classmethod
    async def _display_intro_text(cls) -> None:
        st.write(
            "This page demonstrates ensemble forecasting using multiple forecasters. "
            "Select which forecasters to include and how to weight them."
        )

    @classmethod
    async def _get_input(cls) -> EnsembleForecastInput | None:
        cls.__display_metaculus_url_input()
        
        with st.form("ensemble_form"):
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
            
            st.subheader("Forecaster Configuration")
            
            # Create 4 columns (2 rows of 4)
            col1, col2, col3, col4 = st.columns(4)
            
            # First row
            with col1:
                use_llm = st.checkbox("Standard LLM", 
                                   value=True,
                                   key=cls.USE_LLM)
                if use_llm:
                    llm_weight = st.slider("Weight", 
                                        min_value=0.0, 
                                        max_value=1.0, 
                                        value=0.2,
                                        key=cls.LLM_WEIGHT)
                else:
                    llm_weight = 0.0
            
            with col2:
                use_enhanced_llm = st.checkbox("Enhanced LLM",
                                           value=True,
                                           key=cls.USE_ENHANCED_LLM)
                if use_enhanced_llm:
                    enhanced_llm_weight = st.slider("Weight",
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=0.2,
                                                key=cls.ENHANCED_LLM_WEIGHT)
                else:
                    enhanced_llm_weight = 0.0
            
            with col3:
                use_expert = st.checkbox("Expert Forecaster",
                                      value=True,
                                      key=cls.USE_EXPERT)
                if use_expert:
                    expert_weight = st.slider("Weight",
                                           min_value=0.0,
                                           max_value=1.0,
                                           value=0.2,
                                           key=cls.EXPERT_WEIGHT)
                else:
                    expert_weight = 0.0
            
            with col4:
                use_historical = st.checkbox("Historical Data",
                                         value=True,
                                         key=cls.USE_HISTORICAL)
                if use_historical:
                    historical_weight = st.slider("Weight",
                                              min_value=0.0,
                                              max_value=1.0,
                                              value=0.2,
                                              key=cls.HISTORICAL_WEIGHT)
                else:
                    historical_weight = 0.0
            
            # Second row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                use_calibrated = st.checkbox("Calibrated LLM",
                                         value=False,
                                         key=cls.USE_CALIBRATED)
                if use_calibrated:
                    calibrated_weight = st.slider("Weight",
                                              min_value=0.0,
                                              max_value=1.0,
                                              value=0.1,
                                              key=cls.CALIBRATED_WEIGHT)
                else:
                    calibrated_weight = 0.0
            
            with col6:
                use_dynamic = st.checkbox("Dynamic Selection",
                                       value=False,
                                       key=cls.USE_DYNAMIC)
                if use_dynamic:
                    dynamic_weight = st.slider("Weight",
                                           min_value=0.0,
                                           max_value=1.0,
                                           value=0.1,
                                           key=cls.DYNAMIC_WEIGHT)
                else:
                    dynamic_weight = 0.0
            
            with col7:
                use_synthetic = st.checkbox("Synthetic",
                                        value=False,
                                        key=cls.USE_SYNTHETIC)
                if use_synthetic:
                    synthetic_weight = st.slider("Weight",
                                              min_value=0.0,
                                              max_value=1.0,
                                              value=0.1,
                                              key=cls.SYNTHETIC_WEIGHT)
                else:
                    synthetic_weight = 0.0
            
            with col8:
                ensemble_method = st.selectbox("Ensemble Method",
                                           options=["weighted_average", "simple_average", 
                                                   "dynamic_weighting", "stacking"],
                                           key=cls.ENSEMBLE_METHOD)
            
            with st.expander("Advanced Options", expanded=False):
                st.info("These options affect how the forecasters behave.")
                
                # Synthetic forecaster settings
                st.subheader("Synthetic Forecaster")
                synthetic_mode = st.selectbox("Mode", options=["random", "fixed"], value="fixed")
                synthetic_probability = st.slider("Fixed Probability", min_value=0.0, max_value=1.0, value=0.7)
                
                # Historical forecaster settings
                st.subheader("Historical Forecaster")
                similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.95, value=0.75)
                time_decay = st.slider("Time Decay Factor", min_value=0.5, max_value=1.0, value=0.9)
                
                # Expert forecaster settings
                st.subheader("Expert Forecaster")
                default_domain = st.selectbox("Default Domain", 
                                          options=[None, "politics", "economics", "technology", "science", 
                                                 "health", "climate", "geopolitics"])
                
                # Calibration settings
                st.subheader("Calibration")
                recalibration_method = st.selectbox("Recalibration Method", 
                                                options=["platt", "isotonic", "none"])
                
                # Confidence interval settings
                st.subheader("Confidence Intervals")
                confidence_interval_method = st.selectbox(
                    "Interval Estimation Method",
                    options=["bootstrapping", "variance_propagation"],
                    index=0,
                    help="Bootstrapping is more robust but computationally intensive"
                )
                with st.expander("About Confidence Interval Methods"):
                    st.write(
                        "**Bootstrapping**: Resamples forecasts to estimate uncertainty while accounting for "
                        "correlations between forecasters. More robust but computationally intensive."
                    )
                    st.write(
                        "**Variance Propagation**: Traditional approach that averages bounds assuming "
                        "independent errors. Faster but may underestimate uncertainty."
                    )
                
                # Dynamic weighting settings
                st.subheader("Dynamic Weighting")
                adjustment_rate = st.slider("Adjustment Rate", 
                                        min_value=0.01, 
                                        max_value=0.5, 
                                        value=0.1,
                                        help="How quickly weights adjust based on performance")
                
                # Stacking settings
                st.subheader("Stacking")
                meta_model_type = st.selectbox("Meta-model Type",
                                           options=["logistic_regression", "random_forest"],
                                           help="Model used to combine forecaster outputs")
            
            submitted = st.form_submit_button("Run Ensemble Forecast")

            if submitted:
                if not question_text:
                    st.error("Question Text is required.")
                    return None
                    
                # Ensure at least one forecaster is selected
                if not any([use_llm, use_enhanced_llm, use_expert, use_historical, 
                           use_calibrated, use_dynamic, use_synthetic]):
                    st.error("Please select at least one forecaster.")
                    return None
                
                question = BinaryQuestion(
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria=resolution_criteria,
                    fine_print=fine_print,
                    page_url="",
                    api_json={},
                )
                
                # Store forecaster selections in session state
                st.session_state['forecaster_config'] = {
                    'use_llm': use_llm,
                    'use_enhanced_llm': use_enhanced_llm,
                    'use_expert': use_expert,
                    'use_historical': use_historical,
                    'use_calibrated': use_calibrated,
                    'use_dynamic': use_dynamic,
                    'use_synthetic': use_synthetic,
                    
                    'llm_weight': llm_weight,
                    'enhanced_llm_weight': enhanced_llm_weight,
                    'expert_weight': expert_weight,
                    'historical_weight': historical_weight,
                    'calibrated_weight': calibrated_weight,
                    'dynamic_weight': dynamic_weight,
                    'synthetic_weight': synthetic_weight,
                    
                    'ensemble_method': ensemble_method,
                    
                    # Advanced options
                    'synthetic_mode': synthetic_mode,
                    'synthetic_probability': synthetic_probability,
                    'similarity_threshold': similarity_threshold,
                    'time_decay': time_decay,
                    'default_domain': default_domain,
                    'recalibration_method': recalibration_method,
                    'adjustment_rate': adjustment_rate,
                    'meta_model_type': meta_model_type,
                    'confidence_interval_method': confidence_interval_method
                }
                
                return EnsembleForecastInput(question=question)
        return None

    @classmethod
    async def _run_tool(cls, input: EnsembleForecastInput) -> BinaryReport:
        with st.spinner("Building ensemble..."):
            # Get forecaster configuration from session state
            config = st.session_state.get('forecaster_config', {})
            
            forecasters = []
            weights = []
            
            # Create standard LLM
            if config.get('use_llm', False):
                st.text("Adding Standard LLM to ensemble...")
                forecasters.append(GeneralLlm(model="openai/o1", temperature=0.2))
                weights.append(config.get('llm_weight', 0.2))
            
            # Create enhanced LLM
            if config.get('use_enhanced_llm', False):
                st.text("Adding Enhanced LLM to ensemble...")
                forecasters.append(EnhancedLLMForecaster(model_name="openai/o1", temperature=0.2))
                weights.append(config.get('enhanced_llm_weight', 0.2))
            
            # Create synthetic forecaster
            if config.get('use_synthetic', False):
                st.text("Adding Synthetic Forecaster to ensemble...")
                synthetic_mode = config.get('synthetic_mode', 'fixed')
                synthetic_probability = config.get('synthetic_probability', 0.7)
                forecasters.append(SyntheticForecaster(
                    mode=synthetic_mode, 
                    fixed_probability=synthetic_probability
                ))
                weights.append(config.get('synthetic_weight', 0.1))
            
            # Create expert forecaster
            if config.get('use_expert', False):
                st.text("Adding Expert Forecaster to ensemble...")
                default_domain = config.get('default_domain')
                forecasters.append(ExpertForecaster(
                    model_name="openai/o1",
                    temperature=0.1,
                    default_domain=default_domain
                ))
                weights.append(config.get('expert_weight', 0.2))
            
            # Create historical forecaster
            base_forecaster_for_historical = None
            if config.get('use_enhanced_llm', False):
                # Use enhanced LLM as base if available
                base_forecaster_for_historical = EnhancedLLMForecaster(model_name="openai/o1", temperature=0.2)
            elif config.get('use_llm', False):
                # Fallback to standard LLM
                base_forecaster_for_historical = GeneralLlm(model="openai/o1", temperature=0.2)
                
            if config.get('use_historical', False):
                st.text("Adding Historical Forecaster to ensemble...")
                similarity_threshold = config.get('similarity_threshold', 0.75)
                time_decay = config.get('time_decay', 0.9)
                forecasters.append(HistoricalForecaster(
                    similarity_threshold=similarity_threshold,
                    time_decay_factor=time_decay,
                    base_forecaster=base_forecaster_for_historical
                ))
                weights.append(config.get('historical_weight', 0.2))
            
            # Create calibrated forecaster
            if config.get('use_calibrated', False):
                st.text("Adding Calibrated Forecaster to ensemble...")
                # Use enhanced LLM as base if available, otherwise standard LLM
                base_forecaster = None
                if config.get('use_enhanced_llm', False):
                    base_forecaster = EnhancedLLMForecaster(model_name="openai/o1", temperature=0.2)
                elif config.get('use_llm', False):
                    base_forecaster = GeneralLlm(model="openai/o1", temperature=0.2)
                
                if base_forecaster:
                    recalibration_method = config.get('recalibration_method', 'platt')
                    forecasters.append(CalibratedForecaster(
                        base_forecaster=base_forecaster,
                        recalibration_method=recalibration_method
                    ))
                    weights.append(config.get('calibrated_weight', 0.1))
            
            # Create dynamic forecaster
            if config.get('use_dynamic', False):
                st.text("Adding Dynamic Forecaster to ensemble...")
                # Create a registry of available forecasters
                forecaster_registry = {}
                
                if config.get('use_llm', False):
                    forecaster_registry["general_llm"] = GeneralLlm(model="openai/o1", temperature=0.2)
                
                if config.get('use_enhanced_llm', False):
                    forecaster_registry["enhanced_llm"] = EnhancedLLMForecaster(model_name="openai/o1", temperature=0.2)
                
                if config.get('use_expert', False):
                    forecaster_registry["expert"] = ExpertForecaster(
                        model_name="openai/o1",
                        temperature=0.1,
                        default_domain=config.get('default_domain')
                    )
                
                if config.get('use_synthetic', False):
                    forecaster_registry["synthetic"] = SyntheticForecaster(
                        mode=config.get('synthetic_mode', 'fixed'),
                        fixed_probability=config.get('synthetic_probability', 0.7)
                    )
                
                # Only add if we have at least 2 forecasters in the registry
                if len(forecaster_registry) >= 2:
                    forecasters.append(DynamicForecaster(
                        forecaster_registry=forecaster_registry,
                        default_forecaster="enhanced_llm" if "enhanced_llm" in forecaster_registry else "general_llm"
                    ))
                    weights.append(config.get('dynamic_weight', 0.1))
            
            # If no forecasters are selected (unlikely due to form validation), use a default
            if not forecasters:
                st.warning("No forecasters selected, using Enhanced LLM as default")
                forecasters.append(EnhancedLLMForecaster())
                weights.append(1.0)
            
            # Normalize weights if using weighted methods
            if config.get('ensemble_method') in ['weighted_average', 'dynamic_weighting']:
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                else:
                    # Equal weights if sum is 0
                    weights = [1.0/len(forecasters) for _ in forecasters]
            
            ensemble_method = config.get('ensemble_method', 'weighted_average')
            
            # Create the appropriate ensemble based on method
            if ensemble_method == 'dynamic_weighting':
                st.text("Creating Dynamic Weighting Ensemble...")
                adjustment_rate = config.get('adjustment_rate', 0.1)
                ensemble = DynamicWeightingEnsemble(
                    forecasters=forecasters,
                    initial_weights=weights,
                    adjustment_rate=adjustment_rate
                )
            elif ensemble_method == 'stacking':
                st.text("Creating Stacking Ensemble...")
                meta_model_type = config.get('meta_model_type', 'logistic_regression')
                ensemble = StackingEnsemble(
                    forecasters=forecasters,
                    meta_model_type=meta_model_type
                )
            else:
                # Standard ensemble methods (weighted_average or simple_average)
                ensemble = EnsembleForecaster(
                    forecasters=forecasters,
                    weights=weights,
                    ensemble_method=ensemble_method
                )
            
            # Set the confidence interval method if specified
            confidence_interval_method = config.get('confidence_interval_method', 'bootstrapping')
            if hasattr(ensemble, 'confidence_interval_method'):
                ensemble.confidence_interval_method = confidence_interval_method
            
            with st.spinner(f"Running {ensemble_method} ensemble forecast with {len(forecasters)} forecasters..."):
                # Use progress indicators
                progress_container = st.empty()
                progress_container.text("Getting ensemble prediction...")
                
                # Get the ensemble prediction
                probability = await ensemble.predict(input.question)
                progress_container.text(f"Ensemble Prediction: {probability:.3f}")
                
                # Get the explanation
                explanation_container = st.empty()
                explanation_container.text("Getting ensemble explanation...")
                rationale = await ensemble.explain(input.question)
                explanation_container.text("Got ensemble explanation! ✓")
                
                # Get the confidence interval
                interval_container = st.empty()
                interval_container.text("Getting confidence interval...")
                interval = await ensemble.confidence_interval(input.question)
                interval_container.text(f"Confidence Interval: [{interval[0]:.2f}, {interval[1]:.2f}]")
                
                # Additional info for advanced ensemble methods
                ensemble_info = ""
                if ensemble_method == 'dynamic_weighting':
                    ensemble_info = (
                        "\n\n## Dynamic Weighting Information\n\n"
                        f"Current weights: {[float(f'{w:.3f}') for w in ensemble.weights]}\n"
                        f"Adjustment rate: {adjustment_rate}\n\n"
                        "Note: Weights will adjust over time as outcomes are recorded."
                    )
                elif ensemble_method == 'stacking':
                    training_samples = len(ensemble.training_data.get("outcomes", []))
                    ensemble_info = (
                        "\n\n## Stacking Information\n\n"
                        f"Meta-model type: {meta_model_type}\n"
                        f"Training samples: {training_samples}\n"
                        f"Meta-model status: {'Trained' if training_samples >= ensemble.min_training_samples else 'Not yet trained'}\n\n"
                        "Note: The meta-model will improve as more outcomes are recorded."
                    )
                
                # Create the report
                report = BinaryReport(
                    question=input.question,
                    prediction=probability,
                    explanation=f"# {ensemble_method.replace('_', ' ').title()} Ensemble Forecast\n\n{rationale}{ensemble_info}",
                    other_notes=f"Confidence Interval: [{interval[0]:.2f}, {interval[1]:.2f}]",
                )
                
                return report

    @classmethod
    async def _display_outputs(cls, outputs: list[BinaryReport]) -> None:
        ReportDisplayer.display_report_list(outputs)
        
        # Add additional visualizations for ensemble
        st.subheader("Ensemble Metrics")
        
        # If the report contains confidence intervals in other_notes
        if outputs and outputs[0].other_notes and "Confidence Interval" in outputs[0].other_notes:
            try:
                # Extract interval from other_notes
                interval_str = outputs[0].other_notes
                pattern = r"\[([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+)\]"
                match = re.search(pattern, interval_str)
                if match:
                    lower = float(match.group(1))
                    upper = float(match.group(2))
                    
                    # Display the confidence interval as a chart
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    fig, ax = plt.subplots(figsize=(10, 3))
                    x = np.linspace(0, 1, 100)
                    
                    # Create a narrow normal distribution around the prediction
                    mean = outputs[0].prediction
                    std = (upper - lower) / 3.92  # approx 95% confidence
                    y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                    
                    # Plot the distribution
                    ax.plot(x, y)
                    ax.fill_between(x, 0, y, alpha=0.2)
                    
                    # Add vertical lines for prediction and interval
                    ax.axvline(x=mean, color='r', linestyle='-', alpha=0.7, label=f'Prediction: {mean:.2f}')
                    ax.axvline(x=lower, color='g', linestyle='--', alpha=0.7, label=f'Lower: {lower:.2f}')
                    ax.axvline(x=upper, color='g', linestyle='--', alpha=0.7, label=f'Upper: {upper:.2f}')
                    
                    ax.set_xlabel('Probability')
                    ax.set_ylabel('Density')
                    ax.set_title('Ensemble Forecast Distribution')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Add explanation of calibration
                    st.info(
                        "The chart above shows the probability distribution of the ensemble forecast. "
                        "The red line represents the point prediction, while the green lines show the "
                        "confidence interval bounds. The shaded area represents the probability density."
                    )
                    
                    # Display information about selected ensemble method and confidence interval method
                    config = st.session_state.get('forecaster_config', {})
                    ensemble_method = config.get('ensemble_method', 'weighted_average')
                    confidence_interval_method = config.get('confidence_interval_method', 'bootstrapping')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"Ensemble Method: {ensemble_method.replace('_', ' ').title()}")
                        if ensemble_method == 'dynamic_weighting':
                            st.write(
                                "Dynamic weighting automatically adjusts forecaster weights based on their recent "
                                "performance. Weights adjust over time as outcomes are recorded, favoring forecasters "
                                "that have been more accurate recently."
                            )
                        elif ensemble_method == 'stacking':
                            st.write(
                                "Stacking uses a meta-model (machine learning model) to learn the optimal way to "
                                "combine forecasts. The meta-model improves as more outcomes are recorded, learning "
                                "which forecasters perform best in which situations."
                            )
                            meta_model_type = config.get('meta_model_type', 'logistic_regression')
                            st.text(f"Meta-model type: {meta_model_type}")
                        elif ensemble_method == 'simple_average':
                            st.write(
                                "Simple average gives equal weight to all forecasters, regardless of their "
                                "historical performance. This method is robust when there is limited historical data."
                            )
                        else:  # weighted_average
                            st.write(
                                "Weighted average assigns different importance to each forecaster based on "
                                "user-defined weights. This allows manual tuning of the ensemble based on "
                                "domain knowledge or prior performance."
                            )
                    
                    with col2:
                        st.subheader(f"Confidence Interval Method: {confidence_interval_method.replace('_', ' ').title()}")
                        if confidence_interval_method == 'bootstrapping':
                            st.write(
                                "Bootstrapping resamples predictions to estimate uncertainty while accounting for "
                                "correlations between forecasters. This provides more robust intervals than "
                                "traditional methods, especially when forecasters may have correlated errors."
                            )
                        else:  # variance_propagation
                            st.write(
                                "Variance propagation uses a weighted average of individual confidence intervals. "
                                "This method is computationally efficient but assumes forecaster errors are "
                                "independent and normally distributed."
                            )
                    
                    # Add forecaster contributions if available
                    if hasattr(outputs[0], 'metadata') and outputs[0].metadata:
                        st.subheader("Forecaster Contributions")
                        if "forecaster_metrics" in outputs[0].metadata:
                            st.json(outputs[0].metadata["forecaster_metrics"])
            except Exception as e:
                st.warning(f"Could not display confidence interval chart: {e}")

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
    dotenv.load_dotenv()
    EnsembleForecastPage.main() 