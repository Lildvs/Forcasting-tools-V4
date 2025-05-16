import logging
import re
import asyncio
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import dotenv
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.ai_models.model_interfaces.advanced_ensemble_methods import DynamicWeightingEnsemble, StackingEnsemble

logger = logging.getLogger(__name__)

class EnsembleMetricsPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "📈 Ensemble Metrics"
    URL_PATH: str = "/ensemble-metrics"
    INPUT_TYPE = Jsonable
    OUTPUT_TYPE = Jsonable

    # Form input keys
    ENSEMBLE_TYPE_SELECT = "ensemble_type_select"
    TIME_RANGE = "time_range_ensemble"
    REFRESH_BUTTON = "refresh_button_ensemble"
    
    # Default data paths
    DYNAMIC_WEIGHTS_PATH = "forecasting_tools/data/dynamic_weights_history.pkl"
    STACKING_DATA_PATH = "forecasting_tools/data/stacking_ensemble_data.pkl"

    @classmethod
    async def _display_intro_text(cls) -> None:
        st.write(
            "This dashboard displays performance metrics for ensemble forecasting methods. "
            "View weight evolution for dynamic weighting and model learning for stacking."
        )

    @classmethod
    async def _get_input(cls) -> Jsonable | None:
        # Create a form for selecting display options
        with st.form("ensemble_metrics_form"):
            ensemble_type = st.selectbox(
                "Select Ensemble Type",
                options=["dynamic_weighting", "stacking"],
                key=cls.ENSEMBLE_TYPE_SELECT
            )
            
            refresh = st.form_submit_button("Refresh Data")
            
            if refresh:
                return Jsonable(value={"ensemble_type": ensemble_type, "refresh": True})
        
        # Allow viewing without submitting the form
        return Jsonable(value={"ensemble_type": "dynamic_weighting", "refresh": False})

    @classmethod
    async def _run_tool(cls, input: Jsonable) -> Jsonable:
        ensemble_type = input.value.get("ensemble_type", "dynamic_weighting")
        
        # Check if data files exist
        dynamic_weights_exists = os.path.exists(cls.DYNAMIC_WEIGHTS_PATH)
        stacking_data_exists = os.path.exists(cls.STACKING_DATA_PATH)
        
        if ensemble_type == "dynamic_weighting" and not dynamic_weights_exists:
            st.warning("No dynamic weighting data available yet. Try making some forecasts first.")
            return Jsonable(value={"success": False, "message": "No data available"})
        
        if ensemble_type == "stacking" and not stacking_data_exists:
            st.warning("No stacking ensemble data available yet. Try making some forecasts first.")
            return Jsonable(value={"success": False, "message": "No data available"})
        
        # Load and process the appropriate data
        if ensemble_type == "dynamic_weighting":
            try:
                import pickle
                with open(cls.DYNAMIC_WEIGHTS_PATH, 'rb') as f:
                    performance_history = pickle.load(f)
                    
                # Convert to more usable format for display
                forecaster_count = len(performance_history)
                history_data = {
                    "forecaster_count": forecaster_count,
                    "history": performance_history
                }
                
                return Jsonable(value={
                    "ensemble_type": ensemble_type,
                    "data": history_data,
                    "success": True
                })
            except Exception as e:
                st.error(f"Error loading dynamic weighting data: {e}")
                return Jsonable(value={"success": False, "message": str(e)})
        
        elif ensemble_type == "stacking":
            try:
                import pickle
                with open(cls.STACKING_DATA_PATH, 'rb') as f:
                    training_data = pickle.load(f)
                    
                return Jsonable(value={
                    "ensemble_type": ensemble_type,
                    "data": training_data,
                    "success": True
                })
            except Exception as e:
                st.error(f"Error loading stacking data: {e}")
                return Jsonable(value={"success": False, "message": str(e)})
        
        return Jsonable(value={"success": False, "message": "Invalid ensemble type"})

    @classmethod
    async def _display_outputs(cls, outputs: list[Jsonable]) -> None:
        if not outputs or not outputs[0].value.get("success", False):
            st.info("Select an ensemble type and click 'Refresh Data' to view metrics.")
            
            # Display information about ensemble methods
            st.subheader("About Ensemble Methods")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Dynamic Weighting")
                st.write(
                    "Dynamic weighting adjusts forecaster weights based on recent performance. "
                    "Weights evolve over time as questions are resolved, favoring forecasters "
                    "that have been most accurate recently."
                )
                
                st.write("**Benefits:**")
                st.write("- Adapts to changing forecaster performance")
                st.write("- Works well with limited historical data")
                st.write("- Easily interpretable")
                
                st.write("**When to use:**")
                st.write("- When you have at least 10-20 resolved questions")
                st.write("- When forecaster performance varies over time")
            
            with col2:
                st.write("### Stacking")
                st.write(
                    "Stacking uses a meta-model (machine learning model) to learn the optimal "
                    "way to combine forecasts. It can discover complex relationships between "
                    "forecaster outputs to maximize accuracy."
                )
                
                st.write("**Benefits:**")
                st.write("- Can learn complex patterns")
                st.write("- Often produces more accurate ensembles")
                st.write("- Adapts to different question types")
                
                st.write("**When to use:**")
                st.write("- When you have 100+ resolved questions")
                st.write("- When you want maximum accuracy")
            
            # Add explanation about confidence interval methods
            st.subheader("Confidence Interval Methods")
            
            st.write(
                "Our ensemble forecasting system offers advanced confidence interval estimation "
                "using bootstrapping techniques. This provides more robust uncertainty estimates "
                "than traditional methods."
            )
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("### Simple Averaging / Variance Propagation")
                st.write(
                    "Traditional methods combine confidence intervals by averaging the bounds "
                    "or using variance formulas that assume independence between forecasters."
                )
                
                st.write("**Characteristics:**")
                st.write("- Fast and computationally efficient")
                st.write("- Works well when forecaster errors are independent")
                st.write("- May underestimate uncertainty when forecasters make correlated errors")
                st.write("- Assumes normally distributed errors")
            
            with col4:
                st.write("### Bootstrapping")
                st.write(
                    "Bootstrapping resamples predictions to empirically estimate the ensemble's "
                    "confidence interval, capturing complex relationships between forecasters."
                )
                
                st.write("**Characteristics:**")
                st.write("- More computationally intensive but more robust")
                st.write("- Captures correlations between forecaster errors")
                st.write("- Handles non-normal error distributions")
                st.write("- Provides realistic uncertainty estimates for decision-making")
                st.write("- Particularly valuable when combining diverse forecasting methods")
            
            st.info(
                "**Recommendation:** Bootstrapping is preferred for most applications as it provides "
                "more reliable uncertainty estimates, especially when forecasters may be using similar "
                "information or methodologies."
            )
            
            return
        
        output = outputs[0].value
        ensemble_type = output.get("ensemble_type")
        data = output.get("data", {})
        
        if ensemble_type == "dynamic_weighting":
            cls._display_dynamic_weighting_metrics(data)
        elif ensemble_type == "stacking":
            cls._display_stacking_metrics(data)

    @classmethod
    def _display_dynamic_weighting_metrics(cls, data: dict) -> None:
        """Display metrics for dynamic weighting ensemble."""
        st.subheader("Dynamic Weighting Performance")
        
        history = data.get("history", [])
        forecaster_count = data.get("forecaster_count", 0)
        
        if not history or forecaster_count == 0:
            st.warning("No performance history available yet.")
            return
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Weight Evolution", "Performance Metrics", "Predictions"])
        
        with tab1:
            st.write("### Weight Evolution Over Time")
            
            # Extract resolved questions for each forecaster
            all_resolved = []
            for model_idx, model_history in enumerate(history):
                resolved = [(q_hash, pred, outcome, score) 
                           for q_hash, pred, outcome, score in model_history 
                           if outcome is not None]
                
                if resolved:
                    # Sort by order of addition
                    all_resolved.append(resolved)
            
            if not all_resolved or len(all_resolved) < 2:
                st.info("Not enough resolved questions available to show weight evolution.")
                return
            
            # Calculate weight evolution based on performance
            weights_over_time = cls._calculate_weight_evolution(all_resolved)
            
            # Plot weight evolution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i in range(len(weights_over_time[0])):
                ax.plot([weights[i] for weights in weights_over_time], 
                       label=f"Forecaster {i+1}")
            
            ax.set_xlabel("Questions Resolved")
            ax.set_ylabel("Weight")
            ax.set_title("Ensemble Weight Evolution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.info(
                "This chart shows how weights would evolve as questions are resolved. "
                "Higher weights indicate better recent performance."
            )
        
        with tab2:
            st.write("### Performance Metrics by Forecaster")
            
            # Calculate performance metrics for each forecaster
            metrics = []
            
            for model_idx, model_history in enumerate(history):
                resolved = [(pred, outcome, score) 
                           for _, pred, outcome, score in model_history 
                           if outcome is not None]
                
                if resolved:
                    preds, outcomes, scores = zip(*resolved)
                    
                    # Calculate metrics
                    brier_score = np.mean(scores)
                    accuracy = np.mean([1 if (p > 0.5 and o) or (p < 0.5 and not o) else 0 
                                      for p, o in zip(preds, outcomes)])
                    
                    metrics.append({
                        "Forecaster": f"Forecaster {model_idx+1}",
                        "Questions": len(resolved),
                        "Brier Score": brier_score,
                        "Accuracy": accuracy
                    })
            
            if metrics:
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df)
                
                # Plot Brier scores
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(metrics_df["Forecaster"], metrics_df["Brier Score"])
                ax.set_ylabel("Brier Score (lower is better)")
                ax.set_title("Forecaster Brier Scores")
                st.pyplot(fig)
            else:
                st.info("No performance metrics available yet.")
        
        with tab3:
            st.write("### Recent Predictions")
            
            # Display recent predictions for each forecaster
            recent_predictions = []
            
            for model_idx, model_history in enumerate(history):
                for q_hash, pred, outcome, score in model_history[-10:]:  # Last 10 predictions
                    if outcome is not None:
                        recent_predictions.append({
                            "Forecaster": f"Forecaster {model_idx+1}",
                            "Question Hash": q_hash[:8],
                            "Prediction": pred,
                            "Outcome": "Yes" if outcome else "No",
                            "Brier Score": score
                        })
            
            if recent_predictions:
                recent_df = pd.DataFrame(recent_predictions)
                st.dataframe(recent_df)
            else:
                st.info("No predictions available yet.")

    @classmethod
    def _display_stacking_metrics(cls, data: dict) -> None:
        """Display metrics for stacking ensemble."""
        st.subheader("Stacking Ensemble Performance")
        
        features = data.get("features", [])
        outcomes = data.get("outcomes", [])
        
        if not features or not outcomes:
            st.warning("No training data available yet.")
            return
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Model Learning", "Feature Importance", "Training Data"])
        
        with tab1:
            st.write("### Meta-Model Learning Curve")
            
            # Calculate learning curve (accuracy as training data increases)
            if len(features) > 5:
                # Calculate cumulative accuracy as training data increases
                accuracies = []
                
                for i in range(5, len(features), max(1, len(features) // 20)):
                    X_train = np.array(features[:i])
                    y_train = np.array(outcomes[:i])
                    
                    try:
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.model_selection import cross_val_score
                        
                        model = LogisticRegression(solver='lbfgs')
                        scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)))
                        accuracies.append((i, np.mean(scores)))
                    except Exception as e:
                        st.warning(f"Error calculating learning curve: {e}")
                        break
                
                if accuracies:
                    sizes, scores = zip(*accuracies)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(sizes, scores, 'o-')
                    ax.set_xlabel("Training Examples")
                    ax.set_ylabel("Cross-Validation Accuracy")
                    ax.set_title("Meta-Model Learning Curve")
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    st.info(
                        "This chart shows how the meta-model's accuracy improves as more "
                        "training data becomes available. A rising curve indicates the model "
                        "is learning from the data."
                    )
                else:
                    st.info("Not enough data to calculate learning curve.")
            else:
                st.info("Need at least 5 training examples to show learning curve.")
        
        with tab2:
            st.write("### Feature Importance")
            
            # Calculate feature importance (which forecasters the meta-model relies on)
            if len(features) > 0 and len(features[0]) > 0:
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.ensemble import RandomForestClassifier
                    
                    X = np.array(features)
                    y = np.array(outcomes)
                    
                    # Try both logistic regression and random forest
                    lr_model = LogisticRegression(solver='lbfgs')
                    lr_model.fit(X, y)
                    
                    rf_model = RandomForestClassifier(n_estimators=100)
                    rf_model.fit(X, y)
                    
                    # Get coefficients/importance
                    lr_coefs = lr_model.coef_[0]
                    rf_importance = rf_model.feature_importances_
                    
                    # Create DataFrame for display
                    importance_df = pd.DataFrame({
                        "Forecaster": [f"Forecaster {i+1}" for i in range(len(lr_coefs))],
                        "Logistic Regression Coefficient": lr_coefs,
                        "Random Forest Importance": rf_importance
                    })
                    
                    st.dataframe(importance_df)
                    
                    # Plot feature importance
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Logistic regression coefficients
                    ax1.bar(importance_df["Forecaster"], importance_df["Logistic Regression Coefficient"])
                    ax1.set_ylabel("Coefficient")
                    ax1.set_title("Logistic Regression Coefficients")
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Random forest importance
                    ax2.bar(importance_df["Forecaster"], importance_df["Random Forest Importance"])
                    ax2.set_ylabel("Importance")
                    ax2.set_title("Random Forest Feature Importance")
                    ax2.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info(
                        "These charts show which forecasters are most important to the meta-model. "
                        "Higher values indicate forecasters that have more influence on the final prediction."
                    )
                except Exception as e:
                    st.warning(f"Error calculating feature importance: {e}")
            else:
                st.info("Not enough data to calculate feature importance.")
        
        with tab3:
            st.write("### Training Data")
            
            # Display the training data
            if features and outcomes:
                # Create DataFrame with features and outcomes
                data_rows = []
                
                for i, (feature, outcome) in enumerate(zip(features, outcomes)):
                    row = {"Example": i+1, "Outcome": "Yes" if outcome else "No"}
                    for j, pred in enumerate(feature):
                        row[f"Forecaster {j+1}"] = pred
                    data_rows.append(row)
                
                training_df = pd.DataFrame(data_rows)
                st.dataframe(training_df)
                
                # Display summary statistics
                st.write(f"Number of training examples: {len(features)}")
                st.write(f"Number of forecasters: {len(features[0]) if features else 0}")
                st.write(f"Positive outcomes: {sum(outcomes)} ({sum(outcomes)/len(outcomes)*100:.1f}%)")
            else:
                st.info("No training data available yet.")

    @classmethod
    def _calculate_weight_evolution(cls, all_resolved) -> list:
        """Calculate weight evolution over time for dynamic weighting."""
        # Initialize with equal weights
        model_count = len(all_resolved)
        current_weights = [1.0 / model_count] * model_count
        
        # Find minimum length (number of resolved questions) across all forecasters
        min_length = min(len(resolved) for resolved in all_resolved)
        
        # Track weight evolution
        weights_over_time = [current_weights.copy()]
        
        # Parameters from DynamicWeightingEnsemble
        adjustment_rate = 0.1
        performance_window = 10
        
        # Calculate weight updates for each step
        for step in range(min_length):
            # Get recent performance for each model
            recent_performance = []
            
            for model_idx, resolved in enumerate(all_resolved):
                # Get data for this step
                _, _, _, score = resolved[step]
                
                # Use window of past scores if available
                start_idx = max(0, step - performance_window + 1)
                window_scores = [r[3] for r in resolved[start_idx:step+1]]
                
                # Calculate average score (lower is better)
                avg_score = np.mean(window_scores)
                recent_performance.append(avg_score)
            
            # Convert scores to weights (lower score = higher weight)
            # Add small constant to avoid division by zero
            epsilon = 1e-10
            inverted_scores = [1.0 / (score + epsilon) for score in recent_performance]
            
            # Normalize to get new weights
            total = sum(inverted_scores)
            new_weights = [score / total for score in inverted_scores]
            
            # Apply adjustment rate to smooth weight changes
            for i in range(len(current_weights)):
                current_weights[i] = (1 - adjustment_rate) * current_weights[i] + adjustment_rate * new_weights[i]
            
            # Renormalize weights
            total = sum(current_weights)
            current_weights = [w / total for w in current_weights]
            
            # Add to history
            weights_over_time.append(current_weights.copy())
        
        return weights_over_time


if __name__ == "__main__":
    dotenv.load_dotenv()
    EnsembleMetricsPage.main() 