import logging
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.ai_models.model_interfaces.ensemble_forecaster import EnsembleForecaster

logger = logging.getLogger(__name__)

class DynamicWeightingEnsemble(EnsembleForecaster):
    """
    An ensemble forecaster that dynamically adjusts weights based on recent performance.
    
    This forecaster tracks the performance of each component model and adjusts
    weights to favor models that have performed well recently.
    """
    
    def __init__(
        self,
        forecasters: List[ForecasterBase],
        initial_weights: Optional[List[float]] = None,
        adjustment_rate: float = 0.1,
        performance_window: int = 10,
        performance_metric: str = "brier",  # "brier" or "log_loss"
        performance_history_path: Optional[str] = None,
        confidence_interval_method: str = "bootstrapping"
    ):
        """
        Initialize the dynamic weighting ensemble.
        
        Args:
            forecasters: List of forecaster instances to ensemble
            initial_weights: Initial weights for each forecaster (normalized to sum to 1)
            adjustment_rate: How quickly to adjust weights (0-1)
            performance_window: Number of recent predictions to consider
            performance_metric: Metric to use for adjusting weights
            performance_history_path: Path to save/load performance history
            confidence_interval_method: Method for calculating confidence intervals
                                      ("bootstrapping", "variance_propagation")
        """
        # Initialize with equal weights if none provided
        if initial_weights is None:
            initial_weights = [1.0 / len(forecasters) for _ in forecasters]
        
        # Ensure weights sum to 1
        total = sum(initial_weights)
        normalized_weights = [w / total for w in initial_weights]
        
        # Initialize base ensemble with initial weights
        super().__init__(
            forecasters=forecasters,
            weights=normalized_weights,
            ensemble_method="weighted_average",
            confidence_interval_method=confidence_interval_method
        )
        
        self.model_name = "DynamicWeightingEnsemble"
        self.adjustment_rate = adjustment_rate
        self.performance_window = performance_window
        self.performance_metric = performance_metric
        
        # Performance history for each forecaster [model_idx][question_idx] = (prediction, outcome, score)
        self.performance_history = [[] for _ in forecasters]
        
        # Path for saving/loading performance history
        self.performance_history_path = performance_history_path or "forecasting_tools/data/dynamic_weights_history.pkl"
        
        # Try to load existing performance history
        self._load_performance_history()
        
        logger.info(f"Initialized DynamicWeightingEnsemble with {len(forecasters)} forecasters")
    
    def _load_performance_history(self):
        """Load performance history from disk if available."""
        try:
            if os.path.exists(self.performance_history_path):
                with open(self.performance_history_path, 'rb') as f:
                    saved_history = pickle.load(f)
                
                # Only use if structure matches
                if (isinstance(saved_history, list) and 
                    len(saved_history) == len(self.forecasters)):
                    self.performance_history = saved_history
                    logger.info(f"Loaded performance history from {self.performance_history_path}")
        except Exception as e:
            logger.warning(f"Error loading performance history: {e}")
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        try:
            os.makedirs(os.path.dirname(self.performance_history_path), exist_ok=True)
            with open(self.performance_history_path, 'wb') as f:
                pickle.dump(self.performance_history, f)
            logger.debug(f"Saved performance history to {self.performance_history_path}")
        except Exception as e:
            logger.warning(f"Error saving performance history: {e}")
    
    def _calculate_brier_score(self, prediction: float, outcome: bool) -> float:
        """Calculate Brier score (lower is better)."""
        outcome_value = 1.0 if outcome else 0.0
        return (prediction - outcome_value) ** 2
    
    def _calculate_log_loss(self, prediction: float, outcome: bool) -> float:
        """Calculate log loss (lower is better)."""
        outcome_value = 1.0 if outcome else 0.0
        # Clip prediction to avoid log(0)
        p = max(min(prediction, 0.9999), 0.0001)
        if outcome:
            return -np.log(p)
        else:
            return -np.log(1 - p)
    
    def record_outcome(self, question_hash: str, outcome: bool):
        """
        Record the actual outcome for a previously predicted question.
        
        Args:
            question_hash: Hash of the question text
            outcome: Whether the event actually occurred (True/False)
        """
        # Check if we have predictions for this question in our performance history
        for model_idx, model_history in enumerate(self.performance_history):
            for i, (q_hash, pred, _) in enumerate(model_history):
                if q_hash == question_hash:
                    # Calculate performance score
                    if self.performance_metric == "brier":
                        score = self._calculate_brier_score(pred, outcome)
                    else:  # log_loss
                        score = self._calculate_log_loss(pred, outcome)
                    
                    # Update history with outcome and score
                    self.performance_history[model_idx][i] = (q_hash, pred, outcome, score)
                    break
        
        # Update weights based on recent performance
        self._update_weights()
        
        # Save updated history
        self._save_performance_history()
        
        logger.info(f"Recorded outcome {outcome} for question hash {question_hash[:8]}...")
    
    def _update_weights(self):
        """Update weights based on recent performance."""
        # Only update if we have enough data
        min_history_length = min(len(history) for history in self.performance_history)
        if min_history_length < 2:  # Need at least a couple of data points
            logger.debug("Not enough data to update weights")
            return
        
        # Calculate recent performance for each model
        recent_performance = []
        
        for model_idx, model_history in enumerate(self.performance_history):
            # Filter to entries with outcomes
            resolved_history = [(p, o, s) for _, p, o, s in model_history if o is not None]
            
            if len(resolved_history) == 0:
                # No resolved questions for this model yet
                recent_performance.append(float('inf'))  # Worst possible performance
                continue
            
            # Sort by recency (assuming later entries are more recent)
            resolved_history = resolved_history[-self.performance_window:]
            
            # Calculate average score (lower is better for both brier and log loss)
            avg_score = np.mean([s for _, _, s in resolved_history])
            recent_performance.append(avg_score)
        
        # Convert scores to weights (lower score = higher weight)
        if all(score == float('inf') for score in recent_performance):
            # No resolved questions yet, keep equal weights
            return
        
        # Replace infinities with worst finite score * 2
        finite_scores = [s for s in recent_performance if s != float('inf')]
        if finite_scores:
            worst_finite = max(finite_scores)
            recent_performance = [worst_finite * 2 if s == float('inf') else s for s in recent_performance]
        
        # Invert scores (lower is better, so invert for weights)
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        inverted_scores = [1.0 / (score + epsilon) for score in recent_performance]
        
        # Normalize to get new weights
        total = sum(inverted_scores)
        new_weights = [score / total for score in inverted_scores]
        
        # Apply adjustment rate to smooth weight changes
        for i in range(len(self.weights)):
            self.weights[i] = (1 - self.adjustment_rate) * self.weights[i] + self.adjustment_rate * new_weights[i]
        
        # Renormalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        logger.debug(f"Updated weights: {self.weights}")
    
    async def predict(self, question, context=None):
        """
        Return a probability forecast using the ensemble with current weights.
        Also records the individual model predictions for later performance tracking.
        """
        # Get predictions from each forecaster
        predictions = []
        question_hash = str(hash(question.question_text))
        
        for i, forecaster in enumerate(self.forecasters):
            try:
                prediction = await forecaster.predict(question, context)
                predictions.append(prediction)
                
                # Record prediction in performance history
                self.performance_history[i].append((question_hash, prediction, None))
            except Exception as e:
                logger.error(f"Error from forecaster {i}: {e}")
                # Use 0.5 as fallback
                predictions.append(0.5)
        
        # Limit history length if needed
        max_history = 1000
        for i in range(len(self.performance_history)):
            if len(self.performance_history[i]) > max_history:
                self.performance_history[i] = self.performance_history[i][-max_history:]
        
        # Save updated history
        self._save_performance_history()
        
        # Calculate weighted average
        weighted_prediction = sum(p * w for p, w in zip(predictions, self.weights))
        
        logger.debug(f"DynamicWeightingEnsemble prediction: {weighted_prediction:.3f} (weights: {self.weights})")
        return weighted_prediction


class StackingEnsemble(ForecasterBase):
    """
    A stacking ensemble that uses a meta-model to combine forecaster outputs.
    
    This forecaster trains a meta-model (e.g., logistic regression) on the outputs
    of base forecasters to learn optimal combination weights.
    """
    
    def __init__(
        self,
        forecasters: List[ForecasterBase],
        meta_model_type: str = "logistic_regression",  # "logistic_regression" or "random_forest"
        training_data_path: Optional[str] = None,
        min_training_samples: int = 20,
        retrain_frequency: int = 10,
        confidence_interval_method: str = "bootstrapping"
    ):
        """
        Initialize the stacking ensemble.
        
        Args:
            forecasters: List of forecaster instances to ensemble
            meta_model_type: Type of meta-model to use
            training_data_path: Path to save/load training data
            min_training_samples: Minimum samples before training meta-model
            retrain_frequency: How often to retrain the meta-model (in questions)
            confidence_interval_method: Method for calculating confidence intervals
                                      ("bootstrapping", "variance_propagation")
        """
        self.forecasters = forecasters
        self.meta_model_type = meta_model_type
        self.training_data_path = training_data_path or "forecasting_tools/data/stacking_ensemble_data.pkl"
        self.min_training_samples = min_training_samples
        self.retrain_frequency = retrain_frequency
        self.confidence_interval_method = confidence_interval_method
        self.model_name = "StackingEnsemble"
        
        # Initialize meta-model
        self.meta_model = None
        self._initialize_meta_model()
        
        # Training data: List of (features, outcome) pairs
        # features = list of forecaster predictions
        self.training_data = {"features": [], "outcomes": []}
        
        # Question tracking for new predictions
        self.pending_predictions = {}  # {question_hash: [forecaster_predictions]}
        
        # Counter for knowing when to retrain
        self.questions_since_retrain = 0
        
        # Load existing training data if available
        self._load_training_data()
        
        # Train initial model if sufficient data
        if len(self.training_data["outcomes"]) >= self.min_training_samples:
            self._train_meta_model()
        
        logger.info(f"Initialized StackingEnsemble with {len(forecasters)} forecasters")
    
    def _initialize_meta_model(self):
        """Initialize the meta-model based on the selected type."""
        if self.meta_model_type == "logistic_regression":
            self.meta_model = LogisticRegression(solver='lbfgs', C=1.0)
        elif self.meta_model_type == "random_forest":
            self.meta_model = RandomForestClassifier(n_estimators=100, max_depth=3)
        else:
            logger.warning(f"Unknown meta_model_type: {self.meta_model_type}, using logistic regression")
            self.meta_model = LogisticRegression(solver='lbfgs', C=1.0)
    
    def _load_training_data(self):
        """Load training data from disk if available."""
        try:
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'rb') as f:
                    saved_data = pickle.load(f)
                
                # Only use if structure matches
                if (isinstance(saved_data, dict) and 
                    "features" in saved_data and 
                    "outcomes" in saved_data):
                    self.training_data = saved_data
                    logger.info(f"Loaded {len(saved_data['outcomes'])} training samples")
        except Exception as e:
            logger.warning(f"Error loading training data: {e}")
    
    def _save_training_data(self):
        """Save training data to disk."""
        try:
            os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
            with open(self.training_data_path, 'wb') as f:
                pickle.dump(self.training_data, f)
            logger.debug(f"Saved training data with {len(self.training_data['outcomes'])} samples")
        except Exception as e:
            logger.warning(f"Error saving training data: {e}")
    
    def _train_meta_model(self):
        """Train the meta-model on available data."""
        if len(self.training_data["outcomes"]) < self.min_training_samples:
            logger.debug(f"Not enough samples ({len(self.training_data['outcomes'])}) to train meta-model")
            return
        
        try:
            # Convert to numpy arrays
            X = np.array(self.training_data["features"])
            y = np.array(self.training_data["outcomes"])
            
            # Train the meta-model
            self.meta_model.fit(X, y)
            
            # Reset counter
            self.questions_since_retrain = 0
            
            logger.info(f"Trained meta-model on {len(y)} samples")
        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
    
    def record_outcome(self, question_hash: str, outcome: bool):
        """
        Record the actual outcome for a previously predicted question.
        
        Args:
            question_hash: Hash of the question text
            outcome: Whether the event actually occurred (True/False)
        """
        # Check if we have predictions for this question
        if question_hash in self.pending_predictions:
            # Add to training data
            features = self.pending_predictions[question_hash]
            self.training_data["features"].append(features)
            self.training_data["outcomes"].append(1 if outcome else 0)
            
            # Remove from pending
            del self.pending_predictions[question_hash]
            
            # Increment counter and check if we should retrain
            self.questions_since_retrain += 1
            if self.questions_since_retrain >= self.retrain_frequency:
                self._train_meta_model()
            
            # Save updated data
            self._save_training_data()
            
            logger.info(f"Recorded outcome {outcome} for question hash {question_hash[:8]}...")
    
    async def predict(self, question, context=None):
        """
        Return a probability forecast using the stacking ensemble.
        """
        # Get predictions from each forecaster
        predictions = []
        
        for forecaster in self.forecasters:
            try:
                prediction = await forecaster.predict(question, context)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error from forecaster: {e}")
                # Use 0.5 as fallback
                predictions.append(0.5)
        
        # Store predictions for later training
        question_hash = str(hash(question.question_text))
        self.pending_predictions[question_hash] = predictions
        
        # If meta-model is not trained yet, use simple average
        if (self.meta_model is None or 
            len(self.training_data["outcomes"]) < self.min_training_samples):
            logger.debug("Meta-model not trained yet, using average")
            return sum(predictions) / len(predictions)
        
        # Use meta-model to combine predictions
        try:
            # Reshape for sklearn
            X = np.array([predictions])
            
            # Get probability
            probability = self.meta_model.predict_proba(X)[0][1]
            
            logger.debug(f"StackingEnsemble prediction: {probability:.3f}")
            return probability
        except Exception as e:
            logger.error(f"Error using meta-model: {e}, falling back to average")
            return sum(predictions) / len(predictions)
    
    async def explain(self, question, context=None):
        """
        Return an explanation of the stacking ensemble forecast.
        """
        # Get predictions and explanations from each forecaster
        predictions = []
        explanations = []
        
        for i, forecaster in enumerate(self.forecasters):
            try:
                prediction = await forecaster.predict(question, context)
                explanation = await forecaster.explain(question, context)
                
                predictions.append(prediction)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Error from forecaster {i}: {e}")
                predictions.append(0.5)
                explanations.append(f"Error from forecaster {i}: {e}")
        
        # Generate meta-model explanation
        meta_explanation = "## Stacking Ensemble Forecast\n\n"
        
        if (self.meta_model is None or 
            len(self.training_data["outcomes"]) < self.min_training_samples):
            meta_explanation += (
                f"*Meta-model not yet trained (have {len(self.training_data['outcomes'])} samples, "
                f"need {self.min_training_samples}). Using simple average.*\n\n"
            )
        else:
            # Add information about the meta-model
            meta_explanation += f"*Using {self.meta_model_type} meta-model trained on "
            meta_explanation += f"{len(self.training_data['outcomes'])} historical questions.*\n\n"
            
            # For logistic regression, show coefficients
            if self.meta_model_type == "logistic_regression":
                try:
                    coefs = self.meta_model.coef_[0]
                    intercept = self.meta_model.intercept_[0]
                    
                    meta_explanation += "**Model Coefficients:**\n\n"
                    meta_explanation += f"Intercept: {intercept:.4f}\n"
                    
                    for i, coef in enumerate(coefs):
                        meta_explanation += f"Forecaster {i+1}: {coef:.4f}\n"
                        
                    meta_explanation += "\n"
                except:
                    pass
            
            # For random forest, show feature importance
            elif self.meta_model_type == "random_forest":
                try:
                    importances = self.meta_model.feature_importances_
                    
                    meta_explanation += "**Feature Importance:**\n\n"
                    for i, importance in enumerate(importances):
                        meta_explanation += f"Forecaster {i+1}: {importance:.4f}\n"
                        
                    meta_explanation += "\n"
                except:
                    pass
        
        # Add individual forecaster predictions
        meta_explanation += "**Individual Forecaster Predictions:**\n\n"
        
        for i, prediction in enumerate(predictions):
            meta_explanation += f"Forecaster {i+1}: {prediction:.4f}\n"
            
        meta_explanation += "\n"
        
        # Add individual explanations (shortened)
        meta_explanation += "**Individual Forecaster Explanations:**\n\n"
        
        for i, explanation in enumerate(explanations):
            # Truncate explanations to avoid extremely long output
            max_length = 500
            if len(explanation) > max_length:
                truncated = explanation[:max_length] + "... [truncated]"
            else:
                truncated = explanation
                
            meta_explanation += f"### Forecaster {i+1}:\n{truncated}\n\n"
        
        return meta_explanation
    
    async def confidence_interval(self, question, context=None):
        """
        Return a confidence interval for the stacking ensemble forecast using bootstrapping.
        
        Supports two methods for interval estimation:
        - bootstrapping: Resamples training data to account for model uncertainty (more robust)
        - variance_propagation: Simpler method that averages individual intervals
        
        The method uses bootstrapping by default to account for model uncertainty and correlations
        between forecaster errors, providing more robust uncertainty estimates.
        """
        # Get individual forecaster predictions and confidence intervals
        predictions = []
        intervals = []
        
        for forecaster in self.forecasters:
            try:
                prediction = await forecaster.predict(question, context)
                interval = await forecaster.confidence_interval(question, context)
                predictions.append(prediction)
                intervals.append(interval)
            except Exception as e:
                logger.error(f"Error getting prediction/interval: {e}")
                predictions.append(0.5)
                intervals.append((0.3, 0.7))  # Default fallback
        
        # Get the ensemble prediction
        ensemble_prediction = await self.predict(question, context)
        
        # If meta-model not trained, use a simple averaging approach
        if (self.meta_model is None or 
            len(self.training_data["outcomes"]) < self.min_training_samples):
            logger.debug("Meta-model not trained yet, using average of intervals")
            avg_lower = sum(interval[0] for interval in intervals) / len(intervals)
            avg_upper = sum(interval[1] for interval in intervals) / len(intervals)
            return (avg_lower, avg_upper)
        
        # Calculate average interval width for fallback method
        avg_width = sum(interval[1] - interval[0] for interval in intervals) / len(intervals)
        lower = max(0.0, ensemble_prediction - avg_width/2)
        upper = min(1.0, ensemble_prediction + avg_width/2)
        
        # If using variance propagation or not enough training data, return simpler method
        if (self.confidence_interval_method == "variance_propagation" or
            len(self.training_data["outcomes"]) < 50):  # Need substantial data for reliable bootstrapping
            logger.debug("Using variance propagation method for confidence interval")
            return (lower, upper)
        
        # For trained meta-model, use bootstrapping to estimate uncertainty
        try:
            # Number of bootstrap samples
            n_bootstrap = 1000
            # Create array for bootstrap predictions
            bootstrap_predictions = []
            
            # Convert to numpy array for efficiency
            X = np.array([predictions])
            
            # Generate bootstrap samples
            for _ in range(n_bootstrap):
                # Sample with replacement from the training data
                indices = np.random.randint(0, len(self.training_data["features"]), size=len(self.training_data["features"]))
                bootstrap_features = [self.training_data["features"][i] for i in indices]
                bootstrap_outcomes = [self.training_data["outcomes"][i] for i in indices]
                
                # Train a bootstrap model
                if self.meta_model_type == "logistic_regression":
                    bootstrap_model = LogisticRegression(solver='lbfgs', C=1.0)
                elif self.meta_model_type == "random_forest":
                    bootstrap_model = RandomForestClassifier(n_estimators=50, max_depth=3)
                else:
                    bootstrap_model = LogisticRegression(solver='lbfgs', C=1.0)
                
                # Train on bootstrap sample
                bootstrap_model.fit(np.array(bootstrap_features), np.array(bootstrap_outcomes))
                
                # Predict with bootstrap model
                bootstrap_pred = bootstrap_model.predict_proba(X)[0][1]
                bootstrap_predictions.append(bootstrap_pred)
            
            # Calculate percentiles for 95% confidence interval
            bootstrap_lower = np.percentile(bootstrap_predictions, 2.5)
            bootstrap_upper = np.percentile(bootstrap_predictions, 97.5)
            
            # Ensure interval is centered around actual prediction
            # This addresses any potential bias in the bootstrapping
            bootstrap_mean = np.mean(bootstrap_predictions)
            adjustment = ensemble_prediction - bootstrap_mean
            bootstrap_lower += adjustment
            bootstrap_upper += adjustment
            
            # Ensure bounds stay within [0, 1]
            bootstrap_lower = max(0.0, min(1.0, bootstrap_lower))
            bootstrap_upper = max(0.0, min(1.0, bootstrap_upper))
            
            logger.info(f"Bootstrap confidence interval: ({bootstrap_lower}, {bootstrap_upper})")
            return (bootstrap_lower, bootstrap_upper)
            
        except Exception as e:
            logger.warning(f"Error in bootstrapping: {e}, falling back to simpler method")
            return (lower, upper)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete forecast result with all components.
        """
        probability = await self.predict(question, context)
        rationale = await self.explain(question, context)
        interval = await self.confidence_interval(question, context)
        
        # Create metadata about the stacking ensemble
        metadata = {
            "forecaster_count": len(self.forecasters),
            "meta_model_type": self.meta_model_type,
            "training_samples": len(self.training_data["outcomes"]),
            "is_trained": (self.meta_model is not None and 
                          len(self.training_data["outcomes"]) >= self.min_training_samples),
            "confidence_interval_method": self.confidence_interval_method
        }
        
        return ForecastResult(
            probability=probability,
            confidence_interval=interval,
            rationale=rationale,
            model_name=self.model_name,
            metadata=metadata
        ) 