"""
GuardrailManager provides guardrails functionality for forecasting tools.
Since the actual nemo-guardrails library might not be available in all environments,
this implementation provides compatibility interfaces with graceful fallbacks.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Union
import re
import importlib.util

logger = logging.getLogger(__name__)

# Check if nemo-guardrails is available
HAS_NEMO_GUARDRAILS = False
try:
    spec = importlib.util.find_spec("nemoguardrails")
    if spec is not None:
        HAS_NEMO_GUARDRAILS = True
        from nemoguardrails import RailsConfig, LLMRails
except ImportError:
    logger.warning("nemo-guardrails not found, using simulated implementation")


class GuardrailManager:
    """
    Manages guardrails for forecasting tools.
    
    This class provides an interface for applying guardrails to LLM-generated content.
    If the nemo-guardrails library is available, it will use that; otherwise,
    it falls back to a simulated implementation.
    """
    
    _instance = None  # Singleton pattern
    
    def __init__(self):
        self.rails = None
        self.initialized = False
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "guardrails", "config")
        
        # Jailbreak patterns for fallback implementation
        self.jailbreak_patterns = [
            r"ignore (previous|your) instructions",
            r"ignore all (previous|your) constraints",
            r"pretend to be",
            r"you are now in developer mode",
            r"act as if you are",
            r"no longer (bound by|have to follow)",
            r"disregard (ethical|safety) guidelines",
            r"bypass (guardrails|restrictions)",
            r"output the (following|next) (text|content) exactly"
        ]
        
        # Topic boundaries for fallback implementation
        self.approved_topics = [
            "forecasting", "predictions", "probabilities", "trends", 
            "statistics", "analysis", "data", "future events", "likelihood",
            "estimates", "scenarios", "modeling", "risk assessment",
            "uncertainty", "numerical predictions", "confidence intervals",
            "historical data", "accuracy metrics", "calibration"
        ]
        
        # Hallucination indicators
        self.hallucination_patterns = [
            r"will definitely", r"absolutely will", r"I guarantee", 
            r"there is no doubt", r"will certainly", r"100% certain",
            r"impossible", r"never happen", r"always happens", 
            r"without question", r"undoubtedly", r"unquestionably"
        ]
        
        # Uncertainty indicators that should be present in forecasts
        self.uncertainty_indicators = [
            "uncertainty", "probability", "confidence interval", 
            "estimate", "approximately", "likely", "possibly", 
            "evidence suggests", "based on available data",
            "confidence level", "margin of error"
        ]
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of GuardrailManager."""
        if cls._instance is None:
            cls._instance = GuardrailManager()
        return cls._instance
        
    async def initialize(self):
        """Initialize guardrails with configurations."""
        if self.initialized:
            return
            
        try:
            if HAS_NEMO_GUARDRAILS:
                # Initialize with actual nemo-guardrails if available
                config = RailsConfig.from_path(self.config_dir)
                self.rails = LLMRails(config)
                self._register_actions()
            else:
                # Log that we're using fallback implementation
                logger.info("Using fallback guardrails implementation")
            
            self.initialized = True
            logger.info("GuardrailManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GuardrailManager: {e}")
            # Continue with fallback implementation instead of raising
            logger.info("Continuing with fallback implementation")
            self.initialized = True
            
    def _register_actions(self):
        """Register custom actions with NeMo Guardrails if available."""
        if not HAS_NEMO_GUARDRAILS or self.rails is None:
            return
            
        try:
            # Try to import and register forecast-specific actions
            actions_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "guardrails", "actions"
            )
            
            # Create actions directory if it doesn't exist
            if not os.path.exists(actions_dir):
                os.makedirs(actions_dir, exist_ok=True)
                
            # Check for forecast actions file
            forecast_actions_path = os.path.join(actions_dir, "forecast_actions.py")
            
            if os.path.exists(forecast_actions_path):
                # If actions file exists, import and register the actions
                module_name = "forecasting_tools.guardrails.actions.forecast_actions"
                spec = importlib.util.spec_from_file_location(module_name, forecast_actions_path)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Register available actions
                    if hasattr(module, "validate_forecast"):
                        self.rails.register_action(module.validate_forecast)
                        
                    if hasattr(module, "retrieve_historical_data"):
                        self.rails.register_action(module.retrieve_historical_data)
                    
                    logger.info("Registered forecast actions successfully")
            else:
                logger.info("No forecast actions file found, skipping action registration")
                
        except Exception as e:
            logger.error(f"Failed to register actions: {e}")
    
    async def validate_response(self, user_input: str, model_response: str) -> str:
        """
        Apply guardrails to validate and possibly modify a model response.
        
        Args:
            user_input: The user's input that triggered the model response
            model_response: The original response from the model
            
        Returns:
            str: The validated (and possibly modified) response
        """
        if not self.initialized:
            await self.initialize()
            
        if HAS_NEMO_GUARDRAILS and self.rails is not None:
            try:
                # Use actual nemo-guardrails if available
                result = await self.rails.generate_async(
                    messages=[
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": model_response}
                    ]
                )
                
                return result["content"]
            except Exception as e:
                logger.error(f"Error using NeMo Guardrails: {e}. Falling back to simulated implementation.")
                return self._fallback_validate_response(user_input, model_response)
        else:
            # Use fallback implementation
            return self._fallback_validate_response(user_input, model_response)
    
    def _fallback_validate_response(self, user_input: str, model_response: str) -> str:
        """
        Fallback implementation of response validation when nemo-guardrails is not available.
        
        Args:
            user_input: The user's input that triggered the model response
            model_response: The original response from the model
            
        Returns:
            str: The validated response
        """
        # Check for jailbreak attempts in user input
        if self._detect_jailbreak(user_input):
            return (
                "I'm designed to provide forecasting insights within ethical boundaries. "
                "I cannot fulfill requests that attempt to bypass safety measures. "
                "If you have legitimate forecasting questions, I'm happy to assist with those."
            )
                
        # Check if user input is off-topic and the response contains potentially harmful content
        if not self._is_on_topic(user_input):
            if self._contains_harmful_content(model_response):
                return (
                    "I'm designed to help with forecasting and predictions. "
                    "I can analyze probabilities and trends, but I don't provide "
                    "information on potentially harmful or dangerous topics. "
                    "Would you like help with a forecasting question instead?"
                )
                
        # Remove hallucination patterns (overly definitive language)
        modified_response = self._reduce_hallucinations(model_response)
            
        # Ensure forecasts include uncertainty language
        if self._is_forecast_request(user_input) and not self._has_uncertainty_language(modified_response):
            modified_response = self._add_uncertainty_language(modified_response)
                
        return modified_response
    
    def _detect_jailbreak(self, text: str) -> bool:
        """Detect potential jailbreak attempts in the text."""
        text = text.lower()
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_on_topic(self, text: str) -> bool:
        """Check if the text is on-topic for forecasting."""
        text = text.lower()
        return any(topic in text for topic in self.approved_topics)
    
    def _contains_harmful_content(self, text: str) -> bool:
        """Check if the text contains potentially harmful content."""
        harmful_keywords = [
            "hack", "illegal", "terrorist", "bomb", "weapon", "steal", 
            "murder", "exploit", "vulnerability", "phishing", "malware"
        ]
        text = text.lower()
        return any(keyword in text for keyword in harmful_keywords)
    
    def _reduce_hallucinations(self, text: str) -> str:
        """Replace overly definitive language with more appropriate uncertainty."""
        for pattern in self.hallucination_patterns:
            text = re.sub(
                pattern, 
                "may potentially", 
                text, 
                flags=re.IGNORECASE
            )
        return text
    
    def _is_forecast_request(self, text: str) -> bool:
        """Determine if the text is requesting a forecast or prediction."""
        forecast_indicators = [
            "forecast", "predict", "likelihood", "probability", 
            "what are the chances", "how likely", "will happen", 
            "future", "estimate"
        ]
        text = text.lower()
        return any(indicator in text for indicator in forecast_indicators)
    
    def _has_uncertainty_language(self, text: str) -> bool:
        """Check if the text contains appropriate uncertainty language."""
        text = text.lower()
        return any(indicator in text for indicator in self.uncertainty_indicators)
    
    def _add_uncertainty_language(self, text: str) -> str:
        """Add appropriate uncertainty language to a forecast."""
        uncertainty_statement = (
            "\n\nPlease note that this forecast involves uncertainty and "
            "should be interpreted as an estimate based on available data. "
            "The actual outcome may differ from this prediction."
        )
        return text + uncertainty_statement
        
    async def check_topic_compliance(self, text: str) -> bool:
        """
        Check if the given text complies with the topical rails.
        
        Args:
            text: The text to check
            
        Returns:
            bool: True if the text is on-topic, False otherwise
        """
        if not self.initialized:
            await self.initialize()
            
        if HAS_NEMO_GUARDRAILS and self.rails is not None:
            try:
                result = await self.rails.check_async(text)
                return result["valid"]
            except Exception as e:
                logger.error(f"Error checking topic compliance with NeMo Guardrails: {e}")
                return self._is_on_topic(text)
        else:
            return self._is_on_topic(text) 