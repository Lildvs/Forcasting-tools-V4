"""
Custom action definitions for NeMo Guardrails.

These functions are callable from colang flows and provide functionality
specific to the forecasting domain.
"""

import re
from typing import Dict, Any, List, Optional


def is_forecast_request(text: str) -> bool:
    """
    Determine if a user message is requesting a forecast or prediction.
    
    Args:
        text: The text to analyze
        
    Returns:
        True if the text appears to be requesting a forecast, False otherwise
    """
    forecast_indicators = [
        "forecast", "predict", "likelihood", "probability", 
        "what are the chances", "how likely", "will happen", 
        "future", "estimate", "projected", "anticipate",
        "expected to", "outlook", "in the coming", "potential"
    ]
    
    text = text.lower()
    return any(indicator in text for indicator in forecast_indicators)


def contains_harmful_content(text: str) -> bool:
    """
    Check if text contains potentially harmful content.
    
    Args:
        text: The text to analyze
        
    Returns:
        True if harmful content is detected, False otherwise
    """
    harmful_keywords = [
        "hack", "illegal", "terrorist", "bomb", "weapon", "steal", 
        "murder", "exploit", "vulnerability", "phishing", "malware",
        "attack", "kill", "fraud", "criminal", "suicide", "assassinate"
    ]
    
    text = text.lower()
    return any(keyword in text for keyword in harmful_keywords)


def has_uncertainty_language(text: str) -> bool:
    """
    Check if text contains appropriate uncertainty language for forecasts.
    
    Args:
        text: The text to analyze
        
    Returns:
        True if uncertainty language is present, False otherwise
    """
    uncertainty_terms = [
        "uncertainty", "probability", "confidence interval", "estimate", 
        "approximately", "likely", "possibly", "evidence suggests", 
        "based on available data", "margin of error", "may", "might",
        "could", "uncertain", "not guaranteed", "potential", "projected"
    ]
    
    text = text.lower()
    return any(term in text for term in uncertainty_terms)


def check_harmful_content(message: str) -> bool:
    """
    Check if a user message contains references to harmful content.
    Used to filter inappropriate user queries.
    
    Args:
        message: The user message to check
        
    Returns:
        True if harmful content is detected, False otherwise
    """
    return contains_harmful_content(message) 