"""
Custom actions for NeMo Guardrails that support forecasting functionality.

These actions can be called from colang flows to implement guardrails for forecasting.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Regular expressions for identifying numerical predictions
NUMERIC_PREDICTION_PATTERN = r"(\d+(\.\d+)?%|\d+(\.\d+)?\s*(percent|percentage))"
PROBABILITY_CLAIM_PATTERN = r"(likelihood|probability|chance)\s+of\s+.+\s+is\s+(\d+(\.\d+)?%|\d+(\.\d+)?)"
DATE_PREDICTION_PATTERN = r"by\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"

async def validate_forecast(
    forecast_text: str, 
    min_probability: float = 0.0, 
    max_probability: float = 1.0
) -> Dict[str, Any]:
    """
    Validates if a forecast text contains valid probabilistic estimates.
    
    This action checks if probability values in the text are within expected ranges
    and ensures forecasts have appropriate uncertainty language.
    
    Args:
        forecast_text: The text containing the forecast to validate
        min_probability: The minimum valid probability value (0.0-1.0)
        max_probability: The maximum valid probability value (0.0-1.0)
        
    Returns:
        Dict containing validation results and modified text if needed
    """
    logger.info(f"Validating forecast: {forecast_text[:100]}...")
    
    original_text = forecast_text
    is_valid = True
    issues = []
    
    # Extract probability values
    probability_values = []
    
    # Look for percentage expressions
    percent_matches = re.finditer(NUMERIC_PREDICTION_PATTERN, forecast_text, re.IGNORECASE)
    for match in percent_matches:
        percent_str = match.group(1)
        if "%" in percent_str:
            value = float(percent_str.replace("%", "")) / 100
        else:
            # Handle "XX percent" format
            value = float(re.search(r"\d+(\.\d+)?", percent_str).group(0)) / 100
        probability_values.append(value)
    
    # Look for probability/likelihood statements
    prob_matches = re.finditer(PROBABILITY_CLAIM_PATTERN, forecast_text, re.IGNORECASE)
    for match in prob_matches:
        prob_str = match.group(2)
        if "%" in prob_str:
            value = float(prob_str.replace("%", "")) / 100
        else:
            value = float(prob_str)
        probability_values.append(value)
    
    # Validate all extracted probability values
    for value in probability_values:
        if value < min_probability or value > max_probability:
            is_valid = False
            issues.append(f"Probability value {value} is outside the valid range [{min_probability}, {max_probability}]")
            
            # Replace invalid probability with a valid one
            if value < min_probability:
                replacement = min_probability
            else:
                replacement = max_probability
                
            # Format for replacement (keep original format)
            if "%" in forecast_text:
                replacement_str = f"{replacement * 100}%"
            else:
                replacement_str = str(replacement)
                
            forecast_text = forecast_text.replace(str(value), replacement_str)
    
    # Check for uncertainty language
    uncertainty_terms = [
        "uncertainty", "probability", "confidence interval", "estimate", 
        "approximately", "likely", "possibly", "evidence suggests", 
        "based on available data", "margin of error"
    ]
    
    has_uncertainty_language = any(term in forecast_text.lower() for term in uncertainty_terms)
    
    if not has_uncertainty_language:
        is_valid = False
        issues.append("Forecast lacks appropriate uncertainty language")
        
        # Add uncertainty language
        forecast_text += "\n\nPlease note that this forecast involves uncertainty and should be interpreted as an estimate based on available data."
    
    # Check for date predictions and validate them
    date_matches = re.finditer(DATE_PREDICTION_PATTERN, forecast_text, re.IGNORECASE)
    for match in date_matches:
        date_str = match.group(0)
        # No need to validate the actual date format as we're only checking for presence
        # but you could add more specific validation here if needed
        
    result = {
        "valid": is_valid,
        "issues": issues,
        "modified_text": forecast_text if forecast_text != original_text else None
    }
    
    logger.info(f"Forecast validation complete: valid={is_valid}, issues={len(issues)}")
    return result

async def retrieve_historical_data(
    topic: str, 
    time_period: Optional[str] = "last 5 years"
) -> Dict[str, Any]:
    """
    Simulates retrieving historical data for a forecasting topic.
    
    In a real implementation, this would connect to a database or API to fetch actual
    historical data. This is a placeholder implementation.
    
    Args:
        topic: The topic to retrieve historical data for
        time_period: The time period to retrieve data for
        
    Returns:
        Dict containing simulated historical data
    """
    logger.info(f"Retrieving historical data for topic: {topic}, period: {time_period}")
    
    # This is a placeholder implementation that returns simulated data
    # In a real implementation, this would query a database or API
    
    # Parse time period
    years_to_go_back = 5  # default
    if time_period:
        match = re.search(r"(\d+)\s+years?", time_period, re.IGNORECASE)
        if match:
            years_to_go_back = int(match.group(1))
    
    # Current date
    current_date = datetime.now()
    
    # Generate simulated data points
    data_points = []
    for i in range(years_to_go_back * 4):  # Quarterly data points
        date = current_date - timedelta(days=(years_to_go_back * 365) - (i * 90))
        date_str = date.strftime("%Y-%m-%d")
        
        # Simulated value - in a real implementation, this would be actual historical data
        # This just generates slightly random but trending values
        value = 50 + (i * 2) + ((i % 4) * 5) - ((i // 8) * 10)
        
        data_points.append({
            "date": date_str,
            "value": value
        })
    
    return {
        "topic": topic,
        "time_period": time_period,
        "data_points": data_points,
        "summary": f"Historical data for {topic} shows a general upward trend with seasonal fluctuations over the {time_period}.",
        "note": "This is simulated data for demonstration purposes."
    } 