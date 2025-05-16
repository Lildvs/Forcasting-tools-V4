# NeMo Guardrails for Forecasting Tools

This directory contains the implementation of NVIDIA NeMo Guardrails for forecasting tools. The guardrails help ensure that forecasts are appropriate, well-calibrated, and contain proper expressions of uncertainty.

## Installation

The GuardrailManager class is designed to work with or without the NeMo Guardrails package installed. If the package is available, it will use the full guardrails functionality. If not, it will fall back to a simulated implementation.

To install NeMo Guardrails:

```bash
pip install nemoguardrails
```

## Directory Structure

- `actions/`: Contains custom Python actions for forecasting
- `config/`: Contains guardrails configuration files
  - `config.yml`: Main configuration file
  - `rails/`: Contains colang files defining rails
    - `input_rails.co`: Rails for user input validation
    - `output_rails.co`: Rails for output content validation
    - `actions.py`: Custom action implementations
- `kb/`: Knowledge base files containing forecasting principles

## Usage

The guardrails are accessed through the GuardrailManager class, which is implemented as a singleton:

```python
from forecasting_tools.forecast_helpers.guardrail_manager import GuardrailManager

# Get the singleton instance
guardrail_manager = GuardrailManager.get_instance()

# Initialize the guardrails (this is done automatically on the first validation)
await guardrail_manager.initialize()

# Validate a model response
user_input = "What's the probability of X happening next year?"
model_response = "The probability is definitely 100%."
validated_response = await guardrail_manager.validate_response(user_input, model_response)
```

## Guardrail Types

The implementation includes several types of guardrails:

1. **Topical Rails**: Ensure conversations stay on forecasting-related topics
2. **Jailbreak Prevention**: Prevent attempts to bypass system safeguards
3. **Hallucination Reduction**: Replace overly confident language with appropriate uncertainty
4. **Harmful Content Filter**: Prevent generation of potentially harmful content
5. **Uncertainty Enforcement**: Ensure forecasts include appropriate uncertainty language

## Customization

To customize the guardrails:

1. Modify the `config.yml` file to change the model used or rail flows
2. Add or modify rail definitions in the `rails/` directory
3. Add new knowledge base files in the `kb/` directory
4. Extend the action implementations in `actions/forecast_actions.py`

## Fallback Implementation

If NeMo Guardrails is not installed, the GuardrailManager falls back to a simulated implementation that:

- Detects jailbreak attempts using regular expressions
- Enforces topic boundaries
- Filters harmful content
- Reduces hallucinations by replacing overly definitive language
- Ensures forecasts include uncertainty language 