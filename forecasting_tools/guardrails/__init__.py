"""
NVIDIA NeMo Guardrails integration for forecasting tools.

This package provides guardrails functionality to ensure forecasts are appropriate,
well-calibrated, and include proper expressions of uncertainty.
"""

from pathlib import Path

# Define the paths to important guardrails directories
CONFIG_DIR = Path(__file__).parent / "config"
RAILS_DIR = CONFIG_DIR / "rails" 
KB_DIR = Path(__file__).parent / "kb"
ACTIONS_DIR = Path(__file__).parent / "actions"

# Ensure the directories exist
CONFIG_DIR.mkdir(exist_ok=True, parents=True)
RAILS_DIR.mkdir(exist_ok=True, parents=True)
KB_DIR.mkdir(exist_ok=True, parents=True)
ACTIONS_DIR.mkdir(exist_ok=True, parents=True) 