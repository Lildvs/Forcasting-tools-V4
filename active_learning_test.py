#!/usr/bin/env python
"""
Test script for demonstrating the active learning functionality.
This script creates a few sample forecasts with varying levels of uncertainty
and shows how the active learning system flags and handles those forecasts.
"""

import asyncio
import sys
import os
from pprint import pprint
from datetime import datetime

from forecasting_tools.ai_models.model_interfaces.active_learning_manager import ActiveLearningManager
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.data_models.questions import BinaryQuestion

async def main():
    print("Active Learning System Test")
    print("--------------------------")
    
    # Initialize the active learning manager
    active_learning_manager = ActiveLearningManager(
        data_dir="./test_active_learning_data",
        uncertainty_threshold=0.2,
        middle_prob_threshold=0.15
    )
    
    # Create some sample questions with varying levels of uncertainty
    questions = [
        BinaryQuestion(
            question_text="Will global average temperatures exceed 1.5°C above pre-industrial levels before 2030?",
            background_info="Climate scientists have warned that exceeding 1.5°C warming could lead to more severe climate impacts.",
            resolution_criteria="Based on official NASA GISTEMP data.",
            api_json={}
        ),
        BinaryQuestion(
            question_text="Will there be a Mars landing mission by 2030?",
            background_info="Various space agencies and private companies have announced plans for Mars missions.",
            resolution_criteria="A successful landing of a crewed or uncrewed vehicle on the Mars surface.",
            api_json={}
        ),
        BinaryQuestion(
            question_text="Will the Dow Jones Industrial Average surpass a level of 40,000 by the end of 2025?",
            background_info="The Dow reached 30,000 for the first time in November 2020.",
            resolution_criteria="Based on the official closing price as reported by S&P Dow Jones Indices.",
            api_json={}
        ),
        BinaryQuestion(
            question_text="Will OpenAI release a successor to GPT-4 before the end of 2023?",
            background_info="OpenAI has a history of releasing major model updates approximately every 1-2 years.",
            resolution_criteria="The model must be officially released and available to at least some users.",
            api_json={}
        )
    ]
    
    # Create sample forecast results with varying levels of uncertainty
    forecast_results = [
        ForecastResult(  # High uncertainty - wide confidence interval
            probability=0.65,
            confidence_interval=(0.45, 0.85),
            rationale="Climate models show significant uncertainty...",
            model_name="EnsembleForecaster",
            metadata={"confidence_score": 0.6},
            high_uncertainty=True,
            confidence_score=0.6
        ),
        ForecastResult(  # High uncertainty - probability near 0.5
            probability=0.52,
            confidence_interval=(0.4, 0.65),
            rationale="Multiple competing factors affect Mars mission timelines...",
            model_name="ExpertForecaster",
            metadata={"confidence_score": 0.7},
            high_uncertainty=True,
            confidence_score=0.7
        ),
        ForecastResult(  # Medium uncertainty
            probability=0.78,
            confidence_interval=(0.65, 0.9),
            rationale="Market trends suggest continued growth...",
            model_name="TimeSeriesForecaster",
            metadata={"confidence_score": 0.8},
            high_uncertainty=False,
            confidence_score=0.8
        ),
        ForecastResult(  # Low uncertainty
            probability=0.88,
            confidence_interval=(0.82, 0.94),
            rationale="Based on company statements and past release patterns...",
            model_name="HistoricalForecaster",
            metadata={"confidence_score": 0.9},
            high_uncertainty=False,
            confidence_score=0.9
        )
    ]
    
    print("\n1. Evaluating forecasts for flagging...")
    
    # Evaluate each forecast to see if it should be flagged
    for i, (question, forecast) in enumerate(zip(questions, forecast_results)):
        should_flag = active_learning_manager.evaluate_forecast(forecast, question)
        print(f"Question {i+1}: '{question.question_text[:50]}...'")
        print(f"  Probability: {forecast.probability:.2f}, Interval: [{forecast.confidence_interval[0]:.2f}, {forecast.confidence_interval[1]:.2f}]")
        print(f"  Flagged for review: {'YES' if should_flag else 'NO'}\n")
    
    print("2. Getting flagged questions for review...")
    
    # Get the flagged questions
    flagged_questions = active_learning_manager.get_flagged_questions(sort_by_importance=True)
    print(f"Found {len(flagged_questions)} flagged questions:")
    
    for i, question in enumerate(flagged_questions):
        print(f"\nFlagged Question {i+1}: '{question.get('question_text', '')[:50]}...'")
        print(f"  Importance: {question.get('importance', 0):.2f}")
        print(f"  Model: {question.get('model_name', 'Unknown')}")
        print(f"  Probability: {question.get('probability', 0):.2f}")
    
    print("\n3. Simulating human review...")
    
    # Simulate human review for the first flagged question
    if flagged_questions:
        question_id = flagged_questions[0].get('question_id', '')
        human_prob = 0.7  # Different from the model's prediction
        feedback = "Based on recent climate data, I believe there's a higher likelihood of exceeding 1.5°C warming before 2030."
        
        success = active_learning_manager.submit_review(
            question_id=question_id,
            human_probability=human_prob,
            feedback=feedback,
            update_model=True
        )
        
        print(f"Submitted review for question '{flagged_questions[0].get('question_text', '')[:50]}...'")
        print(f"  Human probability: {human_prob:.2f}")
        print(f"  Feedback: {feedback}")
        print(f"  Success: {'YES' if success else 'NO'}")
    
    print("\n4. Getting training data for model retraining...")
    
    # Get training data for model improvement
    training_data = active_learning_manager.get_training_data()
    print(f"Found {len(training_data)} training examples from reviewed questions.")
    
    if training_data:
        example = training_data[0]
        print("\nExample training data:")
        print(f"  Question: '{example.get('question_text', '')[:50]}...'")
        print(f"  Model probability: {example.get('probability', 0):.2f}")
        print(f"  Human probability: {example.get('human_probability', 0):.2f}")
        print(f"  Feedback: {example.get('review_feedback', '')[:100]}...")
    
    print("\nActive learning test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 