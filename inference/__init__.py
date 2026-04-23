"""Inference stage: (well, date) -> MLOutput for consumption by the LLM stage."""

from .contract import MLOutput, StepPrediction

__all__ = ["MLOutput", "StepPrediction"]
