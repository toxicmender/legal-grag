"""
Explainability and interpretability module.

This module provides tools for explaining model decisions and reasoning,
including circuit tracing, attribution, and visualization.
"""

from .circuit_tracer import CircuitTracer
from .shap_wrapper import SHAPWrapper
from .attribution import AttributionAnalyzer
from .visualization import ExplanationVisualizer

__all__ = [
    'CircuitTracer',
    'SHAPWrapper',
    'AttributionAnalyzer',
    'ExplanationVisualizer',
]

