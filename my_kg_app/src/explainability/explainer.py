"""Single explainer facade.

Defines interfaces and concrete implementations for explainability methods.
"""

from abc import ABC, abstractmethod
from typing import Any

class Explainer(ABC):
    @abstractmethod
    def explain(self, input_data: Any, output: Any) -> Any:
        """Return explanation object (e.g. feature importance, traces)."""
        pass

class SHAPExplainer(Explainer):
    def explain(self, input_data: Any, output: Any) -> Any:
        # placeholder: wrap SHAP library calls
        raise NotImplementedError()

class CircuitTracerExplainer(Explainer):
    def explain(self, input_data: Any, output: Any) -> Any:
        # placeholder: wrap circuit-tracer logic
        raise NotImplementedError()
