"""Abstract interface for explainability tools."""

class TracerInterface:
    def explain(self, *args, **kwargs):
        raise NotImplementedError
