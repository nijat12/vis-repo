"""
VIS Pipeline Package

Contains modular pipeline implementations:
- baseline: YOLO with 4x3 tiled inference
- strategy_7: Motion compensation + CNN verifier
- strategy_8: YOLO on ROIs
"""

from typing import Dict, Callable, Any, List

# Pipeline registry
PIPELINES: Dict[str, Callable] = {}

def register_pipeline(name: str):
    """Decorator to register a pipeline function."""
    def decorator(func: Callable) -> Callable:
        PIPELINES[name] = func
        return func
    return decorator

# Import modules to trigger registration
from . import baseline, strategy_2, strategy_7, strategy_8, strategy_9

def get_pipeline(name: str) -> Callable:
    """Get a pipeline function by name."""
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(PIPELINES.keys())}")
    return PIPELINES[name]

def list_pipelines() -> List[str]:
    """List all registered pipelines."""
    return list(PIPELINES.keys())
