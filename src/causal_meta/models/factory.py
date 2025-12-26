from typing import Any, Dict, Type

import torch.nn as nn

from causal_meta.models.base import BaseModel

# Registry to store model classes
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls: Type[BaseModel]):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


class ModelFactory:
    """Factory to instantiate models from configuration."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance from configuration.
        
        Args:
            config: Dictionary containing 'type' and model-specific arguments.
            
        Returns:
            An instance of a class inheriting from BaseModel.
        """
        if "type" not in config:
            raise ValueError("Model configuration must contain a 'type' key.")
        
        model_type = config.pop("type")
        
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
        
        return MODEL_REGISTRY[model_type](**config)
