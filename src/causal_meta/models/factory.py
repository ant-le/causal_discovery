from typing import Any, Mapping, Type, TypedDict, cast

from causal_meta.models.base import BaseModel

# Registry to store model classes
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {}


class ModelConfig(TypedDict, total=False):
    type: str


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls: Type[BaseModel]):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


class ModelFactory:
    """Factory to instantiate models from configuration."""

    @staticmethod
    def create(config: Mapping[str, Any] | ModelConfig) -> BaseModel:
        """
        Create a model instance from configuration.
        
        Args:
            config: Dictionary containing 'type' and model-specific arguments.
            
        Returns:
            An instance of a class inheriting from BaseModel.
        """
        if "type" not in config or not isinstance(config.get("type"), str):
            raise ValueError("Model configuration must contain a 'type' key.")

        model_type = cast(str, config["type"])
        
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

        # Do not mutate the input config (Hydra configs or shared dicts may be reused).
        kwargs = {k: v for k, v in config.items() if k != "type"}
        return MODEL_REGISTRY[model_type](**kwargs)
