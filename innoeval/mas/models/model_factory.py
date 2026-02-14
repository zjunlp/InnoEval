"""
Model Provider Factory

A dynamic system for managing and instantiating language model backends.
Enables seamless switching between different LLM providers and handles
configuration, caching, and fallback strategies.
"""

import importlib
import logging
import time
from typing import Dict, Any, List, Tuple

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Available model provider implementations
MODEL_PROVIDER_MAP = {
    "openai": "innoeval.mas.models.openai_model.OpenAIModel",
    "interns1": "innoeval.mas.models.s1_model.S1Model",
    "dsr1": "innoeval.mas.models.r1_model.R1Model",
}


class ModelFactory:
    """Central factory for creating and managing language model instances."""
    
    registered_models = {}
    _model_cache = {}
    _model_stats = {}
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on provided configuration.
        
        Args:
            config: Configuration dictionary for model creation
                
        Returns:
            Configured model instance
            
        Raises:
            ValueError: For unsupported providers or configuration errors
        """
        provider = config.get("provider")
        if provider == "default" or provider is None:
            provider = config.get("default_provider", "openai")

        if "models" in config and provider in config["models"]:
            provider_config = config["models"][provider].copy()
            logger.info(f"Using provider-specific config for {provider}: {provider_config}")
            
            # Merge with general configuration
            provider_config.update({
                k: v for k, v in config.items()
                if k not in ["models", "agents", "workflow", "tools", "memory", "execution"]
            })
            config_to_use = provider_config
        else:
            logger.info(f"Using fallback config for provider {provider}")
            config_to_use = config
        
        cache_key = ModelFactory._create_cache_key(provider, config_to_use)
        
        if cache_key in ModelFactory._model_cache:
            logger.debug(f"Reusing cached model for provider: {provider}")
            return ModelFactory._model_cache[cache_key]
            
        if provider not in MODEL_PROVIDER_MAP and provider not in ModelFactory.registered_models:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        if provider in ModelFactory.registered_models:
            model_class = ModelFactory.registered_models[provider]
        else:
            try:
                module_path, class_name = MODEL_PROVIDER_MAP[provider].rsplit(".", 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load model implementation: {e}")
                raise ValueError(f"Implementation not available for provider: {provider}")
        
        try:
            model = model_class.from_config(config_to_use)
            ModelFactory._model_cache[cache_key] = model
            ModelFactory._model_stats[cache_key] = {
                "created_at": time.time(),
                "success_count": 0,
                "failure_count": 0,
                "total_tokens": 0,
                "total_latency": 0.0
            }
            return model
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise
    
    @staticmethod
    def create_model_for_agent(agent_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance tailored for a specific agent type.
        
        Args:
            agent_type: Type identifier for the agent
            config: Agent configuration parameters
                
        Returns:
            Model instance configured for the specified agent
        """
        model_provider = config.get("model_provider", "default")
        global_config = config.get("_global_config", {})
        model_config = {}
        
        # Apply provider configuration
        if model_provider != "default" and model_provider in global_config.get("models", {}):
            model_config.update(global_config.get("models", {}).get(model_provider, {}))
        
        # Handle default provider case
        if model_provider == "default":
            default_provider = global_config.get("models", {}).get("default_provider", "openai")
            model_config.update(global_config.get("models", {}).get(default_provider, {}))
        
        model_config["provider"] = model_provider
        model_config["default_provider"] = global_config.get("models", {}).get("default_provider", "openai")
        
        # Apply agent-specific overrides
        for key in ["temperature", "max_tokens"]:
            if key in config:
                model_config[key] = config[key]
        
        # Include remaining agent settings
        for k, v in config.items():
            if k not in ["_global_config", "count"] and k not in model_config:
                model_config[k] = v
                
        return ModelFactory.create_model(model_config)
    
    @classmethod
    def register_model(cls, provider_name: str, model_class) -> None:
        """Add a custom model implementation to the factory registry."""
        cls.registered_models[provider_name] = model_class
        logger.info(f"New model provider registered: {provider_name}")
    
    @classmethod
    def create_with_fallbacks(cls, 
                             config: Dict[str, Any], 
                             fallback_providers: List[str] = None,
                             max_retries: int = 3) -> Tuple[BaseModel, str]:
        """
        Create a model with automatic fallback to alternative providers.
        
        Attempts to create a model with the primary provider, falling back
        to alternatives if the primary fails. Each provider can be tried
        multiple times before moving to the next option.
        
        Args:
            config: Model configuration
            fallback_providers: Ordered list of backup providers
            max_retries: Maximum attempts per provider
            
        Returns:
            (model_instance, successful_provider_name)
            
        Raises:
            ValueError: When all providers fail
        """
        primary_provider = config.get("provider", config.get("default_provider", "openai"))
        
        # Setup fallback sequence
        if fallback_providers is None:
            fallback_providers = [p for p in MODEL_PROVIDER_MAP.keys() if p != primary_provider]
            # Prioritize known reliable providers
            if "openai" in fallback_providers:
                fallback_providers.remove("openai")
                fallback_providers.insert(0, "openai")
            if "ollama" in fallback_providers:
                fallback_providers.remove("ollama")
                fallback_providers.insert(1, "ollama")
        
        providers_to_try = [primary_provider] + fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    provider_config = dict(config)
                    provider_config["provider"] = provider
                    
                    model = cls.create_model(provider_config)
                    logger.info(f"Successfully initialized {provider} model")
                    return model, provider
                except Exception as e:
                    retry_count += 1
                    last_error = e
                    logger.warning(f"Attempt {retry_count} failed for provider {provider}: {e}")
                    time.sleep(1)
        
        raise ValueError(f"Failed to create model with any provider. Last error: {last_error}")
    
    @classmethod
    def update_model_stats(cls, cache_key: str, success: bool, tokens: int = 0, latency: float = 0.0) -> None:
        """Update performance metrics for a model instance."""
        if cache_key not in cls._model_stats:
            return
            
        stats = cls._model_stats[cache_key]
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
            
        stats["total_tokens"] += tokens
        stats["total_latency"] += latency
    
    @classmethod
    def get_model_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all model instances."""
        return cls._model_stats
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get all registered model providers and their implementations."""
        available_models = {provider: path for provider, path in MODEL_PROVIDER_MAP.items()}
        
        for provider, model_class in cls.registered_models.items():
            available_models[provider] = model_class.__name__
            
        return available_models
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances and statistics."""
        cls._model_cache.clear()
        cls._model_stats.clear()
        logger.info("Model cache cleared")
    
    @staticmethod
    def _create_cache_key(provider: str, config: Dict[str, Any]) -> str:
        """
        Generate a unique identifier for model instance caching.
        
        Different configurations of the same provider are treated as
        separate instances to ensure correct behavior.
        """
        provider_config = config.get(provider, {})
        model_name = provider_config.get("model_name", "default")
        
        if provider == "local":
            model_path = provider_config.get("model_path", "")
            return f"{provider}:{model_name}:{model_path}"
        
        if provider == "ollama":
            api_base = provider_config.get("api_base", "http://localhost:11434")
            return f"{provider}:{model_name}:{api_base}"
            
        return f"{provider}:{model_name}"
