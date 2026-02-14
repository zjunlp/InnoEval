"""
Agent Factory for InnoEval

This module provides functionality for registering and creating specialized
agent instances based on configuration.
"""
import logging
from typing import Dict, Any, Type

from ..models.model_factory import ModelFactory
from .base_agent import BaseAgent
from .extraction_agent import ExtractionAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating agent instances based on configuration.
    
    This factory maintains a registry of available agent types
    and creates properly configured instances as needed.
    """
    
    # Registry of agent types
    _agent_registry: Dict[str, Type[BaseAgent]] = {
        "extraction": ExtractionAgent,
    }
    
    # Cache of created agent instances
    _agent_cache: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class to register
        """
        if agent_type in cls._agent_registry:
            logger.warning(f"Overriding existing agent type: {agent_type}")
            
        cls._agent_registry[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    @classmethod
    def create_agent(cls, 
                  agent_type: str, 
                  config: Dict[str, Any],
                  model_factory: 'ModelFactory') -> BaseAgent:
        """
        Create an agent instance of the specified type.
        
        Args:
            agent_type: Type of agent to create
            config: Configuration for the agent
            model_factory: ModelFactory instance for creating models
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If the agent type is not registered
        """
        # Check if agent type is registered
        if agent_type not in cls._agent_registry:
            raise ValueError(f"Agent type not registered: {agent_type}")
        
        # Create a cache key
        cache_key = cls._create_cache_key(agent_type, config)
        
        # Check if we have a cached instance
        if cache_key in cls._agent_cache:
            logger.debug(f"Using cached agent instance for {agent_type}")
            return cls._agent_cache[cache_key]
        
        # Get the agent class
        agent_class = cls._agent_registry[agent_type]
        
        # Create the model instance using model_provider from agent config
        model_provider = config.get("model_provider", "default")
        
        try:
            # Use the model factory to create a model for this agent
            model = model_factory.create_model_for_agent(agent_type, config)
            
            # Create the agent instance
            agent = agent_class(model, config)
            
            # Cache the instance
            cls._agent_cache[cache_key] = agent
            
            logger.info(f"Created agent instance of type: {agent_type} with provider: {model_provider}")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent {agent_type}: {e}")
            raise
    
    @classmethod
    def create_all_agents(cls, 
                       config: Dict[str, Any],
                       model_factory: 'ModelFactory') -> Dict[str, BaseAgent]:
        """
        Create all configured agent instances.
        
        Args:
            config: Configuration dictionary with agent configurations
            model_factory: ModelFactory instance for creating models
            
        Returns:
            Dictionary mapping agent types to agent instances
        """
        agents = {}
        agent_configs = config.get("agents", {})
        for agent_type, agent_config in agent_configs.items():
            if agent_type in cls._agent_registry:
                try:
                    # Create a merged config with agent-specific settings
                    merged_config = agent_config.copy()
                    
                    # Add reference to global config sections that might be needed
                    merged_config["_global_config"] = config
                    
                    agents[agent_type] = cls.create_agent(
                        agent_type=agent_type,
                        config=merged_config,
                        model_factory=model_factory
                    )
                except Exception as e:
                    logger.error(f"Error creating agent {agent_type}: {str(e)}")
        
        return agents
    
    @classmethod
    def get_available_agent_types(cls) -> Dict[str, str]:
        """
        Get available agent types.
        
        Returns:
            Dictionary mapping agent types to their class names
        """
        return {agent_type: agent_class.__name__ 
                for agent_type, agent_class in cls._agent_registry.items()}
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the agent cache."""
        cls._agent_cache.clear()
        logger.info("Agent cache cleared")
    
    @staticmethod
    def _create_cache_key(agent_type: str, config: Dict[str, Any]) -> str:
        """
        Create a cache key for an agent configuration.
        
        Args:
            agent_type: Type of agent
            config: Agent configuration
            
        Returns:
            Cache key string
        """
        # For simplicity, use a combination of agent type and model provider
        model_config = config.get("model", {})
        model_provider = model_config.get("provider", "default")
        
        return f"{agent_type}_{model_provider}"
