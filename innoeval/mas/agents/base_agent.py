"""
Base Agent Interface for InnoEval Multi-Agent System

This module provides the foundational abstract base class (BaseAgent) that defines
the interface and common functionality for all specialized agents in the InnoEval
system. It establishes a standardized pattern for agent initialization, execution,
model interaction, and error handling that all derived agents must follow.

The module includes:
- BaseAgent: Abstract base class with template methods for agent operations
- AgentExecutionError: Custom exception for agent-specific failures
- Common utilities for model calls, retry logic, and context formatting
"""

import abc
import logging
import os
import time
from typing import Dict, Any, Optional, List, Union, Callable
import asyncio
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """
    Custom exception raised when an agent encounters an unrecoverable execution failure.

    This exception is raised after all retry attempts have been exhausted or when
    a critical error occurs that cannot be resolved through retries. It provides
    context about which agent failed and the nature of the failure.
    """
    pass


class BaseAgent(abc.ABC):
    """
    Abstract base class defining the interface and common functionality for all agents.

    BaseAgent establishes a standardized architecture for specialized agents within
    the InnoEval multi-agent system. Each agent encapsulates a specific cognitive
    task (e.g., hypothesis generation, critical evaluation, method development) and
    interacts with language models to perform that task.

    Key Responsibilities:
        - Define the contract that all concrete agents must implement
        - Provide model interaction utilities with automatic retry logic
        - Track execution metrics (timing, call counts)
        - Handle errors gracefully with configurable retry policies
        - Format context data for agent consumption

    Attributes:
        model (BaseModel): Language model instance for text generation
        config (Dict[str, Any]): Configuration parameters for the agent
        name (str): Human-readable name for the agent
        description (str): Brief description of agent's purpose
        system_prompt (str): Default system-level instructions for the model
        max_retries (int): Maximum number of retry attempts on failures
        last_execution_time (float): Duration of most recent execution in seconds
        total_calls (int): Cumulative count of agent invocations

    Abstract Methods:
        execute: Must be implemented by subclasses to define agent-specific logic

    Usage:
        Subclass BaseAgent and implement the execute() method to create a new
        specialized agent. Use _call_model() for all language model interactions
        to benefit from automatic retries and error handling.
    """
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        """
        Initialize a new agent instance with model and configuration.

        Sets up the agent with a language model backend and configuration parameters
        that control its behavior. Initializes execution tracking metrics and extracts
        commonly used configuration values for quick access.

        Args:
            model (BaseModel): Language model instance that this agent will use for
                text generation and structured output. The model handles the actual
                interaction with the LLM API.
            config (Dict[str, Any]): Configuration dictionary containing agent-specific
                settings. Common keys include:
                - name (str): Agent's display name
                - description (str): Purpose and capabilities description
                - system_prompt (str): Default system-level instructions
                - max_retries (int): Maximum retry attempts on failures (default: 10)

        Returns:
            None

        Note:
            Derived classes should call super().__init__() before performing their
            own initialization to ensure proper setup of base functionality.
        """
        self.model = model
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.description = config.get("description", "")
        self.system_prompt = config.get("system_prompt", "")
        self.max_retries = config.get("max_retries", 10)
        
        # Additional metrics and settings
        self.last_execution_time = 0.0
        self.total_calls = 0
        
    @abc.abstractmethod
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (must be implemented by subclasses).

        This abstract method defines the core interface that all concrete agents must
        implement. Each agent type (generation, reflection, evolution, etc.) provides
        its own implementation that performs its specialized cognitive task using the
        provided context and parameters.

        The method is asynchronous to support efficient concurrent execution of multiple
        agents and non-blocking model API calls.

        Args:
            context (Dict[str, Any]): Contextual information needed for task execution.
                The structure depends on the agent type but typically includes:
                - goal: Research goal or objective
                - hypotheses: Current hypotheses being evaluated
                - iteration: Current iteration number
                - history: Previous execution results
                Agent-specific context keys are documented in each concrete implementation.

            params (Dict[str, Any]): Task-specific parameters that control execution
                behavior. May include overrides for default configuration values or
                runtime-specific settings.

        Returns:
            Dict[str, Any]: Execution results in a standardized dictionary format.
                Common keys include:
                - result: Main output of the agent's task
                - metadata: Execution metadata (timing, success status, etc.)
                Specific return structure is documented in each concrete implementation.

        Raises:
            AgentExecutionError: When execution fails and cannot be recovered through
                retries. Contains details about the failure cause.
            ValueError: When required context or parameter keys are missing or invalid.

        Note:
            Implementations should use self._call_model() for all language model
            interactions to benefit from automatic retry logic and error handling.
        """
        pass
    
    async def run_with_timing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task with automatic timing and metrics collection.

        This wrapper method calls execute() while automatically tracking execution
        time, incrementing call counters, and enriching results with metadata. It
        also provides graceful error handling that captures exceptions and returns
        them in a structured format rather than propagating them.

        Args:
            input_data (Dict[str, Any]): Input data containing both context and
                parameters for the agent task. This dictionary is passed directly
                to the execute() method.

        Returns:
            Dict[str, Any]: Execution results with added metadata fields:
                - Original execution results (from execute())
                - metadata.execution_time: Time taken in seconds
                - metadata.agent_type: Type of agent that executed
                - metadata.timestamp: Unix timestamp of execution
                - metadata.success: Boolean indicating success (only on errors)
                - error: Error message if execution failed

        Note:
            Unlike execute(), this method catches exceptions and returns them in
            the result dictionary rather than raising them. This is useful for
            orchestration scenarios where you want to continue workflow execution
            even if individual agents fail.
        """
        start_time = time.time()
        try:
            result = await self.execute(input_data)
            
            # Log execution time
            self.last_execution_time = time.time() - start_time
            self.total_calls += 1
            
            # Add execution metadata to result
            result["metadata"] = {
                "execution_time": self.last_execution_time,
                "agent_type": self.agent_type,
                "timestamp": time.time()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error executing {self.agent_type} agent: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "execution_time": time.time() - start_time,
                    "agent_type": self.agent_type,
                    "timestamp": time.time(),
                    "success": False
                }
            }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'BaseAgent':
        """
        Factory class method to create an agent instance from configuration.

        Provides an alternative constructor pattern for creating agents from
        configuration dictionaries. This is particularly useful when dynamically
        instantiating agents from configuration files or programmatic setup.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing all
                agent settings. Passed directly to __init__().
            model (BaseModel): Language model instance for the agent to use.

        Returns:
            BaseAgent: Newly created instance of the concrete agent class.

        Example:
            >>> config = {"name": "MyAgent", "max_retries": 5}
            >>> model = OpenAIModel(api_key="...")
            >>> agent = MyAgentClass.from_config(config, model)
        """
        return cls(model, config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Retrieve metadata and information about this agent instance.

        Returns agent identification and descriptive information useful for
        logging, debugging, and runtime introspection of the multi-agent system.

        Returns:
            Dict[str, Any]: Dictionary containing agent metadata with keys:
                - name (str): Agent's configured display name
                - type (str): Agent's class name (e.g., "GenerationAgent")
                - description (str): Human-readable purpose description

        Example:
            >>> agent.get_info()
            {'name': 'hypothesis_generator', 'type': 'GenerationAgent',
             'description': 'Generates novel research hypotheses'}
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "description": self.description
        }
    
    async def _call_model(self,
                        prompt: str,
                        system_prompt: Optional[str] = None,
                        schema: Optional[Dict[str, Any]] = None,
                        temperature: Optional[float] = None) -> Union[str, Dict[str, Any]]:
        """
        Protected method to call the language model with automatic retry logic.

        This is the primary interface for agents to interact with their language model.
        It provides robust error handling with exponential backoff retries, automatic
        selection between text and structured (JSON) generation based on the schema
        parameter, and comprehensive logging of failures.

        All concrete agent implementations should use this method rather than calling
        the model directly to benefit from standardized error handling.

        Args:
            prompt (str): The main user prompt describing the task for the model.
                Should be clear, specific, and include all necessary context.
            system_prompt (Optional[str]): System-level instructions that guide the
                model's behavior and response style. If not provided, uses the agent's
                default system_prompt from configuration. Defaults to None.
            schema (Optional[Dict[str, Any]]): JSON Schema definition for structured
                output. When provided, enforces the model to return JSON matching this
                schema. When None, returns freeform text. Defaults to None.
            temperature (Optional[float]): Sampling temperature for model generation.
                Higher values (e.g., 0.8-1.0) increase creativity, lower values
                (e.g., 0.1-0.3) increase determinism. If None, uses model default.

        Returns:
            Union[str, Dict[str, Any]]: Model's response in one of two formats:
                - str: Freeform text response when schema is None
                - Dict[str, Any]: Structured JSON response when schema is provided

        Raises:
            AgentExecutionError: When model calls fail consistently after exhausting
                all retry attempts (max_retries). Contains details of the final error.

        Note:
            The method sleeps for 1 second between retry attempts to avoid hammering
            the API and potentially triggering rate limits. Consider this latency when
            designing time-sensitive operations.
        """
        system_prompt = system_prompt or self.system_prompt
        remaining_retries = self.max_retries
        
        while True:
            try:
                if schema:
                    return await self.model.generate_json(
                        prompt=prompt,
                        schema=schema,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )
                else:
                    return await self.model.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )
                    
            except Exception as e:
                # sleep for a short time before retrying, log time count
                await asyncio.sleep(1)
                
                remaining_retries -= 1
                logger.warning(f"Agent {self.name} model call failed: {str(e)}. Retries left: {remaining_retries}")
                
                if remaining_retries <= 0:
                    raise AgentExecutionError(f"Agent {self.name} failed after max retries: {str(e)}")
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Protected utility method to format context dictionaries as readable text.

        Converts nested context dictionaries into a structured, human-readable string
        format suitable for inclusion in prompts or logs. Handles nested dictionaries
        and lists with appropriate indentation and formatting.

        This default implementation provides basic formatting that can be overridden
        by subclasses for custom context rendering needs specific to particular agent
        types.

        Args:
            context (Dict[str, Any]): Context dictionary with arbitrary structure.
                May contain nested dictionaries, lists, or primitive values.

        Returns:
            str: Formatted multi-line string representation of the context with:
                - Section headers for top-level keys (uppercase)
                - Indented nested structures
                - Bullet points for list items
                - Readable key-value pairs

        Example:
            >>> context = {"goal": "Test", "hypotheses": [{"id": 1, "text": "..."}]}
            >>> formatted = agent._format_context(context)
            >>> print(formatted)
            Context:

            GOAL: Test

            HYPOTHESES:
              id: 1
              text: ...

        Note:
            Subclasses may override this method to provide agent-specific formatting
            that better suits their prompt engineering needs.
        """
        # Default implementation - can be overridden by subclasses for custom formatting
        context_str = "Context:\n"
        
        for key, value in context.items():
            if isinstance(value, dict):
                context_str += f"\n{key.upper()}:\n"
                for k, v in value.items():
                    context_str += f"  {k}: {v}\n"
            elif isinstance(value, list):
                context_str += f"\n{key.upper()}:\n"
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            context_str += f"  {k}: {v}\n"
                        context_str += "\n"
                    else:
                        context_str += f"  - {item}\n"
            else:
                context_str += f"\n{key.upper()}: {value}\n"
        
        return context_str 
