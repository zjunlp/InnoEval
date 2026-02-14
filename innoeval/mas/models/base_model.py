"""
Base Model Interface for InnoEval

Defines the core abstraction layer for language model interactions.
This module provides a standardized interface that all model implementations
must adhere to, enabling consistent access patterns across different backends.
"""

import abc
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception class for all model-related errors."""
    pass

class RateLimitError(ModelError):
    """Raised when API quotas or rate limits are exceeded."""
    pass

class TokenLimitError(ModelError):
    """Raised when input or output exceeds model context window."""
    pass

class AuthenticationError(ModelError):
    """Raised when API credentials are invalid or expired."""
    pass

class ServiceUnavailableError(ModelError):
    """Raised when model service endpoints cannot be reached."""
    pass

class BaseModel(abc.ABC):
    """
    Abstract foundation for all language model implementations.
    
    This class defines the contract that all model providers must implement,
    providing a unified interface for text generation, structured output,
    and embedding creation regardless of the underlying model service.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize model with performance tracking metrics.
        
        Args:
            **kwargs: Provider-specific configuration parameters
        """
        # Performance tracking metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        
        # Optional telemetry callback
        self._on_completion: Optional[Callable] = None
    
    @abc.abstractmethod
    async def generate(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       stop_sequences: Optional[List[str]] = None,
                       **kwargs) -> str:
        """
        Generate text completion from the model.
        
        Args:
            prompt: The main input text to complete
            system_prompt: Context or instructions for the model
            temperature: Creativity control (higher = more random)
            max_tokens: Maximum generation length
            stop_sequences: Strings that will halt generation when produced
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text completion
            
        Raises:
            ModelError: On generation failure
        """
        pass
    
    @abc.abstractmethod
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON output conforming to a schema.
        
        Args:
            prompt: The input text to process
            schema: JSON schema specification for the expected output format
            system_prompt: Context or instructions for the model
            temperature: Creativity control (higher = more random)
            default: Fallback response if generation fails
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Structured data as a Python dictionary
            
        Raises:
            ModelError: On generation failure when no default is provided
        """
        pass
    
    @abc.abstractmethod
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Create vector embeddings from text input.
        
        Converts text into high-dimensional vector representations
        suitable for semantic search, clustering, or similarity analysis.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Vector representation(s) as floating point arrays
            
        Raises:
            ModelError: On embedding generation failure
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModel':
        """
        Instantiate a model from configuration dictionary.
        
        Factory method that each implementation must provide to
        create properly configured instances from standardized
        configuration format.
        
        Args:
            config: Configuration parameters dictionary
            
        Returns:
            Configured model instance
        """
        pass
    
    def set_completion_callback(self, callback: Callable) -> None:
        """
        Register a telemetry callback for monitoring model performance.
        
        The callback will be invoked after each model operation with
        statistics about the operation's performance and outcome.
        
        Args:
            callback: Function to call after model operations
        """
        self._on_completion = callback
    
    async def _timed_generate(self,
                             func: Callable,
                             *args,
                             **kwargs) -> Any:
        """
        Execution wrapper with performance monitoring and error handling.
        
        Tracks timing, success rates, and token usage while providing
        standardized error classification for all model operations.
        
        Args:
            func: Async function to execute and monitor
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result from the wrapped function
            
        Raises:
            ModelError: Categorized exception on failure
        """
        start_time = time.time()
        self.total_calls += 1
        success = False
        token_count = 0
        result = None
        
        try:
            # Execute the model operation
            result = await func(*args, **kwargs)
            success = True
            self.successful_calls += 1

            if isinstance(result, str):
                token_count = len(result) // 4
            elif isinstance(result, dict) and kwargs.get('return_token_count'):
                token_count = kwargs.get('return_token_count', 0)
                
            self.total_tokens += token_count
            return result
            
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"Model call failed: {str(e)}")
            
            # Convert to appropriate ModelError subclass
            if "rate limit" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded: {str(e)}")
            elif "token" in str(e).lower() and ("limit" in str(e).lower() or "exceed" in str(e).lower()):
                raise TokenLimitError(f"Token limit exceeded: {str(e)}")
            elif "auth" in str(e).lower() or "key" in str(e).lower() or "credential" in str(e).lower():
                raise AuthenticationError(f"Authentication failed: {str(e)}")
            elif "unavailable" in str(e).lower() or "down" in str(e).lower() or "connect" in str(e).lower():
                raise ServiceUnavailableError(f"Service unavailable: {str(e)}")
            else:
                raise ModelError(f"Model error: {str(e)}")
                
        finally:
            # Finalize performance tracking
            elapsed_time = time.time() - start_time
            self.total_time += elapsed_time
            
            # Report telemetry if callback is registered
            if self._on_completion:
                try:
                    self._on_completion(
                        success=success,
                        elapsed_time=elapsed_time,
                        token_count=token_count,
                        model_type=self.__class__.__name__
                    )
                except Exception as callback_error:
                    logger.warning(f"Error in completion callback: {callback_error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this model.
        
        Returns:
            Dictionary of statistics
        """
        avg_time = 0 if self.total_calls == 0 else self.total_time / self.total_calls
        success_rate = 0 if self.total_calls == 0 else (self.successful_calls / self.total_calls) * 100
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "average_time_per_call": avg_time,
            "model_type": self.__class__.__name__
        } 
