"""
OpenAI Model Adapter for InnoEval

Implements the BaseModel interface for OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None,
                base_url: Optional[str] = None,
                model_name: str = "gpt-4o", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the OpenAI model adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.")
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:

            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            print(f"key{self.api_key}&url{self.base_url}")
            logger.info(f"OpenAI client initialized with model: {self.model_name} via {self.base_url}")
        except TypeError as e:
            logger.warning(f"Error initializing OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAIModel':
        """
        Create an OpenAI model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured OpenAIModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )
