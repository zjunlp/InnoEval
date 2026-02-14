"""
DeepSeek R1 Model Adapter for InnoEval

This module implements the BaseModel interface for DeepSeek R1 reasoning models.
DeepSeek R1 models provide advanced reasoning capabilities with explicit thinking
processes that are separated from the final answer using XML-style tags.
"""

import json
import logging
import os
import asyncio
from typing import Dict, List, Optional, Any, Union

import openai
from openai import AsyncOpenAI
import jsonschema
from jsonschema import validate, ValidationError

from .base_model import BaseModel
from .usage_tracker import get_current_tracker

logger = logging.getLogger(__name__)


class R1Model(BaseModel):
    """
    DeepSeek R1 model implementation with reasoning capabilities.
    
    This model adapter interfaces with DeepSeek R1 models that provide explicit
    reasoning traces. The model outputs its thinking process wrapped in <think> tags
    followed by the final answer. This implementation automatically extracts and
    returns only the answer portion while discarding the thinking trace.
    
    The model is compatible with the OpenAI API format and uses AsyncOpenAI client
    for async operations.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                base_url: Optional[str] = None,
                model_name: str = "DeepSeek-V3.2", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60,
                max_schema_retries: int = 3,
                max_generate_retries: int = 2):
        """
        Initialize the OpenAI model adapter.
        
        Args:
            api_key: OpenAI API key (defaults to DS_API_KEY environment variable)
            model_name: Model name to use (e.g., "DeepSeek-R1")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
            max_schema_retries: Maximum number of retries when JSON doesn't match schema (default: 3)
        """
        self.api_key = api_key or os.environ.get("DS_API_KEY")
        if not self.api_key:
            logger.warning("DS API key not provided. Please set DS_API_KEY environment variable.")
        self.base_url = base_url or os.environ.get("DS_API_BASE_URL")
        if not self.base_url:
            logger.warning("DS base URL not provided. Please set DS_API_BASE_URL environment variable.")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_schema_retries = max_schema_retries
        # 针对 generate 的通用重试次数（仅对暂时性错误生效）
        self.max_generate_retries = max_generate_retries
        
        # Initialize the client with only the supported parameters for version 1.3.3
        try:
            # The AsyncOpenAI in version 1.3.3 doesn't support 'proxies' parameter
            logger.info(f"Initializing DeepSeek {self.model_name} client with API key: {self.api_key} and base URL: {self.base_url}")
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        except TypeError as e:
            logger.warning(f"Error initializing DeepSeek client: {e}")
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
        
        # 通用重试机制：仅对网络错误、超时、429 和 5xx 进行重试
        attempts_remaining = self.max_generate_retries
        last_error: Optional[Exception] = None

        while attempts_remaining >= 0:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    **kwargs
                )

                # Accumulate provider-reported token usage for current run (if enabled)
                try:
                    usage = getattr(response, "usage", None)
                    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
                    if total_tokens is None and isinstance(usage, dict):
                        total_tokens = usage.get("total_tokens")
                        if total_tokens is None:
                            pt = usage.get("prompt_tokens")
                            ct = usage.get("completion_tokens")
                            if pt is not None or ct is not None:
                                total_tokens = (pt or 0) + (ct or 0)
                    tracker = get_current_tracker()
                    if tracker is not None:
                        logger.info(f"R1Model.generate: total_tokens: {total_tokens}")
                        tracker.add_tokens(total_tokens)
                except Exception:
                    # Never fail generation due to usage accounting
                    pass
                
                response_text = response.choices[0].message.content
                
                # Handle R1 model reasoning tags if present
                if "</think>" in response_text:
                    think_text, answer_text = response_text.split("</think>\n\n", 1)
                else:
                    answer_text = response_text
                
                return answer_text

            except openai.RateLimitError as e:
                last_error = e
                logger.warning(
                    f"R1Model.generate hit rate limit (attempt {self.max_generate_retries - attempts_remaining + 1}/"
                    f"{self.max_generate_retries + 1}): {e}"
                )
            except openai.APIConnectionError as e:
                last_error = e
                logger.warning(
                    f"R1Model.generate API connection error (attempt {self.max_generate_retries - attempts_remaining + 1}/"
                    f"{self.max_generate_retries + 1}): {e}"
                )
            except openai.APITimeoutError as e:
                last_error = e
                logger.warning(
                    f"R1Model.generate API timeout (attempt {self.max_generate_retries - attempts_remaining + 1}/"
                    f"{self.max_generate_retries + 1}): {e}"
                )
            except openai.APIStatusError as e:
                last_error = e
                status = getattr(e, "status_code", None)
                # 只对 429 和 5xx 的 HTTP 状态码进行重试；400 这类 client error（包括 rix_api_error/bad_response_status_code）
                # 通常代表请求本身有问题，继续重试无意义，直接抛出。
                if status == 400:
                    # DeepSeek RIX: 输入字符数超过限制（如 "Invalid param: input characters limit is 393216"）
                    error_text = str(e)
                    if "input characters limit is" in error_text:
                        # 尝试截断 user content 到 370000 字符以内，再重试
                        # 只处理最后一个 user 消息，避免误伤 system/assistant
                        for i in range(len(messages) - 1, -1, -1):
                            msg = messages[i]
                            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                                original_len = len(msg["content"])
                                if original_len > 370000:
                                    messages[i] = {
                                        **msg,
                                        "content": msg["content"][:370000],
                                    }
                                    logger.warning(
                                        "R1Model.generate: Truncated user content from "
                                        f"{original_len} to {len(messages[i]['content'])} characters "
                                        "due to input characters limit error (400)."
                                    )
                                break
                if status not in (429, 400) and not (status and 500 <= status < 600):
                    logger.error(
                        f"R1Model.generate received non-retriable API status error (status={status}): {e}"
                    )
                    raise
                logger.warning(
                    f"R1Model.generate API status error (status={status}) with retry "
                    f"(attempt {self.max_generate_retries - attempts_remaining + 1}/"
                    f"{self.max_generate_retries + 1}): {e}"
                )
            except Exception as e:
                # 其它未知异常：记录后直接抛出，避免死循环
                logger.error(f"Error generating response from DeepSeek R1: {e}")
                raise

            # 如果还能重试，则等待一会儿；否则跳出循环并抛出最后一次错误
            if attempts_remaining > 0:
                await asyncio.sleep(1)
                attempts_remaining -= 1
                continue
            else:
                # 用最后一次异常作为失败原因
                if last_error is not None:
                    logger.error(
                        f"R1Model.generate failed after {self.max_generate_retries + 1} attempts: {last_error}"
                    )
                    raise last_error
                # 理论上不会走到这里，但为了安全兜底
                raise RuntimeError("R1Model.generate failed with unknown error and no retry information")
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        This method includes automatic retry logic to ensure the returned JSON
        conforms to the provided schema. If the response doesn't match the schema,
        it will retry up to max_schema_retries times.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
            
        Raises:
            ValueError: If JSON doesn't match schema after all retries
            ValidationError: If schema validation fails after all retries
        """
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        remaining_retries = self.max_schema_retries
        last_error = None
        
        while remaining_retries >= 0:
            try:
                # API 调用层面的重试（处理网络错误、超时、429、5xx）
                api_attempts = self.max_generate_retries
                last_api_error: Optional[Exception] = None
                response = None

                # 在 API 调用重试内部维护当前使用的 prompt，方便在 400 超长错误时进行截断重试
                current_prompt = prompt

                while api_attempts >= 0:
                    try:
                        response = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": enhanced_system_prompt},
                                {"role": "user", "content": current_prompt}
                            ],
                            temperature=temperature if temperature is not None else self.temperature,
                            response_format={"type": "json_object"},
                            **kwargs
                        )

                        # Accumulate provider-reported token usage for current run (if enabled)
                        try:
                            usage = getattr(response, "usage", None)
                            total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
                            if total_tokens is None and isinstance(usage, dict):
                                total_tokens = usage.get("total_tokens")
                                if total_tokens is None:
                                    pt = usage.get("prompt_tokens")
                                    ct = usage.get("completion_tokens")
                                    if pt is not None or ct is not None:
                                        total_tokens = (pt or 0) + (ct or 0)
                            tracker = get_current_tracker()
                            if tracker is not None:
                                logger.info(f"R1Model.generate_with_json_output: total_tokens: {total_tokens}")
                                tracker.add_tokens(total_tokens)
                        except Exception:
                            pass

                        break
                    except openai.RateLimitError as e:
                        last_api_error = e
                        logger.warning(
                            f"R1Model.generate_with_json_output hit rate limit "
                            f"(attempt {self.max_generate_retries - api_attempts + 1}/"
                            f"{self.max_generate_retries + 1}): {e}"
                        )
                    except openai.APIConnectionError as e:
                        last_api_error = e
                        logger.warning(
                            f"R1Model.generate_with_json_output API connection error "
                            f"(attempt {self.max_generate_retries - api_attempts + 1}/"
                            f"{self.max_generate_retries + 1}): {e}"
                        )
                    except openai.APITimeoutError as e:
                        last_api_error = e
                        logger.warning(
                            f"R1Model.generate_with_json_output API timeout "
                            f"(attempt {self.max_generate_retries - api_attempts + 1}/"
                            f"{self.max_generate_retries + 1}): {e}"
                        )
                    except openai.APIStatusError as e:
                        last_api_error = e
                        status = getattr(e, "status_code", None)
                        # 只对 429 和 5xx 的 HTTP 状态码进行重试；400 这类 client error（包括 rix_api_error/bad_response_status_code）
                        # 通常代表请求本身有问题，继续重试无意义，直接抛出。
                        if status == 400:
                            # DeepSeek RIX: 输入字符数超过限制（如 "Invalid param: input characters limit is 393216"）
                            error_text = str(e)
                            if "input characters limit is" in error_text and isinstance(current_prompt, str):
                                original_len = len(current_prompt)
                                if original_len > 370000:
                                    current_prompt = current_prompt[:370000]
                                    logger.warning(
                                        "R1Model.generate_with_json_output: Truncated user prompt from "
                                        f"{original_len} to {len(current_prompt)} characters "
                                        "due to input characters limit error (400)."
                                    )
                        if status not in (429, 400) and not (status and 500 <= status < 600):
                            logger.error(
                                f"R1Model.generate_with_json_output received non-retriable API status error "
                                f"(status={status}): {e}"
                            )
                            raise
                        logger.warning(
                            f"R1Model.generate_with_json_output API status error (status={status}) with retry "
                            f"(attempt {self.max_generate_retries - api_attempts + 1}/"
                            f"{self.max_generate_retries + 1}): {e}"
                        )

                    # API 调用失败且可重试
                    if api_attempts > 0:
                        await asyncio.sleep(1)
                        api_attempts -= 1
                        continue
                    else:
                        # API 层面多次重试仍失败，抛出最后一次错误
                        if last_api_error is not None:
                            logger.error(
                                f"R1Model.generate_with_json_output failed after {self.max_generate_retries + 1} "
                                f"API attempts: {last_api_error}"
                            )
                            raise last_api_error
                        raise RuntimeError(
                            "R1Model.generate_with_json_output failed with unknown API error and no retry information"
                        )

                logger.info(f"R1Model: response: {response}")
                response_text = response.choices[0].message.content
                
                # Handle R1 model reasoning tags if present
                if "</think>" in response_text:
                    think_text, answer_text = response_text.split("</think>\n\n", 1)
                else:
                    answer_text = response_text
                
                # Remove markdown code block markers if present
                answer_text = answer_text.strip()
                if answer_text.startswith("```json"):
                    answer_text = answer_text[7:]  # Remove ```json
                elif answer_text.startswith("```"):
                    answer_text = answer_text[3:]  # Remove ```
                if answer_text.endswith("```"):
                    answer_text = answer_text[:-3]  # Remove trailing ```
                answer_text = answer_text.strip()
                
                # Parse JSON
                try:
                    result_dict = json.loads(answer_text)
                except json.JSONDecodeError as e:
                    last_error = ValueError(f"Model did not return valid JSON: {e}")
                    logger.warning(f"Failed to decode JSON response (attempt {self.max_schema_retries - remaining_retries + 1}/{self.max_schema_retries + 1}): {e}")
                    if remaining_retries > 0:
                        await asyncio.sleep(1)  # Wait before retry
                        remaining_retries -= 1
                        continue
                    raise last_error
                
                # Validate against schema
                try:
                    validate(instance=result_dict, schema=json_schema)
                    logger.info(f"JSON response successfully validated against schema (attempt {self.max_schema_retries - remaining_retries + 1}/{self.max_schema_retries + 1})")
                    return result_dict
                except ValidationError as e:
                    last_error = e
                    error_message = str(e) if hasattr(e, '__str__') else getattr(e, 'message', str(e))
                    logger.warning(f"JSON response does not match schema (attempt {self.max_schema_retries - remaining_retries + 1}/{self.max_schema_retries + 1}): {error_message}")
                    if remaining_retries > 0:
                        # Add feedback to prompt for next retry
                        error_feedback = f"\n\nPrevious attempt failed schema validation: {error_message}. Please ensure the JSON strictly matches the schema."
                        prompt_with_feedback = prompt + error_feedback
                        await asyncio.sleep(1)  # Wait before retry
                        remaining_retries -= 1
                        # Update prompt for next iteration
                        prompt = prompt_with_feedback
                        continue
                    raise ValueError(f"JSON response does not match schema after {self.max_schema_retries + 1} attempts: {error_message}")
                    
            except json.JSONDecodeError as e:
                last_error = ValueError(f"Model did not return valid JSON: {e}")
                logger.error(f"Failed to decode JSON response: {e}")
                if remaining_retries > 0:
                    await asyncio.sleep(1)
                    remaining_retries -= 1
                    continue
                raise last_error
            except ValidationError as e:
                # This should not happen here as we catch it above, but just in case
                last_error = e
                error_message = str(e) if hasattr(e, '__str__') else getattr(e, 'message', str(e))
                if remaining_retries > 0:
                    await asyncio.sleep(1)
                    remaining_retries -= 1
                    continue
                raise ValueError(f"JSON response does not match schema after {self.max_schema_retries + 1} attempts: {error_message}")
            except Exception as e:
                logger.error(f"Error generating JSON response from OpenAI: {e}")
                raise
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise ValueError("Failed to generate valid JSON response")
    
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
    def from_config(cls, config: Dict[str, Any]) -> 'R1Model':
        """
        Create an OpenAI model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured R1Model instance
        """
        return cls(
            api_key=config.get("api_key"),
            base_url=config.get("base_url") or config.get("api_base"),
            model_name=config.get("model_name", "DeepSeek-V3.2"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60),
            max_schema_retries=config.get("max_schema_retries", 3)
        ) 
