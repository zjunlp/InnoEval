"""
BGE model singleton cache.

Goal:
- Avoid repeated loading of SentenceTransformer/CrossEncoder across multiple agent instances.
- Support explicit preload before starting parallel pipelines (especially before forking).

Notes:
- This is a per-process singleton. On Linux with multiprocessing start method "fork",
  preloading in the parent process can significantly reduce repeated downloads and
  can share memory pages copy-on-write in child processes.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Tuple

from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_embedding_model_instance: Optional[SentenceTransformer] = None
_reranker_model_instance: Optional[CrossEncoder] = None
_embedding_model_name: Optional[str] = None
_reranker_model_name: Optional[str] = None
_model_device: Optional[str] = None


def _get_default_device() -> str:
    """Get default device for BGE models, supports environment variable override."""
    # Check environment variable first
    env_device = os.environ.get("BGE_DEVICE")
    if env_device:
        return env_device
    # Auto-detect: prefer CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_bge_models(
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    reranker_model_name: str = "BAAI/bge-reranker-base",
    *,
    hf_endpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[SentenceTransformer, CrossEncoder]:
    """
    Get (embedding_model, reranker_model) with singleton caching.

    If model names differ from the cached ones, models are reloaded.
    """
    global _embedding_model_instance, _reranker_model_instance
    global _embedding_model_name, _reranker_model_name, _model_device

    # Use default device if not specified
    if device is None:
        device = _get_default_device()

    with _lock:
        if hf_endpoint and "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        need_reload = (
            _embedding_model_instance is None
            or _reranker_model_instance is None
            or _embedding_model_name != embedding_model_name
            or _reranker_model_name != reranker_model_name
            or _model_device != device
        )
        if need_reload:
            logger.info(
                "Loading BGE models: %s and %s on device %s",
                embedding_model_name,
                reranker_model_name,
                device,
            )
            _embedding_model_instance = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
            _reranker_model_instance = CrossEncoder("BAAI/bge-reranker-base", device=device)
            _embedding_model_name = embedding_model_name
            _reranker_model_name = reranker_model_name
            _model_device = device
            logger.info("BGE models loaded successfully on device %s", device)

        # mypy: instances guaranteed by need_reload logic
        return _embedding_model_instance, _reranker_model_instance  # type: ignore[return-value]


def preload_bge_models(
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    reranker_model_name: str = "BAAI/bge-reranker-base",
    *,
    hf_endpoint: str = "https://hf-mirror.com",
    device: Optional[str] = None,
) -> None:
    """Explicitly preload models (useful before starting parallel execution / forking)."""
    get_bge_models(
        embedding_model_name=embedding_model_name,
        reranker_model_name=reranker_model_name,
        hf_endpoint=hf_endpoint,
        device=device,
    )


