"""
Repo Analysis Toolkit
=====================

A toolkit for analyzing code repositories and extracting contextual information.

Main Components:
- RepoAnalyzer: Static analysis and view generation (HCT, MCG, FCG)
- ImportanceScorer: Component importance scoring
- ContextBuilder: Context generation for core modules
- RepoContextPipeline: Unified workflow

Quick Start:
    >>> from innoeval.mas.tools.repo_analysis import SimplePipeline
    >>> pipeline = SimplePipeline('/path/to/repo')
    >>> context = pipeline.get_context()
    >>> key_modules = pipeline.get_key_modules()
"""

from .repo_analyzer import RepoAnalyzer
from .importance_scorer import ImportanceScorer
from .context_builder import ContextBuilder
from .pipeline import RepoContextPipeline, SimplePipeline
from .config import (
    PipelineConfig,
    AnalysisConfig,
    ScoringConfig,
    ContextConfig,
    DEFAULT_CONFIG,
    load_config_from_dict,
    load_config_from_json
)

__version__ = "0.1.0"

__all__ = [
    # Core components
    "RepoAnalyzer",
    "ImportanceScorer",
    "ContextBuilder",

    # Pipelines
    "RepoContextPipeline",
    "SimplePipeline",

    # Configuration
    "PipelineConfig",
    "AnalysisConfig",
    "ScoringConfig",
    "ContextConfig",
    "DEFAULT_CONFIG",
    "load_config_from_dict",
    "load_config_from_json",
]
