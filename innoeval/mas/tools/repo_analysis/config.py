"""
Configuration module for repo_analysis.

Provides default settings and configuration management.
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """Configuration for repository analysis."""

    # Repository parsing settings
    max_depth: int = 3  # Maximum directory depth
    max_files_per_dir: int = 40  # Maximum files per directory
    max_file_size_mb: int = 10  # Maximum file size in MB

    # Ignored patterns
    ignored_dirs: List[str] = field(default_factory=lambda: [
        '__pycache__', '.git', '.vscode', 'venv', 'env', 'node_modules',
        '.pytest_cache', 'build', 'dist', '.github', 'logs', '.idea',
        '.mypy_cache', '.tox', 'htmlcov', '.coverage', 'eggs', '.eggs'
    ])

    ignored_file_patterns: List[str] = field(default_factory=lambda: [
        r'.*\.pyc$', r'.*\.pyo$', r'.*\.pyd$', r'.*\.so$', r'.*\.dll$',
        r'.*\.class$', r'.*\.egg-info$', r'.*~$', r'.*\.swp$'
    ])


@dataclass
class ScoringConfig:
    """Configuration for importance scoring."""

    # Scoring weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'usage': 2.0,
        'imports_relationships': 3.0,
        'complexity': 1.0,
        'semantic': 0.5,
        'git_history': 4.0,
    })

    # Important keywords for semantic analysis
    important_keywords: List[str] = field(default_factory=lambda: [
        'main', 'core', 'engine', 'api', 'service',
        'controller', 'manager', 'handler', 'processor',
        'factory', 'builder', 'provider', 'repository',
        'executor', 'scheduler', 'config', 'security'
    ])


@dataclass
class ContextConfig:
    """Configuration for context building."""

    max_tokens: int = 8000  # Maximum tokens for context
    max_abstract_tokens: int = 1000  # Maximum tokens per code abstract
    max_modules: int = 50  # Maximum modules in hierarchical structure
    max_dependencies: int = 20  # Maximum external dependencies to list
    top_k_modules: int = 20  # Number of key modules to extract


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    # Multi-repo processing
    max_workers: int = 4  # Number of parallel workers
    show_progress: bool = True  # Show progress bars


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()


def load_config_from_dict(config_dict: Dict) -> PipelineConfig:
    """
    Load configuration from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        PipelineConfig instance
    """
    config = PipelineConfig()

    if 'analysis' in config_dict:
        for key, value in config_dict['analysis'].items():
            if hasattr(config.analysis, key):
                setattr(config.analysis, key, value)

    if 'scoring' in config_dict:
        for key, value in config_dict['scoring'].items():
            if hasattr(config.scoring, key):
                setattr(config.scoring, key, value)

    if 'context' in config_dict:
        for key, value in config_dict['context'].items():
            if hasattr(config.context, key):
                setattr(config.context, key, value)

    return config


def load_config_from_json(json_file: str) -> PipelineConfig:
    """
    Load configuration from JSON file.

    Args:
        json_file: Path to JSON configuration file

    Returns:
        PipelineConfig instance
    """
    import json

    with open(json_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    return load_config_from_dict(config_dict)
