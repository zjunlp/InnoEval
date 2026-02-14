"""
Repo Context Pipeline - Unified workflow for repository analysis.

This module provides a high-level pipeline that integrates all components:
RepoAnalyzer, ImportanceScorer, and ContextBuilder.
"""

import json
import logging
import os
import re
import subprocess
from typing import Dict, List, Optional
from pathlib import Path

from .repo_analyzer import RepoAnalyzer
from .importance_scorer import ImportanceScorer
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)

# Global storage directory: InnoEval/saved/downloaded_repos
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAVED_DIR = PROJECT_ROOT / "saved"
REPO_SAVE_DIR = SAVED_DIR / "downloaded_repos"
SAVED_DIR.mkdir(parents=True, exist_ok=True)
REPO_SAVE_DIR.mkdir(parents=True, exist_ok=True)


class RepoContextPipeline:
    """
    Unified pipeline for repository analysis and context extraction.

    Workflow:
    1. Analyze repository (static analysis + view generation)
    2. Score components by importance
    3. Build context
    4. Export results
    """

    def __init__(self, repo_path: str, importance_weights: Optional[Dict] = None):
        """
        Initialize pipeline.

        Args:
            repo_path: Path to the repository
            importance_weights: Optional custom weights for importance scoring
        """
        self.repo_path = str(Path(repo_path).resolve())
        self.importance_weights = importance_weights

        # Components (initialized lazily)
        self.analyzer = None
        self.scorer = None
        self.context_builder = None

        # Results
        self.analysis_results = None
        self.key_modules = None
        self.context = None

    def run(self, max_tokens: int = 8000,
            output_file: Optional[str] = None, format: str = 'json') -> Dict:
        """
        Run complete pipeline.

        Args:
            max_tokens: Maximum tokens for context generation
            output_file: Optional output file path
            format: Output format ('json' or 'string')

        Returns:
            Complete results dictionary
        """
        logger.info(f"Starting pipeline for repository: {self.repo_path}")

        # Step 1: Analyze repository
        logger.info("Step 1: Analyzing repository...")
        self.analyze()

        # Step 2: Score components
        logger.info("Step 2: Scoring components...")
        self.score_importance()

        # Step 3: Build context
        logger.info(f"Step 3: Building context (max_tokens={max_tokens})...")
        self.build_context(max_tokens=max_tokens)

        # Step 4: Prepare results
        results = {
            'repo_path': self.repo_path,
            'analysis': {
                'total_modules': len(self.analysis_results['modules']),
                'total_classes': len(self.analysis_results['classes']),
                'total_functions': len(self.analysis_results['functions']),
            },
            'key_modules': self.key_modules,
            'context': self.context,
        }

        # Export if requested
        if output_file:
            self.export(output_file, format=format)

        logger.info("Pipeline completed successfully!")
        return results

    def analyze(self) -> Dict:
        """
        Run repository analysis (HCT, MCG, FCG).

        Returns:
            Analysis results
        """
        if self.analyzer is None:
            self.analyzer = RepoAnalyzer(self.repo_path)

        self.analysis_results = self.analyzer.analyze()
        return self.analysis_results

    def score_importance(self, top_k: int = 20) -> List[Dict]:
        """
        Score component importance.

        Args:
            top_k: Number of top modules to return

        Returns:
            List of key modules with scores
        """
        if self.analysis_results is None:
            self.analyze()

        if self.scorer is None:
            self.scorer = ImportanceScorer(
                repo_path=self.repo_path,
                modules=self.analysis_results['modules'],
                classes=self.analysis_results['classes'],
                functions=self.analysis_results['functions'],
                imports=self.analysis_results['imports'],
                code_tree=self.analysis_results['code_tree'],
                call_graph=self.analysis_results['call_graph'],
                weights=self.importance_weights
            )

        self.key_modules = self.scorer.get_key_modules(top_k=top_k)

        # Update code_tree with key_modules
        self.analysis_results['code_tree']['key_modules'] = self.key_modules

        return self.key_modules

    def build_context(self, max_tokens: int = 8000) -> Dict:
        """
        Build context from analysis results.

        Args:
            max_tokens: Maximum tokens for context

        Returns:
            Context dictionary
        """
        if self.analysis_results is None:
            self.analyze()

        if self.key_modules is None:
            self.score_importance()

        if self.context_builder is None:
            self.context_builder = ContextBuilder(
                repo_path=self.repo_path,
                analysis_results=self.analysis_results,
                key_modules=self.key_modules
            )

        self.context = self.context_builder.build_context(max_tokens=max_tokens)
        return self.context

    def export(self, output_file: str, format: str = 'json') -> None:
        """
        Export results to file.

        Args:
            output_file: Output file path
            format: Output format ('json' or 'string')
        """
        if format == 'json':
            results = {
                'repo_path': self.repo_path,
                'key_modules': self.key_modules,
                'context': self.context,
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Results exported to JSON: {output_file}")

        elif format == 'string':
            if self.context_builder is None:
                raise ValueError("Context not built yet")

            content = self.context_builder.export_to_string()

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Results exported to text: {output_file}")

        else:
            raise ValueError(f"Unknown format: {format}")

    def get_summary(self) -> str:
        """
        Get a summary of the pipeline results.

        Returns:
            Summary string
        """
        if self.analysis_results is None:
            return "Pipeline not run yet"

        lines = []
        lines.append(f"Repository: {self.repo_path}")
        lines.append(f"Total Modules: {len(self.analysis_results['modules'])}")
        lines.append(f"Total Classes: {len(self.analysis_results['classes'])}")
        lines.append(f"Total Functions: {len(self.analysis_results['functions'])}")

        if self.key_modules:
            lines.append(f"\nTop 5 Key Modules:")
            for i, module in enumerate(self.key_modules[:5], 1):
                lines.append(f"  {i}. {module['path']} (score: {module['importance_score']:.2f})")

        return "\n".join(lines)


class SimplePipeline:
    """
    Simplified pipeline for quick analysis.

    Usage:
        pipeline = SimplePipeline('/path/to/repo')
        context = pipeline.get_context()
    """

    def __init__(self, repo_path: str):
        """Initialize simple pipeline."""
        self.pipeline = RepoContextPipeline(repo_path)

    def get_readme(self) -> Optional[str]:
        """
        Locate and return README content as text.

        Searches the repository directory for a README file (case-insensitive,
        matching names like readme.md/readme). Returns file content if found,
        otherwise None.
        """
        repo_dir = Path(self.pipeline.repo_path)
        if not repo_dir.exists():
            logger.warning(f"Repository path does not exist: {repo_dir}")
            return None

        candidates = []
        for root, _, files in os.walk(repo_dir):
            for fname in files:
                name_lower = fname.lower()
                if name_lower in {"readme.md", "readme"}:
                    candidates.append(Path(root) / fname)
            # Prefer README near the root; stop descending after first level
            if root == str(repo_dir):
                # continue to walk deeper only if not found at root
                if candidates:
                    break
        if not candidates:
            return None

        # Choose the first candidate (root-level preferred)
        readme_path = sorted(candidates, key=lambda p: len(p.parts))[0]
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read README at {readme_path}: {e}")
            return None

    def get_context(self, max_tokens: int = 8000, format: str = 'dict') -> any:
        """
        Get repository context.

        Args:
            max_tokens: Maximum tokens
            format: Return format ('dict', 'json', or 'string')

        Returns:
            Context in requested format
        """
        self.pipeline.analyze()
        self.pipeline.score_importance()
        context = self.pipeline.build_context(max_tokens=max_tokens)

        if format == 'dict':
            return context
        elif format == 'json':
            return json.dumps(context, ensure_ascii=False, indent=2)
        elif format == 'string':
            return self.pipeline.context_builder.export_to_string()
        else:
            raise ValueError(f"Unknown format: {format}")

    def get_key_modules(self, top_k: int = 10) -> List[Dict]:
        """
        Get key modules.

        Args:
            top_k: Number of modules to return

        Returns:
            List of key modules
        """
        self.pipeline.analyze()
        return self.pipeline.score_importance(top_k=top_k)

    @staticmethod
    def _extract_github_repo_url(url: str) -> Optional[str]:
        """
        Extract the base GitHub repository URL from various GitHub URL formats.

        Handles URLs like:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - https://github.com/owner/repo/discussions/538
        - https://github.com/owner/repo/issues/172
        - https://github.com/owner/repo/blob/master/path/to/file.md
        - https://github.com/owner/repo/releases
        - https://github.com/owner/repo/tree/branch/path

        Args:
            url: GitHub URL in any format

        Returns:
            Git clone URL (e.g., https://github.com/owner/repo.git) or None if invalid
        """
        # Pattern to match github.com/owner/repo
        # Matches: github.com (or www.github.com) / owner / repo
        pattern = r'(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)'
        match = re.search(pattern, url)

        if not match:
            logger.warning(f"Invalid GitHub URL format: {url}")
            return None

        owner = match.group(1)
        repo = match.group(2)

        # Remove .git suffix if present in repo name
        repo = repo.replace('.git', '')

        # Construct the git URL
        git_url = f"https://github.com/{owner}/{repo}.git"

        return git_url

    @staticmethod
    def download_github_repo(repo_url: str, target_dir: Optional[str] = None) -> Optional[str]:
        """
        Download GitHub repository to a target directory. If the repo already
        exists (by repo_name) under the target directory, reuse it directly.

        Args:
            repo_url: GitHub repository URL (can be in various formats)
            target_dir: Target directory for cloning. Defaults to the global
                        cached path `saved/downloaded_repos`.

        Returns:
            Local path to the cloned repository, or None if download fails
        """
        base_dir = Path(target_dir) if target_dir else REPO_SAVE_DIR
        base_dir.mkdir(parents=True, exist_ok=True)

        # Extract the base GitHub repository URL
        git_url = SimplePipeline._extract_github_repo_url(repo_url)
        if git_url is None:
            logger.error(f"Failed to extract GitHub repository URL from: {repo_url}")
            return None

        # Extract repo name for local path
        repo_name = git_url.split('/')[-1].replace('.git', '')
        local_path = base_dir / repo_name

        # Reuse if already exists
        if local_path.exists():
            logger.info(f"Repository already exists, reuse cached path: {local_path}")
            return str(local_path)

        logger.info(f"Downloading GitHub repository from {git_url} to {local_path}")

        try:
            # Clone repository with depth=1 for faster download
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', git_url, str(local_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            logger.info(f"Successfully downloaded repository to {local_path}")
            return str(local_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Timeout while cloning repository")
            return None
        except FileNotFoundError:
            logger.error("git command not found. Please ensure git is installed.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while downloading repository: {e}")
            return None
