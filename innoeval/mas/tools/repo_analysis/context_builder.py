"""
Context Builder - Generate context for core modules based on analysis results.

This module extracts and builds contextual information for LLM consumption.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .utils import get_code_abs_token, get_code_abstract

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Build context from repository analysis results.
    
    Generates structured context including:
    - Repository overview
    - Hierarchical structure (HCT)
    - Dependency graphs (MCG, FCG)
    - Core component abstracts
    - Execution context
    """
    
    def __init__(self, repo_path: str, analysis_results: Dict,
                 key_modules: List[Dict] = None):
        """
        Initialize context builder.
        
        Args:
            repo_path: Path to the repository
            analysis_results: Results from RepoAnalyzer
            key_modules: List of key modules with importance scores (optional)
        """
        self.repo_path = repo_path
        self.modules = analysis_results.get('modules', {})
        self.classes = analysis_results.get('classes', {})
        self.functions = analysis_results.get('functions', {})
        self.imports = analysis_results.get('imports', {})
        self.call_graph = analysis_results.get('call_graph')
        self.code_tree = analysis_results.get('code_tree', {})
        self.key_modules = key_modules or []
    
    def build_context(self, max_tokens: int = 8000, 
                     include_readme: bool = True) -> Dict:
        """
        Build complete context.
        
        Args:
            max_tokens: Maximum token limit for context
            include_readme: Whether to include README files
            
        Returns:
            Dictionary containing structured context
        """
        logger.info(f"Building context (max_tokens={max_tokens})")
        
        context = {
            'metadata': self._build_metadata(),
            'repository_overview': self._build_overview(),
            'hierarchical_structure': self._build_hierarchical_structure(),
            'dependency_graphs': self._build_dependency_graphs(),
            'core_components': self._build_core_components(max_tokens),
            'execution_context': self._build_execution_context()
        }
        
        return context
    
    def _build_metadata(self) -> Dict:
        """Build metadata section."""
        return {
            'repo_path': self.repo_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzer_version': '0.1.0'
        }
    
    def _build_overview(self) -> Dict:
        """Build repository overview."""
        stats = self.code_tree.get('stats', {})
        
        # Extract key technologies from imports
        key_technologies = set()
        for imports_list in self.imports.values():
            for imp in imports_list:
                module_name = imp.get('name') or imp.get('module', '')
                if module_name:
                    # Extract top-level package name
                    top_level = module_name.split('.')[0]
                    if top_level and not top_level.startswith('_'):
                        key_technologies.add(top_level)
        
        return {
            'total_modules': stats.get('total_modules', 0),
            'total_classes': stats.get('total_classes', 0),
            'total_functions': stats.get('total_functions', 0),
            'total_lines': stats.get('total_lines', 0),
            'key_technologies': sorted(list(key_technologies))[:20]
        }
    
    def _build_hierarchical_structure(self) -> Dict:
        """Build hierarchical structure (HCT) summary."""
        return {
            'packages': self._extract_packages(self.code_tree.get('modules', {})),
            'modules': list(self.modules.keys())[:50],  # Limit to 50 modules
            'importance_filtered': True
        }
    
    def _extract_packages(self, tree: Dict, depth: int = 0, max_depth: int = 2) -> List[str]:
        """Extract package names from tree."""
        packages = []
        if depth > max_depth:
            return packages
        
        for name, node in tree.items():
            if isinstance(node, dict) and node.get('type') == 'package':
                packages.append(name)
                if 'children' in node:
                    child_packages = self._extract_packages(node['children'], depth + 1, max_depth)
                    packages.extend([f"{name}.{p}" for p in child_packages])
        
        return packages
    
    def _build_dependency_graphs(self) -> Dict:
        """Build dependency graphs (MCG and FCG)."""
        # MCG: Module imports
        module_imports = {}
        for module_id, imports_list in self.imports.items():
            imported = []
            for imp in imports_list:
                if imp['type'] == 'import':
                    imported.append(imp['name'])
                elif imp['type'] == 'importfrom':
                    imported.append(imp['module'])
            module_imports[module_id] = imported[:10]  # Limit per module
        
        # FCG: Function calls (limited to key functions)
        function_calls = {}
        if self.call_graph:
            for edge in list(self.call_graph.edges())[:100]:  # Limit to 100 edges
                caller, callee = edge
                if caller not in function_calls:
                    function_calls[caller] = []
                function_calls[caller].append(callee)
        
        return {
            'module_imports': module_imports,
            'function_calls': function_calls
        }
    
    def _build_core_components(self, max_tokens: int) -> List[Dict]:
        """Build core components with abstracts."""
        core_components = []
        current_tokens = 0
        
        # Prioritize key modules
        for module_info in self.key_modules:
            if current_tokens >= max_tokens * 0.7:  # Use 70% of tokens for core components
                break
            
            module_id = module_info['id']
            if module_id not in self.modules:
                continue
            
            module_data = self.modules[module_id]
            content = module_data.get('content', '')
            path = module_data.get('path', '')
            
            # Generate abstract using tree-sitter if available
            try:
                abstract = get_code_abstract(path, content, max_token=1000)
            except Exception as e:
                logger.warning(f"Error generating abstract for {path}: {e}")
                # Fallback: use first N lines
                lines = content.splitlines()
                abstract = '\n'.join(lines[:30]) + '\n...(truncated)'
            
            component = {
                'type': 'module',
                'id': module_id,
                'path': path,
                'importance_score': module_info.get('importance_score', 0.0),
                'abstract': abstract,
                'summary': module_data.get('docstring', '')[:200],  # Limit docstring
                'num_classes': len(module_data.get('classes', [])),
                'num_functions': len(module_data.get('functions', []))
            }
            
            component_str = json.dumps(component, ensure_ascii=False)
            component_tokens = get_code_abs_token(component_str)
            
            if current_tokens + component_tokens < max_tokens * 0.7:
                core_components.append(component)
                current_tokens += component_tokens
            else:
                break
        
        return core_components
    
    def _build_execution_context(self) -> Dict:
        """Build execution context with entry points and dependencies."""
        entry_points = []
        
        # Find entry points (files with __main__ or main function)
        for module_id, module_data in self.modules.items():
            content = module_data.get('content', '')
            if '__main__' in content or 'if __name__' in content:
                entry_points.append(module_data.get('path', module_id))
        
        return {
            'entry_points': entry_points[:5],  # Limit to 5
            'key_dependencies': self._extract_key_dependencies()
        }
    
    def _extract_key_dependencies(self) -> List[str]:
        """Extract key external dependencies."""
        external_deps = set()
        
        # Common Python standard library modules to filter out
        stdlib_modules = {
            'os', 'sys', 're', 'json', 'time', 'datetime', 'collections',
            'itertools', 'functools', 'math', 'random', 'logging', 'argparse',
            'pathlib', 'typing', 'io', 'subprocess', 'threading', 'multiprocessing'
        }
        
        for imports_list in self.imports.values():
            for imp in imports_list:
                module_name = imp.get('name') or imp.get('module', '')
                if module_name:
                    top_level = module_name.split('.')[0]
                    if (top_level and 
                        not top_level.startswith('_') and
                        top_level not in stdlib_modules and
                        top_level not in self.modules):
                        external_deps.add(top_level)
        
        return sorted(list(external_deps))[:20]  # Limit to 20
    
    def export_to_json(self, output_file: str, max_tokens: int = 8000) -> None:
        """
        Export context to JSON file.
        
        Args:
            output_file: Path to output file
            max_tokens: Maximum token limit
        """
        context = self.build_context(max_tokens=max_tokens)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Context exported to: {output_file}")
    
    def export_to_string(self, max_tokens: int = 8000) -> str:
        """
        Export context as formatted string for LLM consumption.
        
        Args:
            max_tokens: Maximum token limit
            
        Returns:
            Formatted context string
        """
        context = self.build_context(max_tokens=max_tokens)
        
        lines = []
        lines.append("# Repository Context\n")
        
        # Overview
        lines.append("## Overview")
        overview = context['repository_overview']
        lines.append(f"- Modules: {overview['total_modules']}")
        lines.append(f"- Classes: {overview['total_classes']}")
        lines.append(f"- Functions: {overview['total_functions']}")
        lines.append(f"- Lines of Code: {overview['total_lines']}")
        lines.append(f"- Key Technologies: {', '.join(overview['key_technologies'][:10])}\n")
        
        # Core components
        lines.append("## Core Components\n")
        for comp in context['core_components']:
            lines.append(f"### {comp['path']} (Score: {comp['importance_score']:.2f})")
            lines.append(f"**Summary:** {comp['summary']}")
            lines.append(f"\n```python\n{comp['abstract']}\n```\n")
        
        # Entry points
        exec_ctx = context['execution_context']
        if exec_ctx['entry_points']:
            lines.append("## Entry Points")
            for ep in exec_ctx['entry_points']:
                lines.append(f"- {ep}")
            lines.append("")
        
        # Dependencies
        if exec_ctx['key_dependencies']:
            lines.append("## Key Dependencies")
            lines.append(", ".join(exec_ctx['key_dependencies']))
            lines.append("")
        
        return "\n".join(lines)

