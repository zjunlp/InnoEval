"""
Importance Scorer - Component importance scoring based on multiple dimensions.

This module is extracted and refactored from the original ImportanceAnalyzer.
"""

import os
import re
import subprocess
import networkx as nx
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """
    Code importance analyzer for evaluating component importance.
    
    Scoring dimensions:
    - Usage frequency: How often modules/functions are imported/called
    - Module relationships: PageRank, in-degree/out-degree, betweenness centrality
    - Code complexity: Branches, nesting depth
    - Semantic importance: Keyword matching (main, core, api, etc.)
    - Git history: Commit count, recency
    """
    
    def __init__(self, repo_path: str, modules: Dict, classes: Dict,
                 functions: Dict, imports: Dict, code_tree: Dict,
                 call_graph: Optional[nx.DiGraph] = None,
                 weights: Optional[Dict] = None):
        """
        Initialize importance scorer.
        
        Args:
            repo_path: Path to the code repository
            modules: Module information dictionary
            classes: Class information dictionary
            functions: Function information dictionary
            imports: Import information dictionary
            code_tree: Code tree structure
            call_graph: Function call graph (optional)
            weights: Custom weights for importance calculation (optional)
        """
        self.repo_path = repo_path
        self.modules = modules
        self.classes = classes
        self.functions = functions
        self.imports = imports
        self.code_tree = code_tree
        self.call_graph = call_graph
        
        # Default weights
        default_weights = {
            'usage': 2.0,
            'imports_relationships': 3.0,
            'complexity': 1.0,
            'semantic': 0.5,
            'git_history': 4.0,
        }
        
        self.weights = default_weights
        if weights:
            self.weights.update(weights)
        
        # Important semantic keywords
        self.important_keywords = [
            'main', 'core', 'engine', 'api', 'service',
            'controller', 'manager', 'handler', 'processor',
            'factory', 'builder', 'provider', 'repository',
            'executor', 'scheduler', 'config', 'security'
        ]
        
        # Build module dependency graph
        self.module_dependency_graph = self._build_module_dependency_graph()
    
    def _build_module_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph between modules."""
        graph = nx.DiGraph()
        
        # Add all modules as nodes
        for module_id in self.modules:
            graph.add_node(module_id)
        
        # Add import relationships as edges
        for module_id, imports_list in self.imports.items():
            for imp in imports_list:
                if imp['type'] == 'import':
                    imported_module = imp['name']
                    if imported_module in self.modules:
                        graph.add_edge(module_id, imported_module)
                elif imp['type'] == 'importfrom':
                    imported_module = imp['module']
                    if imported_module in self.modules:
                        graph.add_edge(module_id, imported_module)
        
        return graph
    
    def calculate_node_importance(self, node: Dict) -> float:
        """
        Calculate importance score of a node.
        
        Args:
            node: Node information (must have 'type' field)
            
        Returns:
            Importance score (0.0 - 10.0)
        """
        if 'type' not in node:
            return 0.0
        
        if node['type'] == 'module':
            return self._calculate_module_importance(node)
        elif node['type'] == 'package':
            return self._calculate_package_importance(node)
        else:
            return 0.0
    
    def _calculate_module_importance(self, node: Dict) -> float:
        """Calculate importance score of a module."""
        importance = 0.0
        
        # Usage frequency analysis
        usage_score = self._analyze_usage(node)
        importance += usage_score * self.weights['usage']
        
        # Inter-module reference relationship analysis
        imports_score = self._analyze_imports_relationships(node)
        importance += imports_score * self.weights['imports_relationships']
        
        # Code complexity analysis
        complexity_score = self._analyze_complexity(node)
        importance += complexity_score * self.weights['complexity']
        
        # Semantic importance analysis
        semantic_score = self._analyze_semantic_importance(node)
        importance += semantic_score * self.weights['semantic']
        
        # Git history analysis
        git_score = self._analyze_git_history(node)
        importance += git_score * self.weights['git_history']
        
        return min(importance, 10.0)
    
    def _calculate_package_importance(self, node: Dict) -> float:
        """Calculate package importance score."""
        importance = 0.0
        
        # Semantic importance
        if 'name' in node:
            semantic_score = self._semantic_importance(node['name'])
            importance += semantic_score * self.weights['semantic']
        
        # Importance of child nodes
        if 'children' in node and node['children']:
            child_scores = [self.calculate_node_importance(child) 
                          for child in node['children'].values()]
            if child_scores:
                max_score = max(child_scores)
                avg_score = sum(child_scores) / len(child_scores)
                importance += (max_score * 0.7 + avg_score * 0.3) * 1.5
        
        # Special package names
        if 'name' in node and node['name'] in ['src', 'core', 'main', 'api']:
            importance += 2.0
        
        return min(importance, 10.0)
    
    def _analyze_imports_relationships(self, node: Dict) -> float:
        """Analyze inter-module reference relationships importance."""
        score = 0.0
        
        if node['type'] == 'module' and 'id' in node:
            module_id = node['id']
            
            if module_id in self.module_dependency_graph:
                in_degree = self.module_dependency_graph.in_degree(module_id)
                out_degree = self.module_dependency_graph.out_degree(module_id)
                
                # Calculate PageRank
                if len(self.module_dependency_graph.nodes()) > 0:
                    try:
                        pagerank = nx.pagerank(
                            self.module_dependency_graph,
                            alpha=0.85,
                            personalization={n: 2.0 if n == module_id else 1.0 
                                           for n in self.module_dependency_graph.nodes()}
                        )
                        pagerank_score = pagerank.get(module_id, 0.0) * 10
                    except:
                        pagerank_score = 0.0
                else:
                    pagerank_score = 0.0
                
                # Calculate betweenness centrality
                betweenness = 0.0
                if len(self.module_dependency_graph.nodes()) > 1:
                    try:
                        between_dict = nx.betweenness_centrality(
                            self.module_dependency_graph,
                            k=min(20, len(self.module_dependency_graph.nodes())),
                            normalized=True
                        )
                        betweenness = between_dict.get(module_id, 0.0)
                    except:
                        betweenness = 0.0
                
                # Combine scores
                in_degree_score = min(in_degree / 5.0, 1.0) * 0.5
                out_degree_score = min(out_degree / 10.0, 1.0) * 0.2
                pagerank_score = min(pagerank_score, 1.0) * 0.6
                betweenness_score = min(betweenness * 10, 1.0) * 0.4
                
                score = (in_degree_score + out_degree_score + 
                        pagerank_score + betweenness_score) / 1.7
                
                # Bonus for root modules
                if in_degree > 2 and out_degree <= 1:
                    score += 0.3
                
                # Bonus for integration modules
                if in_degree > 2 and out_degree > 2:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_usage(self, node: Dict) -> float:
        """Analyze node usage frequency."""
        score = 0.0
        
        if node['type'] == 'module' and 'id' in node:
            module_id = node['id']
            
            # Count imports
            import_count = 0
            for imports in self.imports.values():
                for imp in imports:
                    if ((imp['type'] == 'import' and imp['name'] == module_id) or
                        (imp['type'] == 'importfrom' and imp['module'] == module_id)):
                        import_count += 1
            
            score = min(import_count / 5.0, 1.0)
            
            # Count function/class calls
            if 'functions' in node:
                func_call_count = 0
                for func_ref in node['functions']:
                    if isinstance(func_ref, dict) and 'id' in func_ref:
                        func_id = func_ref['id']
                        if func_id in self.functions:
                            func_call_count += len(self.functions[func_id].get('called_by', []))
                score += min(func_call_count / 10.0, 1.0) * 0.5
            
            if 'classes' in node:
                class_call_count = 0
                for class_ref in node['classes']:
                    if isinstance(class_ref, dict) and 'id' in class_ref:
                        class_id = class_ref['id']
                        if class_id in self.classes:
                            class_call_count += len(self.classes[class_id].get('called_by', []))
                score += min(class_call_count / 10.0, 1.0) * 0.5
        
        return score
    
    def _analyze_complexity(self, node: Dict) -> float:
        """Analyze node code complexity."""
        score = 0.0
        
        if node['type'] == 'module' and 'id' in node:
            module_id = node['id']
            if module_id in self.modules and 'content' in self.modules[module_id]:
                content = self.modules[module_id]['content']
                lines = content.splitlines()
                
                # Count branches and loops
                if_count = sum(1 for line in lines if re.search(r'\bif\b', line))
                for_count = sum(1 for line in lines if re.search(r'\bfor\b', line))
                while_count = sum(1 for line in lines if re.search(r'\bwhile\b', line))
                except_count = sum(1 for line in lines if re.search(r'\bexcept\b', line))
                
                branch_count = if_count + for_count + while_count + except_count
                score = min(branch_count / 50.0, 1.0)
        
        return score
    
    def _analyze_semantic_importance(self, node: Dict) -> float:
        """Analyze node semantic importance."""
        score = 0.0
        
        if 'name' in node:
            score += self._semantic_importance(node['name'])
        
        if 'id' in node:
            module_parts = node['id'].split('.')
            for part in module_parts:
                score += self._semantic_importance(part) * 0.5
        
        return min(score, 1.0)
    
    def _semantic_importance(self, name: str) -> float:
        """Semantic importance analysis based on name."""
        score = 0.0
        name_lower = name.lower()
        
        # Check for important keywords
        for keyword in self.important_keywords:
            if keyword in name_lower:
                score += 0.3
                break
        
        # Entry points
        if name == '__main__' or name == 'main':
            score += 0.7
        
        # Common important file names
        if name in ['__init__', 'app', 'settings', 'config', 'utils', 'constants']:
            score += 0.5
        
        return min(score, 1.0)
    
    def _analyze_git_history(self, node: Dict) -> float:
        """Analyze node Git history."""
        score = 0.0
        
        if node['type'] == 'module' and 'id' in node:
            module_id = node['id']
            if module_id in self.modules and 'path' in self.modules[module_id]:
                file_path = os.path.join(self.repo_path, self.modules[module_id]['path'])
                score = self._get_file_history_importance(file_path)
        
        return score
    
    def _get_file_history_importance(self, file_path: str) -> float:
        """Calculate importance based on file's Git history."""
        try:
            if not os.path.exists(file_path):
                return 0.0
            
            # Check if in Git repository
            if not os.path.exists(os.path.join(self.repo_path, '.git')):
                return 0.0
            
            # Get commit count
            try:
                rel_path = os.path.relpath(file_path, self.repo_path)
                cmd = ['git', '-C', self.repo_path, 'log', '--oneline', '--', rel_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    commit_lines = result.stdout.strip().split('\n')
                    commit_count = len([line for line in commit_lines if line])
                    score = min(commit_count / 20.0, 1.0)
                    
                    # Get last modification time
                    cmd_last = ['git', '-C', self.repo_path, 'log', '-1', '--format=%at', '--', rel_path]
                    result_last = subprocess.run(cmd_last, capture_output=True, text=True, check=False)
                    
                    if result_last.returncode == 0 and result_last.stdout.strip():
                        import time
                        try:
                            last_commit_time = int(result_last.stdout.strip())
                            current_time = int(time.time())
                            days_since_last_commit = (current_time - last_commit_time) / (60 * 60 * 24)
                            recency_score = max(0, 1.0 - (days_since_last_commit / 365))
                            score = (score * 0.7) + (recency_score * 0.3)
                        except:
                            pass
                    
                    return score
                
                return 0.0
            except subprocess.SubprocessError:
                return 0.0
        except Exception:
            return 0.0
    
    def score_all_modules(self) -> Dict[str, float]:
        """
        Score all modules and return a dictionary of module_id -> score.
        
        Returns:
            Dictionary mapping module IDs to importance scores
        """
        scores = {}
        
        for module_id in self.modules:
            node = {
                'type': 'module',
                'id': module_id,
                'name': module_id.split('.')[-1],
                **self.modules[module_id]
            }
            scores[module_id] = self.calculate_node_importance(node)
        
        return scores
    
    def get_key_modules(self, top_k: int = 20, min_score: float = 0.0) -> List[Dict]:
        """
        Get key modules sorted by importance.
        
        Args:
            top_k: Number of top modules to return
            min_score: Minimum score threshold
            
        Returns:
            List of module information with scores, sorted by importance
        """
        scores = self.score_all_modules()
        
        key_modules = []
        for module_id, score in scores.items():
            if score >= min_score:
                key_modules.append({
                    'id': module_id,
                    'name': module_id.split('.')[-1],
                    'path': self.modules[module_id]['path'],
                    'importance_score': score
                })
        
        # Sort by importance score
        key_modules.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return key_modules[:top_k]

