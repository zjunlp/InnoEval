"""
Repository Analyzer - Static analysis and view generation (HCT, MCG, FCG).

This module is extracted and refactored from the original GlobalCodeTreeBuilder.
"""

import os
import ast
import json
import logging
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from collections import defaultdict

from .utils import should_ignore_path, get_code_abstract, IGNORED_DIRS, IGNORED_FILE_PATTERNS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RepoAnalyzer:
    """
    Repository Analyzer for static code analysis.
    
    Generates three views:
    - HCT (Hierarchical Component Tree): Package -> Module -> Class -> Function
    - MCG (Module Call Graph): Module import relationships
    - FCG (Function Call Graph): Function call relationships
    """
    
    def __init__(self, repo_path: str, ignored_dirs: List[str] = None, 
                 ignored_file_patterns: List[str] = None):
        """
        Initialize repository analyzer.
        
        Args:
            repo_path: Path to the code repository
            ignored_dirs: List of directories to ignore
            ignored_file_patterns: List of file patterns to ignore
        """
        self.repo_path = repo_path
        self.call_graph = nx.DiGraph()  # FCG: Function call graph
        self.modules = {}  # Module information
        self.functions = {}  # Function information
        self.classes = {}  # Class information
        self.imports = defaultdict(list)  # MCG: Import information
        
        # HCT: Hierarchical code tree
        self.code_tree = {
            'modules': {},
            'stats': {
                'total_modules': 0,
                'total_classes': 0,
                'total_functions': 0,
                'total_lines': 0
            },
            'key_components': []
        }
        
        self.ignored_dirs = ignored_dirs or IGNORED_DIRS
        self.ignored_file_patterns = ignored_file_patterns or IGNORED_FILE_PATTERNS
        
        # For importance analysis
        self.importance_analyzer = None
    
    def analyze(self) -> Dict:
        """
        Perform complete repository analysis.
        
        Returns:
            Analysis results containing modules, classes, functions, and views
        """
        logger.info(f"Starting to analyze repository: {self.repo_path}")
        
        # Step 1: Parse repository
        self._parse_repository()
        
        # Step 2: Build relationships
        self._build_call_relationships()
        self._build_hierarchical_code_tree()
        
        # Step 3: Identify key components
        self._identify_key_components()
        
        logger.info(f"Analysis completed: {len(self.modules)} modules, "
                   f"{len(self.classes)} classes, {len(self.functions)} functions")
        
        return {
            'modules': self.modules,
            'classes': self.classes,
            'functions': self.functions,
            'imports': dict(self.imports),
            'call_graph': self.call_graph,
            'code_tree': self.code_tree
        }
    
    def _parse_repository(self) -> None:
        """Parse all Python files in the repository."""
        for root, dirs, files in os.walk(self.repo_path):
            # Calculate directory depth
            rel_path = os.path.relpath(root, self.repo_path)
            current_depth = 0 if rel_path == '.' else len(rel_path.split(os.sep))
            
            # Limit depth to 3
            if current_depth > 3:
                dirs[:] = []
                continue
            
            # Filter ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            # Limit files per directory
            if len(files) > 100:
                continue
            elif len(files) > 50:
                files = files[:5]
            
            file_count = 0
            max_files_per_dir = 40
            
            for file in files:
                if file_count >= max_files_per_dir:
                    break
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                if should_ignore_path(rel_path, self.ignored_dirs, self.ignored_file_patterns):
                    continue
                
                # Skip large files (>10MB)
                file_size = os.path.getsize(file_path)
                if file_size > 10 * 1024 * 1024:
                    continue
                
                try:
                    if file.endswith('.py'):
                        self._parse_python_file(file_path, rel_path)
                        file_count += 1
                except Exception as e:
                    logger.error(f"Error parsing file {rel_path}: {e}")
    
    def _parse_python_file(self, file_path: str, rel_path: str) -> None:
        """Parse a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            module_node = ast.parse(content, filename=rel_path)
            module_docstring = ast.get_docstring(module_node) or ""
            
            # Create module ID
            module_id = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')
            self.modules[module_id] = {
                'path': rel_path,
                'docstring': module_docstring,
                'content': content,
                'functions': [],
                'classes': [],
                'lines': len(content.splitlines())
            }
            
            # Process imports
            self._process_imports(module_node, module_id)
            
            # Parse classes and functions
            for node in ast.walk(module_node):
                if isinstance(node, ast.FunctionDef):
                    if not hasattr(node, 'parent_class'):
                        self._process_function(node, module_id, None)
                
                elif isinstance(node, ast.ClassDef):
                    self._process_class(node, module_id, content)
        
        except SyntaxError as e:
            logger.warning(f"File {rel_path} has syntax errors: {e}")
        except Exception as e:
            logger.error(f"Error processing file {rel_path}: {e}")
    
    def _process_imports(self, module_node: ast.Module, module_id: str) -> None:
        """Process import statements."""
        for node in module_node.body:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self.imports[module_id].append({
                        'type': 'import',
                        'name': name.name,
                        'alias': name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    self.imports[module_id].append({
                        'type': 'importfrom',
                        'module': module,
                        'name': name.name,
                        'alias': name.asname
                    })
    
    def _process_class(self, node: ast.ClassDef, module_id: str, content: str) -> None:
        """Process a class definition."""
        class_id = f"{module_id}.{node.name}"
        class_docstring = ast.get_docstring(node) or ""
        
        # Analyze inheritance
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(self._get_attribute_path(base))
        
        self.classes[class_id] = {
            'name': node.name,
            'module': module_id,
            'docstring': class_docstring,
            'methods': [],
            'base_classes': base_classes,
            'source': self._get_source(content, node),
            'called_by': []
        }
        
        self.modules[module_id]['classes'].append(class_id)
        
        # Process methods
        for class_node in node.body:
            if isinstance(class_node, ast.FunctionDef):
                class_node.parent_class = class_id
                self._process_function(class_node, module_id, class_id)
    
    def _process_function(self, node: ast.FunctionDef, module_id: str, 
                         class_id: Optional[str]) -> None:
        """Process a function or method definition."""
        function_name = node.name
        if class_id:
            function_id = f"{class_id}.{function_name}"
            self.classes[class_id]['methods'].append(function_id)
        else:
            function_id = f"{module_id}.{function_name}"
            self.modules[module_id]['functions'].append(function_id)
        
        docstring = ast.get_docstring(node) or ""
        source = self._get_source(self.modules[module_id]['content'], node)
        
        # Extract parameters
        parameters = [{'name': arg.arg, 'type': None} for arg in node.args.args]
        
        # Extract function calls
        calls = self._extract_function_calls(node)
        
        self.functions[function_id] = {
            'name': function_name,
            'module': module_id,
            'class': class_id,
            'docstring': docstring,
            'parameters': parameters,
            'calls': calls,
            'called_by': [],
            'source': source
        }
        
        self.call_graph.add_node(function_id)
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[Dict]:
        """Extract function calls from function body."""
        calls = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                call_info = self._analyze_call(subnode)
                if call_info:
                    calls.append(call_info)
        return calls
    
    def _analyze_call(self, node: ast.Call) -> Optional[Dict]:
        """Analyze a function call expression."""
        if isinstance(node.func, ast.Name):
            return {'type': 'simple', 'name': node.func.id}
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return {
                    'type': 'attribute',
                    'object': node.func.value.id,
                    'attribute': node.func.attr
                }
            return {
                'type': 'nested_attribute',
                'full_path': self._get_attribute_path(node.func)
            }
        return None
    
    def _get_attribute_path(self, node: ast.Attribute) -> str:
        """Get complete attribute path."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _get_source(self, content: str, node: ast.AST) -> str:
        """Extract source code for a node."""
        try:
            lines = content.splitlines()
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                start = node.lineno - 1
                end = node.end_lineno
                return '\n'.join(lines[start:end])
        except:
            pass
        return ""
    
    def _build_call_relationships(self) -> None:
        """Build function call relationships for FCG."""
        logger.info("Building function call relationships...")
        
        for func_id, func_info in self.functions.items():
            calls = func_info['calls']
            module_id = func_info['module']
            
            for call in calls:
                called_func_id = self._resolve_call(call, module_id, func_info['class'])
                
                if called_func_id and called_func_id in self.functions:
                    self.call_graph.add_edge(func_id, called_func_id)
                    if func_id not in self.functions[called_func_id]['called_by']:
                        self.functions[called_func_id]['called_by'].append(func_id)
    
    def _resolve_call(self, call: Dict, module_id: str, class_id: Optional[str]) -> Optional[str]:
        """Resolve a function call to its ID."""
        if call['type'] == 'simple':
            # Check same module
            direct_func_id = f"{module_id}.{call['name']}"
            if direct_func_id in self.functions:
                return direct_func_id
            
            # Check same class
            if class_id:
                method_id = f"{class_id}.{call['name']}"
                if method_id in self.functions:
                    return method_id
        
        return None
    
    def _build_hierarchical_code_tree(self) -> None:
        """Build hierarchical code tree (HCT)."""
        logger.info("Building hierarchical code tree...")
        
        # Create package structure
        tree = {}
        for module_id, module_info in self.modules.items():
            parts = module_id.split('.')
            current = tree
            
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {'type': 'package', 'children': {}, 'name': part}
                current = current[part]['children']
            
            # Add module
            module_name = parts[-1]
            current[module_name] = {
                'type': 'module',
                'name': module_name,
                'id': module_id,
                'path': module_info['path'],
                'docstring': module_info['docstring'],
                'functions': [{'id': f, 'name': f.split('.')[-1]} 
                            for f in module_info['functions']],
                'classes': [{'id': c, 'name': c.split('.')[-1]} 
                           for c in module_info['classes']],
                'lines': module_info.get('lines', 0)
            }
        
        self.code_tree['modules'] = tree
        
        # Update stats
        self.code_tree['stats'] = {
            'total_modules': len(self.modules),
            'total_classes': len(self.classes),
            'total_functions': len(self.functions),
            'total_lines': sum(m.get('lines', 0) for m in self.modules.values())
        }
    
    def _identify_key_components(self) -> None:
        """Identify key components (classes and functions with high importance)."""
        logger.info("Identifying key components...")
        
        key_components = []
        
        # Simple heuristic: classes with many methods or referenced frequently
        for class_id, class_info in self.classes.items():
            if len(class_info['methods']) >= 3 or len(class_info.get('called_by', [])) > 0:
                key_components.append({
                    'type': 'class',
                    'id': class_id,
                    'name': class_info['name'],
                    'module': class_info['module']
                })
        
        # Functions that are called frequently
        for func_id, func_info in self.functions.items():
            if len(func_info.get('called_by', [])) >= 2:
                key_components.append({
                    'type': 'function',
                    'id': func_id,
                    'name': func_info['name'],
                    'module': func_info['module']
                })
        
        self.code_tree['key_components'] = key_components[:20]  # Limit to top 20
    
    def get_module_dependency_graph(self) -> nx.DiGraph:
        """
        Get module dependency graph (MCG).
        
        Returns:
            NetworkX DiGraph representing module dependencies
        """
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
    
    def export_to_json(self, output_file: str) -> None:
        """
        Export analysis results to JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        data = {
            'modules': self.modules,
            'classes': self.classes,
            'functions': self.functions,
            'imports': dict(self.imports),
            'code_tree': self.code_tree,
            'call_graph_edges': list(self.call_graph.edges())
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis results exported to: {output_file}")

