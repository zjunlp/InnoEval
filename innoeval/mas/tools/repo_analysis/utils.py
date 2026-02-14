"""Utility functions for repository analysis."""

import re
import os
import tiktoken
from typing import List

# Default directories and file patterns to ignore
IGNORED_DIRS = [
    '__pycache__', '.git', '.vscode', 'venv', 'env', 'node_modules',
    '.pytest_cache', 'build', 'dist', '.github', 'logs', '.idea',
    '.mypy_cache', '.tox', 'htmlcov', '.coverage', 'eggs', '.eggs'
]

IGNORED_FILE_PATTERNS = [
    r'.*\.pyc$', r'.*\.pyo$', r'.*\.pyd$', r'.*\.so$', r'.*\.dll$',
    r'.*\.class$', r'.*\.egg-info$', r'.*~$', r'.*\.swp$'
]


def get_code_abs_token(content: str) -> int:
    """
    Calculate token count of content using tiktoken.
    
    Args:
        content: Text content to calculate tokens for
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(content))


def should_ignore_path(path: str, ignored_dirs: List[str] = None, 
                       ignored_patterns: List[str] = None) -> bool:
    """
    Determine whether a given path should be ignored.
    
    Args:
        path: File or directory path to check
        ignored_dirs: List of directory names to ignore
        ignored_patterns: List of regex patterns for files to ignore
        
    Returns:
        True if path should be ignored, False otherwise
    """
    if ignored_dirs is None:
        ignored_dirs = IGNORED_DIRS
    if ignored_patterns is None:
        ignored_patterns = IGNORED_FILE_PATTERNS
    
    # Special handling for .ipynb files - we want to parse them
    if path.endswith('.ipynb') and not any(part in ignored_dirs for part in path.split(os.sep)):
        return False
    
    # Ignore hidden files and __pycache__
    if path.startswith('.') or path.startswith('__'):
        return True
    
    # Ignore media files
    media_extensions = (
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico', '.webp',  # Images
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mpeg', '.mpg', '.m4v', '.mkv', '.webm',  # Videos
        '.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.wma', '.m4b', '.m4p'  # Audio
    )
    if path.endswith(media_extensions):
        return True
    
    # Ignore compressed files
    compressed_extensions = (
        '.zip', '.rar', '.tar', '.gz', '.bz2', '.7z', '.iso', '.dmg',
        '.pkg', '.deb', '.rpm', '.msi', '.exe', '.app'
    )
    if path.endswith(compressed_extensions):
        return True
    
    # Ignore document files
    doc_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')
    if path.endswith(doc_extensions):
        return True
    
    # Check path components for ignored directories
    path_parts = path.split(os.sep)
    for part in path_parts:
        if part in ignored_dirs:
            return True
    
    # Check file name against patterns
    file_name = os.path.basename(path)
    for pattern in ignored_patterns:
        if re.match(pattern, file_name):
            return True
    
    return False


def get_code_abstract(filename: str, source_code: str, max_token: int = 3000, 
                     child_context: bool = False) -> str:
    """
    Generate code structure abstract using grep_ast.TreeContext.
    
    Args:
        filename: Name of the file
        source_code: Source code content
        max_token: Maximum token limit
        child_context: Whether to include child context
        
    Returns:
        Code structure abstract
    """
    try:
        from grep_ast import TreeContext
    except ImportError:
        # Fallback: return first N lines if grep_ast not available
        lines = source_code.splitlines()
        return '\n'.join(lines[:50]) + '\n...(truncated)'
    
    context = TreeContext(
        filename,
        source_code,
        color=False,
        line_number=False,
        child_context=child_context,
        last_line=False,
        margin=0,
        mark_lois=False,
        loi_pad=0,
        show_top_of_file_parent_scope=False,
    )
    
    structure_lines = []
    important_lines = []
    
    for i, line in enumerate(context.lines):
        # Match function definitions, class definitions
        if re.match(r'^\s*(def|class)\s+', line):
            important_lines.append(i)
            
            # Check for single-line docstring
            if ('"""' in line and line.count('"""') >= 2) or ("'''" in line and line.count("'''") >= 2):
                pass
            else:
                # Check if next line is docstring start
                docstring_start = i + 1
                if docstring_start < len(context.lines):
                    next_line = context.lines[docstring_start]
                    triple_double = '"""' in next_line
                    triple_single = "'''" in next_line
                    
                    if triple_double or triple_single:
                        quote_type = '"""' if triple_double else "'''"
                        
                        if next_line.count(quote_type) >= 2:
                            important_lines.append(docstring_start)
                        else:
                            # Multi-line docstring
                            for j in range(docstring_start, len(context.lines)):
                                important_lines.append(j)
                                if j > docstring_start and quote_type in context.lines[j]:
                                    break
                                    
        elif re.match(r'^\s*(import|from)\s+', line) and i < 50:
            # Only focus on import statements at the beginning
            structure_lines.append(i)
    
    # Add found lines as lines of interest
    context.lines_of_interest = set(important_lines)
    context.add_lines_of_interest(structure_lines)
    context.add_context()
    
    # Format and output
    formatted_code = context.format()
    formatted_code = '\n'.join([line[1:] if line.startswith(' ') else line 
                                for line in formatted_code.split('\n')])
    
    return formatted_code


def cut_logs_by_token(logs_all: str, max_token: int = 4000) -> str:
    """
    Cut logs based on token count limit, keeping head and tail.
    
    Args:
        logs_all: Complete logs
        max_token: Maximum token limit
        
    Returns:
        Truncated logs
    """
    if get_code_abs_token(logs_all) <= max_token:
        return logs_all
    
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(logs_all)
    
    half_token = max_token // 2
    head_tokens = tokens[:half_token]
    tail_tokens = tokens[-half_token:]
    
    head_text = encoding.decode(head_tokens)
    tail_text = encoding.decode(tail_tokens)
    
    return f"{head_text}\n\n>>> ...omitted content... <<<\n\n{tail_text}"

