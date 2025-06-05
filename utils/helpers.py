"""
Helper utilities for Research Q&A Bot
"""
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_response_time(seconds: float) -> str:
    """
    Format response time in human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple regex
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction - find words longer than 3 characters
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been',
        'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there',
        'what', 'about', 'more', 'very', 'could', 'only', 'other', 'after',
        'first', 'well', 'also', 'where', 'much', 'through', 'when', 'before'
    }
    
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequency and return most common
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> bool:
    """
    Validate file path
    
    Args:
        path: File path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path_obj = Path(path)
        
        if must_exist:
            return path_obj.exists() and path_obj.is_file()
        else:
            # Check if parent directory exists
            return path_obj.parent.exists()
    except Exception:
        return False


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\[\]{}"]', '', text)
    
    return text


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source citations for display
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted sources string
    """
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        if isinstance(source, dict):
            # Extract relevant fields
            title = source.get('title', 'Unknown')
            page = source.get('page', '')
            score = source.get('score', 0)
            
            source_str = f"{i}. {title}"
            if page:
                source_str += f" (Page {page})"
            if score > 0:
                source_str += f" [Relevance: {score:.2f}]"
                
            formatted.append(source_str)
        else:
            formatted.append(f"{i}. {str(source)}")
    
    return "\n".join(formatted)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        return True
    except Exception:
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def measure_time(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that returns (result, execution_time)
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    return wrapper


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Format timestamp to readable string
    
    Args:
        timestamp: Unix timestamp (defaults to current time)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = time.time()
    
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_research_mode(mode_name: str, config_modes: List[Dict]) -> Optional[Dict]:
    """
    Parse research mode configuration
    
    Args:
        mode_name: Name of the research mode
        config_modes: List of mode configurations
        
    Returns:
        Mode configuration or None if not found
    """
    for mode in config_modes:
        if mode.get('name') == mode_name:
            return mode
    
    return None


def create_response_metadata(
    query: str,
    mode: str,
    response_time: float,
    sources_count: int = 0
) -> Dict[str, Any]:
    """
    Create metadata for response
    
    Args:
        query: Original query
        mode: Research mode used
        response_time: Time taken to generate response
        sources_count: Number of sources used
        
    Returns:
        Metadata dictionary
    """
    return {
        "timestamp": format_timestamp(),
        "query": truncate_text(query, 200),
        "mode": mode,
        "response_time": format_response_time(response_time),
        "sources_count": sources_count,
        "keywords": extract_keywords(query, 5)
    }
