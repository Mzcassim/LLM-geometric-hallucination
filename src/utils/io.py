"""I/O utilities for JSONL and other file formats."""

import json
from pathlib import Path
from typing import Any, Union


def read_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    """
    Read JSONL file into a list of dictionaries.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    path = Path(path)
    data = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    return data


def write_jsonl(path: Union[str, Path], data: list[dict[str, Any]]) -> None:
    """
    Write list of dictionaries to JSONL file.
    
    Args:
        path: Path to output JSONL file
        data: List of dictionaries to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(path: Union[str, Path], item: dict[str, Any]) -> None:
    """
    Append a single item to a JSONL file.
    
    Args:
        path: Path to JSONL file
        item: Dictionary to append
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
