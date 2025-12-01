"""Configuration management for the manifold-bends project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, Optional
import yaml


@dataclass
class ProjectConfig:
    """Main configuration object for the project."""
    
    # Model configurations
    embedding_model: str = "text-embedding-3-large"
    generation_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    
    # Data and results paths
    data_dir: str = "data"
    results_dir: str = "results"
    
    # Experiment parameters
    max_prompts_per_category: int = 300
    seed: int = 42
    
    # Geometry computation parameters
    n_neighbors_id: int = 20  # for intrinsic dimension estimation
    n_neighbors_curvature: int = 30  # for curvature proxy
    n_pca_components: int = 10  # for oppositeness score
    n_flip_components: int = 3  # for oppositeness score
    
    # API parameters
    api_timeout: int = 60
    max_retries: int = 3
    batch_size: int = 100
    embedding_batch_size: int = 100  # For embedding API calls
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
    
    @property
    def prompts_dir(self) -> Path:
        """Return prompts directory path."""
        return self.data_dir / "prompts"
    
    @property
    def processed_dir(self) -> Path:
        """Return processed data directory path."""
        return self.data_dir / "processed"
    
    @property
    def figures_dir(self) -> Path:
        """Return figures directory path."""
        return self.results_dir / "figures"
    
    @property
    def tables_dir(self) -> Path:
        """Return tables directory path."""
        return self.results_dir / "tables"


def load_config(path: Union[str, Path]) -> ProjectConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        ProjectConfig object with loaded settings
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ProjectConfig(**config_dict)


def save_config(config: ProjectConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: ProjectConfig object to save
        path: Path to save YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict, handling Path objects
    config_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in config.__dict__.items()
    }
    
    with open(path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
