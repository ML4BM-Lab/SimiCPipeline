#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Pipeline Class
Provides common functionality shared across all SimiC pipeline modules.
"""

from pathlib import Path
from typing import Union, Optional
from simicpipeline.utils.io import format_time, print_tree
import pickle

class SimiCBase:
    """
    Base class for all SimiC pipeline modules.
    
    Provides common functionality for:
    - Project directory initialization
    - Time formatting
    - Project structure visualization

    All pipeline classes (SimiCPipeline, SimiCVisualization, SimiCPreprocess, etc.)
    should inherit from this class.
    """
    
    def __init__(self, project_dir: Union[str, Path]):
        """
        Initialize base pipeline with project directory.
        
        Args:
            project_dir (str or Path): Working directory for input/output files
        """
        self.project_dir = Path(project_dir)
        # Create project directory if it doesn't exist
        if not self.project_dir.exists():
            print(f"Creating project directory: {self.project_dir}")
            self.project_dir.mkdir(parents=True, exist_ok=True)

    def format_time(self, seconds: float) -> str:
        return format_time(seconds)
  
    def print_project_info(self) -> dict:
        """
        Get information about the project structure.
        
        Returns:
            dict: Dictionary with project paths and status
        """
        return print_tree(self.project_dir, prefix="", max_depth=3)

    def set_paths_custom(self, 
                         force = False,
                         p2df: Optional[Union[str, Path]] = None,
                         p2tf: Optional[Union[str, Path]] = None,
                         p2assignment: Optional[Union[str, Path]] = None,
                         p2simic_matrices: Optional[Union[str, Path]] = None, 
                         p2filtered_matrices: Optional[Union[str, Path]] = None, 
                         p2auc_raw: Optional[Union[str, Path]] = None, 
                         p2auc_filtered: Optional[Union[str, Path]] = None,
                         p2figures: Optional[Union[str, Path]] = None):
        """
        Set up all file paths for input and output with custom paths (intended for loading previous results).
        
        Args:
            force (bool): If True, set paths without checking if they exist and raising error. Defaults to False.
            magic_output_file (str or Path): Path to MAGIC output file.
            
            p2df (str or Path): Path to expression matrix file.
            p2tf (str or Path): Path to TF list file.
            p2assignment (str or Path, optional): Path to phenotype assignment file.
            p2simic_matrices (str or Path, optional): Path to SimiC raw matrices file.
            p2filtered_matrices (str or Path, optional): Path to filtered matrices file.
            p2auc_raw (str or Path, optional): Path to raw AUC matrices file.
            p2auc_filtered (str or Path, optional): Path to filtered AUC matrices file.
            p2figures (str or Path, optional): Path to figures output directory.
        
        Raises:
            FileNotFoundError: If paths do not exist.
            Self for method chaining
        """
        print("\n" + "="*70)
        print("SETTING CUSTOM PATHS")
        print("="*70 + "\n")
        if force:
            print("Force mode enabled: Skipping path existence checks.")
            self.p2df = Path(p2df)
            self.p2assignment = Path(p2assignment)
            self.p2tf = Path(p2tf)
            self.p2simic_matrices = Path(p2simic_matrices)
            self.p2filtered_matrices = Path(p2filtered_matrices)
            self.p2auc_raw = Path(p2auc_raw)
            self.p2auc_filtered = Path(p2auc_filtered)
            self.p2figures = Path(p2figures) 
            print("\n" + "="*70 + "\n")
            return self
        # Input paths - use custom paths if provided, otherwise raise error
        if p2df is not None and not Path(p2df).exists():
            raise FileNotFoundError("Invalid path to expression matrix dataframe (p2df).")
        self.p2df = Path(p2df)

        if p2assignment is not None and not Path(p2assignment).exists():
            raise FileNotFoundError("Invalid path to phenotype assignment file (p2assignment).")
        self.p2assignment = Path(p2assignment)
        
        if p2tf is not None and not Path(p2tf).exists():
            raise FileNotFoundError("Invalid path to TF list file (p2tf).")
        self.p2tf = Path(p2tf)
        
        if p2simic_matrices is not None and not Path(p2simic_matrices).exists():
            raise FileNotFoundError("Invalid path to SimiC raw matrices file (p2simic_matrices).")
        self.p2simic_matrices = Path(p2simic_matrices)
        if p2filtered_matrices is not None and not Path(p2filtered_matrices).exists():
            raise FileNotFoundError("Invalid path to filtered matrices file (p2filtered_matrices).\n" \
                                     "Provide a valid path or run `SimiCPipeline.filter_weights()`.")
        self.p2filtered_matrices = Path(p2filtered_matrices)
        if p2auc_raw is not None and not Path(p2auc_raw).exists():
            raise FileNotFoundError("Invalid path to raw AUC matrices file (p2auc_raw).\n"
                                    "Provide a valid path or run `SimiCPipeline.calculate_auc()`.")
        self.p2auc_raw = Path(p2auc_raw)
        if p2auc_filtered is not None and not Path(p2auc_filtered).exists():
            raise FileNotFoundError("Invalid path to filtered AUC matrices file (p2auc_filtered). \n " \
                                     "Provide a valid path or run `SimiCPipeline.calculate_auc(use_filtered = True)`.")
        self.p2auc_filtered = Path(p2auc_filtered)
        if p2figures is not None and not Path(p2figures).exists():
            print("Could not find path to figures output directory (p2figures). Creting it.")
            Path(p2figures).mkdir(parents=True, exist_ok=True)  
        self.p2figures = Path(p2figures)
        
        print("\n" + "="*70 + "\n")
        return self
    
    def load_results(self, result_type='Ws_filtered'):
        """
        Load pipeline results.

        Args:
            result_type (str): Type of results to load ('Ws_raw', 'Ws_filtered', 'auc_raw', 'auc_filtered')

        Returns:
            dict: Loaded results
        """
        if not hasattr(self, 'p2simic_matrices'):
            raise AttributeError("Paths to simic matrices not set. Please run `set_paths_custom()` first.")
        if not hasattr(self, 'p2filtered_matrices'):
            raise AttributeError("Paths to filtered simic matrices not set. Please run `set_paths_custom()` first.")
        if not hasattr(self, 'p2auc_raw'):
            raise AttributeError("Paths to AUC files not set. Please run `set_paths_custom()` first.")
        if not hasattr(self, 'p2auc_filtered'):
            raise AttributeError("Paths to filtered AUC files not set. Please run `set_paths_custom()` first.")
        
        file_map = {
            'Ws_raw': self.p2simic_matrices,
            'Ws_filtered': self.p2filtered_matrices,
            'auc_raw': self.p2auc_raw,
            'auc_filtered': self.p2auc_filtered
        }
        
        if result_type not in file_map:
            raise ValueError(f"result_type must be one of {list(file_map.keys())}")
        
        result_file = file_map[result_type]
        
        if not result_file.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")
        
        with open(result_file, 'rb') as f:
            return pickle.load(f)
        
    def available_results(self):
        """
        Check which results are available.
        
        Returns:
            dict: Availability of each result type
        """
        
        print(f"\nAvailable Results:")
        for result_type in ['Ws_raw', 'Ws_filtered', 'auc_raw', 'auc_filtered']:
            try:
                self.load_results(result_type)
                print(f"✓ {result_type}")
            except FileNotFoundError:
                print(f"✗ {result_type}")
        print("\n" + "="*70)