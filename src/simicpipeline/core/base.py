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
  
    def print_project_info(self, max_depth:int = 3) -> None:
        """
        Get information about the project structure.
        
        Returns:
            dict: Dictionary with project paths and status
        """
        print_tree(directory = self.project_dir, prefix="", max_depth=max_depth)
    
    def assign_path(self, attr_name, path_value, force_flag, must_exist=True, create_dir=False):
        """
        Assigns self.attr_name only if path_value is not None.
        Checks existence unless force=True.
        Optionally creates directory.
        """
        if path_value is None:
            return  # skip entirely

        path_obj = Path(path_value)

        if not force_flag and must_exist and not path_obj.exists():
            raise FileNotFoundError(
                f"Invalid path for {attr_name}: {path_obj}"
            )
        # Directory creation if needed
        if create_dir:
            path_obj.mkdir(parents=True, exist_ok=True)

        # Assign to self
        setattr(self, attr_name, path_obj)

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
        self.assign_path("p2df", p2df, force)
        self.assign_path("p2tf", p2tf, force)
        self.assign_path("p2assignment", p2assignment, force)

        self.assign_path("p2simic_matrices", p2simic_matrices, force)
        self.assign_path("p2filtered_matrices", p2filtered_matrices, force)

        self.assign_path("p2auc_raw", p2auc_raw, force)
        self.assign_path("p2auc_filtered", p2auc_filtered, force)

        # Figures directory: create if missing
        self.assign_path("p2figures", p2figures, force, must_exist=False, create_dir=True)

        print("✓ Custom paths successfully set.")
        print("\n" + "=" * 70 + "\n")
        
    
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