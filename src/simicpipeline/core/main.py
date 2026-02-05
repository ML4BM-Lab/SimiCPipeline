#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Pipeline Class
Orchestrates the complete SimiC workflow including regression and AUC calculation.
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from importlib.resources import files

from simicpipeline.core import SimiCBase

class SimiCPipeline(SimiCBase):
    """
    Main pipeline class for running the complete SimiC analysis workflow.
    Handles data loading, regression, filtering, and AUC calculation.
    """

    def __init__(self, 
                 project_dir: str,
                 run_name: Optional[str] = None) -> None:
        """
        Initialize the SimiC pipeline.

        Args:
            project_dir (str): Working directory for input/output
            run_name (str): Optional name for this run (used in output filenames)
        """
        # Initialize base pipeline
        super().__init__(project_dir=project_dir)
        
        # Initialize directory paths
        self.input_path = self.project_dir / "inputFiles"
        self.output_path = self.project_dir / "outputSimic"
        self.matrices_path = self.output_path / "matrices"
        
        # Create output directories if they don't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.matrices_path.mkdir(parents=True, exist_ok=True)
        
        # Set run name
        self.run_name = run_name if run_name else "simic_run"
        
        # Default parameters
        self.n_tfs = ""
        self.n_targets = ""
        self.lambda1 = 1e-2
        self.lambda2 = 1e-5
        self.num_rep = 1
        self.similarity = True
        self.max_rcd_iter = 500000
        self.df_with_label = False
        self.cross_val = False
        self.k_cross_val = 5
        self.max_rcd_iter_cv = 10000
        self.list_of_l1 = [1e-1, 1e-2, 1e-3, 10]
        self.list_of_l2 = [1e-1, 1e-2, 1e-3, 10]
        
        # Timing
        self.timing = {}
        
        # Input file paths (to be set by set_input_paths)
        self.p2df = None
        self.p2assignment = None
        self.p2tf = None
        
        # Output file paths (to be set by set_output_paths)
        self.p2simic_matrices = None
        self.p2filtered_matrices = None
        self.p2auc_raw = None
        self.p2auc_filtered = None
        
# Path management functions       
    def set_input_paths(self, 
                  p2df: Union[str, Path],
                  p2tf: Union[str, Path],
                  p2assignment: Optional[Union[str, Path]] = None) -> None:
        """
        Set up all file paths for input and output.
        
        Args:
            p2df (str): Custom path to expression matrix file. Must be a pickle file with a dataframe.
                        dataframe should be like (with optional label column):
                                    gene1, gene2, ..., genek, label
                            cell1:  x,     x,   , ..., x,   , type1
                            cell2:  x,     x,   , ..., x,   , type2
            p2assignment (str): Custom path to phenotype assignment file. Must be a plain text file 1 col with numbers (no header). 0 for baseline/control phenotype group. Should match cell order as in expression matrix.
            p2tf (str): Custom path to TF list file. Must be a pickle file with a list of TF names or a pdDataframe with 1 col with gene symbol names (matching expression matrix colnames) and no header.
        """
        # Input paths - use custom paths if provided, otherwise raise error
        if not Path(p2df).exists():
            raise FileNotFoundError("Invalid path to expression matrix dataframe (p2df).")
        self.p2df = Path(p2df)

        if p2assignment:
            self.p2assignment = Path(p2assignment)
        else:
            print("WARNING: Path to phenotype assignment file (p2assignment) not provided. df_with_label should be set to True and p2df should contain a 'label' column.")
            
        if not Path(p2tf).exists():
            print("WARNING: Path to TF list file (p2tf) not found. Using default mouse TF list.")
            default_TFmouse = files("simicpipeline.data").joinpath("Mus_musculus_TF.txt")
            mouse_TF_df = pd.read_csv(default_TFmouse, sep='\t')
            mouse_TF = mouse_TF_df['Symbol'].to_list()
            self.input_path.mkdir(parents=True, exist_ok=True)
            with open(self.input_path / "TF_list.pickle", 'wb') as f:
                pickle.dump(mouse_TF, f)
            self.p2tf = self.input_path / "TF_list.pickle"
        else: 
            self.p2tf = Path(p2tf)
        # Set output paths based on run name and default parameters
        self.set_output_paths()
    
    def set_output_paths(self) -> None:
        # Output paths
        self.run_path = self.matrices_path / self.run_name
        self.run_path.mkdir(parents=True, exist_ok=True)
        base_name = f"{self.run_name}_L1_{self.lambda1}_L2_{self.lambda2}"
        self.p2simic_matrices = self.run_path / f"{base_name}_simic_matrices.pickle"
        self.p2filtered_matrices = self.run_path / f"{base_name}_simic_matrices_filtered_BIC.pickle"
        self.p2auc_raw = self.run_path / f"{base_name}_wAUC_matrices.pickle"
        self.p2auc_filtered = self.run_path / f"{base_name}_wAUC_matrices_filtered_BIC.pickle"

    def validate_inputs(self):
        """Validate that all required input files exist."""
        if self.p2assignment is None and self.df_with_label is False:
            raise ValueError("Either a phenotype assignment file (p2assignment) must be provided or the expression dataframe must contain labels in the last column (set df_with_label=True).")
        required_files = [self.p2df, self.p2tf]
        if self.p2assignment is not None:
            required_files.append(self.p2assignment)
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required input files: {missing_files}")
        
        print("✓ All required input files found")
# Parameter management functions 
    def set_parameters(self,
                       lambda1: Optional[float] = None,
                       lambda2: Optional[float] = None,
                       similarity: Optional[bool] = None,
                       max_rcd_iter: Optional[int] = None,
                       cross_val: Optional[bool] = None,
                       k_cross_val: Optional[int] = None,
                       max_rcd_iter_cv: Optional[int] = None,
                       num_rep: Optional[int] = None,
                       list_of_l1: Optional[list] = None,
                       list_of_l2: Optional[list] = None,
                       _NF: Optional[float] = None,
                       run_name: str = None):
        """
        Set pipeline parameters.

        Args:
            lambda1 (float): L1 regularization parameter (sparsity)
            lambda2 (float): L2 regularization parameter (network similarity)
            similarity (bool): Enables similarity constraint for Lipswitz constant (RCD process)
            max_rcd_iter (int): Maximum RCD iterations
            cross_val (bool): Whether to perform cross-validation to select optimal lambdas
            k_cross_val (int): Number of folds for cross-validation
            max_rcd_iter_cv (int): Maximum number of RCD iterations in the cross-validation step
            num_rep (int): Number of repetitions for test evaluation
            list_of_l1 (list): List of L1 values for cross-validation
            list_of_l2 (list): List of L2 values for cross-validation
            _NF (float): # Normalization factor for expression data (default 1.0, we expect data to be preprocessed).
            run_name (str): Name for this run
        """
        # Update parameters if provided
        if lambda1 is not None:
            self.lambda1 = lambda1
        if lambda2 is not None:
            self.lambda2 = lambda2
        if similarity is not None:
            self.similarity = similarity
        if max_rcd_iter is not None:
            self.max_rcd_iter = max_rcd_iter
        if cross_val is not None:
            self.cross_val = cross_val
        if list_of_l1 is not None:
            self.list_of_l1 = list_of_l1
        if list_of_l2 is not None:
            self.list_of_l2 = list_of_l2
        if k_cross_val is not None:
            self.k_cross_val = k_cross_val
        if max_rcd_iter_cv is not None:
            self.max_rcd_iter_cv = max_rcd_iter_cv
        if num_rep is not None:
            self.num_rep = num_rep
        if _NF is not None:
            self._NF = _NF
        if run_name is not None:
            self.run_name = run_name
        
        # Update paths based on new parameters
        self.set_output_paths()

# Core pipeline functions
    def run_simic_regression(self):
        """
        Run the SimiC LASSO regression. 
        This function uses the parameters and paths set in the pipeline instance and
        the simicLASSO_op function from the clus_regression_fixed module 
        (almost identical to what is found in Jianhao's github: SimiC/code/simiclasso/clus_regression.py ).
        """
        from simicpipeline.core.clus_regression_fixed import simicLASSO_op
        import numpy as np
        print("\n" + "="*50)
        print(f"Running SimiC Regression")
        print(f"Run name: {self.run_name}")
        self.validate_inputs()

        if self.cross_val:
            print(f"Running cross-validation with following lambdas: {self.list_of_l1} (L1), {self.list_of_l2} (L2)")
        else:
            print(f"Lambda1: {self.lambda1}, Lambda2: {self.lambda2}")
        print("="*50 + "\n")
        
        ts = time.time()
        # To ensure reproducibility
        np.random.seed(123)
        simicLASSO_op(
            p2df=str(self.p2df),
            p2tf=str(self.p2tf),
            p2assignment=str(self.p2assignment) if self.p2assignment is not None else None,
            p2saved_file=str(self.p2simic_matrices),
            df_with_label=self.df_with_label, 
            similarity=self.similarity,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            max_rcd_iter=self.max_rcd_iter,
            cross_val=self.cross_val,
            k_cv=self.k_cross_val,
            max_rcd_iter_cv=self.max_rcd_iter_cv,
            num_rep=self.num_rep,
            list_of_l1=self.list_of_l1,
            list_of_l2=self.list_of_l2
        )
        
        te = time.time()
        self.timing['simic_regression'] = te - ts
        print(f"\n✓ SimiC regression completed in {self.format_time(te - ts)}")
        simic_results = self.load_results('Ws_raw')
        self.n_tfs = len(simic_results['TF_ids'])
        self.n_targets = len(simic_results['query_targets'])

    def filter_weights(self, variance_threshold: float = 0.9):
        """
        Filter weights using BIC criterion.
        
        This method filters the weight matrix by keeping only the top TFs that explain
        at least `variance_threshold` of the variance for each target gene.
        
        Args:
            variance_threshold (float): Threshold for cumulative explained variance (default: 0.9)
        
        Returns:
            bool: True if filtering succeeded, False otherwise
        """
        print("\n" + "="*50)
        print("Filtering weights using BIC criterion")
        print(f"Variance threshold: {variance_threshold}")
        print("="*50 + "\n")
        
        ts = time.time()
        
        # Load the weights file
        if not self.p2simic_matrices.exists():
            print(f"Error: Weight file not found: {self.p2simic_matrices}")
            return False
        
        try:
            with open(self.p2simic_matrices, 'rb') as f:
                weights_dict = pickle.load(f)
            
            weight_dic = weights_dict['weight_dic']
            TF_ids = weights_dict['TF_ids']
            target_ids = weights_dict['query_targets']
            
            print(f"Loaded weights for {len(weight_dic)} phenotype labels")
            print(f"Number of TFs: {len(TF_ids)}")
            print(f"Number of targets: {len(target_ids)}")
            
            # Process each phenotype label
            for label in weight_dic.keys():
                print(f"\nProcessing label {label}...")
                
                # Get weight matrix for this label: (n_tfs + 1) x n_targets
                # Last row is the bias term
                weight_matrix = weight_dic[label]
                
                # Convert to DataFrame
                all_data = pd.DataFrame(
                    weight_matrix,
                    columns=target_ids
                )
                
                # Remove columns (targets) with all zero weights
                all_data = all_data.loc[:, (all_data.abs().sum(axis=0) > 0)]
                remaining_targets = all_data.columns.tolist()
                print(f"  Targets with non-zero weights: {len(remaining_targets)}/{len(target_ids)}")
                
                # Separate bias term (last row)
                bias_row = all_data.iloc[-1, :].copy()
                all_data = all_data.iloc[:-1, :]
                all_data.index = TF_ids
                
                # Scale the data by column (normalize each target by its RMS)
                # This is equivalent to R's scale(all_data, center=FALSE)
                n, p = all_data.shape 
                root_mean_sq = np.sqrt((all_data ** 2).sum(axis=0) / (n - 1))
                scaled_data = all_data / root_mean_sq
                
                max_l = []
                filtered_data = scaled_data.copy()
                
                # For each target gene
                for target in scaled_data.columns:
                    # Get absolute weights for all TFs for this target
                    target_weights = scaled_data[target].abs()
                    
                    # Sort TFs by absolute weight (descending)
                    sorted_tfs = target_weights.sort_values(ascending=False)
                    
                    # Calculate total variance for this target
                    total_variance = (scaled_data[target] ** 2).sum()
                    
                    # Find minimum number of TFs to explain variance_threshold of variance
                    cumulative_variance = 0
                    l = 0
                    
                    for tf_idx, tf in enumerate(sorted_tfs.index, 1):
                        l = tf_idx
                        # Calculate cumulative variance explained by top l TFs
                        top_l_tfs = sorted_tfs.index[:l]
                        cumulative_variance = (scaled_data.loc[top_l_tfs, target] ** 2).sum()
                        
                        # Check if we've explained enough variance
                        if total_variance > 0 and (cumulative_variance / total_variance) >= variance_threshold:
                            break
                    
                    max_l.append(l)
                    
                    # Keep only top l TFs, set others to zero
                    tfs_to_keep = sorted_tfs.index[:l]
                    tfs_to_zero = [tf for tf in scaled_data.index if tf not in tfs_to_keep]
                    filtered_data.loc[tfs_to_zero, target] = 0
                
                # Add bias term back as last row
                filtered_data = pd.concat([filtered_data, bias_row.to_frame().T])
                
                # Restore original column order (add back zero-weight targets)
                for target in target_ids:
                    if target not in filtered_data.columns:
                        filtered_data[target] = 0
                filtered_data = filtered_data[target_ids]
                
                # Convert back to numpy array
                weight_dic[label] = filtered_data.values
                
                print(f"  TFs kept per target: Mean={np.mean(max_l):.2f}, "
                      f"Median={np.median(max_l):.0f}, "
                      f"Max={np.max(max_l)}, Min={np.min(max_l)}")
            
            # Update weights_dict with filtered weights
            weights_dict['weight_dic'] = weight_dic
            
            # Save filtered weights
            with open(self.p2filtered_matrices, 'wb') as f:
                pickle.dump(weights_dict, f)
            
            te = time.time()
            self.timing['filtering'] = te - ts
            print(f"\n✓ Weight filtering completed in {self.format_time(te - ts)}")
            print(f"Filtered weights saved to: {self.p2filtered_matrices}")
            
            return True
            
        except Exception as e:
            print(f"Error during weight filtering: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_auc(self,
                      use_filtered=False, 
                      adj_r2_threshold: float = 0.7, 
                      select_top_k_targets: int = None,
                      percent_of_target: float = 1, 
                      sort_by:str ='expression', 
                      num_cores: int = 0):
        """
        Calculate AUC matrices.

        Args:
            use_filtered (bool): Whether to use filtered weights
            adj_r2_threshold (float): R-squared threshold for filtering
            select_top_k_targets (int): Number of top targets to select
            percent_of_target (float): Percentage of targets to consider
            sort_by (str): Sorting criterion ('expression', 'weight', 'adj_r2')
            num_cores (int): Number of cores for parallel processing
        """
        from simicpipeline.core.aucprocessor import AUCProcessor
        if sort_by not in ['expression', 'weight', 'adj_r2']:
            raise ValueError("sort_by must be one of 'expression', 'weight', or 'adj_r2'")
        
        weight_file = self.p2filtered_matrices if use_filtered else self.p2simic_matrices
        p2auc_file = self.p2auc_filtered if use_filtered else self.p2auc_raw
        
        file_type = "filtered" if use_filtered else "raw"
        sys.stdout.flush()
        print("\n" + "="*50)
        print(f"Calculating AUC matrices ({file_type} weights)")
        print("="*50 + "\n")
        
        ts = time.time()
        
        processor = AUCProcessor(self.project_dir, self.p2df, str(weight_file))
        processor.normalized_by_target_norm()
        processor.save_AUC_dict(
            p2auc_file=str(p2auc_file),
            adj_r2_threshold=adj_r2_threshold,
            select_top_k_targets=select_top_k_targets,
            percent_of_target=percent_of_target,
            sort_by=sort_by,
            num_cores=num_cores
        )
        te = time.time()
        self.timing[f'auc_{file_type}'] = te - ts
        
        print(f"\n✓ AUC calculation completed in {self.format_time(te - ts)}")

        # Automatically save concatenated AUC for all labels
        print("\nCollecting AUC for all labels...")
        res = self.load_results('Ws_filtered')
        labels=list(res['weight_dic'].keys())
        auc_subset_list = []
        for label in labels:
            auc_subset = self.subset_label_specific_auc(f'auc_{file_type}',label=label)
            auc_subset_list.append(auc_subset)
        auc_subset_all = pd.concat(auc_subset_list, axis=0)
        
        out_file= p2auc_file.with_name(p2auc_file.stem + "_collected.csv")
        auc_subset_all.to_csv(out_file)
        print(f"✓ Collected AUC for all labels saved to: {out_file.with_name(out_file.stem + '_collected.csv')}")
# Wrapper function to run the complete pipeline
    def run_pipeline(self,
                     skip_filtering=False,
                     calculate_raw_auc=False,
                     calculate_filtered_auc=True,
                     variance_threshold=0.9,
                     auc_params=None):
        """
        Run the complete SimiC pipeline.

        Args:
            skip_filtering (bool): Skip weight filtering step. If True, it will filter out (set to 0) weights for TFs not meeting the variance threshold for that target gene.
            calculate_raw_auc (bool): Calculate AUC for raw weights
            calculate_filtered_auc (bool): Calculate AUC for filtered weights
            variance_threshold (float): Threshold for filtering (default: 0.9)
            auc_params (dict): Parameters for AUC calculation
        """
        if auc_params is None:
            auc_params = {}
        
        total_start = time.time()
        
        print("\n" + "="*70)
        print("STARTING SIMIC PIPELINE")
        print("="*70)
        
        # Run regression
        self.run_simic_regression()
        
        if skip_filtering and not calculate_raw_auc:
            print("\nOnly calculating simic weights! \n ✗ No filtering applied. \n ✗ No AUC calculated")
        
        # Calculate raw AUC if requested
        if calculate_raw_auc:
            self.calculate_auc(use_filtered = False, **auc_params)
        
        # Filter weights and calculate filtered AUC
        if not skip_filtering:
            filtered_success = self.filter_weights(variance_threshold=variance_threshold)
            if calculate_filtered_auc:
                if filtered_success:
                    self.calculate_auc(use_filtered=True, **auc_params)
                else:
                    print("\n✗ Skipping filtered AUC calculation due to filtering error!!")
            else:
                if filtered_success:
                    print("\n✗ Skipping filtered AUC calculation.")

        total_end = time.time()
        self.timing['total'] = total_end - total_start
        
        # Print summary
        sys.stdout.flush()
        self._print_summary()

    def _print_summary(self):
        try:
            res = self.load_results("Ws_raw")  # Load results to get number of TFs and targets for summary
            self.n_tfs = len(res['TF_ids'])
            self.n_targets = len(res['query_targets'])
        except:
            self.n_tfs = "Unknown (results not found)"
            self.n_targets = "Unknown (results not found)"
        """Print pipeline execution summary."""
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"\nRun name: {self.run_name}")
        print(f"Project directory: {self.project_dir}")
        print(f"\nParameters:")
        print(f"  - Lambda1: {self.lambda1}")
        print(f"  - Lambda2: {self.lambda2}")
        print(f"  - Number of TFs: {self.n_tfs}")
        print(f"  - Number of targets: {self.n_targets}")
        print(f"\nTiming:")
        for step, duration in self.timing.items():
            print(f"  - {step}: {self.format_time(duration)}")
        self.available_results()
        print("\n" + "="*70)


# Functions for analyzing results
    def get_TF_network(self, TF_name: str, stacked: bool = False):
        """
        Retrieve the TF-target weight matrix.
        Args:
            TF_name (str): Name of the transcription factor.
            stacked (bool): If True, returns a DataFrame with targets as rows and labels as columns.
        Output:
            pd.DataFrame or dict: If stacked is True, returns a DataFrame with targets as rows and labels as columns.
                                  If stacked is False, returns a dictionary with labels as keys and Series of target weights as values.
        
        """
        simic_results = self.load_results('Ws_filtered')
        weight_dic = simic_results['weight_dic']
        TF_ids = simic_results['TF_ids']
        target_ids = simic_results['query_targets']
        if TF_name not in TF_ids:
            raise ValueError(f"TF '{TF_name}' not found in TF list.")
        print(f"Retrieving network for TF: {TF_name}")
        TF_index = TF_ids.index(TF_name)
        if stacked:
            tf_weights = [weight_dic[label][TF_index, :] for label in weight_dic.keys()]
            ws_df = pd.DataFrame(np.column_stack(tf_weights), index=target_ids, columns=weight_dic.keys())
            ws_df = ws_df.loc[(ws_df!=0).any(axis=1)]
            return ws_df

        tf_weights_filtered = {}
        for label in weight_dic.keys():
            weights = weight_dic[label]
            tf_weights = weights[TF_index, :]  # Get weights for the specified TF
            print(f"\nLabel {label}:")
            target_tupl = [(tf_weights[i], target_ids[i]) for i in range(len(target_ids)) if tf_weights[i] != 0]
            target_ws, target_names = zip(*target_tupl)
            target_ws_df = pd.Series(data=list(target_ws), index=list(target_names))
            tf_weights_filtered[label] = target_ws_df
        return tf_weights_filtered
    
    def get_TF_auc(self, TF_name: str, stacked: bool = False):
        """
        Retrieve the TF-target AUC matrix.
        Args:
            TF_name (str, list): List with name(s) of the transcription factor(s).
            stacked (bool): If True, returns a DataFrame with targets as rows and labels as columns.
        Output:
            pd.DataFrame or dict: If stacked is True, returns a DataFrame with targets as rows and labels as columns.
                                  If stacked is False, returns a dictionary with labels as keys and pd.DataFrame of target AUCs as values.
                                  """
        auc_results = self.load_results('auc_filtered')
        if isinstance(TF_name, str):
            TF_name = [TF_name]
        missing = [tf for tf in TF_name if tf not in auc_results[0].columns]
        if missing:
            raise ValueError(f"TF(s) not found in AUC results: {missing}")
        print(f"Retrieving AUC for TF: {TF_name}")
        # Analyze each label
        tf_aucs_filtered = {}
        for label in auc_results.keys():
            # Subset AUC dataframe to only cells in this label
            auc_subset = self.subset_label_specific_auc('auc_filtered',label)
            tf_aucs_filtered[label] = auc_subset.loc[:, TF_name]            
        if stacked:
            # concatenate all labels into a single DataFrame of one column and add label column
            aucs_df = pd.DataFrame()
            for label in tf_aucs_filtered.keys():
                label_df = pd.DataFrame(tf_aucs_filtered[label])
                label_df['label'] = label
                aucs_df = pd.concat([aucs_df, label_df], axis=0)
            return aucs_df
        return tf_aucs_filtered


    def analyze_weights(self):
        """
        Run weight analysis.
        Ouptut: 
            Prints Ws matrices sparsity before and after filtering.
        """
        print("\n" + "="*70)
        print("ANALYZING WEIGHT MATRICES")
        print("="*70 + "\n")
        
        # Load raw and filtered weights
        try:
            simic_results = self.load_results('Ws_raw')
            filtered_results = self.load_results('Ws_filtered')
        except FileNotFoundError as e:
            print(f"Error loading results: {e}")
            print(f"Need raw and filtered SimiC weight results to analyze weights.")
            self.available_results()
            return
        
        weight_dic_raw = simic_results['weight_dic']
        weight_dic_filtered = filtered_results['weight_dic']
        
        # Compare sparsity before and after filtering
        for label in weight_dic_raw.keys():
            weights_raw = weight_dic_raw[label]
            weights_filtered = weight_dic_filtered[label]
            
            # Remove bias term (last row) for analysis
            weights_raw_no_bias = weights_raw[:-1, :]
            weights_filtered_no_bias = weights_filtered[:-1, :] 
            # Calculate sparsity (percentage of zero weights)
            sparsity_raw = (weights_raw_no_bias == 0).sum() / weights_raw_no_bias.size * 100
            sparsity_filtered = (weights_filtered_no_bias == 0).sum() / weights_filtered_no_bias.size * 100
            
            # Calculate number of non-zero weights per target
            nonzero_per_target_raw = (weights_raw_no_bias != 0).sum(axis=0)
            nonzero_per_target_filtered = (weights_filtered_no_bias != 0).sum(axis=0)
            
            print(f"Label {label}:")
            print(f"  Raw weights:")
            print(f"    - Sparsity: {sparsity_raw:.2f}%")
            print(f"    - Avg non-zero TFs per target: {nonzero_per_target_raw.mean():.2f}")
            print(f"  Filtered weights:")
            print(f"    - Sparsity: {sparsity_filtered:.2f}%")
            print(f"    - Avg non-zero TFs per target: {nonzero_per_target_filtered.mean():.2f}")
            print(f"  Reduction: {((sparsity_filtered - sparsity_raw) / (100 - sparsity_raw) * 100):.2f}% more sparse")
            print()

    def subset_label_specific_auc(self, result_type: str, label: int):
        """
        Extract from auc results the label specific AUC dataframe.
        Args:
            result_type (str): Type of AUC results to load ('auc_raw' or 'auc_filtered')
            label (str or int): Phenotype label to subset
        Output:
            pd.DataFrame: AUC dataframe for the specified label
        """    
        auc_dic = self.load_results(result_type)
        
        auc_df = auc_dic[label]
        # Load cell assignments to get which cells belong to which label
        assignment_df = pd.read_csv(self.p2assignment, sep='\t', header=None, names=['label'])
        # Get cells that belong to this label
        cells_idx_in_label = assignment_df[assignment_df['label'] == int(label)].index.to_list()
        # Subset AUC dataframe to only cells in this label
        auc_subset = auc_df.iloc[cells_idx_in_label,]
        return auc_subset

    def analyze_auc_scores(self):
        """
        Run AUC score analysis.
        Ouptut: 
            Prints basic statistics and top TFs by average activity scores.
        """
        print("\n" + "="*70)
        print("ANALYZING AUC SCORES")
        print("="*70 + "\n")
        
        # Load AUC results
        try:
            auc_filtered = self.load_results('auc_filtered')
        except FileNotFoundError as e:
            print(f"Error loading results: {e}")
            print(f"Need auc_filtered results to analyze AUC scores.")
            self.available_results()
            return
        
        # Analyze each label
        for label in auc_filtered.keys():
            # Subset AUC dataframe to only cells in this label
            auc_subset = self.subset_label_specific_auc('auc_filtered',label)
            
            print(f"Label {label}:")
            print(f"  Shape: {auc_subset.shape} (cells x TFs)")
            print(f"  AUC score statistics:")
            print(f"    - Mean: {np.nanmean(auc_subset.values):.4f}")
            print(f"    - Median: {np.nanmedian(auc_subset.values):.4f}")
            print(f"    - Std: {np.nanstd(auc_subset.values):.4f}")
            print(f"    - Min: {np.nanmin(auc_subset.values):.4f}")
            print(f"    - Max: {np.nanmax(auc_subset.values):.4f}")
            
            # Find top 5 TFs with highest average AUC
            mean_auc_per_tf = auc_subset.mean(axis=0)
            top_tfs = mean_auc_per_tf.nlargest(5)
            print(f"  Top 5 TFs by average AUC:")
            for tf, score in top_tfs.items():
                print(f"    - {tf}: {score:.4f}")
            print()

    def calculate_dissimilarity(self, select_labels=None, verbose=True):
        """
        Compare AUC scores between different labels calculating dissimilarity score (0 = similar distributions, higher = more dissimilarity)
        Args:
            select_labels (list): List of labels to compare. If None, compare all available labels
        
        Output: 
            pd.DataFrame with TFs and their dissimilarity scores (sorted in descending order).
        """
        if verbose:
            print("\n" + "="*70)
            print("CALCULATING DISSIMILARITY SCORES ACROSS LABELS")
            print("="*70 + "\n")
        try:
            auc_filtered = self.load_results('auc_filtered')
        except FileNotFoundError as e:
            print(f"Error loading results: {e}")
            print(f"Need auc_filtered results to compare labels.")
            self.available_results()
            return
        
        if select_labels:
            print(f"Comparing labels {select_labels}.")
            labels = select_labels
        else:
            labels = list(auc_filtered.keys())
        if len(labels) < 2:
            print("Only one label, cannot compare!")
            return
        
        auc_dic = {}
        for label in labels:
            auc_dic[label] = self.subset_label_specific_auc('auc_filtered', label)
        if verbose:
            print(f"\nCalculating dissimilarity scores...")
        n_breaks = 100
        MinMax_val = []
        tf_names = auc_dic[labels[0]].columns.tolist()
        
        for tf in tf_names:
            # Get AUC values for this TF from both labels
            Wauc_dist = {}
            for label in labels:
                Wauc_dist[label] = np.histogram(auc_dic[label][tf].dropna(), bins=np.linspace(0, 1, n_breaks + 1), density=True)[0]
            # Create matrix of distributions
            # Extract the .values from each DataFrame (2D numpy arrays)
            arrays = [df for df in Wauc_dist.values()]
            mat = np.vstack(arrays)
            # Remove columns with all NaN
            mat = mat[:, ~np.isnan(mat).all(axis=0)]
            
            if mat.shape[1] > 0:
                # Calculate minmax difference
                minmax_diff = np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0)
                variant = np.sum(np.abs(minmax_diff)) / n_breaks
                
                # Normalize by number of non-zero rows
                non_zero_rows = np.sum(np.sum(mat, axis=1) != 0)
                if non_zero_rows > 0:
                    variant = variant / non_zero_rows
            else:
                variant = 0.0
            
            MinMax_val.append(variant)
        
        # Create DataFrame with dissimilarity scores
        MinMax_df = pd.DataFrame({
            'TF': tf_names,
            'MinMax_score': MinMax_val
        }).set_index('TF')
        
        # Sort by dissimilarity score
        MinMax_df_sorted = MinMax_df.sort_values('MinMax_score', ascending=False)
        if verbose:
            print(f"\nTop 10 TFs by MinMax dissimilarity score:")
            for tf, row in MinMax_df_sorted.head(10).iterrows():
                print(f"  {tf}: {row['MinMax_score']:.4f}")
        
        return MinMax_df_sorted
