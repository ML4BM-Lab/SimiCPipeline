#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Preprocessing Pipeline
This script handles data loading, MAGIC imputation, gene selection, and file preparation for SimiCPipeline run.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
import pickle

try:
    import anndata as ad
except ImportError:
    raise ImportError("Anndata is required. Please install.")

from simicpipeline.core import SimiCBase


class MagicPipeline(SimiCBase):
    """
    MAGIC imputation pipeline.  
    Handles loading raw expression data, filtering, normalization, and MAGIC imputation.  
    """
            
    def __init__(self, 
                 input_data: ad.AnnData,
                 project_dir: Union[str, Path],
                 magic_output_file: str = 'magic_data_allcells_sqrt.pickle',
                 filtered: bool = False):
        """
        Initialize MAGIC pipeline. Takes AnnData and sparse matrix if needed. Creates output directory.

        Args:
            project_dir:   Directory for output files
            magic_output_file:  Filename for MAGIC-imputed matrix (saved in project_dir)
            filtered: Flag to indicate if the data is filtered
        """
        super().__init__(project_dir)
        # Initialize data containers
        if not hasattr(input_data.raw, 'X'):
            print("Warning: No raw attribute found in AnnData object")

        self.adata = input_data
            
        # Create MAGIC output directory
        self.magic_output_dir = self.project_dir / 'magic_output'
        self.magic_output_dir.mkdir(exist_ok=True)
        
        # Set output file path
        self.magic_output_file = self.magic_output_dir / magic_output_file
        
        self.magic_adata = None
        
        # Track pipeline state
        self._filtered = filtered
        self._imputed = False

    def __repr__(self) -> str:
        """
        String representation showing pipeline state and data dimensions.
        
        Returns:
            Formatted string with pipeline information
        """
        lines = ["MagicPipeline("]
        
        # Data status
        if self.adata is None:
            lines.append("  data = None,")
        else:
            lines.append(f"  data = AnnData object with (n_obs × n_vars) = {self.adata.shape[0]} × {self.adata.shape[1]},")
        
        # Filtered status
        lines.append(f"  filtered = {self._filtered},")
        
        # Imputed status
        lines.append(f"  imputed = {self._imputed},")
        
        # MAGIC data dimensions if available
        if self.magic_adata is not None:
            lines.append(f"  magic_data = AnnData object with n_obs × n_vars = {self.magic_adata.shape[0]} × {self.magic_adata.shape[1]},")
        else:
            lines.append("  magic_data = None,")
        
        # Output directory
        lines.append(f"  project_dir = '{self.project_dir}'")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def filter_cells_and_genes(self,
                               min_cells_per_gene: int = 10,
                               min_umis_per_cell: int = 500) -> 'MagicPipeline':
        """
        Filters raw input data based on desired thresholds.
        Genes are filtered by number of cells expressing them,
        Cells are filtered by total UMI counts per cell after gene filtering.

        Args:
            min_cells_per_gene: Minimum number of cells that must express a gene (gene filtering)
            min_umis_per_cell: Minimum total UMIs per cell (cell filtering)
            
        Returns:
            Self for method chaining
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first using load_from_matrix_market() or load_from_anndata().")
        
        if not hasattr(self.adata.raw, 'X'):
            raise ValueError("No raw attribute found in AnnData object. Please ensure raw counts are stored in adata.raw.X before filtering. You can manually set it using: adata.raw = adata.copy()")
        
        if self._filtered:
            print("Warning: Data has already been filtered. Filtering again!")
        
        print("\nFiltering cells and genes...")
        print(f"Before filtering: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        
        # Get Raw counts (cells x genes)
        X = self.adata.raw.X.copy()

        # 1. Filter genes - keep genes expressed in more than min_cells_per_gene cells
        bool_mat = X > 0
        cells_per_gene = np.array(bool_mat.sum(axis=0)).reshape(-1)
        keep_genes = cells_per_gene > min_cells_per_gene
        if sum(keep_genes) == 0:
            raise ValueError("All genes filtered out!  Consider lowering min_cells_per_gene.")
        
        assert keep_genes.shape[0] == X.shape[1], "Gene mask shape mismatch"
        pct_g_kept = sum(keep_genes) / len(keep_genes) * 100
        print(f"Keeping {sum(keep_genes)}/{len(keep_genes)} genes ({pct_g_kept:.2f}%)")
        
        X = X[:, keep_genes]
        
        # 2. Filter cells - keep cells with more than min_umis_per_cell UMIs
        umis_per_cell = np.array(X.sum(axis=1)).reshape(-1)
        keep_cells = umis_per_cell > min_umis_per_cell
        if sum(keep_cells) == 0:
            raise ValueError("All cells filtered out! Consider lowering min_umis_per_cell.")
        
        assert keep_cells.shape[0] == X.shape[0], "Cell mask shape mismatch"
        pct_c_kept = sum(keep_cells) / len(keep_cells) * 100
        print(f"Keeping {sum(keep_cells)}/{len(keep_cells)} cells ({pct_c_kept:.2f}%)")
        
        if sum(keep_cells) == len(keep_cells):
            print("All cells pass the filter!")
        X = X[keep_cells,:]
        # Update AnnData object (asumes X and raw.X have same shape and obs/var)
        raw_adata = ad.AnnData(
            X=X,
            obs=self.adata.obs.iloc[keep_cells].copy(),
            var=self.adata.var.iloc[keep_genes].copy()
        )
        self.adata = self.adata[keep_cells, keep_genes].copy()
        self.adata.raw = raw_adata
        print(f"After filtering: {self.adata.raw.shape[0]} cells x {self.adata.raw.shape[1]} genes")
        
        self._filtered = True
        self._imputed = False  # Reset imputation flag since data changed

        return self
    
    def normalize_data(self) -> 'MagicPipeline':
        """
        Normalize data using library size normalization and square root transformation.
        Note: It overides adata.X with normalized data and removes adata.raw slot
        Returns:
            Self for method chaining
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first using load_from_matrix_market() or load_from_anndata().")
        if not hasattr(self.adata.raw, 'X'):
            raise ValueError("No raw attribute found in AnnData object. Please ensure raw counts are stored in adata.raw.X before filtering. You can manually set it using: adata.raw = adata.copy()")
        print("\nNormalizing data...")
        try:
            import scprep
        except ImportError:
            raise ImportError("scprep is required for normalization. Please install or normalize data externally.")        
        
        # Normalize the matrix. Make sure is cells x genes matrix because scprep expects [n_samples, n_features]
        matrix_sparse_norm = scprep.normalize.library_size_normalize(self.adata.raw.X)
        sqrt_mat = np.sqrt(matrix_sparse_norm)
        
        # Update AnnData object (from here on we work with normalized data and not raw counts)
        # Update AnnData object
        self.adata = ad.AnnData(
            X=sqrt_mat,
            obs=self.adata.obs.copy(),
            var=self.adata.var.copy()
        )
        
        print(f"After normalization: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        
        self._imputed = False  # Reset imputation flag since data changed
        return self

    def run_magic(self,
                  save_data: bool = True,
                  n_jobs: Optional[int] = -2,
                   **kwargs) -> 'MagicPipeline':
        """
        Run MAGIC imputation on the data. It uses the normalized data in self.adata.X.

        args:
            save_data: Whether to save the MAGIC-imputed data to file
            n_jobs: Number of jobs for parallel processing. -1 Use all CPUs, -2 use all but one
            **kwargs:  Additional arguments to pass to magic. MAGIC()
                        t: Number of diffusion steps
                        knn: Number of nearest neighbors
                        decay: Decay rate for kernel
            
        Returns:
            Self for method chaining
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first using load_from_matrix_market() or load_from_anndata().")
        
        if not self._filtered:
            print("Warning: Data has not been filtered. Consider running filter_cells_and_genes() first.")
        
        print("\nRunning MAGIC imputation...")
        
        try:
            import magic
        except ImportError:
            raise ImportError("MAGIC is not installed. Please install it with: pip install magic-impute")
        
        magic_op = magic.MAGIC(n_jobs=n_jobs, **kwargs)
        magic_result = magic_op.fit_transform(self.adata.X)
        
        # Create new AnnData with imputed values
        self.magic_adata = ad.AnnData(
            X=magic_result,
            obs=self.adata.obs.copy(),
            var=self.adata.var.copy()
        )
        
        self._imputed = True
        print(f"MAGIC imputation complete:  {self.magic_adata.shape[0]} cells x {self.magic_adata.shape[1]} genes")
        if save_data:
            self.save_magic_data(self.magic_output_file)
        return self
    
    def save_magic_data(self, filepath: Optional[Union[str, Path]] = None) -> 'MagicPipeline':
        """
        Save MAGIC-imputed data to file.

        Args:
            filepath: Optional custom path.  If None, uses self.magic_output_file
            
        Returns:
            Self for method chaining
        """
        if self.magic_adata is None: 
            raise ValueError("No MAGIC-imputed data available. Please run run_magic() first.")
        
        output_file = Path(filepath) if filepath else self.magic_output_file
        # Check extension
        if output_file.suffix not in ['.pickle', '.h5ad']:
            print(f"Unsupported file format: {output_file.suffix}. pickle will be appended to filename.")
            output_file = output_file.with_suffix('.pickle')
        # Check that output_path file exists and print warning
        if output_file.exists():
            print(f"Warning: Output file {output_file} already exists and will be overwritten.")
        
        print(f"\nSaving MAGIC-imputed data to {output_file}")
        
        # Save based on file extension
        if output_file.suffix == '.pickle':
            with open(output_file, 'wb') as f:
                pickle.dump(self.magic_adata, f)
        elif output_file.suffix == '.h5ad':
            self.magic_adata.write_h5ad(output_file)
        else:
            print(f"Unsupported file format: {output_file.suffix}. Will use default name: 'magic_data_allcells_sqrt.pickle'")
            output_file = self.magic_output_dir / 'magic_data_allcells_sqrt.pickle'
            with open(output_file, 'wb') as f:
                pickle.dump(self.magic_adata, f)
        
        print(f"Saved successfully to {output_file}")
    
    def is_filtered(self) -> bool:
        """Check if data has been filtered."""
        return self._filtered
    
    def is_imputed(self) -> bool:
        """Check if MAGIC imputation has been run."""
        return self._imputed


class ExperimentSetup(SimiCBase):
    """
    Minimal experiment setup for SimiC analysis.

    Accepts input expression data (AnnData or pandas DataFrame), converts to NumPy,
    and provides:
      - calculate_mad_genes: compute MAD and select top genes
      - save_experiment_files_csv: save matrix, gene names, and cell names as CSV files
    """

    def __init__(self,
                 input_data: Union[ad.AnnData, pd.DataFrame],
                 tf_path: Union[str, Path],
                 project_dir: Union[str, Path]):
        """
        Initialize ExperimentSetup. Loads TF lists and creates directory structure.

        Args:
            input_data: AnnData (cells x genes) or pandas DataFrame (cells x genes)
            tf_path: Path to transcription factor list file
            project_dir: Directory for output files
        """
        super().__init__(project_dir)

        # Load TF list
        self.tf_list = self._load_tf_list(tf_path)
        
        # Create standard SimiC directory structure (already done in parent __init__)
        self._create_directory_structure()

        # Convert input into NumPy matrix with cell/gene names
        if isinstance(input_data, ad.AnnData):
            X = input_data.X
            # Convert sparse to dense if needed
            if hasattr(X, "toarray"):
                X = X.toarray()
            self.matrix = np.asarray(X)  # cells x genes
            self.cell_names = input_data.obs_names.tolist()
            self.gene_names = input_data.var_names.tolist()
        elif isinstance(input_data, pd.DataFrame):
            # Expect DataFrame as cells x genes
            self.matrix = input_data.values
            self.cell_names = input_data.index.astype(str).tolist()
            self.gene_names = input_data.columns.astype(str).tolist()
        else:
            raise TypeError("input_data must be an AnnData or a pandas DataFrame")

        # Basic validation
        if self.matrix.ndim != 2:
            raise ValueError("Expression matrix must be 2-dimensional (cells x genes)")
        if len(self.cell_names) != self.matrix.shape[0]:
            raise ValueError("Cell names size does not match matrix rows")
        if len(self.gene_names) != self.matrix.shape[1]:
            raise ValueError("Gene names size does not match matrix columns")

    def _load_tf_list(self, tf_path: Union[str, Path]) -> List[str]:
        """Load transcription factor list from file."""
        tf_path = Path(tf_path)
        if not tf_path.exists():
            raise FileNotFoundError(f"TF list file not found: {tf_path}")
        if tf_path.suffix == '.csv':
            tf_df = pd.read_csv(tf_path, header=None, names=["TF"])
        else:
            tf_df = pd.read_csv(tf_path, header=None, names=["TF"], sep='\t')
        return tf_df['TF'].tolist()

    def _create_directory_structure(self) -> None:
        """Create standard SimiC directory structure."""
        # Input files directory
        self.input_files_dir = self.project_dir / 'inputFiles'
        self.input_files_dir.mkdir(parents=True, exist_ok=True)

        # Output SimiC directories
        self.output_simic_dir = self.project_dir / 'outputSimic'
        self.figures_dir = self.output_simic_dir / 'figures'
        self.matrices_dir = self.output_simic_dir / 'matrices'

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.matrices_dir.mkdir(parents=True, exist_ok=True)

    def calculate_mad_genes(self,
                            n_tfs: int,
                            n_targets: int) -> tuple[List[str], List[str]]:
        """
        Calculate Median Absolute Deviation and select top TF and target genes.

        Args:
            n_tfs: Number of top TF genes to select based on MAD
            n_targets: Number of top target genes to select based on MAD

        Returns:
            Tuple of (final_TF_list, final_TARGET_list)
        """
        # Create expression matrix as genes x cells
        expr_matrix = pd.DataFrame(
            self.matrix.T,  # Transpose to genes x cells
            index=self.gene_names,
            columns=self.cell_names
        )
        
        # MAD calculation
        # Calculation of Median Absolute Deviation (MAD)
        mymean = expr_matrix.mean(axis=1).to_numpy()  # Mean expression of each gene (i) across all cells (j)
        dev_mat = expr_matrix.to_numpy() - mymean[:, None]  # Deviation matrix Xij - mean(Xi)
        mymad = np.median(np.abs(dev_mat), axis=1)  # Median of the absolute deviation of each gene (i) across all cells (j)
        MAD = pd.Series(mymad, index=expr_matrix.index)  # Create a pandas series with gene names as index
        
        # Select top MAD TF genes
        TFsmedian = MAD.reindex(set(self.tf_list)).dropna()
        TFsmedian.sort_values(ascending=False, inplace=True)  # Sort the series in descending order
        
        if len(TFsmedian) < n_tfs:
            n_tfs = min(n_tfs, len(TFsmedian))
            if n_tfs == 0:
                raise ValueError("n_tfs must be at least 1.")
            print(f"Only {len(TFsmedian)} TFs found in dataset. Selecting top {n_tfs} TFs based on MAD.")
        
        final_TF_list = TFsmedian.head(n_tfs).index.tolist()
        
        # Select the top MAD target genes
        target_set = set(expr_matrix.index).difference(set(self.tf_list))  # Get target genes
        TARGETs = MAD[list(target_set)]  # No need to reindex as MAD is generated from expr_matrix
        
        print("Removing " + str(sum(TARGETs == 0)) + " targets with MAD = 0")  # remove TARGETs with MAD = 0
        TARGETs = TARGETs[TARGETs > 0]
        TARGETs.sort_values(ascending = False, inplace = True)  # Sort the series in descending order
        
        n_targets = min(n_targets, len(TARGETs))
        print(f"Selecting top {n_targets} targets based on MAD.")
        final_TARGET_list = TARGETs.head(n_targets).index.tolist()  # Top target genes
        
        return final_TF_list, final_TARGET_list

    def save_experiment_files(self,
                              run_data: Union[ad.AnnData, pd.DataFrame],
                              matrix_filename: str = 'expression_matrix.pickle',
                              tf_filename: str = 'TF_list.pickle',
                              annotation: Optional[str] = None,
                              annotation_order: Optional[List[str]] = None) -> None:
        """
        Save matrix (cells x genes) and TF names to pickle files in inputFiles directory.

        Args:
            run_data: AnnData or pandas DataFrame (cells x genes) to save
            matrix_filename: Filename for the expression matrix pickle format (saved in project_dir/inputFiles/)
            tf_filename: Filename for the TF names in pickle format (saved in project_dir/inputFiles/)
            annotation: Optional annotation column name in run_data.obs to save as txt file
            annotation_order: Optional list defining the desired order of annotation categories 
                              (e.g. ['control', 'treated']). The first element maps to 0, second to 1, etc.
                              If None, pd.factorize default order is used.
        """
        # Build paths in inputFiles directory
        matrix_filename = Path(matrix_filename)
        if matrix_filename.suffix not in ['.pickle', '.csv']:
            matrix_filename = matrix_filename.with_suffix('.csv')
            print(f"\nWarning: Matrix filename must have a .pickle or csv suffix. Changing to {matrix_filename}")
        matrix_path = self.input_files_dir / matrix_filename
        if matrix_path.exists():
            print(f"\nWarning: Output file {matrix_path} already exists and will be overwritten.")
        
        tf_filename = Path(tf_filename)
        if tf_filename.suffix not in ['.pickle', '.csv']:
            tf_filename = tf_filename.with_suffix('.csv')
            print(f"\nWarning: TF filename must have a .pickle suffixor csv suffix. Changing to {tf_filename}")
        tf_path = self.input_files_dir / tf_filename
        
        # Generate df from run_data
        if isinstance(run_data, ad.AnnData):
            X = run_data.X
            obs = run_data.obs_names.tolist()
            var = run_data.var_names.tolist()
            # Convert sparse to dense if needed
            if hasattr(X, "toarray"):
                X = X.toarray()
            df = pd.DataFrame(
                X,
                index=obs,
                columns=var
            )
        else:
            df = run_data
        
        # Save matrix in pickle format
        if matrix_path.suffix == '.csv':
            df.to_csv(matrix_path, index=True, header=True)
            print(f"Saved expression matrix to {matrix_path}")
        else:
            with open(matrix_path, 'wb') as f:
                pickle.dump(df, f)
                print(f"Saved expression matrix to {matrix_path}")

        # Save TF list
        # Check TFs found in final dataset
        data_tfs = list(set(self.tf_list).intersection(set(df.columns)))
        if len(data_tfs) == 0:
            raise ValueError("No TFs found in expression matrix from the provided TF list.")
        if tf_path.exists():
            print(f"\nWarning: Output file {tf_path} already exists and will be overwritten.")
        if tf_path.suffix == '.csv':
            pd.DataFrame(data_tfs).to_csv(tf_path, index=False, header=False)
            print(f"Saved {len(data_tfs)} TFs to {tf_path}")
        else: 
            with open(tf_path, 'wb') as f:
                pickle.dump(data_tfs, f)
                print(f"Saved {len(data_tfs)} TFs to {tf_path}")
        
        if annotation and isinstance(run_data, ad.AnnData):
            if annotation in run_data.obs.columns:
                print("\n-------\n")
                print(f"Annotation '{annotation}' found in obs columns!")
                # Check annotation column is numeric
                annot_series = run_data.obs[annotation].copy()
                if not pd.api.types.is_numeric_dtype(annot_series):
                    print("Warning: annotation is not numeric. Will convert from categorical to numeric.")
                    if annotation_order is not None:
                        # Validate that all observed categories are in annotation_order
                        observed_categories = set(annot_series.unique())
                        ordered_set = set(annotation_order)
                        missing_from_order = observed_categories - ordered_set
                        unused_in_order = ordered_set - observed_categories
                        if missing_from_order:
                                raise ValueError(f"The following annotation values are in the data but not in annotation_order: {missing_from_order}.\n"
                                                "If only these annotation_order categories are needed, subset run_data accordingly before running `save_experiment_files`"
                                                )
                        if unused_in_order:
                            raise ValueError(
                                f"The following annotation_order values are not present in data: {unused_in_order}"
                            )
                        # Map categories to integers based on user-defined order
                        category_to_int = {cat: idx for idx, cat in enumerate(annotation_order)}
                        annot_numeric = annot_series.map(category_to_int)
                        print(f"Annotation order applied: {dict(enumerate(annotation_order))}")
                    else:
                        codes, uniques = pd.factorize(annot_series)
                        annot_numeric = pd.Series(codes, index=annot_series.index)
                        category_to_int = {cat: idx for idx, cat in enumerate(uniques)}
                        print(f"Annotation order (auto): {dict(enumerate(uniques))}")
                    
                    # Build a two-column DataFrame with category name and numeric label
                    int_to_category = {v: k for k, v in category_to_int.items()}
                    annot_df = pd.DataFrame({
                        'category': annot_numeric.map(int_to_category),
                        'label': annot_numeric
                    }, index=annot_series.index)
                else:
                    # Already numeric — check for NaNs
                    if annot_series.isna().any():
                        n_na = annot_series.isna().sum()
                        print(f"Warning: {n_na} NaN values found in numeric annotation column '{annotation}'. Dropping NaN rows.")
                        annot_series = annot_series.dropna()
                    # Convert to int if all values are whole numbers
                    if (annot_series == annot_series.astype(int)).all():
                        annot_series = annot_series.astype(int)
                    
                    if annotation_order is not None:
                        # Map sorted unique numeric values to user-defined category names
                        sorted_unique = sorted(annot_series.unique())
                        if len(annotation_order) != len(sorted_unique):
                            raise ValueError(
                                f"annotation_order has {len(annotation_order)} entries but the numeric annotation "
                                f"column has {len(sorted_unique)} unique values: {sorted_unique}. "
                                f"They must match in length."
                            )
                        label_to_category = {val: cat for val, cat in zip(sorted_unique, annotation_order)}
                        annot_df = pd.DataFrame({
                            'category': annot_series.map(label_to_category),
                            'label': annot_series
                        }, index=annot_series.index)
                        print(f"Annotation order applied to numeric labels: {label_to_category}")
                    else:
                        annot_df = pd.DataFrame({
                            'category': annot_series.values,
                            'label': annot_series.values
                        }, index=annot_series.index)
                
                # Print annotation distribution
                print(f"\nAnnotation distribution:\n{annot_df['label'].value_counts().sort_index().to_string()}")
                
                # Save annotation
                annot_path = self.input_files_dir / f"{annotation}_annotation.csv"
                if annot_path.exists():
                    print(f"\nWarning: Output file {annot_path} already exists and will be overwritten.")
                annot_df.to_csv(annot_path, index=True, header=True)
                print(f"Saved annotation to {annot_path}")
            else:
                print(f"\nWarning: Annotation '{annotation}' not found in obs columns.")
                print(f"\nAvailable columns:\n {list(run_data.obs.columns)}")
                print(f"Please manually provide an appropriate annotation file to SimiCPipeline in {self.input_files_dir}")
        elif annotation:
            print(f"Warning: Cannot save annotation. run_data must be AnnData object.")
            print(f"Please manually provide an appropriate annotation file to SimiCPipeline in {self.input_files_dir}")
        
        print("\n-------\n")
        print("Experiment files saved successfully.")
        print("\n-------\n")

