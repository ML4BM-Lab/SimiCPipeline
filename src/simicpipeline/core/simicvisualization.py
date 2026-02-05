#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Visualization Class
Provides visualization methods for SimiC pipeline results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Union, Dict
import warnings

# Required optional dependencies
try:
    import anndata as ad
except ImportError:
    raise ImportError("Anndata is required. Please install: pip install anndata")

try:
    import scanpy as sc
except ImportError:
    raise ImportError("Scanpy is required. Please install: pip install scanpy")

# Import base pipeline (will raise if not installed)
from simicpipeline.core.base import SimiCBase
from simicpipeline.core.main import SimiCPipeline
warnings.filterwarnings('ignore')

class SimiCVisualization(SimiCBase):
    """
    Visualization class for SimiC results.
    
    Inherits from SimiCPipeline to access all pipeline functionality plus visualization methods.
    Requires SimiCPipeline to be fully installed and configured.
    """
    
    def __init__(self, 
                 project_dir: str,
                 run_name: str,
                 search: bool = True,
                 lambda1: Optional[float] = None,
                 lambda2: Optional[float] = None,
                 p2assignment: Optional[Union[str, Path]] = None,
                 label_names: Optional[Dict[Union[int, str], str]] = None,
                 colors: Optional[Dict[Union[int, str], str]] = None,
                 adata: Optional[Union[Path, str, ad.AnnData]] = None):
        """
        Initialize visualization pipeline.
        
        Args:
            project_dir (str): Working directory path where input files are located
            run_name (str): Unique identifier for this analysis run
            search: Whether to automatically search for simic output files in project directory / run_name based on lambda1 and lambda2
            lambda1: Lambda1 regularization parameter (optional)
            lambda2: Lambda2 regularization parameter (optional)
            p2assignment: Path to cell label assignment file (optional)
            label_names: Dictionary mapping labels to custom names (optional)
            adata: AnnData object containing cell metadata (optional)
        
        Example:
          
            # Initialization in one step
            viz = SimiCVisualization(
                project_dir="./SimiCExampleRun/KPB25L/Tumor",
                run_name="experiment_tumor",
                lambda1=1e-1,
                lambda2=1e-2,
                p2assignment="./SimiCExampleRun/KPB25L/annotation.csv",
                label_names={0: 'Control', 1: 'PD-L1', 2: 'DAC', 3: 'Combination'},
                adata=adata_object
            )
        """
        # Initialize parent SimiCPipeline
        super().__init__(project_dir = project_dir)
        # Set paths
        
        self.output_path = self.project_dir / "outputSimic"
        self.run_name = run_name
        # Create figures directory
        self.figures_path = self.output_path / "figures" / self.run_name
        if self.figures_path.exists():
            print("Warning: Figures path already exists. Existing figures may be overwritten.")
        else:
            print(f"Creating figures directory at: {self.figures_path}")
            self.figures_path.mkdir(parents=True, exist_ok=True)

        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p2assignment = p2assignment
        base_name = f"{self.run_name}_L1_{self.lambda1}_L2_{self.lambda2}"
        if search:
            if self.lambda1 is None or self.lambda2 is None:
                raise ValueError("Both lambda1 and lambda2 must be provided when search=True")
            self.run_path = self.output_path / "matrices"  / self.run_name
            self.p2simic_matrices = self.run_path / f"{base_name}_simic_matrices.pickle"
            self.p2filtered_matrices = self.run_path / f"{base_name}_simic_matrices_filtered_BIC.pickle"
            self.p2auc_raw = self.run_path / f"{base_name}_wAUC_matrices.pickle"
            self.p2auc_filtered = self.run_path / f"{base_name}_wAUC_matrices_filtered_BIC.pickle"

       # Store AnnData object if provided (for UMAP visualziations)
        if adata is not None:
            self.set_adata(adata)
        
        # Default plot settings
        self.default_figsize = (12, 6)
        
        # Label names mapping
        if label_names is not None:
            if colors is not None:
                self.set_label_names(p2assignment= self.p2assignment,
                                     label_names= label_names, colors = colors)
            else:   
                self.set_label_names(p2assignment= self.p2assignment,
                                     label_names= label_names)
        else:   
            self.label_names = {}
            self.colors = {}

    # def set_paths        

    def set_adata(self, adata: Union[Path, str, ad.AnnData]):
        """
        Set or update the AnnData object.
        
        Args:
            adata: AnnData object containing cell metadata
        """
        if isinstance(adata, str) or isinstance(adata, Path):
            from simicpipeline import load_from_anndata
            adata = load_from_anndata(adata)

        self.adata = adata
        print(f"AnnData object set with {adata.n_obs} cells and {adata.n_vars} genes")
        
    def set_label_names(self, p2assignment: Union[Path, str],
                        label_names: Dict[Union[int, str], str],
                        colors: Optional[Dict[Union[int, str], str]] = None):
        """
        Set path and custom names and colors for phenotype labels.
        
        Args:
            p2assignment: Path to cell label assignment file. Needed for label-specific visualizations.
            label_names: Dictionary mapping numeric labels to custom names
                        e.g., {0: 'Control', 1: 'Treatment1', 2: 'Treatment2'}
            colors: (Optional) Dictionary mapping numeric labels to colors
                    e.g., {0: 'blue', 1: 'orange', 2: 'green'}
        Example:
            p2assignment = "./SimiCExampleRun/KPB25L/annotation.csv"
            lab_dict = {0: 'Control', 1: 'PD-L1', 2: 'DAC', 3: 'Combination'}
            col_dict = {0: 'blue', 1: 'orange', 2: 'green', 3: 'purple'}
            viz.set_label_names(label_names=lab_dict, colors=col_dict)
        """
        if p2assignment is None:
            raise ValueError("Path to assignment file (p2assignment) must be provided.")
        if p2assignment is not None:
            self.p2assignment = Path(p2assignment)
        if not self.p2assignment.exists():
            raise FileNotFoundError(f"Assignment file not found: {self.p2assignment}")
        self.label_names = {int(k): str(v) for k, v in label_names.items()}
        print(f"Label names set: {self.label_names}")
        if colors is not None:
            self.colors = {int(k): v for k, v in colors.items()}
            print(f"Label colors set: {self.colors}")
        # else: 
        #     self.colors = None
        
    def _get_label_name(self, label: Union[int, str]) -> str:
        """
        Get the display name for a label.
        
        Args:
            label: Numeric label
            
        Returns:
            Custom name if set, otherwise returns 'Label {label}'
        """
        label_int = int(label)
        if label_int in self.label_names:
            return self.label_names[label_int]
        return f'Label {label}'

    # Plotting functions

    def plot_r2_distribution(
        self,
        labels: Optional[List[Union[int, str]]] = None,
        threshold: float = 0.7,
        save: bool = True,
        grid_layout: Optional[tuple] = None,
        filename: Optional[str] = None
    ):
        """
        Plot R² distribution histograms for target genes.

        Args:
            labels: List of phenotype labels to plot (default: all)
            threshold: R² threshold line to display
            save: Whether to save the figure
            grid_layout: Tuple (n_rows, n_cols) for subplot arrangement
            filename: Custom filename for saved figure
        """

        print("\n" + "="*70)
        print("PLOTTING R² DISTRIBUTIONS")
        print("="*70 + "\n")

        # Load results
        results = self.load_results("Ws_filtered")
        adj_r2 = results["adjusted_r_squared"]

        if labels is None:
            labels = list(adj_r2.keys())
        n_labels = len(labels)

        if grid_layout is not None:
            n_rows, n_cols = grid_layout
        else:
            n_rows, n_cols = n_labels, 1  # default vertical layout

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows)
        )

        # Always flatten axes into 1D list for easy indexing
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        # Plot each label
        for idx, label in enumerate(labels):
            # Data to plot
            r2_values = adj_r2[label]
            selected = np.sum(r2_values > threshold)
            mean_r2 = np.mean(r2_values[r2_values > threshold]) if selected > 0 else 0
            label_name = self._get_label_name(label)
            # Figure
            ax = axes[idx]

            ax.hist(
                r2_values,
                bins=100,
                alpha=0.7,
                edgecolor="black"
            )

            ax.axvline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold={threshold}"
            )
            ax.set_xlabel("Adjusted R²", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(
                f"{label_name}\nTargets selected: {selected}, Mean R²: {mean_r2:.3f}",
                fontsize=12
            )
            ax.legend()
            ax.grid(alpha=0.3)

        # Hide unused axes if grid is larger than needed
        for j in range(n_labels, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        if save:
            fname = filename or "r2_distribution.pdf"
            pdf_path = self.figures_path / fname
            plt.savefig(pdf_path, dpi=300)
            plt.show()
            plt.close(fig)
            return fig

        plt.show()
        return fig
    
    def _prepare_weight_plot_data(self, gene_names: Union[str, List[str], None],
                               gene_type: str,  # 'tf' or 'target'
                               labels: Optional[List[Union[int, str]]],
                               r2_threshold: float):
        """
        Prepare data for weight plotting (shared by TF and target plots).
        
        Args:
            gene_names: Gene name(s) to plot (TFs or targets)
            gene_type: Either 'tf' or 'target'
            labels: Phenotype labels to include
            r2_threshold: Minimum adjusted R² threshold
            
        Returns:
            tuple: (genes_to_plot, labels, weight_dic, tf_ids, target_ids, unselected_targets, mute)
        """
        # Load results
        try:
            results = self.load_results('Ws_filtered')
        except FileNotFoundError:
            print("Error: Filtered weights not found!")
            return None
        
        weight_dic = results['weight_dic']
        tf_ids = results['TF_ids']
        target_ids = results['query_targets']
        
        if labels is None:
            labels = list(weight_dic.keys())
        
        # Get unselected targets per label
        adj_r2 = results['adjusted_r_squared']
        unselected_targets = {}
        for label in labels:
            unselected_targets[label] = [target_ids[i] for i, r2 in enumerate(adj_r2[label]) if r2 < r2_threshold]
        
        # Determine which genes to plot
        gene_list = tf_ids if gene_type == 'tf' else target_ids
        
        if gene_names is not None:
            if isinstance(gene_names, str):
                gene_names = [gene_names]
            valid_genes = [g for g in gene_names if g in gene_list]
            if not valid_genes:
                print(f"Error: None of the specified {gene_type}s found in results.")
                return None
            genes_to_plot = valid_genes
            mute = False
        else:
            genes_to_plot = sorted(gene_list)
            mute = True
        
        return genes_to_plot, labels, weight_dic, tf_ids, target_ids, unselected_targets, mute

    def _collect_weight_data(self, gene_name: str, gene_type: str, 
                            labels: List, weight_dic: dict,
                            tf_ids: List[str], target_ids: List[str],
                            unselected_targets: dict) -> pd.DataFrame:
        """
        Collect weight data for a single gene (TF or target).
        
        Args:
            gene_name: Name of the gene (TF or target)
            gene_type: Either 'tf' or 'target'
            labels: Phenotype labels
            weight_dic: Dictionary of weight matrices
            tf_ids: List of TF names
            target_ids: List of target names
            unselected_targets: Dict of unselected targets per label
            
        Returns:
            DataFrame with columns: gene, weight, label, label_name, abs_weight
        """
        plot_data = []
        
        if gene_type == 'tf':
            gene_index = tf_ids.index(gene_name)
            partner_list = target_ids
            axis = 1  # TF row, iterate over target columns
        else:  # target
            gene_index = target_ids.index(gene_name)
            partner_list = tf_ids
            axis = 0  # Target column, iterate over TF rows
        
        for label in labels:
            # Skip if target is unselected for this label
            if gene_type == 'target' and gene_name in unselected_targets[label]:
                continue
            
            try:
                if axis == 1:  # TF
                    weights = weight_dic[label][gene_index, :]
                else:  # Target
                    weights = weight_dic[label][:, gene_index]
                
                for i, partner in enumerate(partner_list):
                    # Skip unselected targets when plotting TFs
                    if gene_type == 'tf' and partner in unselected_targets[label]:
                        continue
                    
                    if weights[i] != 0:
                        plot_data.append({
                            'gene': partner,  # the partner gene (target for TF, TF for target)
                            'weight': weights[i],
                            'label': str(label),
                            'label_name': self._get_label_name(label),
                            'abs_weight': abs(weights[i])
                        })
            except Exception as e:
                print(f"    Error processing label {label}: {e}")
                continue
        
        return pd.DataFrame(plot_data)

    def _plot_weight_barplot(self, ax, df: pd.DataFrame, gene_name: str, 
                            gene_type: str, labels: List, top_n: Optional[int] = None):
        """
        Create a weight barplot on the given axis.
        
        Args:
            ax: Matplotlib axis
            df: DataFrame with weight data
            gene_name: Name of the gene being plotted
            gene_type: Either 'tf' or 'target'
            labels: List of phenotype labels
            top_n: Number of top genes to show (for TFs only)
        """
        if df.empty:
            ax.text(0.5, 0.5, f'No non-zero weights\nfor {gene_name}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            title = f'TF: {gene_name}' if gene_type == 'tf' else f'Target: {gene_name}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            return
        
        # Filter to top genes if specified (for TFs)
        if top_n is not None and gene_type == 'tf':
            top_genes = df.groupby('gene')['abs_weight'].mean().nlargest(top_n).index
            df = df[df['gene'].isin(top_genes)]
        
        # Order genes by max absolute weight
        gene_order = df.groupby('gene')['abs_weight'].max().sort_values(ascending=False).index
        
        # Plot
        x_pos = np.arange(len(gene_order))
        bar_width = 0.8 / len(labels)
        default_colors = ["steelblue", "orange", "green", "purple", "brown"]
        
        for label_idx, label in enumerate(labels):
            label_data = df[df['label'] == str(label)]
            weights_ordered = [
                label_data[label_data['gene'] == g]['weight'].values[0] 
                if g in label_data['gene'].values else 0 
                for g in gene_order
            ]
            
            label_name = self._get_label_name(label)
            bar_color = self.colors.get(int(label), default_colors[label_idx % len(default_colors)])
            
            ax.bar(x_pos + label_idx * bar_width, weights_ordered, 
                bar_width, label=label_name, 
                color=bar_color, edgecolor='black', linewidth=0.5)
        
        # Set labels and title
        xlabel = 'Target Genes' if gene_type == 'tf' else 'Transcription Factors'
        title = f'TF: {gene_name}' if gene_type == 'tf' else f'Target: {gene_name}'
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Weight', fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + bar_width * (len(labels)-1) / 2)
        ax.set_xticklabels(gene_order, rotation=45, ha='right', 
                        fontsize=7 if gene_type == 'tf' else 8)
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)

    def _plot_weights_paginated(self, gene_names: Union[str, List[str], None],
                                gene_type: str,
                                labels: Optional[List[Union[int, str]]] = None,
                                top_n: Optional[int] = None,
                                r2_threshold: float = 0.7,
                                grid_layout: Optional[tuple] = (4, 1),
                                save: bool = True,
                                filename: Optional[str] = None):
        """
        Unified method for plotting TF or target weights with pagination.
        
        Args:
            gene_names: Gene name(s) to plot
            gene_type: Either 'tf' or 'target'
            labels: Phenotype labels to include
            top_n: Number of top genes to show (for TF plots only)
            r2_threshold: Minimum adjusted R² threshold
            grid_layout: Tuple (rows, cols) for subplot grid per page
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        gene_type_label = "TF" if gene_type == 'tf' else "TARGET"
        print("\n" + "="*70)
        print(f"PLOTTING {gene_type_label} WEIGHT BARPLOTS")
        print("="*70 + "\n")
        
        # Prepare data
        prep_result = self._prepare_weight_plot_data(
            gene_names, gene_type, labels, r2_threshold
        )
        if prep_result is None:
            return None
        
        genes_to_plot, labels, weight_dic, tf_ids, target_ids, unselected_targets, mute = prep_result
        
        # Filter targets that are valid
        if gene_type == 'target':
            valid_genes = []
            for gene in genes_to_plot:
                if not all(gene in unselected_targets[label] for label in labels):
                    valid_genes.append(gene)
                elif not mute:
                    print(f"  Skipping {gene}: unselected in all labels")
            genes_to_plot = valid_genes
            
            if not genes_to_plot:
                print("Error: No valid targets to plot!")
                return None
        
        n_genes = len(genes_to_plot)
        print(f"Plotting {n_genes} {gene_type}s...")
        
        # Setup pagination
        from matplotlib.backends.backend_pdf import PdfPages
        
        if grid_layout is None:
            single_page_mode = True
            grid_rows, grid_cols = n_genes, 1
            plots_per_page = n_genes
            print(f"Single page mode: {grid_rows} rows x {grid_cols} col")
        else:
            single_page_mode = False
            grid_rows, grid_cols = grid_layout
            plots_per_page = grid_rows * grid_cols
            n_pages = (n_genes + plots_per_page - 1) // plots_per_page
            print(f"Multi-page mode: {grid_rows}x{grid_cols} per page ({n_pages} pages)")
        
        # Prepare PDF
        if save:
            fname = filename or f"{self.run_name}_{gene_type}_weights.pdf"
            pdf_path = self.figures_path / fname
            pdf_pages = None if single_page_mode else PdfPages(pdf_path)
        else:
            pdf_path = None
            pdf_pages = None
        
        # Process genes in batches
        all_figs = []
        
        if mute:
            print(f"Generating all {gene_type} plots...")
        
        for page_idx, page_start in enumerate(range(0, n_genes, plots_per_page)):
            page_genes = genes_to_plot[page_start:page_start + plots_per_page]
            
            # Calculate figure size
            if single_page_mode:
                fig_width = 16 if gene_type == 'tf' else 12
                fig_height = max(5 if gene_type == 'tf' else 4, 
                            (5 if gene_type == 'tf' else 4) * len(page_genes))
            else:
                fig_width = 16 if grid_cols >= 2 or gene_type == 'tf' else 12
                fig_height = 5 * grid_rows
            
            # Create figure
            current_rows = len(page_genes) if single_page_mode else grid_rows
            current_cols = 1 if single_page_mode else grid_cols
            
            fig, axes = plt.subplots(current_rows, current_cols, 
                                    figsize=(fig_width, fig_height))
            axes = [axes] if current_rows == 1 and current_cols == 1 else np.array(axes).flatten()
            
            # Plot each gene
            for plot_idx, gene_name in enumerate(page_genes):
                if not mute:
                    print(f"  [{page_idx + 1}] Processing {gene_name}...")
                
                # Collect data
                df = self._collect_weight_data(
                    gene_name, gene_type, labels, weight_dic, 
                    tf_ids, target_ids, unselected_targets
                )
                
                # Plot
                self._plot_weight_barplot(
                    axes[plot_idx], df, gene_name, gene_type, labels, 
                    top_n=top_n if gene_type == 'tf' else None
                )
            
            # Hide unused subplots
            if not single_page_mode:
                for idx in range(len(page_genes), len(axes)):
                    axes[idx].axis('off')
            
            plt.tight_layout()
            
            # Save
            if save:
                if single_page_mode:
                    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
                    print(f"\n✓ Saved {n_genes} {gene_type}s to {pdf_path}")
                    plt.show()
                    plt.close(fig)
                elif pdf_pages is not None:
                    pdf_pages.savefig(fig, bbox_inches='tight')
                    if page_idx < 2:
                        print("Showing first 2 pages preview...")
                        plt.show()
                    plt.close(fig)
            else:
                all_figs.append(fig)
                plt.show()
        
        # Close PDF
        if save and pdf_pages is not None:
            pdf_pages.close()
            print(f"\n✓ Saved {n_genes} {gene_type}s to {pdf_path}")
        
        return None if save else (all_figs[0] if len(all_figs) == 1 else all_figs)

    def plot_tf_weights(self, tf_names: Union[str, List[str], None] = None,
                        labels: Optional[List[Union[int, str]]] = None,
                        top_n_targets: int = 50,
                        r2_threshold: float = 0.7,
                        grid_layout: Optional[tuple] = None,
                        save: bool = True,
                        filename: Optional[str] = None):
        """Plot weight barplots for specific transcription factors."""
        return self._plot_weights_paginated(
            tf_names, 'tf', labels, top_n_targets, 
            r2_threshold, grid_layout, save, filename
        )

    def plot_target_weights(self, target_names: Union[str, List[str], None] = None,
                        labels: Optional[List[Union[int, str]]] = None,
                        r2_threshold: float = 0.7,
                        grid_layout: Optional[tuple] = (4, 1),
                        save: bool = True,
                        filename: Optional[str] = None):
        """Plot weight barplots for specific target genes."""
        return self._plot_weights_paginated(
            target_names, 'target', labels, None,
            r2_threshold, grid_layout, save, filename
        )

    def plot_tf_network_heatmap(self, tf_name: str,
                               top_n_targets: int = 10,
                               labels: Optional[List[Union[int, str]]] = None,
                               cmap: str = 'RdBu_r',
                               vmin: Optional[float] = None,
                               vmax: Optional[float] = None,
                               save: bool = True,
                               filename: Optional[str] = None):
        """
        Plot heatmap of TF regulatory network showing weights across phenotypes.
        
        Args:
            tf_name: Transcription factor name
            top_n_targets: Number of top targets to display (by max absolute weight)
            labels: Phenotype labels to include (default: all)
            cmap: Colormap to use (default: 'RdBu_r' - red/white/blue)
                  Red gradient for positive weights, blue gradient for negative weights
            vmin: Minimum value for colormap (default: auto-calculated from data, symmetric around 0)
            vmax: Maximum value for colormap (default: auto-calculated from data, symmetric around 0)
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING TF NETWORK HEATMAP")
        print("="*70 + "\n")
        
        print(f"Processing {tf_name}...")
        
        # Get network for the TF
        try:
            network = self.get_TF_network(tf_name, stacked=True)
        except Exception as e:
            print(f"Error: Could not retrieve network for {tf_name}: {e}")
            return None
        
        if network is None or network.empty:
            print(f"Error: No network data found for {tf_name}")
            return None
        
        # Filter to selected labels if provided
        if labels is not None:
            label_names = [self._get_label_name(l) for l in labels]
            # Find columns that match the label names
            matching_cols = [col for col in network.columns if any(ln in str(col) for ln in label_names)]
            if matching_cols:
                network = network[matching_cols]
        
        print(f"Total targets for {tf_name}: {len(network)}")
        
        # Calculate max absolute weight across all phenotypes
        network['max_abs_weight'] = network.abs().max(axis=1)
        top_targets = network.nlargest(top_n_targets, 'max_abs_weight')
        
        print(f"\nTop {top_n_targets} targets by absolute weight:")
        print(top_targets.drop('max_abs_weight', axis=1).to_string())
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(top_targets.columns) * 1.2), 
                                        max(6, len(top_targets) * 0.4)))
        
        # Drop the max_abs_weight column for plotting
        plot_data = top_targets.drop('max_abs_weight', axis=1)
        
        # Auto-calculate vmin and vmax from data if not provided
        # Always make symmetric around 0 for RdBu scale
        if vmin is None or vmax is None:
            data_min = plot_data.values.min()
            data_max = plot_data.values.max()
            max_abs = max(abs(data_min), abs(data_max))
            vmin = -max_abs
            vmax = max_abs
        
        # Create the heatmap
        im = ax.imshow(plot_data.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set labels
        ax.set_xticks(np.arange(len(plot_data.columns)))
        ax.set_yticks(np.arange(len(plot_data.index)))
        ax.set_xticklabels(plot_data.columns, fontsize=10)
        ax.set_yticklabels(plot_data.index, fontsize=9)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Regulatory Weight', rotation=270, labelpad=20, fontsize=10)
        
        # Add values to cells
        for i in range(len(plot_data.index)):
            for j in range(len(plot_data.columns)):
                value = plot_data.values[i, j]
                text_color = 'white' if abs(value) > (max_abs / 2) else 'black'
                text = ax.text(j, i, f'{value:.2f}',
                             ha='center', va='center',
                             color=text_color, fontsize=8)
        
        # Add title
        ax.set_title(f'Top {top_n_targets} Targets for {tf_name}\nacross phenotypes',
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add gridlines
        ax.set_xticks(np.arange(len(plot_data.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(plot_data.index)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_network_{tf_name}_heatmap.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"\n✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
            finally:
                plt.close(fig) 
        
        return fig

    def plot_auc_distributions(self, tf_names: Union[str, List[str], None] = None,
                            labels: Optional[List[Union[int, str]]] = None,
                            fill: bool = True,
                            alpha: float = 0.5,
                            bw_adjust: Union[str, float] = "scott",
                            rug: bool = False,
                            grid_layout: Optional[tuple] = (4, 2),
                            save: bool = True,
                            filename: Optional[str] = None):
        """
        Plot AUC density distributions for specific TFs across phenotypes.
        
        Args:
            tf_names: TF name(s) to plot. If None, plots all TFs sorted alphabetically.
            labels: Phenotype labels to compare
            fill: Whether to fill the density curves (default: True)
            alpha: Transparency level for filled curves (0-1, default: 0.5)
            bw_adjust: Bandwidth adjustment for density smoothness (default: "scott")
                    Can be 'scott', 'silverman', or a float value
                    Lower values (e.g., 0.2-0.5) = less smooth, more detail
                    Higher values (e.g., 1.0-2.0) = more smooth
            rug: Whether to add rug plot at bottom (default: False)
            grid_layout: Tuple (rows, cols) for subplot grid per page. Default (4,2) = 8 plots per page.
                        Set to None to put ALL plots on a single page (auto grid layout).
            save: Whether to save the figure
            filename: Custom filename for saved figure
        
        Examples:
            # Plot specific TFs, 8 per page in 4x2 grid (multi-page PDF)
            viz.plot_auc_distributions(tf_names=['TF1', 'TF2'])
            
            # Plot all TFs, 6 per page in 3x2 grid (multi-page PDF)
            viz.plot_auc_distributions(grid_layout=(3, 2))
            
            # Plot specific TFs on a single page (auto grid layout)
            viz.plot_auc_distributions(tf_names=['TF1', 'TF2'], grid_layout=None)
        """
        print("\n" + "="*70)
        print(f"PLOTTING AUC DISTRIBUTIONS")
        print("="*70 + "\n")
        
        # Load AUC results
        try:
            auc_data = self.load_results('auc_filtered')
        except FileNotFoundError:
            print("Error: AUC filtered results not found!")
            return None
        
        if labels is None:
            labels = list(auc_data.keys())
        
        # Calculate dissimilarity scores
        try:
            dissim_scores = self.calculate_dissimilarity(select_labels=labels, verbose=False)
        except Exception as e:
            print(f"Warning: Could not calculate dissimilarity scores: {e}")
            dissim_scores = pd.DataFrame()
        
        # Determine which TFs to plot
        if tf_names is not None:
            if isinstance(tf_names, str):
                tf_names = [tf_names]
            tf_names = [tf for tf in tf_names if tf in auc_data[0].columns.to_list()]
            if len(tf_names) == 0:
                print("Error: None of the specified TFs found in AUC results.")
                return None
            mute = False
        else:
            tf_names = sorted(auc_data[0].columns.tolist())
            mute = True
        
        # Filter out TFs that have no data
        valid_tf_names = []
        for tf_name in tf_names:
            has_data = False
            for label in labels:
                try:
                    auc_subset = self.subset_label_specific_auc('auc_filtered', label)
                    if tf_name in auc_subset.columns:
                        values = auc_subset[tf_name].dropna()
                        if len(values) > 1:  # Need at least 2 points for density
                            has_data = True
                            break
                except Exception:
                    continue
            
            if has_data:
                valid_tf_names.append(tf_name)
            elif not mute:
                print(f"Warning: Skipping {tf_name} - insufficient data for plotting")
        
        if not valid_tf_names:
            print("Error: No valid TFs to plot!")
            return None
        
        n_tfs = len(valid_tf_names)
        print(f"Plotting {n_tfs} TFs...")
        
        # Import PdfPages for multi-page output
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Determine single-page vs multi-page mode
        if grid_layout is None:
            # SINGLE PAGE MODE: auto-calculate grid to fit all TFs
            single_page_mode = True
            # Use 2 columns for better layout
            grid_cols = 2
            grid_rows = (n_tfs + 1) // 2
            plots_per_page = n_tfs
            print(f"Single page mode: {grid_rows} rows x {grid_cols} cols (all {n_tfs} TFs on one page)")
        else:
            # MULTI-PAGE MODE: specified grid per page
            single_page_mode = False
            grid_rows, grid_cols = grid_layout
            plots_per_page = grid_rows * grid_cols
            n_pages = (n_tfs + plots_per_page - 1) // plots_per_page
            print(f"Multi-page mode: {grid_rows} rows x {grid_cols} cols per page ({n_pages} pages)")
        
        # Prepare PDF path
        if save:
            fname = filename or f"{self.run_name}_AUC_distributions.pdf"
            pdf_path = self.figures_path / fname
            if not single_page_mode:
                pdf_pages = PdfPages(pdf_path)
            else:
                pdf_pages = None
        else:
            pdf_pages = None
            pdf_path = None
        
        # Process TFs in batches (pages)
        default_colors = ["steelblue", "orange", "green", "purple", "brown"]
        
        all_figs = []
        
        if mute:
            print(f"Generating all AUC distribution plots...")
        
        for page_idx, page_start in enumerate(range(0, n_tfs, plots_per_page)):
            page_tfs = valid_tf_names[page_start:page_start + plots_per_page]
            
            # Calculate figure size
            if single_page_mode:
                # Single page: scale with number of TFs
                fig_width = 14
                fig_height = max(5, 5 * grid_rows)
            else:
                # Multi-page: fixed size per page based on grid
                fig_width = 14
                fig_height = 5 * grid_rows
            
            # Create figure for this page
            if single_page_mode:
                current_rows = grid_rows
                current_cols = grid_cols
            else:
                current_rows = grid_rows
                current_cols = grid_cols
            
            fig, axes = plt.subplots(current_rows, current_cols, figsize=(fig_width, fig_height))
            
            # Flatten axes for easier iteration
            if current_rows == 1 and current_cols == 1:
                axes = [axes]
            else:
                axes = np.array(axes).flatten()
            
            # Plot each TF in this page
            for plot_idx, tf_name in enumerate(page_tfs):
                if not mute:
                    print(f"  [{page_start + plot_idx + 1}/{n_tfs}] Processing {tf_name}...")
                
                ax = axes[plot_idx]
                plotted_any = False
                rug_list = []
                for label_idx, label in enumerate(labels):
                    try:
                        auc_subset = self.subset_label_specific_auc('auc_filtered', label)
                        
                        if tf_name not in auc_subset.columns:
                            if not mute:
                                print(f"    Warning: {tf_name} not found in label {label}")
                            continue
                        
                        values = auc_subset[tf_name].dropna()
                        rug_list.extend(values.tolist())
                        # Check if we have enough valid data points
                        if len(values) < 2:
                            if not mute:
                                print(f"    Warning: Insufficient data for {tf_name} in label {label} (n={len(values)})")
                            continue
                        
                        # Check if all values are the same (would cause density plot to fail)
                        if values.std() == 0:
                            if not mute:
                                print(f"    Warning: No variance in {tf_name} for label {label}, plotting as vertical line")
                            label_name = self._get_label_name(label)
                            
                            # Use custom color if available, otherwise use default
                            if int(label) in self.colors:
                                color = self.colors[int(label)]
                            else:
                                color = default_colors[label_idx % len(default_colors)]
                            
                            ax.axvline(values.iloc[0], color=color, 
                                    linestyle='--', linewidth=2, label=label_name, alpha=alpha)
                            plotted_any = True
                            continue
                        
                        label_name = self._get_label_name(label)
                        
                        # Use custom color if available, otherwise use default
                        if int(label) in self.colors:
                            color = self.colors[int(label)]
                        else:
                            color = default_colors[label_idx % len(default_colors)]
                        
                        # Plot density with optional fill and bandwidth adjustment
                        try:
                            if fill:
                                values.plot.density(ax=ax, label=label_name, 
                                                color=color, alpha=alpha, linewidth=2,
                                                bw_method=bw_adjust)
                                # Fill under the curve
                                line = ax.get_lines()[-1]
                                x_data = line.get_xdata()
                                y_data = line.get_ydata()
                                ax.fill_between(x_data, y_data, alpha=alpha, color=color)
                            else:
                                values.plot.density(ax=ax, label=label_name, 
                                                color=color, alpha=1.0, linewidth=2,
                                                bw_method=bw_adjust)
                              
                            plotted_any = True
                            
                        except Exception as plot_error:
                            if not mute:
                                print(f"    Warning: Could not plot density for {tf_name}, label {label}: {plot_error}")
                    except Exception as e:
                        if not mute:
                            print(f"    Error processing label {label}: {e}")
                        continue
       
                if not plotted_any:
                    ax.text(0.5, 0.5, f'No data available\nfor {tf_name}', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_xlim(0, 1)
                else:
                    # Add dissimilarity score to title
                    if tf_name in dissim_scores.index:
                        dissim_score = dissim_scores.loc[tf_name, 'MinMax_score']
                        ax.set_title(f'{tf_name}\nDissimilarity: {dissim_score:.4f}', 
                                fontsize=12, fontweight='bold')
                    else:
                        ax.set_title(f'{tf_name}', fontsize=12, fontweight='bold')
                    
                    ax.set_xlabel('Activity Score', fontsize=10)
                    ax.set_ylabel('Density', fontsize=10)
                    ax.legend(loc='best')
                    ax.grid(alpha=0.3)
                    ax.set_xlim(0, 1)
                # Add rug plot at bottom
                if rug:
                    yticks = ax.get_yticks()
                    tick_spacing = yticks[1] - yticks[0]
                    rug_length = 0.2 * tick_spacing
                    ax.eventplot(rug_list,
                                orientation="horizontal",
                                colors='tab:gray',
                                linewidths=1,
                                linelengths = rug_length,
                                lineoffsets= -rug_length*0.5
                            ) 
            
            # Hide unused subplots on last page
            for idx in range(len(page_tfs), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # Save page to PDF
            if save:
                if single_page_mode:
                    # Single page: save directly to file
                    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
                    print(f"\n✓ Saved {n_tfs} TFs to {pdf_path}")
                    if page_idx < 2:
                        print("Showing first 2 pages preview...")
                        plt.show()
                    plt.close(fig)
                elif pdf_pages is not None:
                    # Multi-page: add to PDF
                    pdf_pages.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            else:
                all_figs.append(fig)
                plt.show()
        
        # Close PDF file (multi-page mode only)
        if save and pdf_pages is not None:
            pdf_pages.close()
            print(f"\n✓ Saved {n_tfs} TFs to {pdf_path}")
        
        # Return figure(s) if not saving
        if not save:
            return all_figs[0] if len(all_figs) == 1 else all_figs
        return None

    def _prepare_auc_data(self, labels: Optional[List[Union[int, str]]] = None):
        """
        Prepare AUC data and label information for visualization.
        
        Args:
            labels: Phenotype labels to compare (default: all)
            
        Returns:
            tuple: (labels, label_names, colors_list, auc_data_list)
        """
        # Load AUC results
        try:
            auc_data = self.load_results('auc_filtered')
        except FileNotFoundError:
            print("Error: AUC filtered results not found!")
            return None, None, None, None
        
        if labels is None:
            labels = list(auc_data.keys())
        
        # Prepare label names
        label_names = [self._get_label_name(l) for l in labels]
        
        # Get colors for labels
        colors_list = []
        default_colors = ["steelblue", "orange", "green", "purple", "brown"]
        for idx, label in enumerate(labels):
            if int(label) in self.colors:
                colors_list.append(self.colors[int(label)])
            else:
                colors_list.append(default_colors[idx % len(default_colors)])
        
        # Collect AUC data for all labels
        auc_data_list = []
        for label in labels:
            auc_subset = self.subset_label_specific_auc('auc_filtered', label)
            auc_values = auc_subset.values.flatten()
            auc_values = auc_values[~np.isnan(auc_values)]
            auc_data_list.append(auc_values)
            print(f"Label {label_names[labels.index(label)]}: {len(auc_values)} AUC values")
        
        return labels, label_names, colors_list, auc_data_list
    
    def plot_auc_summary_statistics(self, labels: Optional[List[Union[int, str]]] = None,
                                   save: bool = True,
                                   filename: Optional[str] = None):
        """
        Plot AUC summary statistics with boxplot, violin plot, mean bar plot, and high activity count.
        
        Args:
            labels: Phenotype labels to compare (default: all)
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print("PLOTTING AUC SUMMARY STATISTICS")
        print("="*70 + "\n")
        
        # Prepare data
        labels, label_names, colors_list, auc_data_list = self._prepare_auc_data(labels)
        if labels is None:
            return None
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Activity Scores Summary Statistics', fontsize=16, fontweight='bold')
        
        # Plot each subplot using dedicated functions
        self._plot_auc_boxplot(axes[0, 0], auc_data_list, label_names, colors_list)
        self._plot_auc_violin(axes[0, 1], auc_data_list, label_names, colors_list)
        self._plot_auc_mean_bar(axes[1, 0], auc_data_list, label_names, colors_list)
        self._plot_auc_high_activity(axes[1, 1], labels, label_names, colors_list)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_AUC_summary_statistics.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
            finally:
                plt.close(fig)
        
        return fig
    
    def _plot_auc_boxplot(self, ax, auc_data_list: List, label_names: List[str], colors_list: List[str]):
        """
        Plot boxplot of AUC distributions.
        
        Args:
            ax: Matplotlib axis object
            auc_data_list: List of AUC value arrays for each label
            label_names: List of label names
            colors_list: List of colors for each label
        """
        bp = ax.boxplot(auc_data_list, labels=label_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Activity Score', fontsize=10)
        ax.set_title('Activity Score Distribution (Boxplot)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_auc_violin(self, ax, auc_data_list: List, label_names: List[str], colors_list: List[str]):
        """
        Plot violin plot of AUC distributions.
        
        Args:
            ax: Matplotlib axis object
            auc_data_list: List of AUC value arrays for each label
            label_names: List of label names
            colors_list: List of colors for each label
        """
        parts = ax.violinplot(auc_data_list, positions=range(len(label_names)), widths=0.7, 
                               showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names)
        ax.set_ylabel('Activity Score', fontsize=10)
        ax.set_title('Activity Score Distribution (Violin Plot)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_auc_mean_bar(self, ax, auc_data_list: List, label_names: List[str], colors_list: List[str]):
        """
        Plot mean AUC with error bars.
        
        Args:
            ax: Matplotlib axis object
            auc_data_list: List of AUC value arrays for each label
            label_names: List of label names
            colors_list: List of colors for each label
        """
        mean_aucs = [np.mean(data) for data in auc_data_list]
        std_aucs = [np.std(data) for data in auc_data_list]
        x_pos = np.arange(len(label_names))
        ax.bar(x_pos, mean_aucs, yerr=std_aucs, alpha=0.7, color=colors_list,
               edgecolor='black', capsize=5, linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(label_names)
        ax.set_ylabel('Mean Activity Score', fontsize=10)
        ax.set_title('Mean Activity Score by Phenotype', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_auc_high_activity(self, ax, labels: List, label_names: List[str], colors_list: List[str]):
        """
        Plot count of TFs with high activity (Mean AS > 0.5).
        
        Args:
            ax: Matplotlib axis object
            labels: List of phenotype labels
            label_names: List of label names
            colors_list: List of colors for each label
        """
        high_activity_counts = []
        for label in labels:
            auc_subset = self.subset_label_specific_auc('auc_filtered', label)
            mean_auc_per_tf = auc_subset.mean(axis=0)
            high_activity = (mean_auc_per_tf > 0.5).sum()
            high_activity_counts.append(high_activity)
        
        x_pos = np.arange(len(label_names))
        ax.bar(x_pos, high_activity_counts, alpha=0.7, color=colors_list, 
               edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(label_names)
        ax.set_ylabel('Number of TFs', fontsize=10)
        ax.set_title('TFs with High Activity (Mean AS > 0.5)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _calculate_auc_statistics(self, labels: List, label_names: List[str]):
        """
        Calculate AUC statistics for all labels.
        
        Args:
            labels: List of phenotype labels
            label_names: List of label names
            
        Returns:
            list: Table data with statistics and high activity counts
        """
        table_data = []
        table_data.append(['Phenotype', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'TFs AS > 0.5'])
        
        for i, label in enumerate(labels):
            auc_subset = self.subset_label_specific_auc('auc_filtered', label)
            auc_values = auc_subset.values.flatten()
            auc_values = auc_values[~np.isnan(auc_values)]
            
            mean_val = np.mean(auc_values)
            median_val = np.median(auc_values)
            std_val = np.std(auc_values)
            min_val = np.min(auc_values)
            max_val = np.max(auc_values)
            
            # Calculate number of TFs with high activity (AS > 0.5)
            mean_auc_per_tf = auc_subset.mean(axis=0)
            high_activity_count = (mean_auc_per_tf > 0.5).sum()
            
            table_data.append([label_names[i], f'{mean_val:.4f}', f'{median_val:.4f}', 
                              f'{std_val:.4f}', f'{min_val:.4f}', f'{max_val:.4f}', str(high_activity_count)])
        
        return table_data
    
    def plot_auc_statistics_table(self, labels: Optional[List[Union[int, str]]] = None,
                                  save: bool = True,
                                  filename: Optional[str] = None):
        """
        Plot summary statistics table for Activity Scores.
        
        Args:
            labels: Phenotype labels to compare (default: all)
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print("PLOTTING AUC STATISTICS TABLE")
        print("="*70 + "\n")
        
        # Prepare data
        labels, label_names, _, _ = self._prepare_auc_data(labels)
        if labels is None:
            return None
        
        # Calculate statistics
        table_data = self._calculate_auc_statistics(labels, label_names)
        
        # Create figure for table only
        fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.8)))
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.18, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        fig.suptitle('Activity Scores Summary Statistics Table', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_AUC_statistics_table.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
            finally:
                plt.close(fig)
        
        return fig

    def plot_dissimilarity_heatmap(self, labels: Optional[List[Union[int, str]]] = None,
                                   top_n_tfs: Optional[int] = None,
                                   save: bool = True,
                                   filename: Optional[str] = None):
        """
        Plot heatmap of regulatory dissimilarity scores.
        
        Args:
            labels: Phenotype labels to compare
            top_n_tfs: Number of top TFs to display (by dissimilarity)
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING DISSIMILARITY HEATMAP")
        print("="*70 + "\n")
        
        # Calculate dissimilarity scores
        dissim_df = self.calculate_dissimilarity(select_labels=labels)
        
        if top_n_tfs:
            dissim_df = dissim_df.head(top_n_tfs)
        
        # Create heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(8, max(6, len(dissim_df) * 0.3)))
        
        # Create the heatmap
        im = ax.imshow(dissim_df.values, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(dissim_df.columns)))
        ax.set_yticks(np.arange(len(dissim_df.index)))
        ax.set_xticklabels(dissim_df.columns, fontsize=10)
        ax.set_yticklabels(dissim_df.index, fontsize=8)
        
        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Dissimilarity Score', rotation=270, labelpad=20, fontsize=10)
        
        # Annotate cells with values
        for i in range(len(dissim_df.index)):
            for j in range(len(dissim_df.columns)):
                text = ax.text(j, i, f'{dissim_df.values[i, j]:.4f}',
                             ha="center", va="center", color="white", fontsize=6)
        
        # Add title and labels
        ax.set_title('Regulatory Dissimilarity Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Transcription Factors', fontsize=12)
        
        # Add gridlines
        ax.set_xticks(np.arange(len(dissim_df.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(dissim_df.index)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_dissimilarity_heatmap.pdf"
            plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
            print(f"✓ Saved to {self.figures_path / fname}")
        
        return fig
    
# Export functions from SimiCPipeline
SimiCVisualization.calculate_dissimilarity = SimiCPipeline.calculate_dissimilarity
SimiCVisualization.subset_label_specific_auc = SimiCPipeline.subset_label_specific_auc
SimiCVisualization.get_TF_network = SimiCPipeline.get_TF_network
SimiCVisualization.get_TF_auc = SimiCPipeline.get_TF_auc
