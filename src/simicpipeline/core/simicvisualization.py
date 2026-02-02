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
            label_names: Dictionary mapping labels to custom names (optional)
            adata: AnnData object containing cell metadata (optional)
        
        Example:
          
            # Initialization in one step
            viz = SimiCVisualization(
                project_dir="./SimiCExampleRun/KPB25L/Tumor",
                run_name="experiment_tumor",
                p2assignment="./SimiCExampleRun/KPB25L/annotation.csv",
                lambda1=1e-1,
                lambda2=1e-2,
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
        base_name = f"{self.run_name}_L1_{self.lambda1}_L2_{self.lambda2}"
        if search:
            if self.lambda1 is None or self.lambda2 is None:
                raise ValueError("Both lambda1 and lambda2 must be provided when search=True")
            self.run_path = self.output_path / "matrices"  / self.run_name
            self.p2simic_matrices = self.run_path / f"{base_name}_simic_matrices.pickle"
            self.p2filtered_matrices = self.run_path / f"{base_name}_simic_matrices_filtered_BIC.pickle"
            self.p2auc_raw = self.run_path / f"{base_name}_wAUC_matrices.pickle"
            self.p2auc_filtered = self.run_path / f"{base_name}_wAUC_matrices_filtered_BIC.pickle"
        else: 
            self.p2simic_matrices = None
            self.p2filtered_matrices = None
            self.p2auc_raw = None
            self.p2auc_filtered = None

       # Store AnnData object if provided (for UMAP visualziations)
        if adata is not None:
            self.set_adata(adata)
        
        # Default plot settings
        self.default_figsize = (12, 6)
        
        # Label names mapping
        if label_names is not None:
            if colors is not None:
                self.set_label_names(label_names, colors)
            else:   
                self.set_label_names(label_names)
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
        if not p2assignment:
            raise ValueError("Path to assignment file (p2assignment) must be provided.")
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

    def plot_r2_distribution(self, labels: Optional[List[Union[int, str]]] = None, 
                            threshold: float = 0.7,
                            save: bool = True,
                            filename: Optional[str] = None):
        """
        Plot R² distribution histograms for target genes.
        
        Args:
            labels: List of phenotype labels to plot (default: all)
            threshold: R² threshold line to display
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print("PLOTTING R² DISTRIBUTIONS")
        print("="*70 + "\n")
        
        # Load results
        results = self.load_results('Ws_filtered')
        adj_r2 = results['adjusted_r_squared']
        
        if labels is None:
            labels = list(adj_r2.keys())
        
        n_labels = len(labels)
        fig, axes = plt.subplots(1, n_labels, figsize=(6*n_labels, 5))
        if n_labels == 1:
            axes = [axes]
        
        for idx, label in enumerate(labels):
            r2_values = adj_r2[label]
            selected = np.sum(r2_values > threshold)
            mean_r2 = np.mean(r2_values[r2_values > threshold])
            
            label_name = self._get_label_name(label)
            
            axes[idx].hist(r2_values, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
            axes[idx].set_xlabel('Adjusted R²', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{label_name}\nTargets selected: {selected}, Mean R²: {mean_r2:.3f}', 
                               fontsize=12)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_R2_distribution.pdf"
            plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
            print(f"✓ Saved to {self.figures_path / fname}")
        
        return fig

    def plot_tf_weights(self, tf_names: Union[str, List[str]],
                       labels: Optional[List[Union[int, str]]] = None,
                       top_n_targets: int = 50,
                       save: bool = True,
                       filename: Optional[str] = None):
        """
        Plot weight barplots for specific transcription factors.
        
        Args:
            tf_names: TF name(s) to plot (single string or list)
            labels: Phenotype labels to include (default: all)
            top_n_targets: Number of top targets to display
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING TF WEIGHT BARPLOTS")
        print("="*70 + "\n")
        
        if isinstance(tf_names, str):
            tf_names = [tf_names]
        
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
            unselected_targets[label] = [target_ids[i] for i, r2 in enumerate(adj_r2[label]) if r2 < 0.7]
        
        # Filter valid TF names
        valid_tf_names = [tf for tf in tf_names if tf in tf_ids]
        if not valid_tf_names:
            print(f"Error: None of the specified TFs found in results: {tf_names}")
            return None
        
        n_tfs = len(valid_tf_names)
        fig, axes = plt.subplots(n_tfs, 1, figsize=(16, 5*n_tfs))
        if n_tfs == 1:
            axes = [axes]
        
        for tf_idx, tf_name in enumerate(valid_tf_names):
            if tf_name not in tf_ids:
                print(f"Warning: TF '{tf_name}' not found, skipping...")
                continue
            
            print(f"Processing {tf_name}...")
            
            # Get TF index
            tf_index = tf_ids.index(tf_name)
            
            # Collect weights for all labels
            plot_data = []
            for label in labels:
                try:
                    weights = weight_dic[label][tf_index, :]
                    for i, target in enumerate(target_ids):
                        if target not in unselected_targets[label] and weights[i] != 0:
                            plot_data.append({
                                'target': target,
                                'weight': weights[i],
                                'label': str(label),
                                'label_name': self._get_label_name(label),
                                'abs_weight': abs(weights[i])
                            })
                except Exception as e:
                    print(f"  Error processing label {label}: {e}")
                    continue
            
            df = pd.DataFrame(plot_data)
            
            if df.empty:
                print(f"  No non-zero weights found for {tf_name}")
                ax = axes[tf_idx]
                ax.text(0.5, 0.5, f'No non-zero weights\nfor {tf_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'TF: {tf_name}', fontsize=14, fontweight='bold')
                continue
            
            # Get top targets by absolute weight
            top_targets = df.groupby('target')['abs_weight'].max().nlargest(top_n_targets).index
            df_filtered = df[df['target'].isin(top_targets)]
            
            # Order targets by max absolute weight
            target_order = df_filtered.groupby('target')['abs_weight'].max().sort_values(ascending=False).index
            
            # Plot
            ax = axes[tf_idx]
            x_pos = np.arange(len(target_order))
            bar_width = 0.8 / len(labels)
            
            default_colors = ["steelblue", "orange", "green", "purple", "brown"]
            
            for label_idx, label in enumerate(labels):
                label_data = df_filtered[df_filtered['label'] == str(label)]
                weights_ordered = [label_data[label_data['target'] == t]['weight'].values[0] 
                                  if t in label_data['target'].values else 0 
                                  for t in target_order]
                
                label_name = self._get_label_name(label)
                
                # Use custom color if available, otherwise use default
                if int(label) in self.colors:
                    bar_color = self.colors[int(label)]
                else:
                    bar_color = default_colors[label_idx % len(default_colors)]
                
                ax.bar(x_pos + label_idx * bar_width, weights_ordered, 
                      bar_width, label=label_name, 
                      color=bar_color, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Target Genes', fontsize=10)
            ax.set_ylabel('Weight', fontsize=10)
            ax.set_title(f'TF: {tf_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos + bar_width * (len(labels)-1) / 2)
            ax.set_xticklabels(target_order, rotation=45, ha='right', fontsize=7)
            ax.legend(loc='best')
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_TF_weights.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        
        return fig
    
    def plot_target_weights(self, target_names: Union[str, List[str]],
                           labels: Optional[List[Union[int, str]]] = None,
                           save: bool = True,
                           filename: Optional[str] = None):
        """
        Plot weight barplots for specific target genes.
        
        Args:
            target_names: Target gene name(s) to plot
            labels: Phenotype labels to include (default: all)
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING TARGET WEIGHT BARPLOTS")
        print("="*70 + "\n")
        
        if isinstance(target_names, str):
            target_names = [target_names]
        
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
            unselected_targets[label] = [target_ids[i] for i, r2 in enumerate(adj_r2[label]) if r2 < 0.7]
        
        # Filter valid target names
        valid_target_names = [t for t in target_names if t in target_ids]
        if not valid_target_names:
            print(f"Error: None of the specified targets found in results: {target_names}")
            return None
        
        n_targets = len(valid_target_names)
        fig, axes = plt.subplots(n_targets, 1, figsize=(16, 5*n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for tgt_idx, target_name in enumerate(valid_target_names):
            if target_name not in target_ids:
                print(f"Warning: Target '{target_name}' not found, skipping...")
                continue
            
            print(f"Processing {target_name}...")
            
            # Get target index
            target_index = target_ids.index(target_name)
            
            # Collect weights for all labels
            plot_data = []
            for label in labels:
                if target_name in unselected_targets[label]:
                    continue
                
                try:
                    weights = weight_dic[label][:, target_index]
                    for i, tf in enumerate(tf_ids):
                        if weights[i] != 0:
                            plot_data.append({
                                'tf': tf,
                                'weight': weights[i],
                                'label': str(label),
                                'label_name': self._get_label_name(label),
                                'abs_weight': abs(weights[i])
                            })
                except Exception as e:
                    print(f"  Error processing label {label}: {e}")
                    continue
            
            df = pd.DataFrame(plot_data)
            
            if df.empty:
                print(f"  No non-zero weights found for {target_name}")
                ax = axes[tgt_idx]
                ax.text(0.5, 0.5, f'No non-zero weights\nfor {target_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Target: {target_name}', fontsize=14, fontweight='bold')
                continue
            
            # Order TFs by max absolute weight
            tf_order = df.groupby('tf')['abs_weight'].max().sort_values(ascending=False).index
            
            # Plot
            ax = axes[tgt_idx]
            x_pos = np.arange(len(tf_order))
            bar_width = 0.8 / len(labels)
            
            default_colors = ["steelblue", "orange", "green", "purple", "brown"]
            
            for label_idx, label in enumerate(labels):
                label_data = df[df['label'] == str(label)]
                weights_ordered = [label_data[label_data['tf'] == t]['weight'].values[0] 
                                  if t in label_data['tf'].values else 0 
                                  for t in tf_order]
                
                label_name = self._get_label_name(label)
                
                # Use custom color if available, otherwise use default
                if int(label) in self.colors:
                    bar_color = self.colors[int(label)]
                else:
                    bar_color = default_colors[label_idx % len(default_colors)]
                
                ax.bar(x_pos + label_idx * bar_width, weights_ordered, 
                      bar_width, label=label_name, 
                      color=bar_color, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Transcription Factors', fontsize=10)
            ax.set_ylabel('Weight', fontsize=10)
            ax.set_title(f'Target: {target_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos + bar_width * (len(labels)-1) / 2)
            ax.set_xticklabels(tf_order, rotation=45, ha='right', fontsize=8)
            ax.legend(loc='best')
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_target_weights.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        
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

    def plot_auc_distributions(self, tf_names: Union[str, List[str]],
                              labels: Optional[List[Union[int, str]]] = None,
                              fill: bool = True,
                              alpha: float = 0.5,
                              bw_adjust: Union[str, float] = "scott",
                              save: bool = True,
                              filename: Optional[str] = None):
        """
        Plot AUC density distributions for specific TFs across phenotypes.
        
        Args:
            tf_names: TF name(s) to plot
            labels: Phenotype labels to compare
            fill: Whether to fill the density curves (default: True)
            alpha: Transparency level for filled curves (0-1, default: 0.6)
            bw_adjust:  Bandwidth adjustment for density smoothness (default: 0.5)
                      Can be 'scott', 'silverman', or a float value
                      Lower values (e.g., 0.2-0.5) = less smooth, more detail
                      Higher values (e.g., 1.0-2.0) = more smooth
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING AUC DISTRIBUTIONS")
        print("="*70 + "\n")
        
        if isinstance(tf_names, str):
            tf_names = [tf_names]
        
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
            else:
                print(f"Warning: Skipping {tf_name} - insufficient data for plotting")
        
        if not valid_tf_names:
            print("Error: No valid TFs to plot!")
            return None
        
        n_tfs = len(valid_tf_names)
        n_cols = 2
        n_rows = (n_tfs + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        axes = axes.flatten() if n_tfs > 1 else [axes]
        
        default_colors = ["steelblue", "orange", "green", "purple", "brown"]
        
        for tf_idx, tf_name in enumerate(valid_tf_names):
            print(f"Processing {tf_name}...")
            
            ax = axes[tf_idx]
            plotted_any = False
            
            for label_idx, label in enumerate(labels):
                try:
                    auc_subset = self.subset_label_specific_auc('auc_filtered', label)
                    
                    if tf_name not in auc_subset.columns:
                        print(f"  Warning: {tf_name} not found in label {label}")
                        continue
                    
                    values = auc_subset[tf_name].dropna()
                    
                    # Check if we have enough valid data points
                    if len(values) < 2:
                        print(f"  Warning: Insufficient data for {tf_name} in label {label} (n={len(values)})")
                        continue
                    
                    # Check if all values are the same (would cause density plot to fail)
                    if values.std() == 0:
                        print(f"  Warning: No variance in {tf_name} for label {label}, plotting as vertical line")
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
                        print(f"  Warning: Could not plot density for {tf_name}, label {label}: {plot_error}")
                        # Try histogram as fallback
                        try:
                            ax.hist(values, bins=20, density=True, alpha=alpha, 
                                   color=color, label=label_name, 
                                   edgecolor='black')
                            plotted_any = True
                        except Exception:
                            print(f"  Error: Could not plot histogram either")
                            continue
                
                except Exception as e:
                    print(f"  Error processing label {label}: {e}")
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
                
                ax.set_xlabel('AUC Score', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.legend(loc='best')
                ax.grid(alpha=0.3)
                ax.set_xlim(0, 1)
        
        # Hide extra subplots
        for idx in range(n_tfs, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_AUC_distributions.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        
        return fig

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
        fig.suptitle('AUC Score Summary Statistics', fontsize=16, fontweight='bold')
        
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
        ax.set_ylabel('AUC Score', fontsize=10)
        ax.set_title('AUC Distribution (Boxplot)', fontsize=12, fontweight='bold')
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
        ax.set_ylabel('AUC Score', fontsize=10)
        ax.set_title('AUC Distribution (Violin Plot)', fontsize=12, fontweight='bold')
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
        ax.set_ylabel('Mean AUC Score', fontsize=10)
        ax.set_title('Mean AUC Score by Phenotype', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_auc_high_activity(self, ax, labels: List, label_names: List[str], colors_list: List[str]):
        """
        Plot count of TFs with high activity (AUC > 0.5).
        
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
        ax.set_title('TFs with High Activity (AUC > 0.5)', fontsize=12, fontweight='bold')
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
            
            # Calculate number of TFs with high activity (AUC > 0.5)
            mean_auc_per_tf = auc_subset.mean(axis=0)
            high_activity_count = (mean_auc_per_tf > 0.5).sum()
            
            table_data.append([label_names[i], f'{mean_val:.4f}', f'{median_val:.4f}', 
                              f'{std_val:.4f}', f'{min_val:.4f}', f'{max_val:.4f}', str(high_activity_count)])
        
        return table_data
    
    def plot_auc_statistics_table(self, labels: Optional[List[Union[int, str]]] = None,
                                  save: bool = True,
                                  filename: Optional[str] = None):
        """
        Plot summary statistics table for AUC scores.
        
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
        
        fig.suptitle('AUC Score Summary Statistics Table', fontsize=14, fontweight='bold', y=0.98)
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

# Export functions from SimiCPipeline
SimiCVisualization.calculate_dissimilarity = SimiCPipeline.calculate_dissimilarity
SimiCVisualization.subset_label_specific_auc = SimiCPipeline.subset_label_specific_auc
SimiCVisualization.get_TF_network = SimiCPipeline.get_TF_network
SimiCVisualization.get_TF_auc = SimiCPipeline.get_TF_auc
