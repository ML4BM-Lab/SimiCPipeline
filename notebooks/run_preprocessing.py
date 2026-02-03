#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Preprocessing Pipeline

This script runs the preprocessing steps from the tutorial notebook.
Modify the configuration paths below and run: python run_preprocessing.py

Author: Irene Marín-Goñi
"""

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# ============================================================================
WORKDIR = '/path/to/your/working/directory'

# Path to your input data file (h5ad format)
INPUT_DATA = 'path/to/your/input_file.h5ad'

# Project directory where output files will be saved
PROJECT_DIR = './SimiCExampleRun'

# Path to transcription factor list file
TF_LIST_PATH = 'path/to/your/TF_list.txt'

# Gene selection parameters
N_TFS = 100  # Number of top TFs to select
N_TARGETS = 1000  # Number of top target genes to select

# Annotation column from your data (set to None if not available)
ANNOTATION_COLUMN = 'sample_type'  # e.g., 'cell_type' or 'treatment'

# ============================================================================
# IMPORTS
# ============================================================================

import os
import anndata as ad
from simicpipeline import MagicPipeline, ExperimentSetup, load_from_anndata
import simicpipeline

print(f"SimiCPipeline version: {simicpipeline.__version__}")


os.chdir(WORKDIR)
print(f"Current dir files: {os.listdir('.')}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

# Check the notebook for alternative data loading options
print(f"Loading data from: {INPUT_DATA}")
adata = load_from_anndata(INPUT_DATA)
print(f"Data shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

# Ensure raw data is set
if not hasattr(adata.raw, 'X'):
    print("Waring: .raw attribute not found. Setting raw data from current .X matrix.")
    adata.raw = adata.copy()
# ============================================================================
# STEP 2: RUN MAGIC IMPUTATION
# ============================================================================
print("\n" + "="*70)
print("MAGIC IMPUTATION")

# Initialize MAGIC pipeline
magic_pipeline = MagicPipeline(
    input_data=adata,
    project_dir=PROJECT_DIR,
    magic_output_file='magic_imputed.pickle',
    filtered=False
)

print(magic_pipeline)

# Filter cells and genes
print("\nFiltering cells and genes...")
magic_pipeline.filter_cells_and_genes(
    min_cells_per_gene=10,
    min_umis_per_cell=500
)

# Normalize data
print("\nNormalizing data...")
magic_pipeline.normalize_data()

# Run MAGIC
print("\nRunning MAGIC imputation...")
magic_pipeline.run_magic(
    random_state=123,
    n_jobs=-2,  # Use all but 1 CPU cores
    save_data=True
)

print("\nMAGIC imputation complete!")
print(magic_pipeline)
print(magic_pipeline.print_project_info())
# Get the imputed data
imputed_data = magic_pipeline.magic_adata

# ============================================================================
# STEP 3: EXPERIMENT SETUP AND GENE SELECTION
# ============================================================================

print("\n" + "="*70)
print("EXPERIMENT SETUP AND GENE SELECTION")

# Initialize experiment
experiment = ExperimentSetup(
    input_data=imputed_data,
    tf_path=TF_LIST_PATH,
    project_dir=PROJECT_DIR
)

print(f"Matrix shape: {experiment.matrix.shape}")
print(f"Number of cells: {len(experiment.cell_names)}")
print(f"Number of genes: {len(experiment.gene_names)}")
print(f"Number of TFs in list: {len(experiment.tf_list)}")

# Calculate MAD and select genes
print("\nCalculating Median Absolute Deviation and selecting genes...")
tf_list, target_list = experiment.calculate_mad_genes(
    n_tfs=N_TFS,
    n_targets=N_TARGETS
)

print(f"\nSelected {len(tf_list)} TFs")
print(f"Selected {len(target_list)} targets")
print(f"Example TFs: {tf_list[:10]}")
print(f"Example targets: {target_list[:10]}")

# ============================================================================
# STEP 4: PREPARE AND SAVE INPUT FILES
# ============================================================================

print("\n" + "="*70)
print("Save files for SimiC")

# Combine and subset genes
selected_genes = tf_list + target_list

if isinstance(imputed_data, ad.AnnData):
    subset_data = imputed_data[:, selected_genes].copy()
else:
    subset_data = imputed_data[selected_genes].copy()

print(f"Subset data shape: {subset_data.shape}")

# Save experiment files
print("\nSaving experiment files...")
experiment.save_experiment_files(
    run_data=subset_data,
    matrix_filename='expression_matrix.pickle',
    tf_filename='TF_list.pickle',
    annotation=ANNOTATION_COLUMN
)

print(experiment.print_project_info())

print("\n" + "="*70)
print("✓ PREPROCESSING COMPLETE!")
print("\n" + "="*70)
