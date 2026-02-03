#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Pipeline

This script runs the SimiC analysis pipeline from the tutorial notebook.
Modify the configuration paths below and run: python run_simicpipeline.py

Author: Irene Marín-Goñi
"""

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# ============================================================================

# Working directory
WORKDIR = '/path/to/your/working/directory'

# Project directory (where input files are located)
PROJECT_DIR = './SimiCExampleRun'

# Unique identifier for this analysis run
RUN_NAME = 'experiment1'

# Input file paths
INPUT_MATRIX = PROJECT_DIR + '/inputFiles/expression_matrix.pickle'
INPUT_ASSIGNMENT = PROJECT_DIR + '/inputFiles/treatment_annotation.txt'
INPUT_TF_LIST = PROJECT_DIR + '/inputFiles/TF_list.pickle'

# SimiC Parameters
LAMBDA1 = 1e-1  # L1 regularization (sparsity)
LAMBDA2 = 1e-2  # L2 regularization (similarity across phenotypes)
SIMILARITY = True  # Enable similarity constraint
MAX_RCD_ITER = 5000  # Maximum iterations for coordinate descent
CROSVAL = True # Enable cross-validation for parameter tuning
CROSVAL_FOLDS = 4  # Number of folds for cross-validation
LIST_L1 = [1e-2, 1e-1]
LIST_L2 = [ 1e-3, 1e-2]

# AUC Calculation Parameters
ADJ_R2_THRESHOLD = 0.7  # Minimum R² for target genes
NUM_CORES = -2  # Use all but 1 CPU cores


# ============================================================================
# IMPORTS
# ============================================================================

import os
import simicpipeline
from simicpipeline import SimiCPipeline

print(f"SimiCPipeline version: {simicpipeline.__version__}")

# Change to working directory
os.chdir(WORKDIR)
print(f"Working directory: {os.getcwd()}")

# ============================================================================
# STEP 1: INITIALIZE PIPELINE
# ============================================================================

print("\n" + "="*70 +"\n")
pipeline = SimiCPipeline(
    project_dir=PROJECT_DIR,
    run_name=RUN_NAME
)

print(f"Pipeline initialized with workdir: {pipeline.project_dir}")

# ============================================================================
# STEP 2: SET INPUT FILE PATHS
# ============================================================================

pipeline.set_input_paths(
    p2df=INPUT_MATRIX,
    p2assignment=INPUT_ASSIGNMENT,
    p2tf=INPUT_TF_LIST
)

print("Input paths set successfully\n")

# ============================================================================
# STEP 3: SET PARAMETERS
# ============================================================================


pipeline.set_parameters(
    lambda1=LAMBDA1,
    lambda2=LAMBDA2,
    similarity=SIMILARITY,
    max_rcd_iter=MAX_RCD_ITER,
    cross_val=False  # Set to True for cross-validation
)

print(f"Lambda1 (L1 sparsity): {LAMBDA1}\n")
print(f"Lambda2 (L2 similarity): {LAMBDA2}\n")
print(f"Similarity constraint: {SIMILARITY}\n")
print(f"Max iterations: {MAX_RCD_ITER}\n")

# ============================================================================
# STEP 4: VALIDATE INPUTS
# ============================================================================

print("\n" + "="*70)
print("Validate input")

pipeline.validate_inputs()

# ============================================================================
# STEP 5: RUN PIPELINE
# ============================================================================

print("\n RUNNING SimiC PIPELINE")
print("="*70)
print("\nThis may take several minutes depending on data size...\n")

# AUC parameters
auc_params = {
    'adj_r2_threshold': ADJ_R2_THRESHOLD,
    'select_top_k_targets': None,
    'percent_of_target': 1,
    'sort_by': 'expression',
    'num_cores': NUM_CORES
}

# Run the complete pipeline
pipeline.run_pipeline(
    skip_filtering=False,
    calculate_raw_auc=False,
    calculate_filtered_auc=True,
    variance_threshold=0.9,
    auc_params=auc_params
)
print("\nSimiC pipeline run complete!")
# ============================================================================
# STEP 6: CHECK AVAILABLE RESULTS
# ============================================================================

print("\n" + "="*70)
pipeline.available_results()

print("\n" + "="*70)
print(f"\nOutput files saved to:")
print(f"  Matrices: {PROJECT_DIR}/outputSimic/matrices/{RUN_NAME}")

print(pipeline.print_project_info())

# ============================================================================
# SUCCESS MESSAGE
# ============================================================================
print("\n" + "="*70)
print("✓ SimiC PIPELINE COMPLETE!")
print("\n" + "="*70)
# ============================================================================
# SUMMARY
# ============================================================================

