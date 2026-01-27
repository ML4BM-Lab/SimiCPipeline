from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
# Install packages

def install_package(package_name):
   """Install a package using pip3."""
   import subprocess
   import sys
   subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def load_from_anndata(
    path: Union[str, Path],
) -> object:
    """
    Load an AnnData file (.h5ad) and return a DataFrame (cells x genes) with optional obs/var metadata.

    Args:
        path: Path to the .h5ad file.
        cells_index_name: Column name for cells index when returned as DataFrame.

    Returns:
        adata: AnnData object loaded from the file.
    """
    try:
        import anndata as ad  # type: ignore
    except Exception as e:
        raise ImportError("anndata is required to load .h5ad files") from e

    path = Path(path)
    adata = ad.read_h5ad(str(path))
    if adata.raw == None:
        print("No raw attribute found in AnnData object.")
        
    return adata

def load_from_matrix_market(
    matrix_path: Union[str, Path],
    genes_path: Union[str, Path] = None,
    cells_path: Union[str, Path] = None,
    transpose: bool = False,
    cells_index_name: str = "Cell",
) -> object:
    """
    Load a Matrix Market file (.mtx) and genes/cells TSV and return a pandas DataFrame.

    Args:
        matrix_path: Path to matrix.mtx(.gz) or .txt file with cells x genes counts.
        genes_path: Optional path to genes/features list (TSV with first column as names).
        cells_path: Optional path to cells/barcodes list (TSV with first column as names).
        transpose: If True, transpose the matrix (use when input mtx is genes x cells).
        cells_index_name: Index name for cells rows.

    Returns:
        df: Expression matrix DataFrame (cells as rows, genes as columns).
    """
    try: 
        import scipy
    except Exception as e:
        raise ImportError("scipy is required to load Matrix Market files") from e

    matrix_path = Path(matrix_path)

    mtx = scipy.io.mmread(str(matrix_path))
    
    if scipy.sparse.issparse(mtx):
        mat = mtx.tocsr()
    else:
        mat = scipy.sparse.csr_matrix(mtx)

    if transpose:
        mat = mat.T
    # Load names
    genes = None
    cells = None
    if genes_path and genes_path.exists():
        genes = pd.read_csv(genes_path, header=None, sep="\t").iloc[:, 0].astype(str).tolist()
    if cells_path and cells_path.exists():
        cells = pd.read_csv(cells_path, header=None, sep="\t").iloc[:, 0].astype(str).tolist()

    # Convert to dense for simplicity; adjust if large matrices are expected
    arr = mat.toarray()
    df = pd.DataFrame(arr)

    # Assign indices/columns when provided
    if cells is not None and len(cells) == df.shape[0]:
        df.index = cells
    else:
        df.index = [f"cell_{i}" for i in range(df.shape[0])]
    df.index.name = cells_index_name

    if genes is not None and len(genes) == df.shape[1]:
        df.columns = genes
    else:
        df.columns = [f"gene_{j}" for j in range(df.shape[1])]

    return df


def write_pickle(obj, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)
        print(f"Pickle method succeeded, saved to {file_path}")
    except OverflowError as e:
        print(f"Pickle failed due to OverflowError: {e}")
    except Exception as e:
        print(f"Pickle failed with error: {e}")

#### Functions from https://github.com/jianhao2016/SimiC/code/simiclasso/common_io.py #########
# def preprocessing_expression_mat(X_raw):
#     '''
#     Perform log2 and Z-transform on raw input expression matrix
#     '''
#     # X = np.log2(X_raw + 1)
#     # columns standarization
#     # X = (X - np.mean(X, axis = 0))/(np.std(X, axis = 0) + 1e-5)
#     X = X_raw
#     return X

# def extract_df_columns(df, feature_cols):
#     '''
#     Find matching columns in df (case-insensitive), and return extracted DataFrame.
#     '''
#     lower_case_fc = [x.lower() for x in feature_cols]
#     match_idx = df.columns.str.lower().isin(lower_case_fc)
#     extracted_df = df.loc[:, match_idx]
#     return extracted_df


# def load_dataFrame(df_file, feature_cols, df_with_label = True):
#     '''
#     Load a pandas DataFrame from disk and extract expression matrix and labels if available.
#     Args:
#         df_file: Path to pickled DataFrame
#         feature_cols: List of feature column names to extract
#         df_with_label: Whether the DataFrame contains a 'label' column

#     Returns:
#         X: expression matrix, shape (n, m)
#             n: number of cells,
#             m: number of genes [len of feature columns]
#         Y: labels of cells (or None if df_with_label=False).
#         new_feature_cols: List of actual column names extracted
#     '''
#     df = pd.read_pickle(df_file)
#     X_raw_df = extract_df_columns(df, feature_cols)

#     new_feature_cols = X_raw_df.columns.values.tolist()
#     X = X_raw_df.values
#     Y = df['label'].values if df_with_label else None

#     print('Done loading, shape of X = ', X.shape)
#     return X, Y, new_feature_cols


# def load_gene_interaction_net(p2f, p2feat_cols, feature_cols):
#     '''
#     load the gene interaction network, sample the overlapping value between feature_cols
#     convert to a networkx class.
#     input:
#         p2f: path to gene gene interaction network
#         p2feat_col: path to feature column file of ggn
#         feature_cols: feature columns [list of string] in expression matrix X
#     output:
#         a adjacency matrix, GGN
#         feature/genes that are both in the network and in input feature_cols
#     -------
#     to construct a sparse matrix, we need:
#         C: values of [x, y]
#         A: coordinate of x
#         B: coordinate of y
#     mat = sparse.coo_matrix((C, (A, B)), shape = (n, n))
#     '''
#     # A, feat_cols = random_adjacency_matrix(feature_cols)

#     ggn_df = pd.read_csv(p2f, compression='gzip', error_bad_lines=False)
#     ggn_genes = pd.read_csv(p2feat_cols)
#     print('GGN includes {} genes'.format(len(ggn_genes)))

#     edge_list = ggn_df[['protein1', 'protein2']].values
#     weight_list = ggn_df['combined_score'].values

#     # node to index dictionary. Contains every genes in both experssion matrix 
#     # and in GGN network. values of each gene key will be its index on output list.
#     nodes_idx_dict = {}
#     output_gene_list = []
#     idx = 0  # start index of gene list. Add one each time get a new gene.
#     x_coord_list = []
#     y_coord_list = []
#     mat_val_list = []

#     gene_set_in_X = set(feature_cols)

#     for edge, weight in zip(edge_list, weight_list):
#         node_1, node_2 = edge
#         # every nodes should be in the gene list of expression matrix
#         if (node_1 in gene_set_in_X) & (node_2 in gene_set_in_X):
#             # both genes are included, add val of adjacency.
#             mat_val_list.append(weight)

#             # add x coordinate
#             if node_1 in nodes_idx_dict:
#                 tmp_x = nodes_idx_dict[node_1]
#             else:
#                 # node_1 is not in dictionary.
#                 # new gene, add to dict, update idx, and output list
#                 nodes_idx_dict[node_1] = idx
#                 tmp_x = idx
#                 idx += 1
#                 output_gene_list.append(node_1)
#             x_coord_list.append(tmp_x)

#             # add y coordinate
#             if node_2 in nodes_idx_dict:
#                 # gene already in dict
#                 # no need to update output list.
#                 tmp_y = nodes_idx_dict[node_2]
#             else:
#                 # same argument
#                 nodes_idx_dict[node_2] = idx
#                 tmp_y = idx
#                 idx += 1
#                 output_gene_list.append(node_2)
#             y_coord_list.append(tmp_y)

#             assert len(nodes_idx_dict) == idx

#     ggn_sparse = coo_matrix((mat_val_list,(x_coord_list,y_coord_list)),
#             shape=(idx,idx))
#     ggn_sparse += ggn_sparse.T
#     # convert back to dense adjacency matrix
#     GGN_dense = ggn_sparse.toarray()
#     print('GGN generated, size of matrix = {}'.format((idx, idx)))
    
#     return GGN_dense, output_gene_list

def split_df_and_assignment(df_in, assignment, test_proportion = 0.2):
    num_of_cells = len(assignment)
    size_of_test_set = int(num_of_cells * test_proportion)
    random_perm = np.random.RandomState(seed=1).permutation(num_of_cells)
    test_idx = random_perm[:size_of_test_set]
    train_idx = random_perm[size_of_test_set:]

    train_df = df_in.loc[train_idx]
    train_assign = assignment[train_idx]

    test_df = df_in.loc[test_idx]
    test_assign = assignment[test_idx]
    
    return train_df, test_df, train_assign, test_assign
    


# # below are basically useless
# def change_expression_matrix_order(X_df, expect_order):
#     '''
#     change columns in X so that its order is same as expect_order
#     return new order X.
#     input:
#         X: origin expression matrix
#         gene_order_in_X: gene order in original matrix
#         expect_order: expect gene order after change
#     output:
#         X_new: same dimension as X, has gene in order expect_order
#     '''
#     # perm = get_str_permutation(gene_order_in_X, expect_order)
#     X_new = X_df[expect_order].values
#     feat_cols_new = expect_order
#     return X_new, feat_cols_new
    

# def get_gene_list_order_in_int(gene_order):
#     '''
#     convert a string gene list to range(len(gene list)).
#     and provide a dictionary, in 
#     gene_2_int_dict = {'gene1':index}
#     for fast search
#     '''
#     len_of_gene = len(gene_order)
#     gene_order_int = np.arange(len_of_gene)
#     gene_2_int_dict = {gene_name:idx for gene_name, idx in zip(gene_order, gene_order_int)}
#     return gene_order_int, gene_2_int_dict


# def get_int_order_wrt_dict(gene_list, gene_2_int_dict):
#     '''
#     given some gene list in [string], and 
#     a dictionary of form {'gene name': idx}
#     convert the list of gene to list of integer
#     '''
#     gene_list_len = len(gene_list)
#     new_list = np.zeros(gene_list_len)
#     pass
