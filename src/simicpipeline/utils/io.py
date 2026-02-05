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

# Directories


def print_tree(directory: Union[str,Path] = Path('.'),
               prefix: str = "", 
               max_depth:int= 1,
               current_depth: int = 0) -> None:
    """
    Print a tree-like directory structure.
    
    Args:
        directory: Path to the directory to visualize
        prefix: Prefix for tree branches (used internally for recursion)
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current recursion depth (used internally)
    
    Example:
        >>> print_tree(Path("src"))
        src/
        ├── simicpipeline/
        │   ├── __init__.py
        │   ├── core/
        │   │   ├── __init__.py
        │   │   ├── main.py
        │   └── utils/
        │       └── io.py
    """
        #  Print the root directory name at the top
    directory = Path(directory)
    if current_depth == 0:
        print(f"{directory.name}/")
    if max_depth is not None and current_depth >= max_depth:
        return
    
    # Get all items in directory
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}{'/' if item.is_dir() else ''}")
        
        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension, max_depth, current_depth + 1)

# Loaders
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

# Writers
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


@staticmethod
def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted time string (e.g., "1h 30min 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}min {secs}s"
    elif minutes > 0:
        return f"{minutes}min {secs}s"
    else:
        return f"{secs}s"

###########################################################################################
