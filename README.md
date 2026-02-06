# SimiCPipeline

A Python package for running SimiC, a single-cell gene regulatory network inference method that jointly infers distinct, but related, gene regulatory dynamics per phenotype class.

## Installation

This package has been developed with [Poetry](https://python-poetry.org/docs/), a tool for dependency management and packaging in Python. You can install `simicpipeline` locally (need ot install Poetry as well) or use the docker image that has Poetry and the package already installed.

### Option 1: Installing with Poetry (Local)

#### 1. Install Poetry

**Linux, macOS, Windows (WSL):**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Add Poetry to your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 2. Download this repo and install simicpipeline
Poetry will manage all the dependencies

```bash
git clone https://github.com/ML4BM-Lab/SimiCPipeline.git
cd SimiCPipeline
poetry install
```
#### 3.a Start python in poetry environment
```bash
poetry run python
```
#### 3.b Install Jupyter Kernel (useful if working with Jupyter notebooks)

```bash
poetry run jupyter-simic
```

### Option 2: Using Docker

For instructions on how to install Docker Engine for Ubuntu refer to [this page](https://docs.docker.com/engine/install) or Docker Desktop for Windows refer to [this page](https://docs.docker.com/desktop/setup/install/windows-install/).

#### 1. Build the Docker Image

Download the [Dockerfile](./Dockerfile) form this repo and build the image.
*Note you can tag (*`-t`*) the image with any name*

```bash
docker build -t ml4bm/simicpipeline:poetry .
```
*To update your docker with the latest repo commit you can re-build with `--no-cache-filter simic`*
```
docker build --no-cache-filter simic -t ml4bm/simicpipeline:poetry .
```

#### 2. Run the Container

- To work with jupyter notebooks within VS Code or in browser you will need to expose the port wiht `-p`. 
- `--cpuset-cpus` should be set in accordance to your local machine capabilities and the available cpus you want to assign to the docker container.

```bash
docker run -dt -p 8888:8888 --cpuset-cpus=0-32 -v <path-to-host-directory>:/home/workdir --workdir=/home/workdir --name <your-container-name> ml4bm/simicpipeline:poetry
```

#### 3. Access the Container in interactive mode

```bash
docker exec -it <your-container-name> bash
```
#### 4.a Start python in poetry environment
```bash
poetry -P /home/SimiCPipeline run python
```

#### 4.b Start Jupyter Kernel (Inside Container)

```bash
poetry -P /home/SimiCPipeline run jupyter-simic
```

### Connecting to Jupyter Server/KErnel

1. Start the jupyter kernel as indicated above
2. In terminal output look for the message:

        "Jupyter Server 2.17.0 is running at: 
            http://127.0.0.1:8888/tree"
3. Copy the url direction


**For VS Code:**

4. Install the Jupyter extension
5. Open a notebook file (.ipynb)
6. On the top right corner click "Select Kernel" in the top right
7. Choose "Existing Jupyter Server" and paste de url
8. Select "SimiC Pipeline Kernel" from the list

**For Browser:**

4. Open your browser to `http://localhost:8888` and paste the token.

5. Select the "SimiC Pipeline" kernel when creating a new notebook

*If that localhost:8888 does not work look for the `PORTS` tab in the VS Code terminal **--> Fordward Address --> Select "Open in Browser"*** 

## Quick Start

### Example: MAGIC Preprocessing

```python
import anndata as ad
from simicpipeline import MagicPipeline

# Load your data
adata = ad.read_h5ad("path/to/your/data.h5ad")

# Initialize and run pipeline
pipeline = MagicPipeline(
    input_data=adata,
    project_dir="./my_project",
    filtered=False
)

pipeline.filter_cells_and_genes(
    min_cells_per_gene=10,
    min_umis_per_cell=500
).normalize_data().run_magic(
    random_state=123,
    n_jobs=-2,# Use all but 1 CPU cores
    save_data=True
)
imputed_data = magic_pipeline.magic_adata.copy()
# Set up files for SimiC
from simicpipeline import ExperimentSetup
experiment = ExperimentSetup(
    input_data = imputed_data, 
    tf_path = "./data/TF_list.csv", # Should have no header
    project_dir='./my_project'
)
tf_list, target_list = experiment.calculate_mad_genes(
    n_tfs=100,
    n_targets=1000
)

selected_genes = tf_list + target_list
subset_data = imputed_data[:, selected_genes].copy()
experiment.save_experiment_files(
    run_data = subset_data,
    matrix_filename = 'expression_matrix.csv',
    tf_filename = 'TF_list.csv',
    annotation = 'groups' # should be in subset_data.obs_names
)
experiment.print_project_info(max_depth=2)
```

### Example: SimiC Analysis

```python
from simicpipeline import SimiCPipeline

# Initialize pipeline
simic = SimiCPipeline(
    project_dir="./my_project",
    run_name="experiment_1"
)

# Set input paths
simic.set_input_paths(
    p2df="./my_project/magic_output/magic_data.pickle",
    p2assignment="./my_project/inputFiles/phenotype_assignment.txt",
    p2tf="./data/TF_list.csv"
)

# Set parameters and run
simic.set_parameters(
    lambda1=1e-2,
    lambda2=1e-5,
    similarity=True,
    max_rcd_iter=500000,
    cross_val=True,
    k_cross_val=4,
    max_rcd_iter_cv=5000,
    list_of_l1=[1e-1, 1e-2], 
    list_of_l2=[1e-2, 1e-3]
)
auc_params = {
    'adj_r2_threshold': 0.7,
    'select_top_k_targets': None,
    'percent_of_target': 1,
    'sort_by': 'expression',
    'num_cores': -2
}

simic.run_pipeline(
    skip_filtering=False,
    calculate_raw_auc=False,
    calculate_filtered_auc=True,
    variance_threshold=0.9,
    auc_params=auc_params
)
simic.print_project_info(max_depth=3)
```

## Citation

If you use SimiCPipeline in your research, please cite:

```bibtex
@software{simicpipeline,
  author = {Marín-Goñi, Irene, ML4BM- Lab},
  title = {SimiCPipeline: A Python Package for SimiC, a single-cell gene regulatory network inference framework.},
  year = {2025},
  url = {https://github.com/ML4BM-Lab/SimiCPipeline}
}
```

## Contact

Machine Learning 4 Biomedicine group - https://mikelhernaez.github.io/ - mhernaez at unav dot es
Irene Marín-Goñi - imarindot 4 at alumni dot unav dot es

Project Link: [https://github.com/ML4BM-Lab/SimiCPipeline](https://github.com/ML4BM-Lab/SimiCPipeline)