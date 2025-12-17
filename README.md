# SimiCPipeline

A Python package for running SimiC, a single-cell gene regulatory network inference method that jointly infers distinct, but related, gene regulatory dynamics per phenotype class.

## Installation

### Option 1: Installing with Poetry (Local)

1. **Install Poetry**

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

2. **Install the Package**

```bash
git clone https://github.com/irenemaring/SimiCPipeline.git
cd SimiCPipeline
poetry install
```
3.a **Start python in poetry environment**
```bash
poetry run python
```
3.b **Install Jupyter Kernel** (useful if working with Jupyter notebooks)

```bash
poetry run python -m ipykernel install --user --name simicpipeline --display-name "SimiC Pipeline"
```

### Option 2: Using Docker

1. **Build the Docker Image**

```bash
docker build -t irenemaring/simicpipeline:poetry .
```

2. **Run the Container**

```bash
docker run -dt -p 8888:8888 --cpuset-cpus=0-32 -v <path-to-host-directory>:/home/workdir --name <your-container-name> irenemaring/simicpipeline:poetry
```

3. **Access the Container**

```bash
docker exec -it your-container-name bash
```
4.a **Start python in poetry environment**
```bash
poetry -P /home/SimiCPipeline poetry run python
```

4.b **Start Jupyter Kernel (Inside Container)**

```bash
poetry -P /home/SimiCPipeline poetry run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Connecting to Jupyter

**For VS Code:**
1. Install the Jupyter extension
2. Open a notebook
3. Click "Select Kernel" in the top right
4. Choose "SimiC Pipeline" from the list

**For Browser:**
1. Inside the container, start Jupyter:
   ```bash
   poetry run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```
2. Copy the token from the output
3. Open your browser to `http://localhost:8888` and paste the token
4. Select the "SimiC Pipeline" kernel when creating a new notebook

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
    t=3,
    knn=5,
    save_data=True
)
```

### Example: SimiC Analysis

```python
from simicpipeline import SimiCPipeline

# Initialize pipeline
simic = SimiCPipeline(
    workdir="./my_project",
    run_name="experiment_1",
    n_tfs=100,
    n_targets=1000
)

# Set input paths
simic.set_paths(
    p2df="./my_project/magic_output/magic_data.pickle",
    p2assignment="./my_project/inputFiles/phenotype_assignment.txt",
    p2tf="./data/Mus_musculus_TF.txt"
)

# Set parameters and run
simic.set_parameters(
    lambda1=1e-2,
    lambda2=1e-5,
    similarity=True,
    max_rcd_iter=500000
)

simic.run_pipeline(
    skip_filtering=False,
    calculate_raw_auc=False,
    calculate_filtered_auc=True
)
```

## Citation

If you use SimiCPipeline in your research, please cite:

```bibtex
@software{simicpipeline,
  author = {Marín-Goñi, Irene},
  title = {SimiCPipeline: A Python Package for SimiC Analysis},
  year = {2025},
  url = {https://github.com/irenemaring/SimiCPipeline}
}
```

## Contact

Irene Marín-Goñi - imarin.4@alumni.unav.es

Project Link: [https://github.com/irenemaring/SimiCPipeline](https://github.com/irenemaring/SimiCPipeline)