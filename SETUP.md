# PhageHostLearn - Setup Guide

### 1. Installer Python 3.9 avec pyenv

```bash
pyenv install 3.9.18
```

### 2. Configure Python 3.9 for this project

```bash
cd /path/to/PhageHostLearn
pyenv local 3.9.18
python --version  # Verify that it's Python 3.9.18
```

### 3. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install base dependencies

```bash
pip install numpy pandas biopython xgboost scikit-learn matplotlib seaborn joblib tqdm ipykernel
```

### 5. Install bio-embeddings and its dependencies

```bash
# Install bio-embeddings without dependencies to avoid conflicts
pip install --no-deps bio-embeddings transformers torch

# Install required dependencies
pip install h5py "ruamel.yaml>=0.17.10,<0.18.0" plotly umap-learn
pip install appdirs atomicwrites gensim humanize "lock>=2018.3.25,<2019.0.0" python-slugify
pip install regex requests safetensors tokenizers jinja2 networkx sympy
pip install "huggingface-hub>=0.34.0,<1.0"
```

### 6. Install fair-esm (ESM-2)

```bash
pip install fair-esm
```

### 7. Register Jupyter kernel

```bash
python -m ipykernel install --user --name=phagehostlearn --display-name "Python (phagehostlearn)"
```

### 8. Verify installation

```bash
cd code
python -c "import phagehostlearn_processing as phlp; print('Import successful')"
```

## Installation complète en une commande

For a quick installation, you can run all commands at once:

```bash
cd /path/to/PhageHostLearn
pyenv local 3.9.18
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas biopython xgboost scikit-learn matplotlib seaborn joblib tqdm ipykernel
pip install --no-deps bio-embeddings transformers torch
pip install h5py "ruamel.yaml>=0.17.10,<0.18.0" plotly umap-learn
pip install appdirs atomicwrites gensim humanize "lock>=2018.3.25,<2019.0.0" python-slugify
pip install regex requests safetensors tokenizers jinja2 networkx sympy
pip install "huggingface-hub>=0.34.0,<1.0"
pip install fair-esm
python -m ipykernel install --user --name=phagehostlearn --display-name "Python (phagehostlearn)"
```