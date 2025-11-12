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

### 6. Install PHANOTATE

PHANOTATE is required for phage gene calling. Install it inside the virtual environment so the notebook can invoke it directly.

```bash
pip install phanotate
which phanotate.py  
```

### 7. Install fair-esm (ESM-2)

```bash
pip install fair-esm
```

### 8. Install BLAST+

BLAST+ is required for Kaptive to process bacterial genomes.

**macOS (Homebrew):**
```bash
brew install blast
```

**Linux:**
```bash
sudo apt-get install ncbi-blast+
# or
sudo yum install ncbi-blast+
```

Verify installation:
```bash
which makeblastdb
which tblastn
```

### 9. Download Kaptive reference database

Download the Klebsiella K-locus reference database file required for Kaptive:

```bash
cd /Users/eliottvalette/Documents/Clones/PhageHostLearn/data

# Télécharger le fichier directement depuis GitHub
curl -L -o Klebsiella_k_locus_primary_reference.gbk \
  https://raw.githubusercontent.com/klebgenomics/Kaptive/master/reference_database/Klebsiella_k_locus_primary_reference.gbk
```

Verify the file was downloaded:
```bash
ls -lh Klebsiella_k_locus_primary_reference.gbk
```

### 10. Register Jupyter kerne

```bash
python -m ipykernel install --user --name=phagehostlearn --display-name "Python (phagehostlearn)"
```

### 11. Verify installation

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
pip install phanotate
pip install fair-esm
brew install blast  # macOS only, use apt-get/yum on Linux
cd data && curl -L -o Klebsiella_k_locus_primary_reference.gbk \
  https://raw.githubusercontent.com/klebgenomics/Kaptive/master/reference_database/Klebsiella_k_locus_primary_reference.gbk
python -m ipykernel install --user --name=phagehostlearn --display-name "Python (phagehostlearn)"
```