# South Park Dialogue Speaker Classification

<p align="center">
  <img src="assets/south-park.webp" alt="South Park illustration" width="700", height="240">
</p>


This repository is a multi-class NLP project focused on predicting **which South Park character spoke a given line of dialogue**. The work compares lightweight baselines with transformer-based, and pre-trained language-model approaches for speaker classification on an imbalanced dialogue dataset.

## Objective

- Build and compare models for **speaker identification from dialogue text**.
- Measure the trade-off between simple baselines and stronger transformer models.
- Evaluate training strategies for imbalanced multi-class classification, including weighted cross-entropy, label smoothing, and focal loss.

## Data

The raw data is loaded via `utils/data_utils.py` from the public South Park dialogue dataset hosted on GitHub.

The preparation workflow expects at least the following columns:

- `Character`
- `Line`

`02 - Data Load & Processing/01 - load_and_preprocess_data.ipynb` handles the main preprocessing pipeline:

- removing duplicate dialogue lines
- cleaning whitespace and line breaks
- removing empty lines
- selecting target characters
- creating train / validation / test splits
- generating character-to-label mappings
- exporting parquet files for downstream modeling notebooks

`02 - Data Load & Processing/02 - exploratory_data_analysis.ipynb`. 

This notebook summarizes the dataset, reviews season-level distributions, and analyzes character-level coverage using `num_lines`, `num_episodes`, and `num_seasons` to support character-selection decisions.

Prepared inputs are already included in `01 - Inputs/`, including:

- raw and filtered parquet datasets
- train / validation / test splits
- numeric-label variants of those splits
- `char_to_label` and `label_to_char` mappings in JSON and pickle format

## Repository Layout

- `01 - Inputs/` — prepared datasets, split files, and label mappings
- `02 - Data Load & Processing/` — data loading, preprocessing, and exploratory analysis notebooks
- `03 - Model Building/` — training notebooks, scripts, and checkpoints
- `04 - Model Evaluation and Assessment/` — reserved for evaluation material; 
- `05 - Outputs/` — saved experiment outputs and zipped artifacts
- `utils/` — shared utilities for datasets, preprocessing, training, loss functions, and evaluation

## Modeling Approaches

### Baseline Methods

- **Zero-shot BART** (`zero-shot BART.ipynb`)
  Uses Hugging Face zero-shot classification as a non-fine-tuned baseline.

- **Multi Layer Perceptron (MLP)**(`baseline_logistic_regression.ipynb`)
  Tokenization + vocabulary lookup + trainable embeddings + mean pooling + linear classification.

  Used with zero hidden layers acts as a baseline logistic regression model.

- **fastText model** (`fast text models.ipynb`)
  Uses supervised fastText, including autotuning.
    Developed by Facebook, fastText is a library for learning of fast text representations and efficient text classification.

### From-scratch transformer models


- **Custom BERT-style transformer** (`transformer_bert_model.ipynb`)
  - Transformer encoder trained from scratch for classification.
  - It uses the BERT architecture but with a smaller number of heads and layers.

### Fine-tuning pre-trained models

- **Qwen** (`qwen pre-trained fine-tuning.ipynb`, `qwen pre-trained fine-tuning f1_for_early_stopping.ipynb`)
  Fine-tunes the final layer of the 0.6B parameter Qwen model on the South Park dataset.

- **DistilBERT fine-tuning** (`transformer_distilbert_model.ipynb`, `transformer_distilbert_model_val_loss.ipynb`)
  - Fines-tunes a pre-trained Hugging Face DistilBERT classifier.
  - Here we are doing a full fine-tuning of the model as the model is smaller and faster to train.



## Setup

The project includes `requirements.txt` built from dependencies that are used in the notebooks and utility modules.

### Recommended environment

- Python **3.11.x** is the safest starting point. 
- A GPU is recommended for transformer and larger-model notebooks.

### Installation
python environment setup:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

conda environment setup:
```bash
conda create -n sp python=3.11
conda activate sp
conda install -c conda-forge --file requirements.txt
```

### Optional extras

- `03 - Model Building/fast text models.ipynb`

```bash
pip install fasttext-wheel
```

- Parquet files are used throughout the project via pandas, but the repository does not explicitly pin a parquet backend such as `pyarrow` or `fastparquet`. If parquet I/O fails in a fresh environment, install the backend appropriate for your setup.

## Running the Project

### Recommended workflow

1. **Prepare or refresh the data**
   - Run `02 - Data Load & Processing/01 - load_and_preprocess_data.ipynb`

2. **Review the exploratory data analysis**
   - Run `02 - Data Load & Processing/02 - exploratory_data_analysis.ipynb`
   - Use it to inspect dataset structure, season-level summaries, and top-K character coverage before training

3. **Run a modeling notebook** from `03 - Model Building/`, for example:
   - `baseline_logistic_regression.ipynb`
   - `transformer_bert_model.ipynb`
   - `transformer_distilbert_model.ipynb`
   - `fast text models.ipynb`
   - `zero-shot BART.ipynb`

### Execution note

- Most notebooks rely on relative paths such as `../01 - Inputs` and `../utils`.
- Open and run the notebooks from their existing project locations so imports and file paths resolve correctly.
- Some notebooks include commented Google Colab setup cells; use or ignore them depending on your environment.

## Evaluation and Outputs

Evaluation helpers are implemented in `utils/evaluate_models.py`.

Common outputs across notebooks include:

- accuracy, precision, recall, and macro F1-score
- confusion matrices and classification reports
- saved checkpoints (`.pt`)
- training history plots
- experiment artifacts stored in `05 - Outputs/`

## Dependencies and Environment Notes

Core libraries listed in `requirements.txt` include:

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `tqdm`
- `torch`
- `transformers`, `datasets`

Optional notebook-specific dependencies confirmed by repository imports include:

- `fasttext-wheel`


Internet access may be required:

- to fetch the raw South Park data
- to download Hugging Face models on first use



