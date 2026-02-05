# NLP CA4 – Question 1: Advanced NLP Modeling and Analysis

This repository contains **Course Assignment 4 – Question 1 (CA4 Q1)** for the **Natural Language Processing (NLP)** course.  
The assignment focuses on **advanced NLP modeling, training, and evaluation** using Python and Jupyter Notebook, with an emphasis on:

- Data loading and preprocessing
- Model design and training
- Quantitative evaluation with appropriate metrics
- Qualitative analysis of model behavior and errors

All implementations, experiments, explanations, and results are provided in a single Jupyter Notebook.

---

## Repository Structure

```
CA4/
├── NLP_CA4_Q1_Kashani_810101490.ipynb   # Main notebook (implementation + experiments + analysis)
└── README.md                           # This file
```

---

## Setup and Requirements

### Software
- Python 3.9+ recommended
- Jupyter Notebook / JupyterLab

### Python packages

Depending on the notebook, the following packages are typically used:
- `numpy`
- `pandas`
- `torch` (PyTorch) or `tensorflow` / `keras`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `re`, `collections`, `math` (standard library)

Install example:
```bash
pip install numpy pandas torch scikit-learn tqdm matplotlib
```

---

## How to Run

1. Make sure the folder structure matches the one shown above.
2. Open the notebook:
```bash
jupyter notebook NLP_CA4_Q1_Kashani_810101490.ipynb
```
3. Run the cells **top-to-bottom** to reproduce all experiments and results.

---

## Problem Statement (Q1)

In Question 1 of CA4, the goal is to **design and evaluate an NLP model** for a specific task defined in the assignment (e.g., classification, sequence modeling, representation learning, or similar).

The notebook covers:
- A clear definition of the task
- The chosen modeling approach
- The training and evaluation protocol
- A discussion of results and limitations

---

## Data Processing

Typical data processing steps include:

- Loading raw data from file(s)
- Cleaning and normalizing text
- Tokenization
- Building vocabularies or using pre-trained tokenizers
- Converting text into numerical representations (indices, embeddings, etc.)
- Creating training / validation / test splits (if applicable)

---

## Model Architecture

Depending on the assignment requirements, the model may include:

- Embedding layers (randomly initialized or pre-trained)
- Sequence models (e.g., RNN, LSTM, GRU) or Transformer-based components
- Feed-forward layers for prediction
- Appropriate output layers for the task (e.g., softmax for classification)

The notebook documents:
- The full architecture
- The role of each component
- The reasoning behind design choices

---

## Training Procedure

The training pipeline typically includes:

- Mini-batch training
- Loss computation (e.g., cross-entropy)
- Optimization using gradient-based methods (e.g., Adam, SGD)
- Periodic evaluation on a validation set
- Logging of loss and metrics over time

---

## Evaluation

Model performance is evaluated using task-appropriate metrics, such as:

- Accuracy / Precision / Recall / F1-score (for classification tasks)
- Loss or perplexity (for language modeling tasks)
- Confusion matrix or error breakdown (if applicable)

---

## Outputs

Running the notebook produces:

- Training and validation logs
- Final evaluation metrics
- (Optional) Plots of loss and performance curves
- Example predictions and error analysis

---

## Implementation Notes

- The entire experiment is **reproducible** by running the notebook from top to bottom.
- Paths to data files are relative to the notebook directory.
- The focus is on **clarity, correctness, and analysis**, not only raw performance.

---

## Troubleshooting

### `FileNotFoundError`
- Make sure any required dataset files are placed in the correct directory.
- Check and update file paths in the notebook if necessary.

### `ModuleNotFoundError`
Install missing packages, for example:
```bash
pip install torch scikit-learn pandas numpy tqdm
```

### Slow training
- Reduce model size
- Reduce number of epochs
- Use CPU-friendly settings if no GPU is available

---

Suggested improvements (optional):
- Add hyperparameter search or ablation studies
- Add more detailed plots and learning curves
- Try alternative model architectures
- Add more qualitative error analysis examples
