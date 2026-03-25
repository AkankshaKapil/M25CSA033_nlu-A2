# CSL7640 – Assignment 2

This repository contains solutions for both **Problem 1 (Word2Vec)** and **Problem 2 (Character-level Name Generation)**.

---

## 📁 File Structure

### 🔹 Problem 1: Word2Vec (IIT Jodhpur Corpus)

- **Problem1.py**  
  Implementation of Word2Vec (CBOW and Skip-gram with Negative Sampling) **from scratch using NumPy**. Includes training, evaluation, and visualizations.

- **Problem1_Usinglib.py**  
  Implementation of Word2Vec using the **Gensim library** for comparison with the from-scratch model.

- **problem1_scrape.py**  
  Script for **web scraping IIT Jodhpur data** and creating the raw corpus.

- **outputs_final_2.0/**  
  Contains all outputs from Problem 1:
  - Trained embeddings (W1 matrices)
  - Training results (JSON)
  - Word cloud
  - PCA and t-SNE visualizations
  - Cosine similarity heatmap
  - Semantic analysis outputs

---

### 🔹 Problem 2: Character-Level Name Generation

- **Problem2.py**  
  Implementation of:
  - Vanilla RNN  
  - BiLSTM  
  - Attention-based RNN  
  All models are built **from scratch using PyTorch**, including training, evaluation, and generation.

- **problem2_output/**  
  Contains outputs from Problem 2:
  - Generated name samples
  - Evaluation metrics (novelty, diversity)
  - Training loss plots
  - Name length distributions

---

## 📊 Notes

- All models are implemented from scratch where required, following assignment constraints.
- Gensim is used only for comparison in Problem 1.
- Outputs are saved separately for clarity and reproducibility.

---

## 🚀 How to Run

```bash
#installing the requirements.txt
pip install -r requirements.txt
# Problem 1 (from scratch)
python Problem1.py

# Problem 1 (Gensim version)
python Problem1_UsingLib.py

# Problem 2
python Problem2.py
