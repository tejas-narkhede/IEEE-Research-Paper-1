# From Classical ML to EUPN-VAE: Advancing MBTI Personality Classification

## Overview

This project delivers a comprehensive exploration of MBTI personality classification from social media text—starting from classical machine learning baselines and progressing toward a novel deep generative architecture. The central contribution is the **Entropic Uncertainty Personality Variational Autoencoder (EUPN-VAE)**, a model designed not only to predict personality types but also to quantify uncertainty in its decisions. This enables more transparent, interpretable, and trustworthy predictions suitable for real-world psychological or behavioral applications.

---

## What is MBTI and Why Does It Matter?

The Myers–Briggs Type Indicator divides personality into four independent dimensions, forming sixteen personality types. MBTI is widely applied across career guidance, team dynamics, education, and mental-health–related assessments.  
Automating MBTI prediction from text offers immense potential, yet poses challenges due to linguistic diversity, subtle expression patterns, and personal variability.  
This research tackles these challenges using scalable and explainable AI.

---

## Research Workflow

1. **Dataset Acquisition**  
   Used the Kaggle MBTI dataset containing 100,000+ anonymized social media posts labelled with MBTI types.

2. **Text Preprocessing**  
   Performed noise removal (URLs, symbols), lowercasing, tokenization, and lemmatization to create a clean, consistent dataset.

3. **Text Representation**  
   Experimented with multiple encoding approaches:  
   - **TF-IDF** for sparse linguistic patterns  
   - **Word2Vec**, **GloVe**, **FastText** for dense semantic representations  

4. **Classical ML Baselines**  
   Built Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost, and CatBoost classifiers.  
   *TF-IDF + Logistic Regression* emerged as the strongest traditional baseline.

5. **Deep Learning Models**  
   Used LSTM and BiLSTM architectures to capture sequential context.  
   BiLSTM combined with TF-IDF produced highly competitive accuracy.

6. **EUPN-VAE (Proposed Model)**  
   Designed a custom VAE-based architecture that learns a latent distribution of personality traits and expresses *how certain* it is about its predictions.  
   Key innovations include:
   - Entropy-guided latent refinement  
   - KL β-annealing for improved regularization  
   - Genetic algorithm optimization for stable latent exploration  

7. **Evaluation & Benchmarking**  
   Compared models on accuracy, robustness, and uncertainty quality.  
   While deep learning models achieved strong predictive performance, **EUPN-VAE stands out by providing calibrated uncertainty**, making it valuable in ambiguous or multi-personality scenarios.

8. **Future Scope**  
   Potential extensions include reinforcement learning–based reasoning, improved handling of imbalanced MBTI classes, and adaptation for real-time platforms such as Twitter/X and Instagram.

---

## Technologies Used

- **Languages:** Python  
- **Frameworks:** TensorFlow, scikit-learn  
- **ML Algorithms:** Logistic Regression, SVM, Random Forest, Naive Bayes, XGBoost, CatBoost  
- **Deep Learning:** LSTM, BiLSTM, custom VAE architecture  
- **Embeddings:** TF-IDF, Word2Vec, GloVe, FastText  
- **Other Tools:** Genetic Algorithm, Adam optimizer, dropout & regularization  
- **Dataset:** MBTI Personality Types (Kaggle)

---

## Links

- **Research Paper:** https://ieeexplore.ieee.org/document/11211954  
- **Conference Certificate:** https://drive.google.com/file/d/1__eYbmr618xnw5RQGjPEtlS641tRYWs6/view

---

## View Count

![Hits](https://hits.sh/tejas--narkhede/IEEE--Research--Paper--1.svg?style=flat-square)

---

## 📊 Repo Activity & Stats  

![GitHub last commit](https://img.shields.io/github/last-commit/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub repo size](https://img.shields.io/github/repo-size/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub issues](https://img.shields.io/github/issues/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub stars](https://img.shields.io/github/stars/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub forks](https://img.shields.io/github/forks/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  

---

## 🚀 Live Model Demo (Coming Soon)

[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Model%20Demo%20Coming%20Soon-yellow.svg?style=flat-square)](https://huggingface.co/spaces/)

---

## 📖 Citation  

[![DOI](https://zenodo.org/badge/doi/10.1109/AIC66080.2025.11211954.svg)](https://doi.org/10.1109/AIC66080.2025.11211954)
