   # From Classical ML to EUPN-VAE: MBTI Personality Classification

   ## Overview

   This project aims to predict Myers-Briggs Type Indicator (MBTI) personality types from people's social media posts using advanced artificial intelligence techniques. The research combines classical machine learning methods with cutting-edge deep learning models to improve both the accuracy and trustworthiness of personality predictions. Our most innovative contribution is the Entropic Uncertainty Personality Variational Autoencoder (EUPN-VAE), a model that not only predicts personality types but also gives a measure of how confident it is in its predictions—making the results more interpretable and reliable for use in real-world decisions.

   ## What is the MBTI and Why Does It Matter?

   The MBTI is a popular system for categorizing personality using four dimensions such as Extraversion-Introversion and Thinking-Feeling. It is widely used in education, recruitment, and mental health. Understanding personality from social media posts can help create more personalized and meaningful interactions, but doing so automatically with computers is challenging due to the complex nature of language and individual differences.

   ## Step-by-Step Research Process

   1. **Collect Data**: We started with a large, freely-available dataset from Kaggle, containing over 100,000 anonymized social media posts, each labeled with the person's MBTI type.[1]
   2. **Clean the Data**: Before using artificial intelligence, we cleaned up the posts by removing unnecessary information (like links and extra punctuation), making all text lowercase, and simplifying words to their basic form.
   3. **Turn Words into Numbers**: Computers need numbers to analyze text. We converted posts into numbers using different methods:
      - TF-IDF (counts “important” words)
      - Word2Vec, GloVe, FastText (capture word meanings)
   4. **Train Classical ML Models**: We tried traditional methods like Logistic Regression, Random Forest, Naive Bayes, and XGBoost, seeing which could best match posts to personality types. TF-IDF plus Logistic Regression worked very well.
   5. **Deep Learning Models**: To understand the flow and meaning within sentences, we used models called LSTM and BiLSTM. These models can read text forwards and backwards at the same time and often give even better results.
   6. **EUPN-VAE Model**: We designed a brand-new model that can “imagine” possible personalities for a user based on uncertainty and hidden traits. It uses advanced techniques to balance how much detail it learns and how accurate it is. A special genetic algorithm helps avoid mistakes and improves how confident the model is when results are unclear.
   7. **Compare Results**: We checked how well each approach worked. BiLSTM with TF-IDF was the most accurate, but our EUPN-VAE model is unique because it can also say when it is unsure—making it useful when exact results aren’t possible or when we need to understand “grey areas” in personality.
   8. **Challenges & Future Work**: Personality is complex—social, cultural, and ethical factors make it hard to label data and trust model results. Our future improvements include reinforcement learning, better handling imbalanced data, and adapting models to real-time data from platforms like X (Twitter) or Instagram.

   ## Technologies Used

   - **Programming Languages**: Python
   - **Frameworks**: TensorFlow (EUPN-VAE implementation), scikit-learn (classical ML models)
   - **ML Algorithms**: Logistic Regression, XGBoost, CatBoost, Random Forest, Naive Bayes, Support Vector Machines
   - **Deep Learning Models**: LSTM, BiLSTM, custom VAE architecture
   - **Embedding Methods**: TF-IDF, Word2Vec, GloVe, FastText
   - **Other Tools**: Genetic Algorithm for model optimization, Adam optimizer for training, regularization and dropout for stability
   - **Dataset Source**: MBTI Personality Types, Kaggle

   ## Links

   - [Project Paper (PDF)](./From-Classical-ML-to-EUPN-VAE-A-Unified-Framework-for-MBTI-Personality-Classification-2.pdf)
   - [Conference Certification](https://drive.google.com/file/d/1__eYbmr618xnw5RQGjPEtlS641tRYWs6/view)

   ## View Count

   ![Profile Views](https://komarev.com/ghpvc/?username=tejas-narkhede&repo=IEEE-Research-Paper-1&style=flat-square)


   **Paper status:** In process of publishing (under review at IEEE World Conference on Applied Intelligence and Computing 2025)


## 📊 Repo Activity & Stats  

![GitHub last commit](https://img.shields.io/github/last-commit/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub repo size](https://img.shields.io/github/repo-size/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub issues](https://img.shields.io/github/issues/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub stars](https://img.shields.io/github/stars/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  
![GitHub forks](https://img.shields.io/github/forks/tejas-narkhede/IEEE-Research-Paper-1?style=flat-square)  

---

## 🚀 Live Model Demo (Planned)  
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Try%20Model%20Demo-yellow.svg?style=flat-square)](https://huggingface.co/spaces/)  
*(Coming soon: interactive demo of MBTI classifier using Gradio)*  

---

## 📖 Citation (Planned)  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.0000000.svg)]()  
*(DOI badge will be activated once paper is deposited)*  

