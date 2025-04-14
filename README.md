# 🌐 English to Hindi Neural Machine Translation using Transformer

This project implements an English-to-Hindi translation system using the Transformer architecture from scratch in PyTorch. It utilizes the [Tatoeba dataset](https://tatoeba.org) and features a real-time **Streamlit app** for live translation.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Streamlit App](#streamlit-app)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## 📖 Overview

The goal of this project is to translate English sentences into Hindi using a custom-built Transformer model. The architecture follows the **"Attention is All You Need"** paper, with an encoder-decoder structure.

We use a **custom Vocabulary class**, implement positional encoding, multi-head attention, feedforward layers, and layer normalization. The final model is deployed using **Streamlit** for real-time interaction.

---

## 📚 Dataset

- Source: [Tatoeba Parallel Corpus](https://tatoeba.org)
- Format: TSV (Tab Separated) with columns: `id1`, `en`, `id2`, `hi`
- Preprocessing:
  - Removed ID columns
  - Dropped missing values
  - Tokenized using whitespace
  - Built word-level vocabularies (min frequency = 2)
  - Added special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`
  - Maximum sequence length: 50

---

## 🧠 Model Architecture

The Transformer model includes:

- **Encoder**:
  - Embedding + Positional Encoding
  - Multi-head self-attention + FeedForward
  - Layer normalization and dropout

- **Decoder**:
  - Embedding + Positional Encoding
  - Masked self-attention
  - Cross-attention with encoder output
  - FeedForward + normalization

- **Training Objective**:
  - CrossEntropyLoss (ignoring `<pad>`)
  - Optimizer: Adam

---

## ⚙️ Training

- Epochs: 100  
- Batch Size: 32  
- Max Length: 50 tokens  
- Optimizer: Adam (lr = 1e-4)  
- Loss Function: CrossEntropyLoss  
- Device: CUDA / CPU  

Models and vocabularies are saved as:

```bash
transformer_translation_final.pth
en_vocab.pkl
hi_vocab.pkl
```
## 📊 Evaluation
- Manual sentence inspection
- Sample BLEU score evaluation using NLTK
- Common sentence translations tested

```bash
EN: I love you.     → HI: मैं तुमसे प्यार करता हूँ।
EN: How are you?    → HI: आप कैसे हैं?
```

## 🌐 Streamlit App
A user-friendly app was built using Streamlit.

- **Features:**
  - Input an English sentence
  - View Hindi translation instantly
  - Uses pre-trained Transformer model and vocabularies

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install torch streamlit nltk
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

Ensure the following files are present in your project directory:
-transformer_translation_final.pth – Trained Transformer model
-en_vocab.pkl – Saved English vocabulary
-hi_vocab.pkl – Saved Hindi vocabulary
-transformer.py – Transformer model architecture
-vocab.py – Vocabulary class
-app.py – Streamlit interface

## 📈 Results

- High-quality translation for short, common phrases.
- Streamlit app performs real-time inference.
- BLEU score on the test set: ~30–40 (depending on configuration).

## 🔮 Future Work

- Add beam search decoding for improved translation quality.
- Use pretrained embeddings (e.g., fastText) to improve model performance.
- Train on larger multilingual datasets (e.g., OpenSubtitles, CCMatrix).
- Extend the model to support multi-language translation.

## 📚 References

- Vaswani et al., *Attention is All You Need*, NeurIPS 2017. ([Paper Link](https://arxiv.org/abs/1706.03762))
- Tatoeba Project: [https://tatoeba.org](https://tatoeba.org)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NLTK BLEU Score Toolkit](https://www.nltk.org/nltk_data/)

## 👨‍💻 Authors

- Tanesh Gujar

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/english-to-hindi-translation.git
cd english-to-hindi-translation
pip install -r requirements.txt
```
