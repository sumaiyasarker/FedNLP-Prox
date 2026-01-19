# 🚀 Federated Learning for NLP using Flower & DistilBERT

This project demonstrates a **high-profile, research-grade Federated Learning (FL) pipeline** for **Natural Language Processing (NLP)** using **Flower (FLWR)** and **Transformer models**.
We fine-tune **DistilBERT** on the **IMDB sentiment classification dataset** in a **non-IID federated setting** using the **FedProx strategy**.

---

## 🔍 Project Highlights

* ✅ **Federated Learning Simulation** using Flower
* 🧠 **Transformer-based NLP model** (DistilBERT)
* 🔄 **Non-IID client data partitioning**
* ⚙️ **FedProx strategy** for heterogeneous client environments
* 🚀 **GPU-aware resource allocation** (Colab-friendly)
* 📊 **Centralized evaluation with distributed training**



## 🏗️ System Architecture

```
IMDB Dataset
     ↓
Tokenization (HuggingFace)
     ↓
Non-IID Partitioning
     ↓
Federated Clients (DistilBERT)
     ↓
FedProx Aggregation (Flower Server)
     ↓
Centralized Evaluation
```

---

## 🧪 Dataset

* **IMDB Movie Reviews**
* Binary sentiment classification (Positive / Negative)
* Subset used for fast experimentation:

  * 🟢 Train: 5,000 samples
  * 🔵 Test: 1,000 samples

---

## 🧠 Model

* **Checkpoint**: `distilbert-base-uncased`
* **Why DistilBERT?**

  * Lightweight
  * Fast convergence
  * Strong NLP performance
  * Ideal for Federated Learning simulations

---

## ⚙️ Federated Strategy

* **Algorithm**: FedProx
* **Why FedProx?**

  * Handles **Non-IID data**
  * Robust to **client hardware variability**
  * Widely used in modern FL research papers

---

## 🖥️ Requirements

Tested on **Google Colab (Python 3.12)**

```bash
pip install protobuf==5.26.1 ray[default]==2.31.0 cryptography==43.0.3
pip install flwr[simulation] torch torchvision transformers datasets
```

---

## ▶️ How to Run

1. Open **Google Colab**
2. Run the **installation cell**
3. Run the **single self-contained training script**
4. Observe federated rounds:

   ```
   Round 1
   Round 2
   Round 3
   ```
5. Plot accuracy after training

---

## 📈 Results Visualization

The project tracks **federated accuracy across rounds**:

```python
FedProx Accuracy vs Federated Round
```

This demonstrates **stable improvement despite decentralized training**.

---

## 🔬 Research Relevance

This project is relevant for:

* Federated NLP
* Privacy-preserving AI
* Distributed ML systems
* Healthcare / text-based FL extensions

It can be extended to:

* Medical text (clinical notes)
* Multilingual NLP
* Secure aggregation
* Differential privacy

---

## 📌 Future Work

* 🔐 Differential Privacy (DP-SGD)
* 🌍 Heterogeneous client simulation
* 🧬 Domain-specific NLP datasets

