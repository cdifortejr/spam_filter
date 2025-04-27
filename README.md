# spam_filter
# Spam Email Classifier 🚀

This repository contains a complete workflow for building a **Spam Email Classification AI** using **DistilBERT**, fine-tuned with the **Hugging Face Transformers** library on the **SMS Spam Collection dataset**.

The project covers:
- Loading a pre-trained transformer model (`distilbert-base-uncased`)
- Tokenizing and fine-tuning on spam detection data
- Saving and reloading the model for future use
- Deploying for easy inference (with future plans to host on Hugging Face Spaces)

---

## 📚 Project Overview

Traditional spam filters rely on manually engineered rules.  
Here, we leverage **transfer learning** from powerful language models (BERT family) to automatically detect spam based on **context and meaning**.

Key features:
- **Model**: Fine-tuned DistilBERT (`DistilBertForSequenceClassification`)
- **Dataset**: [SMS Spam Collection Dataset](https://huggingface.co/datasets/sms_spam)
- **Training**: Run in Google Colab using free GPUs (T4)
- **Saving**: Model saved in Hugging Face format (`.safetensors`, `config.json`, `tokenizer.json`, etc.)
- **Deployment**: Ready for easy hosting on Hugging Face Spaces or private websites

---

## 📁 Repository Structure


---

## ⚙️ How to Reproduce

1. Open `training_notebook.ipynb` in Colab or locally.
2. Install dependencies:

   ```bash
   pip install transformers datasets evaluate torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("path/to/spam_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/spam_model")
