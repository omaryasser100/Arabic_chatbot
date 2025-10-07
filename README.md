# Arabic Chatbot Experiments – Fine-Tuned Qwen & Unsloth Trials

This repository contains a set of experiments focused on building and analyzing Arabic conversational AI systems using **fine-tuned large language models (LLMs)** such as **Qwen** and **Unsloth**.  
The goal is to explore how well modern LLMs can understand, classify, and generate Arabic text when trained or adapted on a **very small custom Arabic dataset**.

---

## Project Overview

Arabic conversational AI remains a low-resource domain compared to English or Chinese.  
This project investigates how smaller fine-tuned LLMs can perform **intent classification, response generation, and dialogue tasks** in Arabic.

The experiments aim to answer:

- Can a multilingual base model (like Qwen or Unsloth) be adapted effectively on a tiny Arabic dataset?  
- What are the trade-offs between **fine-tuning**, **LoRA adaptation**, and **zero-shot prompting** for Arabic?  
- How does model fluency, accuracy, and coherence change across setups?  

These notebooks collectively serve as an exploration platform for **Arabic NLP, LLM fine-tuning, and chatbot prototyping**.

---

## Notebooks Included

| Notebook | Description |
|-----------|-------------|
| **1️⃣ Arabic_LLM_unsloth.ipynb** | Fine-tunes and evaluates the **Unsloth** variant of an Arabic-capable model. Includes dataset preprocessing, LoRA configuration, training, inference, and sample outputs. |
| **2️⃣ Arabic_rag_psycholigical.ipynb** | Experiments with **Qwen** (multilingual LLM from Alibaba) using PEFT fine-tuning on a small Arabic dialogue dataset. Tests classification vs. generation modes. |
| **3️⃣ psychlogical_health_nlp.ipynb** | Combines supervised **intent classification** and **response generation** into a hybrid chatbot pipeline. Demonstrates fallback logic between a small classifier and a fine-tuned LLM for response synthesis. |

Each notebook focuses on a specific technique — from model adaptation to practical chatbot behavior — while reusing a shared Arabic dataset.

---

## Technologies and Frameworks

This repo integrates a broad set of modern NLP and fine-tuning tools:

- **Transformers:** Hugging Face Transformers for model loading and training  
- **PEFT / LoRA:** Parameter Efficient Fine-Tuning for adapting Qwen/Unsloth on limited hardware  
- **PyTorch:** Deep learning backend for training and inference  
- **Datasets:** Hugging Face Datasets for text management  
- **Accelerate:** Optimized training on GPU  
- **LangChain / PromptTemplate:** Prompt design and experimentation for few-shot learning  
- **SentenceTransformers & scikit-learn:** Used for intent classification baseline  
- **Gradio / Streamlit:** For chatbot and interactive UI trials  
- **Arabic NLP tools:** Tokenization, normalization, and diacritic removal (e.g. `camel_tools`, `pyarabic`)

---

## Dataset

A small, custom **Arabic conversational dataset** designed for:
- Intent classification (e.g. greetings, info requests, farewells)
- Short question–answer pairs for chatbot generation
- Few hundred to a few thousand examples, curated and normalized
- Augmented using template-based expansion and paraphrasing

Due to size limitations, this dataset was used mainly for **LoRA fine-tuning** rather than full-model training.

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/omaryasser100/Arabic_chatbot.git
   cd Arabic_chatbot
