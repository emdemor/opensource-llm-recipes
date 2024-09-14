# 🚀 Open-Source LLM Experiments

### ℹ️ About the Project
This repository serves as a sandbox for experimenting with various open-source LLMs, exploring tasks such as translation, text generation, model fine-tuning, and more. The goal is to explore how LLMs can be applied to real-world problems with a focus on ease of use, reproducibility, and practical outcomes.


### ⚠️ Note on Language
All the notebooks and code in this repository are written in Brazilian Portuguese. If you're not familiar with the language, feel free to use an online translation tool or reach out with any questions.

## 📚 Table of Contents
1. [Translation Experiments](#translation-experiments)
   - [OpenAI Translation](#openai-translation)
   - [Local Translation with NLLB](#local-translation-with-nllb)
2. [Language Model Execution](#language-model-execution)
   - [Local Small Language Models](#local-small-language-models)
   - [Quantized Language Models](#quantized-language-models)
3. [Fine-Tuning](#fine-tuning)
   - [Fine-Tuning LLMs for SQL Generation](#fine-tuning-llms-for-sql-generation)
   - [Fine-Tuning BERT for Text Classification](#fine-tuning-bert-for-text-classification)
   - [Fine-Tuning for Embeddings](#fine-tuning-for-embeddings)
   - [Fine-Tuning for Pairwise Embedding Scores](#fine-tuning-for-pairwise-embedding-scores)
4. [Text-to-Speech](#text-to-speech)
5. [Dataset Preparation](#dataset-preparation)
   - [Q&A Dataset Preparation](#qa-dataset-preparation)
   - [Question Generation Dataset Preparation](#question-generation-dataset-preparation)
6. [Question Generation](#question-generation)
   - [Testing Question Generator v1](#testing-question-generator-v1)
   - [Testing Question Generator v2](#testing-question-generator-v2)
7. [Whisper Transcription](#whisper-transcription)
8. [1984 Project](#1984-project)
   - [Splitting 1984 for Question Generation](#splitting-1984-for-question-generation)
   - [Fine-Tuning a 1984 QA Model](#fine-tuning-a-1984-qa-model)
   - [Using the 1984 QA Model](#using-the-1984-qa-model)
9. [Image Generation with Stable Diffusion](#image-generation-with-stable-diffusion)

---

## 🧪 Experiments

### 🌍 Translation Experiments

#### 1. OpenAI Translation
**Description**: A baseline notebook that uses OpenAI models to translate text from English to Portuguese.  
**Purpose**: Establish a benchmark for translation tasks.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/01-openai-translatIon.ipynb)

#### 2. Local Translation with NLLB
**Description**: This notebook utilizes the `facebook/nllb-200-distilled-1.3B` model to translate sentences from English to Portuguese.  
**Purpose**: Develop a local translation pipeline for text datasets.  
**Dataset Generated**: [emdemor/sql-create-context-pt](https://huggingface.co/datasets/emdemor/sql-create-context-pt)
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/02-local-translation.ipynb)

---

### 🖥️ Language Model Execution

#### 3. Local Small Language Models (SLM)
**Description**: Demonstrates how to execute a small language model locally.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/03-qwen2-hello-world.ipynb)

#### 4. Local Small Language Models (SLM) – Advanced
**Description**: Example of running a more robust small language model locally.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/04-phi3-hello-world.ipynb)

#### 5. Quantized Language Models
**Description**: Execution of quantized language models using Huggingface and BitsAndBytes.  
**Purpose**: Optimize language models for memory and speed efficiency.
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/05-phi3-quant-hello-world.ipynb)

---

### 🔧 Fine-Tuning

#### 6. Fine-Tuning LLMs for SQL Generation
**Description**: Example of fine-tuning a language model to generate SQL queries.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/06-fine-tune-pttext2sql.ipynb)

#### 7. Fine-Tuning BERT for Text Classification
**Description**: Fine-tuning BERT to perform text classification using an autoencoder.  
**External Code**: Based on [Chris McCormick & Nick Ryan's tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/07-bert-classification.ipynb)

#### 10. Fine-Tuning for Embeddings
**Description**: Fine-tuning a model to generate domain-specific embeddings using Huggingface.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/10-simple-embedding-finetune.ipynb)

#### 12. Fine-Tuning for Pairwise Embedding Scores
**Description**: Fine-tuning an embedding model using sentence-pair similarity scores as the target.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/12-call-slm.ipynb)

---

### 🗣️ Text-to-Speech

#### 8. Text-to-Speech
**Description**: Example of generating speech locally using Huggingface and the Parler model.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/08-text-to-speech.ipynb)

---

### 🧑‍💻 Dataset Preparation

#### 9. Q&A Dataset Preparation
**Description**: A notebook for preparing a Portuguese Q&A dataset, which is available at [emdemor/ptbr-question-and-answer](https://huggingface.co/datasets/emdemor/ptbr-question-and-answer).
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/09-prepare-dataset-for-embedding-finetune.ipynb)

#### 13. Question Generation Dataset Preparation
**Description**: This notebook uses GPT models to generate a dataset for question generation based on a given context.
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/13-prepare-dataset-to-question-generator.ipynb)

---

### 🤖 Question Generation

#### 14. Fine-Tuning a Question Generator
**Description**: Fine-tuning a model on Kaggle that generates questions based on a provided context.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/14-kaggle-finetune-question-and-answer-v01.ipynb)

#### 15. Testing Question Generator v1
**Description**: Testing the model [emdemor/question-generator](https://huggingface.co/emdemor/question-generator) that generates random questions from a given context.
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/15-call-question-generator-version-1.ipynb)

#### 16. Testing Question Generator v2
**Description**: Testing the model [emdemor/question-generator-v2](https://huggingface.co/emdemor/question-generator-v2), which generates question-answer pairs from a given context.
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/16-call-question-generator-version-2.ipynb)

---

### 🎤 Whisper Transcription

#### 17. YouTube Transcription with Whisper
**Description**: A notebook demonstrating how to transcribe audio from a YouTube video using Whisper locally.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/17-youtube-transcription-with-openai-whisper.ipynb)

---

### 📚 1984 Project

#### 18. Splitting 1984 for Question Generation
**Description**: Splitting the book *1984* into multiple chunks and using the [emdemor/question-generator-v2](https://huggingface.co/emdemor/question-generator-v2) model to generate questions and answers for each chunk.  
**Purpose**: To train a Q&A model on the book *1984*.
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/18-splitting-1984.ipynb)

#### 19. Fine-Tuning a 1984 QA Model
**Description**: Fine-tuning a Q&A model based on the generated *1984* dataset.
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/19-%5Bkaggle%5D-finetune-q-a-1984.ipynb)

#### 20. Using the 1984 QA Model
**Description**: Instructions for using the trained *1984* Q&A model.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/20-call-guru-1984.ipynb)

---

### 🎨 Image Generation with Stable Diffusion

#### 21. Stable Diffusion Image Generation
**Description**: Initial tests with image generation using Stable Diffusion.  
[**Access notebook**](https://github.com/emdemor/opensource-llm-recipes/blob/master/project/21-test-stable-difusion.ipynb)
