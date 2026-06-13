# 📰 Inshorts News Summariser

> Fine-tuned BART on the Inshorts dataset and compared it against 3 other summarisation approaches — from classical NLP to state-of-the-art LLMs.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🎯 Project Overview

This project explores **abstractive news summarisation** by fine-tuning `facebook/bart-base` on the [Inshorts News Dataset](https://www.kaggle.com/datasets/sunnysai12345/news-summary) and comparing it against multiple summarisation approaches side by side.

The web app lets you paste any news article and instantly see how 4 different models summarise it — making it easy to understand the trade-offs between classical NLP, fine-tuned transformers, and modern LLMs.

---

## 🤖 Models Compared

| # | Model | Type | Training Data | Notes |
|---|-------|------|--------------|-------|
| 1 | **TextRank** | Extractive | None | Graph-based sentence ranking |
| 2 | **BART v2** | Fine-tuned | 3,442 Inshorts samples | Our custom trained model |
| 3 | **bart-large-cnn** | Pretrained | 300k CNN/DailyMail | Facebook's production model |
| 4 | **Llama 3.3 70B** | LLM | Meta proprietary | Via Groq API |

---

## 📈 ROUGE Scores

Evaluated on 500 samples from the Inshorts test set:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Improvement over baseline |
|-------|---------|---------|---------|--------------------------|
| TextRank (baseline) | 0.095 | 0.023 | 0.074 | — |
| **BART v2 (ours)** | **0.482** | **0.267** | **0.363** | **+6x ROUGE-1** |

> ROUGE measures n-gram overlap between generated and reference summaries.
> Higher is better. BART v2 achieves a **6× improvement** over the TextRank baseline.

---

## 🏗️ Architecture

```
Article (300+ words)
        ↓
  Tokenizer (BartTokenizer)
        ↓
  BART Encoder (bidirectional)
        ↓
  BART Decoder (autoregressive)
        ↓
  Summary (~60 words)
```

**Training setup:**
- Base model: `facebook/bart-base` (139M parameters)
- Dataset: Inshorts News — `ctext` (full article) → `text` (60-word summary)
- Epochs: 5 | Batch size: 4 | Learning rate: 2e-5
- Hardware: Kaggle T4 GPU (~50 minutes training)
- Optimizer: AdamW with cosine warmup scheduler

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| ML Framework | PyTorch + HuggingFace Transformers |
| Models | BART base, BART large-CNN |
| LLM API | Groq (Llama 3.3 70B) |
| Extractive NLP | Sumy (TextRank) |
| Frontend | Vanilla HTML + CSS + JavaScript |
| Training Platform | Kaggle (T4 GPU) |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Anaconda or pip
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the repository
```bash
git clone https://github.com/12-kartik66/Inshorts-news-summariser.git
cd Inshorts-news-summariser
```

### 2. Create environment and install dependencies
```bash
conda create -n summariser python=3.10 -y
conda activate summariser
pip install fastapi uvicorn transformers torch sentencepiece sumy groq nltk jinja2 python-multipart
```

### 3. Download the trained model
Download `bart_inshorts_v2` from the [Kaggle Dataset](https://www.kaggle.com/datasets/kartik1321/bart-inshorts-model) and place it at:
```
models/
└── bart_inshorts_v2/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

### 4. Add your Groq API key
Make `.env` and replace:
```python
GROQ_KEY = "YOUR_GROQ_KEY_HERE"
```

### 5. Start the server
```bash
cd app
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## 📁 Project Structure

```
inshorts-summariser/
├── app/
│   ├── main.py                  ← FastAPI backend — all 4 models
│   └── templates/
│       └── index.html           ← Frontend UI with sidebar
├── notebooks/
│   ├── 01_eda.ipynb             ← Data exploration & length analysis
│   ├── 02_textrank.ipynb        ← TextRank baseline + ROUGE evaluation
│   └── 03-bart-finetune.ipynb   ← BART fine-tuning on Kaggle T4 GPU
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💡 Key Learnings

- **Extractive vs Abstractive**: TextRank picks existing sentences (fast, no training) while BART generates new text (slower, better quality)
- **Dataset quality matters more than size**: 3,442 high-quality Inshorts samples gave better style-specific results than 82,000 headline samples
- **Pre-trained models are hard to beat**: `bart-large-cnn` (trained on 300k samples) outperforms our fine-tuned model — showing the value of large-scale pre-training
- **ROUGE has limits**: LLMs like Llama 3.3 generate fluent summaries but score lower on ROUGE because they paraphrase rather than copy n-grams

---

## 🔮 Future Improvements

- [ ] Fine-tune on larger dataset (CNN/DailyMail + Inshorts combined)
- [ ] Add URL input — paste a news article URL and auto-scrape
- [ ] Deploy on cloud (Railway / Render) for public access
- [ ] Add BERTScore evaluation (better than ROUGE for semantic similarity)
- [ ] Support Hindi news summarisation

---

## 👤 Author

**Kartik**  
NLP/ML Portfolio Project | Summer 2025  
🔗 [GitHub](https://github.com/12-kartik66)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
