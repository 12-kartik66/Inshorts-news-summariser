\# 📰 Inshorts News Summariser



A news summarisation app that compares 4 different approaches — from classical NLP to fine-tuned transformers to LLMs.



\## 🔴 Live Demo

> Run locally using the instructions below



\## 📊 Models Compared



| Model | Type | Description |

|-------|------|-------------|

| TextRank | Extractive | Graph-based sentence ranking, no training |

| BART v2 | Fine-tuned | facebook/bart-base fine-tuned on 3,442 Inshorts samples |

| bart-large-cnn | Pretrained | Facebook's BART trained on 300k CNN/DailyMail articles |

| Llama 3.3 70B | LLM | Meta's Llama via Groq API |



\## 📈 ROUGE Scores



| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |

|-------|---------|---------|---------|

| TextRank | 0.095 | 0.023 | 0.074 |

| BART v2 (ours) | 0.482 | 0.267 | 0.363 |



\## 🛠️ Tech Stack



\- \*\*Backend\*\* — FastAPI + Python

\- \*\*Models\*\* — HuggingFace Transformers (BART)

\- \*\*Training\*\* — PyTorch, Kaggle T4 GPU

\- \*\*LLM\*\* — Groq API (Llama 3.3 70B)

\- \*\*Frontend\*\* — Vanilla HTML/CSS/JS



\## 🚀 Run Locally



\*\*1. Clone the repo\*\*

```bash

git clone https://github.com/12-kartik66/Inshorts-news-summariser.git

cd Inshorts-news-summariser

```



\*\*2. Install dependencies\*\*

```bash

pip install fastapi uvicorn transformers torch sentencepiece sumy groq nltk

```



\*\*3. Download the trained model\*\*



Download `bart\_inshorts\_v2` from \[Kaggle Dataset](https://www.kaggle.com/datasets/kartik1321/bart-inshorts-model) and place it in `models/bart\_inshorts\_v2/`



\*\*4. Add your Groq API key\*\*



In `app/main.py`:

```python

GROQ\_KEY = "your-groq-key-here"

```



\*\*5. Run the app\*\*

```bash

cd app

uvicorn main:app --reload

```



Open `http://localhost:8000`



\## 📁 Project Structure



inshorts-summariser/

├── app/

│   ├── main.py              ← FastAPI backend

│   └── templates/

│       └── index.html       ← Frontend UI

├── notebooks/

│   ├── 01\_eda.ipynb         ← Data exploration

│   └── 02\_textrank.ipynb    ← TextRank baseline

├── .gitignore

└── README.md



\## 👤 Author

\*\*Kartik\*\* — NLP/ML Portfolio Project



