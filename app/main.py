from unittest import result

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from groq import Groq
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ── API Keys ──────────────────────────────────────────────
GROQ_KEY   = "YOUR_GROQ_KEY_HERE"

# ── Load models once at startup ───────────────────────────
print("Loading BART v2...")
tokenizer_v2 = BartTokenizer.from_pretrained("../models/bart_inshorts_v2")
model_v2     = BartForConditionalGeneration.from_pretrained("../models/bart_inshorts_v2")
model_v2.eval()

print("Loading bart-large-cnn...")
tokenizer_large = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_large     = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model_large.eval()

print("All models loaded!")

# ── Summarisation functions ───────────────────────────────

def textrank_summarize(text):
    try:
        parser     = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary    = summarizer(parser.document, 3)
        return " ".join([str(s) for s in summary])
    except Exception as e:
        return f"Error: {e}"

def bart_v2_summarize(text):
    try:
        with torch.no_grad():
            inputs = tokenizer_v2(text, max_length=512, truncation=True, return_tensors="pt")
            ids    = model_v2.generate(
                inputs["input_ids"],
                num_beams=2, max_length=180, min_length=50,
                length_penalty=3.0, no_repeat_ngram_size=3, early_stopping=True
            )
        return tokenizer_v2.decode(ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {e}"

def bart_large_summarize(text):
    try:
        with torch.no_grad():
            inputs = tokenizer_large(text, max_length=512, truncation=True, return_tensors="pt")
            ids    = model_large.generate(
                inputs["input_ids"],
                num_beams=2, max_length=130, min_length=60,
                length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True
            )
        return tokenizer_large.decode(ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {e}"


def groq_summarize(text):
    try:
        client   = Groq(api_key=GROQ_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a news editor at Inshorts. Summarise news in exactly 60 words — clear, factual, third-person."},
                {"role": "user",   "content": f"Summarise this article in 60 words:\n\n{text}"}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


# ── Routes ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/summarise")
async def summarise(request: Request):
    data    = await request.json()
    article = data.get("article", "").strip()

    if len(article.split()) < 50:
        return JSONResponse({"error": "Article too short — paste at least 50 words."})

    models  = data.get("models", ["textrank", "bart_v2", "bart_large", "groq"])
    result  = {}
    if "textrank"   in models: result["textrank"]   = textrank_summarize(article)
    if "bart_v2"    in models: result["bart_v2"]    = bart_v2_summarize(article)
    if "bart_large" in models: result["bart_large"] = bart_large_summarize(article)
    if "groq"       in models: result["groq"]       = groq_summarize(article)
    return JSONResponse(result) 