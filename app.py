import os
import torch
import joblib
import re
import uuid
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)

# ======================================================
# 🔐 Hugging Face Token (DO NOT hardcode)
# ======================================================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not set")

# ======================================================
# 🚀 Flask App
# ======================================================
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================================================
# 📥 LOAD MODELS FROM HUGGING FACE HUB (PRIVATE REPOS)
# ======================================================

# ✅ YOUR VERIFIED REPO NAMES
CAT_BERT_REPO = "snehithkumarmatte/category-bert"
CAT_DISTIL_REPO = "snehithkumarmatte/category-distilbert"
PRI_BERT_REPO = "snehithkumarmatte/priority-bert"
PRI_ROBERTA_REPO = "snehithkumarmatte/priority-roberta"

# -------- Category Models --------
cat_bert = BertForSequenceClassification.from_pretrained(
    CAT_BERT_REPO,
    token=HF_TOKEN
).to(device)
cat_bert_tok = BertTokenizer.from_pretrained(
    CAT_BERT_REPO,
    token=HF_TOKEN
)

cat_distil = DistilBertForSequenceClassification.from_pretrained(
    CAT_DISTIL_REPO,
    token=HF_TOKEN
).to(device)
cat_distil_tok = DistilBertTokenizer.from_pretrained(
    CAT_DISTIL_REPO,
    token=HF_TOKEN
)

# -------- Priority Models --------
pri_bert = BertForSequenceClassification.from_pretrained(
    PRI_BERT_REPO,
    token=HF_TOKEN
).to(device)
pri_bert_tok = BertTokenizer.from_pretrained(
    PRI_BERT_REPO,
    token=HF_TOKEN
)

pri_roberta = RobertaForSequenceClassification.from_pretrained(
    PRI_ROBERTA_REPO,
    token=HF_TOKEN
).to(device)
pri_roberta_tok = RobertaTokenizer.from_pretrained(
    PRI_ROBERTA_REPO,
    token=HF_TOKEN
)

# -------- Label Encoders (small files kept locally) --------
le_category = joblib.load("label_encoders/category_encoder.pkl")
le_priority = joblib.load("label_encoders/priority_encoder.pkl")

# Set models to eval mode
cat_bert.eval()
cat_distil.eval()
pri_bert.eval()
pri_roberta.eval()

# ======================================================
# 🧠 HELPER FUNCTIONS
# ======================================================

def is_low_input(text: str) -> bool:
    return len(text.strip().split()) < 3

def is_ticket_like(text: str) -> bool:
    keywords = [
        "vpn", "error", "issue", "not working", "cannot",
        "fail", "crash", "access", "slow", "down"
    ]
    return any(k in text.lower() for k in keywords)

def is_chitchat(text: str) -> bool:
    greetings = ["how are you", "hi", "hello", "hey", "i am"]
    return any(g in text.lower() for g in greetings) and not is_ticket_like(text)

def get_probs(model, tokenizer, text: str):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    return torch.softmax(logits, dim=1).cpu().numpy()

def predict_category(text: str) -> str:
    p1 = get_probs(cat_bert, cat_bert_tok, text)
    p2 = get_probs(cat_distil, cat_distil_tok, text)
    pred = np.argmax((p1 + p2) / 2)
    return le_category.inverse_transform([pred])[0]

def predict_priority(text: str) -> str:
    p1 = get_probs(pri_bert, pri_bert_tok, text)
    p2 = get_probs(pri_roberta, pri_roberta_tok, text)
    pred = np.argmax((p1 + p2) / 2)
    return le_priority.inverse_transform([pred])[0]

def extract_entities(text: str) -> dict:
    return {
        "devices": re.findall(
            r"\b(laptop|desktop|tablet|server|printer|mobile)\b",
            text.lower()
        ),
        "applications": re.findall(
            r"\b(vpn|wifi|teams|zoom|outlook|email|crm|salesforce)\b",
            text.lower()
        ),
        "error_codes": re.findall(
            r"[A-Z]{2,5}-?\d+|0x[A-Fa-f0-9]+",
            text
        )
    }

def generate_ticket(text: str) -> dict:
    return {
        "ticket_id": f"INC-{str(uuid.uuid4())[:4]}",
        "title": "Auto-generated Support Ticket",
        "category": predict_category(text),
        "priority": predict_priority(text).capitalize(),
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": text,
        "entities": extract_entities(text)
    }

# ======================================================
# 🌐 ROUTE
# ======================================================
@app.route("/", methods=["GET", "POST"])
def index():
    ticket = None
    message = None

    if request.method == "POST":
        text = request.form.get("ticket", "").strip()

        if is_low_input(text):
            message = "Input is too short. Please describe the issue clearly."
        elif is_chitchat(text):
            message = "This system handles IT support issues only."
        else:
            ticket = generate_ticket(text)

    return render_template("index.html", ticket=ticket, message=message)

# ======================================================
# ▶️ RUN
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
