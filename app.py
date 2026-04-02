from flask import Flask, request, jsonify, render_template
import torch
import re
import os
from transformers import BertTokenizerFast, BertForSequenceClassification
from groq import Groq

# -----------------------
# INIT
# -----------------------
app = Flask(__name__, template_folder="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# LOAD MODEL (FROM HF OR LOCAL)
# -----------------------
MODEL_PATH = "your-username/fake-news-bert"  # 🔥 CHANGE THIS

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2
)

model.to(device)
model.eval()

# -----------------------
# CLEAN TEXT
# -----------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------
# STRONG FAKE RULES (PRODUCTION)
# -----------------------
def apply_fake_rules(text):
    text_lower = text.lower()

    # 🚨 health scam patterns
    if re.search(r'cure|100% cure|guaranteed cure|miracle', text_lower):
        return True, 0.9

    # 🚨 unrealistic time claims
    if re.search(r'\b\d+\s*(days?|hours?|weeks?)\b', text_lower):
        if any(word in text_lower for word in ["cure", "lose", "gain", "fix"]):
            return True, 0.9

    # 🚨 extreme exaggeration
    if re.search(r'all types|completely|instantly|without effort', text_lower):
        return True, 0.85

    # 🚨 conspiracy / sci-fi
    absurd_patterns = [
        "aliens",
        "time travel",
        "mind control",
        "secret organization",
        "immortal",
        "without oxygen"
    ]

    if any(p in text_lower for p in absurd_patterns):
        return True, 0.9

    # 🚨 clickbait patterns
    if "you won’t believe" in text_lower or "this trick" in text_lower:
        return True, 0.8

    return False, 0.0

# -----------------------
# PREDICT (NOTEBOOK LOGIC + RULES)
# -----------------------
def predict_news(text):
    model.eval()

    text = clean_text(text)
    word_count = len(text.split())

    # minimal safety
    if word_count < 5:
        return {
            "prediction": "UNCERTAIN",
            "confidence": 0.5
        }

    # short text handling
    short_text_flag = False
    if word_count < 20:
        text = text + " " + text
        short_text_flag = True

    # 🔥 APPLY RULES FIRST
    is_fake_rule, rule_conf = apply_fake_rules(text)

    if is_fake_rule:
        return {
            "prediction": "FAKE",
            "confidence": rule_conf
        }

    # -----------------------
    # MODEL PREDICTION
    # -----------------------
    tokens = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    confidence = float(probs[0][pred])

    # calibration
    confidence = min(confidence, 0.95)

    if short_text_flag:
        confidence *= 0.9

    return {
        "prediction": "FAKE" if pred == 0 else "REAL",
        "confidence": round(confidence, 3)
    }

# -----------------------
# LLM EXPLANATION
# -----------------------
client = Groq(api_key=os.getenv("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"))

def generate_explanation(text, prediction, confidence):
    prompt = f"""
A machine learning model predicted this news as {prediction} with confidence {confidence}.

News:
{text}

Explain WHY using ONLY:

1. Writing style  
2. Tone of language  
3. Exaggeration or sensational words  
4. Clarity or vagueness  
5. Sentence structure  

STRICT RULES:
- Do NOT use real-world knowledge
- Do NOT verify facts
- Only analyze text patterns

Format clearly with numbered points.
"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception:
        return "Explanation unavailable"

# -----------------------
# COMBINED
# -----------------------
def predict_with_explanation(text):
    result = predict_news(text)

    explanation = generate_explanation(
        text,
        result["prediction"],
        result["confidence"]
    )

    result["explanation"] = explanation
    return result

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"})

    result = predict_with_explanation(text)
    return jsonify(result)

# -----------------------
# RUN (REPLIT FIX)
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
