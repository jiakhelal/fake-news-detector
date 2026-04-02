from flask import Flask, request, jsonify, render_template
import torch
import re
from transformers import BertTokenizerFast, BertForSequenceClassification
from groq import Groq

# -----------------------
# INIT
# -----------------------
app = Flask(__name__, template_folder="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# LOAD MODEL
# -----------------------
MODEL_PATH = "model"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2
)

model.to(device)
model.eval()

# -----------------------
# CLEAN TEXT (NOTEBOOK SAME)
# -----------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------
# 🚨 PRODUCTION FAKE RULES
# -----------------------
STRONG_FAKE_PATTERNS = [
    # medical scams
    "cure all", "100% cure", "guaranteed cure", "miracle cure",
    "no side effects", "instant cure",

    # weight loss scams
    "lose weight in", "without exercise", "burn fat overnight",

    # conspiracy / unreal
    "secret organization", "government hiding", "mind control",
    "time travel", "immortal", "aliens",

    # clickbait
    "you won't believe", "doctors hate", "one weird trick",
    "shocking truth", "what happens next",

    # impossible science
    "without oxygen", "live forever", "reverse aging"
]

SUSPICIOUS_WORDS = [
    "cure", "secret", "alien", "guarantee",
    "instant", "miracle", "shocking"
]

# -----------------------
# PREDICT (ENHANCED)
# -----------------------
def predict_news(text):
    model.eval()

    text = clean_text(text)
    word_count = len(text.split())
    text_lower = text.lower()

    # 🚨 RULE 1: STRONG FAKE DETECTION (FIRST)
    if any(p in text_lower for p in STRONG_FAKE_PATTERNS):
        return {
            "prediction": "FAKE",
            "confidence": 0.9
        }

    # 🚨 RULE 2: SHORT TEXT
    if word_count < 5:
        return {
            "prediction": "UNCERTAIN",
            "confidence": 0.5
        }

    # 🚨 RULE 3: SHORT TEXT BOOST
    short_text_flag = False
    if word_count < 20:
        text = text + " " + text
        short_text_flag = True

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

    # 🚨 RULE 4: OVERRIDE WRONG REAL
    if pred == 1 and confidence < 0.9:
        if any(w in text_lower for w in SUSPICIOUS_WORDS):
            return {
                "prediction": "FAKE",
                "confidence": 0.85
            }

    return {
        "prediction": "FAKE" if pred == 0 else "REAL",
        "confidence": round(confidence, 3)
    }

# -----------------------
# LLM EXPLANATION
# -----------------------
client = Groq(api_key="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")

def generate_explanation(text, prediction, confidence):
    prompt = f"""
A machine learning model predicted this news as {prediction} with confidence {confidence}.

News:
{text}

Explain ONLY using:
- writing style
- tone
- exaggeration
- clarity
- structure

Do NOT use real-world facts.
"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        return response.choices[0].message.content
    except:
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
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)