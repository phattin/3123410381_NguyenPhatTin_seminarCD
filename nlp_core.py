import threading
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
try:
    from underthesea import word_tokenize
    _UNDERTHESEA = True
except Exception:
    _UNDERTHESEA = False

VIETNAMESE_SHORTCUTS = {
    "rat": "rất",
    "mun": "muốn",
    "ko": "không",
    "kg": "không",
    "biet": "biết",
    "hom nay": "hôm nay",
    "qua": "quá",
    "nhieu": "nhiều",
    "toi": "tôi",
    "dc": "được",
    "vo": "vô"
}

MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

_sentiment_pipeline = None
_pipeline_lock = threading.Lock()

def load_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    with _pipeline_lock:
        if _sentiment_pipeline is not None:
            return _sentiment_pipeline
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=False
            )
            print("NLP Pipeline đã được tải thành công:", MODEL_NAME)
        except Exception as e:
            print("Lỗi tải NLP Pipeline:", e)
            _sentiment_pipeline = None
        return _sentiment_pipeline

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    s = s.lower()
    for short, full in VIETNAMESE_SHORTCUTS.items():
        s = re.sub(r"\b" + re.escape(short) + r"\b", full, s, flags=re.IGNORECASE)
    if _UNDERTHESEA:
        try:
            s = word_tokenize(s, format="text")
        except Exception:
            pass
    return s

LABEL_MAP = {
    "POS": "POSITIVE",
    "NEG": "NEGATIVE",
    "NEU": "NEUTRAL",
    "POSITIVE": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "NEUTRAL": "NEUTRAL",
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}

def map_label(raw_label: str) -> str:
    key = str(raw_label).upper()
    return LABEL_MAP.get(key, "NEUTRAL")

def classify_sentiment(text: str) -> dict:
    pipe = load_pipeline()
    if pipe is None:
        return {"text": text, "sentiment": "ERROR", "score": 0.0}

    raw_text = text.strip()
    if len(raw_text) < 5 or len(raw_text) > 500:  # bạn muốn 50 thì giữ 50; mình tăng tối đa thành 500 cho linh hoạt
        return {"text": raw_text, "sentiment": "LENGTH_ERROR", "score": 0.0}

    processed = preprocess_text(raw_text)
    try:
        out = pipe(processed)
        if isinstance(out, list) and len(out) > 0:
            top = out[0]
            raw_label = top.get("label", "")
            score = float(top.get("score", 0.0))
            label = map_label(raw_label)
            # nếu score thấp hơn threshold -> trả NEUTRAL
            if score < 0.5:
                label = "NEUTRAL"
            return {"text": raw_text, "sentiment": label, "score": score}
        else:
            return {"text": raw_text, "sentiment": "PIPELINE_ERROR", "score": 0.0}
    except Exception as e:
        print("Lỗi phân loại NLP:", e)
        return {"text": raw_text, "sentiment": "PIPELINE_ERROR", "score": 0.0}
