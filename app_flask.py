"""
Movie Review Sentiment Analysis - Flask Application (Single Page App Edition)
=============================================================================

Pages & Endpoints:
    /                  -> SPA Movie Review Sentiment Analyzer
    /predict           -> AJAX/JSON & Form prediction endpoint
    /session-stats     -> Live session statistics endpoint
    /download-report   -> Download prediction report

Model Architecture:
    RoBERTa-base Transformer + Groq Llama 3.3 70B Hybrid Escalation
"""

import time
from io import BytesIO
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
)

from sentiment_analyzer import (
    SentimentSystem,
    DEFAULT_MODEL_PATH,
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# IN-MEMORY SESSION TRACKER
# ============================================================

SESSION_HISTORY = []


def get_session_stats():
    """Calculate running session metrics based on actual predictions."""
    total = len(SESSION_HISTORY)
    if total == 0:
        return {
            "total_reviews": 0,
            "positive_pct": 0,
            "neutral_pct": 0,
            "negative_pct": 0,
            "avg_latency_ms": 0,
            "recent_reviews": [],
        }

    pos_count = sum(1 for item in SESSION_HISTORY if item.get("label", "").lower() == "positive")
    neu_count = sum(1 for item in SESSION_HISTORY if item.get("label", "").lower() == "neutral")
    neg_count = sum(1 for item in SESSION_HISTORY if item.get("label", "").lower() == "negative")

    pos_pct = round((pos_count / total) * 100)
    neu_pct = round((neu_count / total) * 100)
    neg_pct = round((neg_count / total) * 100)

    avg_lat = round(sum(item.get("latency_ms", 0) for item in SESSION_HISTORY) / total, 1)

    return {
        "total_reviews": total,
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "avg_latency_ms": avg_lat,
        "recent_reviews": list(reversed(SESSION_HISTORY[-6:])),
    }


# ============================================================
# MODEL LOADER
# ============================================================

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = SentimentSystem(model_path=DEFAULT_MODEL_PATH)
    return MODEL


# ============================================================
# ORDER PROBABILITIES
# ============================================================

def order_probabilities(probs):
    order = [
        "Positive",
        "Neutral",
        "Negative",
    ]
    return [
        (sentiment, probs[sentiment])
        for sentiment in order
        if sentiment in probs
    ]


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

def get_confidence_strength(probabilities):
    if not probabilities:
        return "Not available"

    highest_probability = max(probability for _, probability in probabilities)
    percentage = highest_probability * 100

    if percentage >= 85:
        return "Extremely confident"
    elif percentage >= 70:
        return "Very confident"
    elif percentage >= 50:
        return "Moderately confident"
    else:
        return "Uncertain - mixed signals"


# ============================================================
# COLOR PALETTE MAPPING
# ============================================================

COLOR_MAP = {
    "Positive": {
        "text": "#3F6B66",
        "bg": "#EBF3F1",
        "border": "#A3C4BE",
        "bg_soft": "linear-gradient(135deg, #EBF3F1 0%, #D8E8E4 100%)",
        "icon": "▲",
    },
    "Negative": {
        "text": "#7A2331",
        "bg": "#FBEAEA",
        "border": "#E4A8B0",
        "bg_soft": "linear-gradient(135deg, #FBEAEA 0%, #F4D5D9 100%)",
        "icon": "▼",
    },
    "Neutral": {
        "text": "#8D6516",
        "bg": "#FBF2E2",
        "border": "#E4C783",
        "bg_soft": "linear-gradient(135deg, #FBF2E2 0%, #F5E3BD 100%)",
        "icon": "◆",
    },
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def index():
    initial_stats = get_session_stats()
    return render_template(
        "index.html",
        stats=initial_stats,
    )


# ============================================================
# SESSION STATS ENDPOINT
# ============================================================

@app.route("/session-stats", methods=["GET"])
def session_stats():
    return jsonify(get_session_stats())


# ============================================================
# PREDICT ENDPOINT (Supports both JSON AJAX & Form POST)
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.perf_counter()

    # Extract fields based on request content type
    if request.is_json:
        data = request.get_json() or {}
        review_text = data.get("review_text", "").strip()
        movie_name = data.get("movie_name", "").strip()
        reviewer_name = data.get("reviewer_name", "").strip()
        hybrid_enabled = data.get("hybrid_enabled", True)
        confidence_threshold = float(data.get("confidence_threshold", 0.80))
        margin_threshold = float(data.get("margin_threshold", 0.15))
        is_ajax = True
    else:
        review_text = request.form.get("review_text", "").strip()
        movie_name = request.form.get("movie_name", "").strip()
        reviewer_name = request.form.get("reviewer_name", "").strip()
        hybrid_enabled_str = request.form.get("hybrid_enabled", "on")
        hybrid_enabled = hybrid_enabled_str in {"on", "true", "1", "True"}
        try:
            confidence_threshold = float(request.form.get("confidence_threshold", 0.80))
        except (ValueError, TypeError):
            confidence_threshold = 0.80
        try:
            margin_threshold = float(request.form.get("margin_threshold", 0.15))
        except (ValueError, TypeError):
            margin_threshold = 0.15
        is_ajax = (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.headers.get("Accept", "")
        )

    # Validate review text
    if not review_text:
        error_msg = "Please enter a movie review before submitting."
        if is_ajax:
            return jsonify({"success": False, "error": error_msg}), 400
        return render_template("index.html", error=error_msg, previous_text=review_text)

    # Load and execute model pipeline
    try:
        system = get_model()
        result = system.predict_hybrid(
            review_text,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            hybrid_enabled=hybrid_enabled,
        )
    except Exception as error:
        error_msg = f"Prediction failed: {str(error)}"
        if is_ajax:
            return jsonify({"success": False, "error": error_msg}), 500
        return render_template("index.html", error=error_msg, previous_text=review_text)

    # Measure exact pipeline latency in milliseconds
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # Format result payload
    label = result.get("label", "Neutral")
    engine = result.get("engine", "roberta")
    engine_used = result.get("engine_used", "roberta")
    is_fast_path = result.get("is_fast_path", True)
    routing_reason = result.get("routing_reason", "")
    roberta_confidence = result.get("roberta_confidence", 0.0)
    roberta_margin = result.get("roberta_margin", 0.0)
    roberta_label = result.get("roberta_label", label)
    llm_rationale = result.get("llm_rationale")
    aspects = result.get("aspects")
    llm_unavailable_notice = result.get("llm_unavailable_notice")

    # Order probabilities
    raw_probs = result.get("probabilities", {})
    probabilities_list = order_probabilities(raw_probs)
    confidence_strength = get_confidence_strength(probabilities_list)

    probabilities_dict = [
        {"sentiment": sentiment, "probability": round(prob, 4), "percentage": round(prob * 100, 1)}
        for sentiment, prob in probabilities_list
    ]

    colors = COLOR_MAP.get(
        label,
        {
            "text": "#5C5042",
            "bg": "#FBF7EC",
            "border": "#E7DAB9",
            "bg_soft": "#FBF7EC",
            "icon": "◆",
        }
    )

    # Store entry in session history
    session_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "movie_name": movie_name if movie_name else "Untitled Film",
        "reviewer_name": reviewer_name if reviewer_name else "Anonymous",
        "review_text": review_text,
        "label": label,
        "confidence": roberta_confidence,
        "engine_used": engine_used,
        "latency_ms": elapsed_ms,
    }
    SESSION_HISTORY.append(session_entry)
    stats = get_session_stats()

    # JSON response for SPA client
    if is_ajax:
        return jsonify({
            "success": True,
            "movie_name": movie_name if movie_name else "Untitled Film",
            "reviewer_name": reviewer_name if reviewer_name else "Anonymous",
            "review_text": review_text,
            "word_count": len(review_text.split()),
            "label": label,
            "confidence_strength": confidence_strength,
            "engine": engine.upper(),
            "engine_used": engine_used,
            "is_fast_path": is_fast_path,
            "routing_reason": routing_reason,
            "roberta_confidence": round(roberta_confidence, 4),
            "roberta_confidence_pct": round(roberta_confidence * 100, 1),
            "roberta_margin": round(roberta_margin, 4),
            "roberta_label": roberta_label,
            "llm_rationale": llm_rationale,
            "aspects": aspects,
            "llm_unavailable_notice": llm_unavailable_notice,
            "probabilities": probabilities_dict,
            "colors": colors,
            "latency_ms": elapsed_ms,
            "stats": stats,
        })

    # Fallback to result.html if traditional form post
    return render_template(
        "result.html",
        movie_name=movie_name if movie_name else "Untitled Film",
        reviewer_name=reviewer_name if reviewer_name else "Anonymous",
        review_text=review_text,
        label=label,
        word_count=len(review_text.split()),
        engine=engine.upper(),
        engine_used=engine_used,
        is_fast_path=is_fast_path,
        routing_reason=routing_reason,
        roberta_confidence=roberta_confidence,
        roberta_margin=roberta_margin,
        roberta_label=roberta_label,
        llm_rationale=llm_rationale,
        aspects=aspects,
        llm_unavailable_notice=llm_unavailable_notice,
        colors=colors,
        probabilities=probabilities_list,
        confidence_strength=confidence_strength,
        latency_ms=elapsed_ms,
    )


# ============================================================
# DOWNLOAD REPORT
# ============================================================

@app.route("/download-report", methods=["POST"])
def download_report():
    review_text = request.form.get("review_text", "")
    movie_name = request.form.get("movie_name", "Untitled Film")
    reviewer_name = request.form.get("reviewer_name", "Anonymous")
    label = request.form.get("label", "Unknown")
    engine_used = request.form.get("engine_used", "roberta")
    roberta_confidence = request.form.get("roberta_confidence", "N/A")
    latency_ms = request.form.get("latency_ms", "N/A")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_content = f"""=====================================================
MARQUEE - MOVIE REVIEW SENTIMENT ANALYSIS REPORT
=====================================================

Date & Time      : {timestamp}
Movie Title      : {movie_name}
Reviewer Name    : {reviewer_name}
Sentiment Label  : {label.upper()}
Pipeline Engine  : {engine_used.upper()}
RoBERTa Conf     : {roberta_confidence}
Inference Latency: {latency_ms} ms

-----------------------------------------------------
REVIEW TEXT:
-----------------------------------------------------
{review_text}

=====================================================
Generated by Marquee Hybrid Sentiment Engine
(RoBERTa-base + Groq Llama 3.3 70B Orchestration)
=====================================================
"""

    buffer = BytesIO()
    buffer.write(report_content.encode("utf-8"))
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="movie_sentiment_report.txt",
        mimetype="text/plain",
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("MOVIE REVIEW SENTIMENT ANALYSIS (SPA)")
    print("=" * 65)
    print(f"\nModel: {DEFAULT_MODEL_PATH}")
    print("\nStarting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print("=" * 65)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )