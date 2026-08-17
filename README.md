---
title: Marquee Movie Review Sentiment Analysis
emoji: 🎬
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎬 Marquee — Movie Review Sentiment Analysis

A modern cinema-themed hybrid sentiment analysis web application powered by a fine-tuned **RoBERTa-base Transformer** (`Shirisha-23/movie-review-roberta`) with intelligent LLM escalation (Groq / Gemini) for nuanced sarcasm and aspect-level film breakdown.

## ✨ Key Features
- **Local RoBERTa Transformer**: Ultra-fast (~45ms) 3-class sentiment prediction (Positive / Neutral / Negative).
- **Hybrid Escalation Routing**: High confidence fast-path, escalating ambiguous or mixed reviews to LLMs for deeper insight.
- **Aspect Breakdown**: Performance ratings across Acting, Direction/Pacing, and Plot.
- **Dual UI Support**:
  - **Gradio Interface** ([app.py](app.py)) — Native Hugging Face Spaces UI.
  - **Flask SPA Dashboard** ([app_flask.py](app_flask.py)) — Standalone cinema-themed web app.

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/dolesh123/sentimental-project.git
cd sentimental-project

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask Web App
python app_flask.py

# Or start the Gradio UI
python app.py
```
