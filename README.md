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

A modern cinema-themed hybrid sentiment analysis application powered by a fine-tuned **RoBERTa-base Transformer** (`Shirisha-23/movie-review-roberta`) with intelligent LLM escalation (Groq / Gemini) for nuanced sarcasm and aspect-level film critique.

---

## ✨ Key Features

- **Local RoBERTa-base Transformer**: Ultra-fast (~45ms) 3-class sentiment prediction (**Positive**, **Neutral**, **Negative**).
- **Hybrid Escalation Routing**: High-confidence fast path routes to RoBERTa; ambiguous, low-margin, or sarcastic reviews escalate to Groq Llama 3.3 70B / Gemini.
- **Aspect Breakdown**: Structured film critique across **Acting**, **Direction/Pacing**, and **Plot/Storyline**.
- **Dual UI Support**:
  - **Flask SPA Dashboard** (`app_flask.py`) — Custom cinema-themed dark interface with glassmorphism, real-time AJAX updates, confidence gauges, and report export.
  - **Gradio Interface** (`app.py`) — Native Hugging Face Spaces interface with interactive threshold controls and click-to-test examples.

---

## 🚀 Running Locally

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/dolesh123/sentimental-project.git
cd sentimental-project

# Initialize virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
# Start the Flask Web Application (Runs on http://127.0.0.1:5000)
python app_flask.py

# Or start the Gradio UI (Runs on http://127.0.0.1:7860)
python app.py
```

### 3. (Optional) Configure Hybrid LLM Reasoning

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🌐 Deployment Options

### Method 1: Instant Live Demo via Cloudflare Tunnel (Zero Cloud Setup)

Broadcast your locally running app to a secure public HTTPS URL instantly:

```powershell
# 1. Start the Flask app in Terminal 1
python app_flask.py

# 2. In Terminal 2, start Cloudflare Tunnel
.\cloudflared.exe tunnel --url http://localhost:5000
```
> Copy the generated `https://xxxx.trycloudflare.com` URL to share your live app with anyone in the world!

---

### Method 2: Render.com (100% Free 24/7 Cloud Hosting)

1. Sign up at **[Render.com](https://render.com)** and click **"New +"** → **"Web Service"**.
2. Connect your GitHub repository: `dolesh123/sentimental-project`.
3. Configure settings:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app_flask:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Plan**: `Free`
4. Click **"Create Web Service"**.

---

### Method 3: Hugging Face Spaces (Free 16GB RAM)

1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)** → Click **"Create new Space"**.
2. Choose **SDK**: `Gradio` (or `Docker`).
3. Hardware: Select **`Free (2 vCPU · 16 GB RAM)`**.
4. In **Settings** → **Space repository**, sync with:
   `https://github.com/dolesh123/sentimental-project`

---

### Method 4: Railway.app / Docker Deployment

Deploy directly using the included `Dockerfile` or `Procfile`:

```bash
# Build the Docker container
docker build -t movie-sentiment-app .

# Run the container
docker run -p 5000:5000 movie-sentiment-app
```

---

## 💻 CLI & Testing Tools

```bash
# Interactive terminal sentiment predictor
python sentiment_analyzer.py interactive

# Single review prediction via CLI
python sentiment_analyzer.py predict --text "A cinematic triumph with brilliant cinematography!"

# Run pipeline smoke tests
python test_hybrid.py

# Run full evaluation benchmark
python evaluate_pipeline.py
```

---

## 📄 License
This project is licensed under the MIT License.
