# 🎬 Movie Review Sentiment Analysis - Project Architecture & Folder Structure

This document provides a comprehensive breakdown of the directories, core modules, data files, models, and assets in this project.

---

## 🗂️ High-Level Directory Tree

```
sentiment_project/
│
├── 🌐 Web Application Layer
│   ├── app_flask.py                  # Flask web server & REST API endpoints
│   ├── templates/                    # Jinja2 HTML templates
│   │   ├── index.html                # Main SPA interface (dashboard, inputs, charts)
│   │   └── result.html               # Result display template
│   └── static/                       # Static web assets
│       ├── styles.css                # CSS design system (glassmorphism, cinema theme)
│       └── images/                   # UI backgrounds, posters, banners & film stills
│
├── 🧠 Core ML & Inference Pipeline
│   ├── sentiment_analyzer.py         # Main sentiment engine & hybrid routing logic
│   ├── llm_client.py                 # Groq API client (Llama 3.3 70B & 3.1 8B)
│   └── final_roberta_sentiment_model/# Fine-tuned local RoBERTa model weights & config
│       ├── config.json               # Transformer architecture config
│       ├── model.safetensors         # PyTorch weights in SafeTensors format
│       ├── tokenizer.json            # Byte-level BPE tokenizer vocabulary
│       ├── tokenizer_config.json     # Tokenizer settings & special tokens
│       └── training_args.bin         # Training parameters & metadata
│
├── 📊 Evaluation & Testing Suite
│   ├── test_hybrid.py                # Quick verification script for hybrid pipeline
│   ├── evaluate_pipeline.py          # Benchmark evaluation script
│   ├── run_full_evaluation.py        # Comprehensive evaluation across models
│   ├── test_all.csv                  # Test dataset with labeled movie reviews
│   ├── accuracy_results.json         # Output metrics from accuracy benchmarks
│   ├── hybrid_evaluation_results.json# Detailed metrics & confusion matrix for hybrid pipeline
│   └── groq_eval_checkpoint.json     # Checkpointed LLM responses to prevent redundant API calls
│
└── ⚙️ Configuration & Environment
    ├── .env                          # Environment variables (Groq API Key, configs)
    ├── .gitignore                    # Git exclusion rules
    ├── requirements.txt              # Python library dependencies
    └── .venv/ (or venv/)             # Python virtual environment
```

---

## 📋 Detailed Component Explanation

### 1. 🌐 Web Application Layer

| File / Folder | Purpose & Details |
| :--- | :--- |
| [app_flask.py](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/app_flask.py) | **Primary Application Entry Point**.<br>• Initializes Flask app.<br>• Endpoints:<br>&nbsp;&nbsp;- `/`: Renders main Single Page Application (SPA).<br>&nbsp;&nbsp;- `/predict`: JSON & Form API handling review classification requests.<br>&nbsp;&nbsp;- `/session-stats`: Returns live sentiment session telemetry.<br>&nbsp;&nbsp;- `/download-report`: Generates downloadable text summary reports. |
| [templates/](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/templates) | **Jinja2 HTML Templates Directory**.<br>• `index.html`: Responsive single-page interface with real-time analysis, interactive sentiment gauges, aspect sentiment breakdowns (Acting, Direction, Plot, Cinematography), and recent history.<br>• `result.html`: Form-based results view for fallback requests. |
| [static/](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/static) | **Static Web Assets**.<br>• `styles.css`: Complete styling rules (dark modern cinema theme, neon accents, glassmorphism, responsive grid).<br>• `images/`: High-resolution background graphics, film posters, cinema badges, and film stills. |

---

### 2. 🧠 Core ML & Inference Pipeline

| File / Folder | Purpose & Details |
| :--- | :--- |
| [sentiment_analyzer.py](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/sentiment_analyzer.py) | **The Core Sentiment Classification Engine**.<br>• Implements `SentimentSystem` & `HybridSentimentPipeline`.<br>• Runs the local fine-tuned RoBERTa model for fast inference (~10ms).<br>• Evaluates prediction confidence and ambiguity scores.<br>• If confidence is below threshold or ambiguity is detected (e.g., mixed sarcasm or subtle critique), escalates to Groq LLM for deep reasoning. |
| [llm_client.py](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/llm_client.py) | **Groq Cloud LLM Integration**.<br>• Communicates with Groq API running `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.<br>• Performs advanced aspect-based sentiment extraction, sarcasm detection, tone analysis, and structured JSON output generation. |
| [final_roberta_sentiment_model/](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/final_roberta_sentiment_model) | **Fine-Tuned RoBERTa Transformer Directory**.<br>• `model.safetensors`: Binary weights for the custom sentiment classification head.<br>• `config.json`: Architecture config (layers, attention heads, hidden dimensions, label mapping).<br>• `tokenizer.json` & `tokenizer_config.json`: BPE vocab mapping and tokenizer config. |

---

### 3. 📊 Evaluation & Testing Suite

| File / Folder | Purpose & Details |
| :--- | :--- |
| [test_hybrid.py](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/test_hybrid.py) | Quick smoke test script. Runs sample positive, negative, sarcastic, and ambiguous reviews through the hybrid analyzer and prints results. |
| [evaluate_pipeline.py](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/evaluate_pipeline.py) | Evaluates accuracy, precision, recall, and F1 score against the benchmark test set. |
| [run_full_evaluation.py](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/run_full_evaluation.py) | In-depth benchmarking script comparing: <br>1. Standalone RoBERTa<br>2. Standalone Groq Llama 3.3 70B<br>3. Hybrid Pipeline (RoBERTa + Groq Escalation). |
| [test_all.csv](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/test_all.csv) | CSV dataset with labeled review texts (`text`, `sentiment`) used for benchmarking. |
| `accuracy_results.json` & `hybrid_evaluation_results.json` | Stores recorded benchmark metrics and confusion matrix results for quick reference. |
| `groq_eval_checkpoint.json` | Checkpoint cache preserving LLM API evaluations to save cost and avoid duplicate calls during re-evaluations. |

---

### 4. ⚙️ Configuration & Environment

| File / Folder | Purpose & Details |
| :--- | :--- |
| [.env](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/.env) | Stores secret keys and runtime variables (e.g., `GROQ_API_KEY`). |
| [requirements.txt](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/requirements.txt) | Python dependencies needed to run the application (`Flask`, `torch`, `transformers`, `scikit-learn`, `pandas`, `datasets`, `accelerate`). |
| [.gitignore](file:///c:/Users/USER/Desktop/CTS-Hackathon/Movie_Review_Project/sentiment_project_main/sentiment_project/.gitignore) | Configures files that should not be tracked by Git (`__pycache__`, virtual environments, checkpoints, `.env`). |
| `.venv/` / `venv/` | Isolated Python virtual environment containing the installed packages. |
