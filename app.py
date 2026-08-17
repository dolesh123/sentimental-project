"""
Marquee Movie Review Sentiment Analysis - Gradio UI
===================================================
Native Hugging Face Spaces entry point (Gradio SDK).
Supports RoBERTa-base classification + Groq LLM Hybrid Escalation.
"""

import gradio as gr
from sentiment_analyzer import SentimentSystem, DEFAULT_MODEL_PATH

# Initialize the sentiment engine (lazy load / singleton)
SYSTEM = None

def get_system():
    global SYSTEM
    if SYSTEM is None:
        SYSTEM = SentimentSystem(model_path=DEFAULT_MODEL_PATH)
    return SYSTEM


def analyze_review(movie_name, reviewer_name, review_text, confidence_threshold, margin_threshold, hybrid_enabled):
    if not review_text or not review_text.strip():
        return "Please enter a movie review.", {}, "⚠️ *Please type a review before analyzing.*"

    system = get_system()
    result = system.predict_hybrid(
        review_text.strip(),
        confidence_threshold=float(confidence_threshold),
        margin_threshold=float(margin_threshold),
        hybrid_enabled=hybrid_enabled,
    )

    label = result.get("label", "Neutral")
    probabilities = result.get("probabilities", {})
    engine_used = result.get("engine_used", "roberta")
    is_fast_path = result.get("is_fast_path", True)
    routing_reason = result.get("routing_reason", "")
    roberta_conf = result.get("roberta_confidence", 0.0)
    aspects = result.get("aspects")
    rationale = result.get("llm_rationale")
    llm_notice = result.get("llm_unavailable_notice")

    # Format sentiment badge & header
    movie_title = movie_name.strip() if movie_name and movie_name.strip() else "Untitled Film"
    reviewer = reviewer_name.strip() if reviewer_name and reviewer_name.strip() else "Anonymous"

    # Markdown Summary Card
    details = f"## 🎬 Verdict for *{movie_title}*\n"
    details += f"**Reviewer:** {reviewer} | **Word Count:** {len(review_text.split())} words\n\n"
    details += f"- **Predicted Sentiment:** **`{label.upper()}`**\n"
    details += f"- **Inference Engine:** `{engine_used.upper()}` ({'Fast-Path' if is_fast_path else 'Escalated'})\n"
    details += f"- **RoBERTa Baseline Confidence:** `{roberta_conf * 100:.1f}%`\n"
    details += f"- **Routing Rationale:** {routing_reason}\n\n"

    if llm_notice:
        details += f"> ℹ️ *{llm_notice}*\n\n"

    if rationale:
        details += f"### 🧠 Critic Rationale\n{rationale}\n\n"

    if aspects and isinstance(aspects, dict):
        details += "### 🎭 Aspect Breakdown\n"
        for aspect_name, aspect_verdict in aspects.items():
            details += f"- **{aspect_name}:** {aspect_verdict}\n"

    # Return structured values for Gradio components
    return label, probabilities, details


# Build Custom Gradio Interface
custom_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
)

with gr.Blocks(theme=custom_theme, title="Marquee — Movie Review Sentiment Analysis") as demo:
    gr.Markdown(
        """
        # 🎬 MARQUEE — Movie Review Sentiment Analysis
        ### AI For Cinema Lovers • Fine-tuned RoBERTa Transformer + LLM Hybrid Escalation
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ✍️ Enter Review Details")
            movie_input = gr.Textbox(
                label="Movie Title",
                placeholder="e.g. Oppenheimer, Dune: Part Two, The Godfather...",
            )
            reviewer_input = gr.Textbox(
                label="Reviewer Name (Optional)",
                placeholder="e.g. Roger Ebert, Jane Doe...",
            )
            review_input = gr.Textbox(
                label="Movie Review Text",
                placeholder="Write or paste your film review here...",
                lines=6,
            )

            with gr.Accordion("⚙️ Hybrid Routing Settings", open=False):
                hybrid_toggle = gr.Checkbox(
                    label="Enable LLM Escalation for Ambiguous Reviews",
                    value=True,
                )
                conf_slider = gr.Slider(
                    minimum=0.50,
                    maximum=0.95,
                    value=0.80,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Escalate to LLM if RoBERTa confidence is below this value.",
                )
                margin_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.30,
                    value=0.15,
                    step=0.05,
                    label="Top-2 Margin Threshold",
                    info="Escalate to LLM if difference between top 2 classes is smaller than this.",
                )

            analyze_btn = gr.Button("⚡ Analyze Sentiment", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis & Telemetry")
            sentiment_badge = gr.Label(label="Overall Sentiment Verdict")
            prob_chart = gr.Label(label="Class Probabilities (RoBERTa / Hybrid)")
            details_card = gr.Markdown("Submit a review to generate insights.")

    analyze_btn.click(
        fn=analyze_review,
        inputs=[
            movie_input,
            reviewer_input,
            review_input,
            conf_slider,
            margin_slider,
            hybrid_toggle,
        ],
        outputs=[sentiment_badge, prob_chart, details_card],
    )

    gr.Markdown("---")
    gr.Markdown("### 💡 Quick Examples (Click to test)")
    gr.Examples(
        examples=[
            [
                "Interstellar",
                "Alice",
                "An incredible cinematic achievement with breathtaking visuals, magnificent score by Hans Zimmer, and heartfelt emotional depth.",
                0.80,
                0.15,
                True,
            ],
            [
                "Disaster Movie",
                "Bob",
                "Completely unwatchable garbage. Terrible acting, pathetic dialogue, and an utter waste of time and money.",
                0.80,
                0.15,
                True,
            ],
            [
                "The Matrix Resurrections",
                "Charlie",
                "Visually stunning and ambitious in concept, but the sluggish narrative and convoluted meta-humor left me feeling somewhat indifferent.",
                0.80,
                0.15,
                True,
            ],
        ],
        inputs=[
            movie_input,
            reviewer_input,
            review_input,
            conf_slider,
            margin_slider,
            hybrid_toggle,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
