"""
Movie Review Sentiment Analysis
================================

Main NLP Model:
    RoBERTa-base fine-tuned for 3-class sentiment classification

Classes:
    0 = Negative
    1 = Neutral
    2 = Positive

Trained model:
    final_roberta_sentiment_model/

This file is used for:
    1. Loading the trained RoBERTa model
    2. Predicting sentiment
    3. Batch prediction
    4. Model evaluation
    5. Interactive prediction
"""

# ============================================================
# IMPORTS
# ============================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# TRANSFORMERS / PYTORCH
# ============================================================

try:
    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    TRANSFORMERS_AVAILABLE = True

except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Hugging Face model repository
DEFAULT_MODEL_PATH = "Shirisha-23/movie-review-roberta"


# ============================================================
# LABEL INFORMATION
# ============================================================

CLASS_ORDER = [
    "Negative",
    "Neutral",
    "Positive",
]

LABEL_TO_ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


# ============================================================
# LABEL NORMALIZATION
# ============================================================

LABEL_MAP = {

    "0": "Negative",
    "1": "Neutral",
    "2": "Positive",

    "negative": "Negative",
    "neg": "Negative",

    "neutral": "Neutral",
    "neu": "Neutral",

    "positive": "Positive",
    "pos": "Positive",
}


def normalize_label(value):

    value = str(value).strip().lower()

    if value in LABEL_MAP:
        return LABEL_MAP[value]

    raise ValueError(
        f"""
Unknown sentiment label: {value}

Expected:

0 = Negative
1 = Neutral
2 = Positive

or:

Negative
Neutral
Positive
"""
    )


# ============================================================
# ROBERTA SENTIMENT ANALYZER
# ============================================================

class MLSentimentAnalyzer:

    """
    Main machine-learning sentiment analyzer.

    Uses:
        RoBERTa-base

    Fine-tuned for:
        Negative / Neutral / Positive

    Model folder:
        final_roberta_sentiment_model/
    """

    MODEL_NAME = "roberta-base"

    DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH

    MAX_LENGTH = 256

    LABEL_TO_ID = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    }

    ID_TO_LABEL = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        model_path=None,
    ):

        self.model_path = (
            Path(model_path)
            if model_path
            else self.DEFAULT_MODEL_PATH
        )

        self.tokenizer = None
        self.model = None
        self.is_trained = False
        self.device = None

    # ========================================================
    # REQUIRE TRANSFORMERS
    # ========================================================

    def _require_transformers(self):

        if not TRANSFORMERS_AVAILABLE:

            raise RuntimeError(
                """
Transformers/PyTorch are not installed.

Activate your virtual environment and run:

pip install torch transformers
"""
            )


    def load(self, path=None):

        self._require_transformers()

        # Use the provided path/repository,
        # otherwise use the default Hugging Face repository.
        model_source = (
            path
            if path is not None
            else self.model_path
        )

        # Convert Path objects to strings
        model_source = str(model_source)

        print("=" * 65)
        print("LOADING ROBERTA SENTIMENT MODEL")
        print("=" * 65)

        # ----------------------------------------------------
        # Detect whether model is local or from Hugging Face
        # ----------------------------------------------------

        local_path = Path(model_source)

        if local_path.is_dir():

            # Local model
            model_source = str(local_path.resolve())

            print(
                "\nModel source: LOCAL"
            )

            print(
                f"Model path:\n{model_source}"
            )

        else:
            # Ensure Hugging Face repo ID uses forward slashes
            model_source = model_source.replace("\\", "/")

            # Hugging Face repository
            print(
                "\nModel source: HUGGING FACE"
            )

            print(
                f"Model repository:\n{model_source}"
            )

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        print("\nLoading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        print("Loading RoBERTa model...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source
        )

        # ----------------------------------------------------
        # Select device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(
            self.device
        )

        # Evaluation mode
        self.model.eval()

        self.is_trained = True

        print(
            f"\nDevice: {self.device}"
        )

        print(
            "RoBERTa model loaded successfully."
        )

        print(
            f"Classes: {CLASS_ORDER}"
        )
    # ========================================================
    # PREDICT ONE REVIEW
    # ========================================================

    def predict(self, text):

        # Load model automatically if necessary
        if not self.is_trained:
            self.load()

        text = str(text).strip()

        # Empty input
        if not text:

            return {
                "label": "Neutral",

                "probabilities": {
                    "Negative": 0.0,
                    "Neutral": 1.0,
                    "Positive": 0.0,
                },

                "engine": "roberta",
            }

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = self.tokenizer(

            text,

            truncation=True,

            max_length=self.MAX_LENGTH,

            padding=True,

            return_tensors="pt",
        )

        # Move tensors to device
        inputs = {

            key: value.to(
                self.device
            )

            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )[0]

        # Convert probabilities to NumPy
        probabilities = (
            probabilities
            .cpu()
            .numpy()
        )

        # Find highest probability class
        predicted_id = int(
            np.argmax(
                probabilities
            )
        )

        # ----------------------------------------------------
        # Probability dictionary
        # ----------------------------------------------------

        probability_dict = {

            self.ID_TO_LABEL[index]:

            round(
                float(
                    probabilities[index]
                ),
                4,
            )

            for index in range(3)
        }

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {

            "label":
                self.ID_TO_LABEL[
                    predicted_id
                ],

            "probabilities":
                probability_dict,

            "engine":
                "roberta",
        }

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(
        self,
        texts,
        batch_size=16,
    ):

        if not self.is_trained:
            self.load()

        texts = [
            str(text)
            for text in texts
        ]

        all_predictions = []

        total = len(texts)

        for start in range(
            0,
            total,
            batch_size,
        ):

            batch = texts[
                start:
                start + batch_size
            ]

            # Tokenize batch
            inputs = self.tokenizer(

                batch,

                padding=True,

                truncation=True,

                max_length=self.MAX_LENGTH,

                return_tensors="pt",
            )

            # Move to device
            inputs = {

                key: value.to(
                    self.device
                )

                for key, value
                in inputs.items()
            }

            # Predict
            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1,
                )

                predictions = torch.argmax(
                    probabilities,
                    dim=1,
                )

            predictions = (
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            for prediction in predictions:

                all_predictions.append(

                    self.ID_TO_LABEL[
                        int(prediction)
                    ]

                )

        return all_predictions

    # ========================================================
    # BATCH NUMERIC PREDICTION
    # ========================================================

    def predict_batch_ids(
        self,
        texts,
        batch_size=16,
    ):

        if not self.is_trained:
            self.load()

        texts = [
            str(text)
            for text in texts
        ]

        all_predictions = []

        total = len(texts)

        for start in range(
            0,
            total,
            batch_size,
        ):

            batch = texts[
                start:
                start + batch_size
            ]

            # Tokenization
            inputs = self.tokenizer(

                batch,

                padding=True,

                truncation=True,

                max_length=self.MAX_LENGTH,

                return_tensors="pt",
            )

            # Move tensors
            inputs = {

                key: value.to(
                    self.device
                )

                for key, value
                in inputs.items()
            }

            # Prediction
            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

                predictions = torch.argmax(
                    outputs.logits,
                    dim=1,
                )

            all_predictions.extend(

                predictions
                .cpu()
                .numpy()
                .tolist()

            )

        return all_predictions

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    def evaluate(
        self,
        test_path,
        batch_size=16,
    ):

        self._require_transformers()

        if not self.is_trained:
            self.load()

        print("\n" + "=" * 65)
        print("ROBERTA MODEL EVALUATION")
        print("=" * 65)

        test_path = Path(test_path)

        if not test_path.is_absolute():
            test_path = BASE_DIR / test_path

        if not test_path.exists():

            raise FileNotFoundError(
                f"""
Test dataset not found:

{test_path}
"""
            )

        print(
            "\nLoading test dataset:"
        )

        print(
            test_path
        )

        # ----------------------------------------------------
        # Load CSV / TSV
        # ----------------------------------------------------

        if test_path.suffix.lower() == ".tsv":

            df = pd.read_csv(
                test_path,
                sep="\t",
            )

        else:

            df = pd.read_csv(
                test_path
            )

        # Normalize column names
        df.columns = [

            str(column)
            .strip()
            .lower()

            for column
            in df.columns

        ]

        # ----------------------------------------------------
        # Detect dataset columns
        # ----------------------------------------------------

        if (
            "sentence" in df.columns
            and "label" in df.columns
        ):

            text_column = "sentence"
            label_column = "label"

        elif (
            "review" in df.columns
            and "sentiment" in df.columns
        ):

            text_column = "review"
            label_column = "sentiment"

        elif (
            "review" in df.columns
            and "label" in df.columns
        ):

            text_column = "review"
            label_column = "label"

        else:

            raise ValueError(
                """
Test dataset must contain one of:

sentence + label

review + sentiment

review + label
"""
            )

        # Remove missing values
        df = df.dropna(
            subset=[
                text_column,
                label_column,
            ]
        )

        # Convert labels
        df["normalized_label"] = (

            df[label_column]
            .apply(normalize_label)

        )

        df[text_column] = (

            df[text_column]
            .astype(str)

        )

        print(
            f"\nTest samples: {len(df):,}"
        )

        # ----------------------------------------------------
        # True labels
        # ----------------------------------------------------

        y_true = [

            self.LABEL_TO_ID[
                label
            ]

            for label
            in df["normalized_label"]

        ]

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        print(
            "\nRunning predictions..."
        )

        y_pred = (

            self.predict_batch_ids(

                df[text_column].tolist(),

                batch_size=batch_size,

            )

        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        report = classification_report(

            y_true,

            y_pred,

            labels=[
                0,
                1,
                2,
            ],

            target_names=CLASS_ORDER,

            digits=4,

            zero_division=0,

        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        matrix = confusion_matrix(

            y_true,

            y_pred,

            labels=[
                0,
                1,
                2,
            ],

        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("FINAL RESULTS")
        print("=" * 65)

        print(
            f"\nAccuracy  : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision : "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall    : "
            f"{recall * 100:.2f}%"
        )

        print(
            f"F1 Score  : "
            f"{f1 * 100:.2f}%"
        )

        print(
            "\nClassification Report:"
        )

        print(
            report
        )

        print(
            "Confusion Matrix:"
        )

        print(
            matrix
        )

        return {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "classification_report":
                report,

            "confusion_matrix":
                matrix.tolist(),

            "n_test":
                len(df),

        }

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save(self, path):

        self._require_transformers()

        if self.model is None:

            raise RuntimeError(
                "No RoBERTa model is loaded."
            )

        path = Path(path)

        if not path.is_absolute():
            path = BASE_DIR / path

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save model
        self.model.save_pretrained(
            path
        )

        # Save tokenizer
        self.tokenizer.save_pretrained(
            path
        )

        # Save configuration
        with open(

            path /
            "sentiment_model_config.json",

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(

                {

                    "model_type":
                        "roberta-base",

                    "max_length":
                        self.MAX_LENGTH,

                    "class_order":
                        CLASS_ORDER,

                },

                file,

                indent=4,

            )

        print(
            "Model saved to:"
        )

        print(
            path.resolve()
        )


# ============================================================
# SENTIMENT SYSTEM
# ============================================================

class SentimentSystem:

    """
    Main application-level sentiment system.

    It loads the already-trained RoBERTa model.

    It does NOT train the model during prediction.
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        auto_load=True,
    ):

        self.ml = MLSentimentAnalyzer(

            model_path=model_path

        )

        if auto_load:

            try:

                self.ml.load(
                    model_path
                )

            except Exception as error:

                print(
                    "\nWARNING: "
                    "Could not load RoBERTa model."
                )

                print(error)

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, text):

        text = str(text).strip()

        if not text:

            return {

                "label":
                    "Neutral",

                "probabilities":
                {
                    "Negative": 0.0,

                    "Neutral": 1.0,

                    "Positive": 0.0,
                },

                "engine":
                    "none",
            }

        # Use RoBERTa
        if self.ml.is_trained:

            return self.ml.predict(
                text
            )

        # Model failed to load
        return {

            "label":
                "Neutral",

            "probabilities":
            {
                "Negative": 0.0,

                "Neutral": 1.0,

                "Positive": 0.0,
            },

            "engine":
                "error",
        }

    # ========================================================
    # PREDICT HYBRID (RoBERTa + LLM Fallback)
    # ========================================================

    def predict_hybrid(
        self,
        text,
        confidence_threshold=0.80,
        margin_threshold=0.15,
        hybrid_enabled=True,
        llm_provider="groq",
    ):
        """
        Hybrid Sentiment Routing:
        1. Run local RoBERTa inference first.
        2. Check confidence and top-2 margin.
        3. IF top_prob >= confidence_threshold AND margin >= margin_threshold:
               Fast path (Return RoBERTa).
           ELSE:
               Escalate to Groq LLM (Llama 3.3 70B) for deep aspect & rationale analysis.
        4. If Groq is unavailable or fails, gracefully fall back to RoBERTa.
        """
        text = str(text).strip()

        if not text:
            return {
                "label": "Neutral",
                "probabilities": {
                    "Negative": 0.0,
                    "Neutral": 1.0,
                    "Positive": 0.0,
                },
                "engine": "none",
                "engine_used": "none",
                "roberta_label": "Neutral",
                "roberta_confidence": 1.0,
                "roberta_margin": 1.0,
                "roberta_probabilities": {
                    "Negative": 0.0,
                    "Neutral": 1.0,
                    "Positive": 0.0,
                },
                "is_fast_path": True,
                "routing_reason": "Empty input",
                "llm_rationale": None,
                "aspects": None,
                "llm_unavailable_notice": None,
            }

        # Step 1: Run RoBERTa
        roberta_res = self.predict(text)
        roberta_label = roberta_res.get("label", "Neutral")
        roberta_probs = roberta_res.get(
            "probabilities",
            {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
        )

        # Step 2: Compute top probability and margin between top-2
        sorted_probs = sorted(
            roberta_probs.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_label, top_prob = sorted_probs[0]
        second_label, second_prob = (
            sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)
        )
        margin = round(float(top_prob - second_prob), 4)
        top_prob = round(float(top_prob), 4)

        # Step 3: Check Fast-Path Condition
        is_confident = (top_prob >= confidence_threshold) and (margin >= margin_threshold)

        if not hybrid_enabled or is_confident:
            return {
                "label": roberta_label,
                "probabilities": roberta_probs,
                "engine": "roberta",
                "engine_used": "roberta",
                "roberta_label": roberta_label,
                "roberta_confidence": top_prob,
                "roberta_margin": margin,
                "roberta_probabilities": roberta_probs,
                "is_fast_path": is_confident,
                "routing_reason": (
                    f"Fast-Path RoBERTa (Confidence: {top_prob*100:.1f}%, Margin: {margin*100:.1f}%)"
                    if is_confident
                    else "Hybrid mode disabled"
                ),
                "llm_rationale": None,
                "aspects": None,
                "llm_unavailable_notice": None,
            }

        # Step 4: Fallback / Escalation to LLM (Groq)
        routing_parts = []
        if top_prob < confidence_threshold:
            routing_parts.append(f"Confidence {top_prob*100:.1f}% < {confidence_threshold*100:.0f}%")
        if margin < margin_threshold:
            routing_parts.append(f"Margin {margin*100:.1f}% < {margin_threshold*100:.0f}%")
        routing_reason_str = " — ".join(routing_parts) if routing_parts else "Ambiguous sentiment"

        try:
            from llm_client import LLMFactory

            client = LLMFactory.get_client("groq")

            if not client.is_available():
                return {
                    "label": roberta_label,
                    "probabilities": roberta_probs,
                    "engine": "roberta",
                    "engine_used": "roberta",
                    "roberta_label": roberta_label,
                    "roberta_confidence": top_prob,
                    "roberta_margin": margin,
                    "roberta_probabilities": roberta_probs,
                    "is_fast_path": False,
                    "routing_reason": f"Escalated ({routing_reason_str}) but GROQ_API_KEY missing",
                    "llm_rationale": None,
                    "aspects": None,
                    "llm_unavailable_notice": "Groq LLM fallback unavailable — showing RoBERTa result.",
                }

            llm_res = client.analyze_sentiment(text)

            if not llm_res.get("success", False):
                return {
                    "label": roberta_label,
                    "probabilities": roberta_probs,
                    "engine": "roberta",
                    "engine_used": "roberta",
                    "roberta_label": roberta_label,
                    "roberta_confidence": top_prob,
                    "roberta_margin": margin,
                    "roberta_probabilities": roberta_probs,
                    "is_fast_path": False,
                    "routing_reason": f"Escalated ({routing_reason_str}) but Groq LLM call failed",
                    "llm_rationale": None,
                    "aspects": None,
                    "llm_unavailable_notice": "Groq LLM fallback unavailable — showing RoBERTa result.",
                    "llm_error": llm_res.get("error"),
                }

            llm_sentiment = llm_res.get("sentiment", roberta_label)
            llm_conf = llm_res.get("confidence", 0.90)

            # Generate synthetic probability distribution matching LLM decision
            rem = max(0.0, round(1.0 - llm_conf, 4))
            other_classes = [
                c for c in ["Negative", "Neutral", "Positive"]
                if c != llm_sentiment
            ]
            synthetic_probs = {
                llm_sentiment: round(llm_conf, 4),
                other_classes[0]: round(rem / 2, 4),
                other_classes[1]: round(rem / 2, 4),
            }

            model_name = llm_res.get("model", "llama-3.3-70b-versatile")

            return {
                "label": llm_sentiment,
                "probabilities": synthetic_probs,
                "engine": "groq",
                "engine_used": "llm",
                "roberta_label": roberta_label,
                "roberta_confidence": top_prob,
                "roberta_margin": margin,
                "roberta_probabilities": roberta_probs,
                "is_fast_path": False,
                "routing_reason": f"Groq ({model_name}) Escalation (RoBERTa was {top_prob*100:.1f}% — {routing_reason_str})",
                "llm_rationale": llm_res.get("rationale"),
                "aspects": llm_res.get("aspects"),
                "llm_model": model_name,
                "llm_unavailable_notice": None,
            }

        except Exception as exc:
            return {
                "label": roberta_label,
                "probabilities": roberta_probs,
                "engine": "roberta",
                "engine_used": "roberta",
                "roberta_label": roberta_label,
                "roberta_confidence": top_prob,
                "roberta_margin": margin,
                "roberta_probabilities": roberta_probs,
                "is_fast_path": False,
                "routing_reason": f"Escalated ({routing_reason_str}) but exception occurred",
                "llm_rationale": None,
                "aspects": None,
                "llm_unavailable_notice": "LLM fallback unavailable — showing RoBERTa result.",
                "llm_error": str(exc),
            }


# ============================================================
# COMMAND: PREDICT
# ============================================================

def cmd_predict(args):

    system = SentimentSystem(

        model_path=args.model

    )

    result = system.predict(
        args.text
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "SENTIMENT PREDICTION"
    )

    print(
        "=" * 65
    )

    print(
        f"\nReview:\n{args.text}"
    )

    print(
        f"\nSentiment: "
        f"{result['label']}"
    )

    print(
        f"Engine: "
        f"{result['engine']}"
    )

    if "probabilities" in result:

        print(
            "\nPrediction Probabilities:"
        )

        for (

            sentiment,
            probability

        ) in result[
            "probabilities"
        ].items():

            print(

                f"{sentiment}: "
                f"{probability * 100:.2f}%"

            )


# ============================================================
# COMMAND: EVALUATE
# ============================================================

def cmd_evaluate(args):

    analyzer = MLSentimentAnalyzer(

        model_path=args.model

    )

    analyzer.load(
        args.model
    )

    result = analyzer.evaluate(

        test_path=args.test,

        batch_size=args.batch_size,

    )

    output = {

        "model":
            "roberta-base fine-tuned for 3-class sentiment",

        "test_samples":
            result["n_test"],

        "accuracy":
            result["accuracy"],

        "precision_weighted":
            result["precision"],

        "recall_weighted":
            result["recall"],

        "f1_weighted":
            result["f1"],

        "classification_report":
            result[
                "classification_report"
            ],

        "confusion_matrix":
            result[
                "confusion_matrix"
            ],

    }

    output_path = (
        BASE_DIR /
        "accuracy_results.json"
    )

    with open(

        output_path,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            output,

            file,

            indent=4,

        )

    print(
        "\nResults saved to:"
    )

    print(
        output_path.resolve()
    )


# ============================================================
# COMMAND: INTERACTIVE
# ============================================================

def cmd_interactive(args):

    system = SentimentSystem(

        model_path=args.model

    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "MOVIE REVIEW SENTIMENT ANALYZER"
    )

    print(
        "=" * 65
    )

    if system.ml.is_trained:

        print(
            "\nEngine: RoBERTa Transformer"
        )

    else:

        print(
            "\nEngine: RoBERTa model unavailable"
        )

    print(
        "\nClasses:"
    )

    print(
        "0 = Negative"
    )

    print(
        "1 = Neutral"
    )

    print(
        "2 = Positive"
    )

    print(
        "\nType 'quit' to exit."
    )

    while True:

        try:

            text = input(
                "\nReview > "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()

            break

        if text.lower() in {
            "quit",
            "exit",
        }:

            break

        if not text:

            continue

        result = system.predict(
            text
        )

        print(
            "\n"
            + "-" * 50
        )

        print(
            f"Sentiment: "
            f"{result['label']}"
        )

        print(
            f"Engine: "
            f"{result['engine']}"
        )

        if "probabilities" in result:

            print(
                "\nPrediction Probabilities:"
            )

            for (

                sentiment,
                probability

            ) in result[
                "probabilities"
            ].items():

                print(

                    f"{sentiment}: "
                    f"{probability * 100:.2f}%"

                )

        print(
            "-" * 50
        )


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_arg_parser():

    parser = argparse.ArgumentParser(

        description=(
            "Movie Review Sentiment Analysis "
            "using fine-tuned RoBERTa"
        )

    )

    subparsers = parser.add_subparsers(

        dest="command",

        required=True,

    )

    # ========================================================
    # PREDICT
    # ========================================================

    predict_parser = (

        subparsers.add_parser(

            "predict",

            help="Predict sentiment",

        )

    )

    predict_parser.add_argument(

        "--text",

        required=True,

    )

    predict_parser.add_argument(

        "--model",

        default=str(
            DEFAULT_MODEL_PATH
        ),

    )

    predict_parser.set_defaults(

        func=cmd_predict

    )

    # ========================================================
    # EVALUATE
    # ========================================================

    evaluate_parser = (

        subparsers.add_parser(

            "evaluate",

            help="Evaluate saved RoBERTa model",

        )

    )

    evaluate_parser.add_argument(

        "--test",

        required=True,

        help="Path to test CSV/TSV file",

    )

    evaluate_parser.add_argument(

        "--model",

        default=str(
            DEFAULT_MODEL_PATH
        ),

    )

    evaluate_parser.add_argument(

        "--batch-size",

        type=int,

        default=16,

    )

    evaluate_parser.set_defaults(

        func=cmd_evaluate

    )

    # ========================================================
    # INTERACTIVE
    # ========================================================

    interactive_parser = (

        subparsers.add_parser(

            "interactive",

            help="Interactive sentiment prediction",

        )

    )

    interactive_parser.add_argument(

        "--model",

        default=str(
            DEFAULT_MODEL_PATH
        ),

    )

    interactive_parser.set_defaults(

        func=cmd_interactive

    )

    return parser


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = build_arg_parser()

    args = parser.parse_args()

    args.func(args)