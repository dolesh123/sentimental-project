"""
Evaluation Script: RoBERTa Baseline vs Hybrid Pipeline (RoBERTa + Groq Llama 3.3 70B)
Dataset: test_all.csv (6,530 rows)
"""

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from dotenv import load_dotenv

# Load .env
load_dotenv()

from sentiment_analyzer import SentimentSystem, normalize_label, ID_TO_LABEL, CLASS_ORDER
from llm_client import GroqClient

def evaluate():
    print("=" * 70, flush=True)
    print("STARTING FULL EVALUATION: ROBERTA BASELINE VS HYBRID (ROBERTA + GROQ)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load test_all.csv
    csv_path = "test_all.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows from {csv_path}.", flush=True)
    df["true_label"] = df["label"].apply(normalize_label)
    df["sentence"] = df["sentence"].astype(str)

    # 2. Check GROQ API Key
    groq_client = GroqClient()
    if not groq_client.is_available():
        raise RuntimeError("GROQ_API_KEY is not set or not available in .env!")
    print("GROQ_API_KEY is set. Model:", groq_client.model, flush=True)

    # 3. Load RoBERTa Model
    system = SentimentSystem()
    ml_model = system.ml

    # 4. RoBERTa Inference on all rows
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: RUNNING ROBERTA INFERENCE ON ALL 6,530 SAMPLES...", flush=True)
    print("=" * 70, flush=True)

    texts = df["sentence"].tolist()
    batch_size = 64
    all_preds = []
    all_top_probs = []
    all_margins = []
    all_escalations = []
    roberta_sample_latencies = []

    t0 = time.time()
    for i in range(0, total_rows, batch_size):
        b_texts = texts[i : i + batch_size]
        t_batch_start = time.time()
        
        inputs = ml_model.tokenizer(
            b_texts,
            padding=True,
            truncation=True,
            max_length=ml_model.MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(ml_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = ml_model.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        batch_time = time.time() - t_batch_start
        per_item_time = batch_time / len(b_texts)

        for p_row in probs:
            s_idx = np.argsort(p_row)[::-1]
            top_id, second_id = s_idx[0], s_idx[1]
            top_prob = float(p_row[top_id])
            second_prob = float(p_row[second_id])
            margin = float(top_prob - second_prob)

            pred_label = ID_TO_LABEL[top_id]
            all_preds.append(pred_label)
            all_top_probs.append(round(top_prob, 4))
            all_margins.append(round(margin, 4))
            roberta_sample_latencies.append(per_item_time)

            # Routing condition: Fast-path if top_prob >= 0.80 AND margin >= 0.15
            is_fast_path = (top_prob >= 0.80) and (margin >= 0.15)
            all_escalations.append(not is_fast_path)

        if (i // batch_size) % 10 == 0 or (i + batch_size) >= total_rows:
            processed = min(i + batch_size, total_rows)
            print(f"  RoBERTa Processed {processed}/{total_rows} samples ({processed/total_rows*100:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    total_roberta_time = time.time() - t0
    avg_roberta_latency_ms = (total_roberta_time / total_rows) * 1000.0

    df["roberta_pred"] = all_preds
    df["roberta_confidence"] = all_top_probs
    df["roberta_margin"] = all_margins
    df["escalate"] = all_escalations

    # Compute RoBERTa Baseline Metrics
    roberta_acc = accuracy_score(df["true_label"], df["roberta_pred"])
    roberta_weighted_f1 = f1_score(df["true_label"], df["roberta_pred"], average="weighted")
    roberta_macro_f1 = f1_score(df["true_label"], df["roberta_pred"], average="macro")
    roberta_rep = classification_report(df["true_label"], df["roberta_pred"], output_dict=True, labels=CLASS_ORDER)
    roberta_neutral_f1 = roberta_rep["Neutral"]["f1-score"]
    roberta_pos_f1 = roberta_rep["Positive"]["f1-score"]
    roberta_neg_f1 = roberta_rep["Negative"]["f1-score"]
    roberta_cm = confusion_matrix(df["true_label"], df["roberta_pred"], labels=CLASS_ORDER)

    escalated_indices = df[df["escalate"]].index.tolist()
    escalated_count = len(escalated_indices)
    escalation_pct = (escalated_count / total_rows) * 100.0

    print("\n--- ROBERTA BASELINE EVALUATION SUMMARY ---", flush=True)
    print(f"  Total Evaluated       : {total_rows} rows", flush=True)
    print(f"  Accuracy              : {roberta_acc * 100:.2f}%", flush=True)
    print(f"  Weighted F1           : {roberta_weighted_f1 * 100:.2f}%", flush=True)
    print(f"  Macro F1              : {roberta_macro_f1 * 100:.2f}%", flush=True)
    print(f"  Neutral F1            : {roberta_neutral_f1 * 100:.2f}%", flush=True)
    print(f"  Positive F1           : {roberta_pos_f1 * 100:.2f}%", flush=True)
    print(f"  Negative F1           : {roberta_neg_f1 * 100:.2f}%", flush=True)
    print(f"  Avg Latency/review    : {avg_roberta_latency_ms:.2f} ms", flush=True)
    print(f"  Escalations Required  : {escalated_count} / {total_rows} ({escalation_pct:.2f}%)", flush=True)

    # 5. Hybrid Escalation via Groq API
    print("\n" + "=" * 70, flush=True)
    print(f"STEP 2: RUNNING HYBRID ROUTING — ESCALATING {escalated_count} SAMPLES TO GROQ...", flush=True)
    print("=" * 70, flush=True)

    checkpoint_file = "groq_eval_checkpoint.json"
    groq_results = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                saved = json.load(f)
                groq_results = {int(k): v for k, v in saved.items()}
                print(f"Loaded {len(groq_results)} cached Groq results from checkpoint.", flush=True)
        except Exception as e:
            print(f"Could not load checkpoint: {e}", flush=True)

    llm_latencies = []
    successful_api_calls = 0
    failed_api_calls = 0

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    def call_groq_with_retry(text, max_retries=5):
        prompt = groq_client._build_prompt(text)
        payload = {
            "model": groq_client.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert film critic and sentiment analysis AI. Always output strictly valid JSON conforming to the requested schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 350,
        }
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            t_start = time.time()
            try:
                resp = requests.post(groq_client.BASE_URL, headers=headers, json=payload, timeout=12)
                elapsed = time.time() - t_start

                if resp.status_code == 200:
                    data = resp.json()
                    raw = data["choices"][0]["message"]["content"]
                    parsed = json.loads(raw)
                    sent = str(parsed.get("sentiment", "Neutral")).strip().capitalize()
                    if sent not in {"Positive", "Neutral", "Negative"}:
                        if "pos" in sent.lower():
                            sent = "Positive"
                        elif "neg" in sent.lower():
                            sent = "Negative"
                        else:
                            sent = "Neutral"
                    conf = float(parsed.get("confidence", 0.90))
                    rat = str(parsed.get("rationale", "")).strip()
                    return {
                        "success": True,
                        "sentiment": sent,
                        "confidence": conf,
                        "rationale": rat,
                        "latency": elapsed,
                        "error": None,
                    }
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after", None)
                    sleep_sec = float(retry_after) if retry_after else (1.5 * (attempt + 1))
                    time.sleep(sleep_sec)
                    continue
                else:
                    return {
                        "success": False,
                        "sentiment": None,
                        "confidence": 0.0,
                        "rationale": None,
                        "latency": elapsed,
                        "error": f"HTTP {resp.status_code}: {resp.text[:100]}",
                    }
            except Exception as ex:
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    return {
                        "success": False,
                        "sentiment": None,
                        "confidence": 0.0,
                        "rationale": None,
                        "latency": time.time() - t_start,
                        "error": str(ex),
                    }
        return {
            "success": False,
            "sentiment": None,
            "confidence": 0.0,
            "rationale": None,
            "latency": 0.0,
            "error": "Exceeded max retries on 429 rate limit",
        }

    t_escalate_start = time.time()
    for count, idx in enumerate(escalated_indices, 1):
        if idx in groq_results and groq_results[idx].get("success"):
            res = groq_results[idx]
            successful_api_calls += 1
            llm_latencies.append(res.get("latency", 0.0))
            continue

        text = df.loc[idx, "sentence"]
        res = call_groq_with_retry(text)
        groq_results[idx] = res

        if res.get("success"):
            successful_api_calls += 1
            llm_latencies.append(res.get("latency", 0.0))
        else:
            failed_api_calls += 1

        # Checkpoint every 50 requests
        if count % 50 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(groq_results, f)
            elapsed_esc = time.time() - t_escalate_start
            req_per_sec = count / elapsed_esc if elapsed_esc > 0 else 0
            print(f"  [Groq Escalation] {count}/{escalated_count} ({count/escalated_count*100:.1f}%) | "
                  f"Success: {successful_api_calls} | Failed: {failed_api_calls} | "
                  f"Avg LLM Latency: {np.mean(llm_latencies)*1000:.1f}ms | Rate: {req_per_sec:.2f} req/s", flush=True)

        # Dynamic small sleep to stay within token / rate limits comfortably
        time.sleep(0.08)

    # Final checkpoint save
    with open(checkpoint_file, "w") as f:
        json.dump(groq_results, f)

    total_escalated_time = time.time() - t_escalate_start
    avg_llm_latency_ms = float(np.mean(llm_latencies) * 1000.0) if llm_latencies else 0.0
    api_success_pct = (successful_api_calls / escalated_count) * 100.0 if escalated_count > 0 else 100.0

    print(f"\nCompleted Groq Escalation in {total_escalated_time:.1f}s.", flush=True)
    print(f"  Successful API calls : {successful_api_calls} / {escalated_count} ({api_success_pct:.2f}%)", flush=True)
    print(f"  Failed / Fallback    : {failed_api_calls} / {escalated_count}", flush=True)
    print(f"  Avg LLM Latency      : {avg_llm_latency_ms:.2f} ms", flush=True)

    # 6. Build Hybrid Predictions
    final_preds = []
    engines_used = []
    llm_rationales = []
    hybrid_latencies = []

    for i in range(total_rows):
        is_escalated = df.loc[i, "escalate"]
        rob_pred = df.loc[i, "roberta_pred"]
        rob_lat = roberta_sample_latencies[i] * 1000.0

        if is_escalated:
            res = groq_results.get(i, {})
            if res.get("success") and res.get("sentiment") in CLASS_ORDER:
                final_preds.append(res.get("sentiment"))
                engines_used.append("llm")
                llm_rationales.append(res.get("rationale", ""))
                hybrid_latencies.append(rob_lat + (res.get("latency", 0.0) * 1000.0))
            else:
                # Fallback to RoBERTa
                final_preds.append(rob_pred)
                engines_used.append("roberta")
                llm_rationales.append(None)
                hybrid_latencies.append(rob_lat + (res.get("latency", 0.0) * 1000.0))
        else:
            final_preds.append(rob_pred)
            engines_used.append("roberta")
            llm_rationales.append(None)
            hybrid_latencies.append(rob_lat)

    df["final_pred"] = final_preds
    df["engine_used"] = engines_used
    df["llm_rationale"] = llm_rationales

    # Compute Hybrid Metrics
    hybrid_acc = accuracy_score(df["true_label"], df["final_pred"])
    hybrid_weighted_f1 = f1_score(df["true_label"], df["final_pred"], average="weighted")
    hybrid_macro_f1 = f1_score(df["true_label"], df["final_pred"], average="macro")
    hybrid_rep = classification_report(df["true_label"], df["final_pred"], output_dict=True, labels=CLASS_ORDER)
    hybrid_neutral_f1 = hybrid_rep["Neutral"]["f1-score"]
    hybrid_pos_f1 = hybrid_rep["Positive"]["f1-score"]
    hybrid_neg_f1 = hybrid_rep["Negative"]["f1-score"]
    hybrid_cm = confusion_matrix(df["true_label"], df["final_pred"], labels=CLASS_ORDER)
    avg_hybrid_latency_ms = float(np.mean(hybrid_latencies))

    # 7. Save Per-Row Predictions CSV
    export_cols = [
        "sentence",
        "true_label",
        "roberta_pred",
        "roberta_confidence",
        "roberta_margin",
        "engine_used",
        "final_pred",
        "llm_rationale",
    ]
    df_export = df[export_cols].rename(columns={"sentence": "review"})
    output_csv = "evaluation_predictions_groq.csv"
    df_export.to_csv(output_csv, index=False)
    print(f"\nSaved per-row predictions to '{output_csv}' ({len(df_export)} rows).", flush=True)

    # 8. Save structured results JSON
    results_json = {
        "dataset": {
            "path": csv_path,
            "total_rows": total_rows,
        },
        "roberta_baseline": {
            "accuracy": roberta_acc,
            "weighted_f1": roberta_weighted_f1,
            "macro_f1": roberta_macro_f1,
            "neutral_f1": roberta_neutral_f1,
            "positive_f1": roberta_pos_f1,
            "negative_f1": roberta_neg_f1,
            "avg_latency_ms": avg_roberta_latency_ms,
            "confusion_matrix": roberta_cm.tolist(),
            "class_labels": CLASS_ORDER,
        },
        "hybrid_pipeline": {
            "accuracy": hybrid_acc,
            "weighted_f1": hybrid_weighted_f1,
            "macro_f1": hybrid_macro_f1,
            "neutral_f1": hybrid_neutral_f1,
            "positive_f1": hybrid_pos_f1,
            "negative_f1": hybrid_neg_f1,
            "avg_latency_ms": avg_hybrid_latency_ms,
            "pct_escalated_to_llm": escalation_pct,
            "pct_escalations_successful": api_success_pct,
            "total_escalations": escalated_count,
            "successful_escalations": successful_api_calls,
            "failed_escalations": failed_api_calls,
            "llm_avg_latency_ms": avg_llm_latency_ms,
            "confusion_matrix": hybrid_cm.tolist(),
            "class_labels": CLASS_ORDER,
        }
    }

    with open("hybrid_evaluation_groq_results.json", "w") as f:
        json.dump(results_json, f, indent=4)
    print("Saved evaluation results to 'hybrid_evaluation_groq_results.json'.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("FINAL BENCHMARK COMPARISON TABLE", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Metric':<45} | {'RoBERTa Baseline':<20} | {'Hybrid (RoBERTa + Groq)':<25}")
    print("-" * 96)
    print(f"{'Accuracy':<45} | {roberta_acc*100:6.2f}%{'':13} | {hybrid_acc*100:6.2f}%")
    print(f"{'Weighted F1':<45} | {roberta_weighted_f1*100:6.2f}%{'':13} | {hybrid_weighted_f1*100:6.2f}%")
    print(f"{'Macro F1':<45} | {roberta_macro_f1*100:6.2f}%{'':13} | {hybrid_macro_f1*100:6.2f}%")
    print(f"{'Neutral-class F1':<45} | {roberta_neutral_f1*100:6.2f}%{'':13} | {hybrid_neutral_f1*100:6.2f}%")
    print(f"{'Positive-class F1':<45} | {roberta_pos_f1*100:6.2f}%{'':13} | {hybrid_pos_f1*100:6.2f}%")
    print(f"{'Negative-class F1':<45} | {roberta_neg_f1*100:6.2f}%{'':13} | {hybrid_neg_f1*100:6.2f}%")
    print(f"{'Avg latency/review':<45} | {avg_roberta_latency_ms:6.2f} ms{'':11} | {avg_hybrid_latency_ms:6.2f} ms")
    print(f"{'% escalated to LLM':<45} | {'0.00%':<20} | {escalation_pct:6.2f}% ({escalated_count}/{total_rows})")
    print(f"{'% escalations with successful API response':<45} | {'N/A':<20} | {api_success_pct:6.2f}% ({successful_api_calls}/{escalated_count})")
    print("=" * 70, flush=True)

    print("\nConfusion Matrix - RoBERTa Baseline (Order: Negative, Neutral, Positive):")
    print(roberta_cm)
    print("\nConfusion Matrix - Hybrid Pipeline (Order: Negative, Neutral, Positive):")
    print(hybrid_cm)

    return results_json

if __name__ == "__main__":
    evaluate()
