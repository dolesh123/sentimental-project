"""
Fast Evaluation Pipeline for RoBERTa Baseline and Hybrid System
===============================================================
Uses batched PyTorch inference on CPU for rapid full-set evaluation (6,530 samples)
and evaluates ambiguous subset with concurrent Gemini API calls.
"""

import time
import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentiment_analyzer import SentimentSystem, normalize_label, ID_TO_LABEL
from llm_client import GeminiClient

def evaluate_fast(max_llm_calls=50):
    print("=" * 65)
    print("EVALUATING SYSTEM ON test_all.csv")
    print("=" * 65)

    df = pd.read_csv("test_all.csv")
    print(f"Loaded {len(df)} test samples.")
    df["true_label"] = df["label"].apply(normalize_label)

    # --- Pre-Flight Quota & LLM Check ---
    gemini_client = GeminiClient()
    preflight = gemini_client.check_quota_preflight(expected_total_samples=len(df), escalation_rate=0.287)
    print("\n--- Pre-Flight LLM Quota Diagnostic ---")
    print(f"  Primary Key Status : {preflight['primary_key_status']}")
    print(f"  Backup Key Status  : {preflight['backup_key_status']}")
    print(f"  Active Key         : {preflight['active_key_name'] or 'None (will use local fallback)'}")
    print(f"  Estimated LLM Calls: ~{preflight['estimated_llm_calls']} (at ~28.7% escalation)")
    if preflight["warning"]:
        print(f"  {preflight['warning']}")

    system = SentimentSystem()
    ml_model = system.ml

    # Fast Batched RoBERTa Inference
    print("\n--- Running Batched RoBERTa Inference on full test set (6,530 samples) ---")
    texts = df["sentence"].astype(str).tolist()
    batch_size = 64
    all_preds = []
    all_top_probs = []
    all_margins = []
    all_escalations = []

    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = ml_model.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=ml_model.MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(ml_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ml_model.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for p_row in probs:
            sorted_idx = np.argsort(p_row)[::-1]
            top_id, second_id = sorted_idx[0], sorted_idx[1]
            top_prob = float(p_row[top_id])
            second_prob = float(p_row[second_id])
            margin = float(top_prob - second_prob)
            
            pred_label = ID_TO_LABEL[top_id]
            all_preds.append(pred_label)
            all_top_probs.append(top_prob)
            all_margins.append(margin)
            
            # Routing condition: Fast-path only if top_prob >= 0.80 and margin >= 0.15
            escalate = not (top_prob >= 0.80 and margin >= 0.15)
            all_escalations.append(escalate)

    total_roberta_time = time.time() - t0
    roberta_latency = (total_roberta_time / len(df)) * 1000  # ms per sample

    df["roberta_pred"] = all_preds
    df["top_prob"] = all_top_probs
    df["margin"] = all_margins
    df["escalate"] = all_escalations

    # Compute RoBERTa Baseline Metrics
    roberta_acc = accuracy_score(df["true_label"], df["roberta_pred"])
    roberta_macro_f1 = f1_score(df["true_label"], df["roberta_pred"], average="macro")
    roberta_report = classification_report(df["true_label"], df["roberta_pred"], output_dict=True)
    roberta_neutral_f1 = roberta_report["Neutral"]["f1-score"]
    escalation_rate = float(df["escalate"].mean() * 100)

    print(f"\n[1] RoBERTa Baseline Results:")
    print(f"  Overall Accuracy     : {roberta_acc * 100:.2f}%")
    print(f"  Macro F1             : {roberta_macro_f1 * 100:.2f}%")
    print(f"  Neutral-class F1     : {roberta_neutral_f1 * 100:.2f}%")
    print(f"  Average Latency      : {roberta_latency:.2f} ms/sample")
    print(f"  Escalation Rate      : {df['escalate'].sum()} / {len(df)} ({escalation_rate:.1f}%)")

    # Evaluate Ambiguous Subset with Gemini with respectful pacing (avoid free-tier 429)
    print(f"\n[2] Evaluating Ambiguous/Escalated Subset with Gemini API ({max_llm_calls} samples)...")
    escalated_indices = df[df["escalate"]].index.tolist()[:max_llm_calls]

    from llm_client import LLMFactory
    gemini_client = LLMFactory.get_client("gemini")

    gemini_results = {}
    llm_latencies = []

    for idx in escalated_indices:
        text = df.loc[idx, "sentence"]
        t_start = time.time()
        res = gemini_client.analyze_sentiment(text)
        lat = time.time() - t_start
        gemini_results[idx] = res
        llm_latencies.append(lat)
        time.sleep(3.5)  # Stay safely under 15 RPM free tier quota

    avg_llm_latency_ms = float(np.mean(llm_latencies) * 1000) if llm_latencies else 0.0

    # Measured accuracy on ambiguous subset
    roberta_correct_subset = sum(1 for idx in escalated_indices if df.loc[idx, "roberta_pred"] == df.loc[idx, "true_label"])
    gemini_correct_subset = sum(1 for idx in escalated_indices if gemini_results[idx].get("sentiment") == df.loc[idx, "true_label"])

    subset_roberta_acc = roberta_correct_subset / len(escalated_indices)
    subset_gemini_acc = gemini_correct_subset / len(escalated_indices)

    print(f"  RoBERTa Accuracy on Ambiguous Subset : {subset_roberta_acc * 100:.2f}% ({roberta_correct_subset}/{len(escalated_indices)})")
    print(f"  Gemini Accuracy on Ambiguous Subset  : {subset_gemini_acc * 100:.2f}% ({gemini_correct_subset}/{len(escalated_indices)})")
    print(f"  Accuracy Gain on Ambiguous Cases     : +{(subset_gemini_acc - subset_roberta_acc) * 100:.2f}%")
    print(f"  Average Gemini API Latency           : {avg_llm_latency_ms:.2f} ms")

    # Evaluate Hybrid configuration on the evaluated portion
    hybrid_eval_df = df.copy()
    for idx in escalated_indices:
        res = gemini_results[idx]
        if res.get("success"):
            hybrid_eval_df.loc[idx, "hybrid_pred"] = res.get("sentiment")
        else:
            hybrid_eval_df.loc[idx, "hybrid_pred"] = df.loc[idx, "roberta_pred"]

    # Hybrid on measured slice (fast-path samples + tested escalated samples)
    eval_slice = df.index.isin(escalated_indices) | (~df["escalate"])
    measured_slice_df = hybrid_eval_df[eval_slice].copy()
    measured_slice_df["final_pred"] = measured_slice_df["hybrid_pred"].fillna(measured_slice_df["roberta_pred"])

    hybrid_acc = accuracy_score(measured_slice_df["true_label"], measured_slice_df["final_pred"])
    hybrid_macro_f1 = f1_score(measured_slice_df["true_label"], measured_slice_df["final_pred"], average="macro")
    hybrid_report = classification_report(measured_slice_df["true_label"], measured_slice_df["final_pred"], output_dict=True)
    hybrid_neutral_f1 = hybrid_report["Neutral"]["f1-score"]

    avg_hybrid_latency = (1 - (escalation_rate/100)) * roberta_latency + (escalation_rate/100) * avg_llm_latency_ms

    results = {
        "roberta_baseline": {
            "total_test_samples": len(df),
            "accuracy": roberta_acc,
            "macro_f1": roberta_macro_f1,
            "neutral_f1": roberta_neutral_f1,
            "avg_latency_ms": roberta_latency,
            "escalation_rate_pct": escalation_rate,
        },
        "ambiguous_subset_benchmark": {
            "samples_evaluated": len(escalated_indices),
            "roberta_accuracy": subset_roberta_acc,
            "gemini_accuracy": subset_gemini_acc,
            "gain_on_ambiguous_pct": (subset_gemini_acc - subset_roberta_acc) * 100,
            "gemini_avg_latency_ms": avg_llm_latency_ms,
        },
        "hybrid_measured": {
            "accuracy": hybrid_acc,
            "macro_f1": hybrid_macro_f1,
            "neutral_f1": hybrid_neutral_f1,
            "avg_latency_ms": avg_hybrid_latency,
            "llm_hit_rate_pct": escalation_rate,
        }
    }

    with open("hybrid_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return results

if __name__ == "__main__":
    evaluate_fast(max_llm_calls=15)
