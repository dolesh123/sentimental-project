"""
Unit tests for Hybrid RoBERTa + Groq Sentiment Analysis
"""

import os
from sentiment_analyzer import SentimentSystem

def run_tests():
    print("=" * 60)
    print("RUNNING HYBRID SENTIMENT SYSTEM TESTS (ROBERTA + GROQ)")
    print("=" * 60)

    system = SentimentSystem()

    # 1. Test High-Confidence Clear Case (RoBERTa Fast Path)
    clear_review = "This movie is a timeless masterpiece! Every actor gave an exceptional performance, and the direction is sheer perfection."
    print("\n[Test 1] Clear Rave Review (Expecting RoBERTa fast-path):")
    res1 = system.predict_hybrid(clear_review, confidence_threshold=0.80, margin_threshold=0.15)
    print(f"  Verdict: {res1['label']}")
    print(f"  Engine Used: {res1['engine_used']}")
    print(f"  Is Fast Path: {res1['is_fast_path']}")
    print(f"  RoBERTa Confidence: {res1['roberta_confidence']}")
    print(f"  RoBERTa Margin: {res1['roberta_margin']}")
    print(f"  Routing Reason: {res1['routing_reason']}")
    assert res1['engine_used'] == 'roberta', f"Expected roberta, got {res1['engine_used']}"
    assert res1['is_fast_path'] is True, "Expected fast path"
    print("  => TEST 1 PASSED!")

    # 2. Test Ambiguous / Mixed Case (LLM Escalation to Groq)
    ambiguous_review = "Visually stunning and the soundtrack was majestic, but the sluggish narrative and hollow characters made the final act feel completely unearned."
    print("\n[Test 2] Mixed Review (Expecting Groq escalation):")
    res2 = system.predict_hybrid(ambiguous_review, confidence_threshold=0.80, margin_threshold=0.15)
    print(f"  Verdict: {res2['label']}")
    print(f"  Engine Used: {res2['engine_used']}")
    print(f"  Is Fast Path: {res2['is_fast_path']}")
    print(f"  RoBERTa Confidence: {res2['roberta_confidence']}")
    print(f"  RoBERTa Margin: {res2['roberta_margin']}")
    print(f"  Routing Reason: {res2['routing_reason']}")
    print(f"  LLM Rationale: {res2.get('llm_rationale')}")
    print(f"  Aspects: {res2.get('aspects')}")
    assert res2['engine_used'] == 'llm', f"Expected llm, got {res2['engine_used']}"
    assert res2['is_fast_path'] is False, "Expected escalation"
    assert res2.get('aspects') is not None, "Expected aspect breakdown"
    print("  => TEST 2 PASSED!")

    # 3. Test Missing API Key (Graceful Degradation)
    print("\n[Test 3] Missing API Key Fallback:")
    orig_gemini = os.environ.get("GEMINI_API_KEY", "")
    orig_groq = os.environ.get("GROQ_API_KEY", "")
    try:
        # Clear factory cached instances and unset keys
        from llm_client import LLMFactory
        LLMFactory._instances.clear()
        os.environ["GEMINI_API_KEY"] = ""
        os.environ["GROQ_API_KEY"] = ""

        res3 = system.predict_hybrid(ambiguous_review, confidence_threshold=0.80, margin_threshold=0.15)
        print(f"  Verdict: {res3['label']}")
        print(f"  Engine Used: {res3['engine_used']}")
        print(f"  Is Fast Path: {res3['is_fast_path']}")
        print(f"  Notice: {res3.get('llm_unavailable_notice')}")
        assert res3['engine_used'] == 'roberta', f"Expected roberta fallback, got {res3['engine_used']}"
        assert res3.get('llm_unavailable_notice') is not None, "Expected fallback notice"
        print("  => TEST 3 PASSED!")
    finally:
        os.environ["GEMINI_API_KEY"] = orig_gemini
        os.environ["GROQ_API_KEY"] = orig_groq
        from llm_client import LLMFactory
        LLMFactory._instances.clear()

    print("\n" + "=" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
