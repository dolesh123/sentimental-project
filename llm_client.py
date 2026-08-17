"""
LLM Client for Hybrid Sentiment Analysis
========================================

Modular LLM integration module supporting Gemini (and structured for future
OpenAI/Groq providers). Reads API key strictly from the environment
(GEMINI_API_KEY).

Output Structure:
    {
        "sentiment": "Positive" | "Neutral" | "Negative",
        "confidence": 0.0 - 1.0,
        "aspects": {
            "Acting": "...",
            "Plot": "...",
            "Direction/Pacing": "..."
        },
        "rationale": "1-2 sentence concise explanation.",
        "model": "gemini-2.5-flash",
        "success": True/False,
        "error": None | "error message"
    }
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

logger = logging.getLogger(__name__)

# Default Gemini model
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class BaseLLMClient(ABC):
    """Abstract Base Class for LLM providers."""

    @abstractmethod
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of review text and return structured verdict."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if API credentials and network are configured."""
        pass


class GeminiClient(BaseLLMClient):
    """
    Google Gemini LLM Client using REST API via requests.
    Supports structured JSON generation, automatic backup key failover on 429 quota exhaustion,
    and pre-flight quota diagnostic checks.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        backup_api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout: int = 10,
    ):
        # Primary and backup keys strictly from environment if not explicitly passed
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.backup_api_key = (backup_api_key or os.getenv("GEMINI_API_KEY_BACKUP", "")).strip()
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if at least one GEMINI_API_KEY is present in environment."""
        return bool(self.api_key or self.backup_api_key)

    def _build_prompt(self, text: str) -> str:
        return f"""Analyze the sentiment of the following movie review.
Classify the sentiment strictly as one of: ["Positive", "Neutral", "Negative"].

Review Text:
\"\"\"{text}\"\"\"

Provide your response in strictly valid JSON format matching this exact schema:
{{
  "sentiment": "Positive" | "Neutral" | "Negative",
  "confidence": 0.95,
  "aspects": {{
    "Acting": "Brief assessment of performances",
    "Plot": "Brief assessment of storyline/narrative",
    "Direction/Pacing": "Brief assessment of directing and pacing"
  }},
  "rationale": "1-2 sentence concise summary explaining the sentiment verdict."
}}"""

    def _send_request(self, key: str, payload: dict) -> requests.Response:
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        
        # If primary model name returns 404, try gemini-flash-latest fallback
        if resp.status_code == 404 and self.model != "gemini-flash-latest":
            fallback_url = f"{self.BASE_URL}/gemini-flash-latest:generateContent?key={key}"
            resp = requests.post(fallback_url, headers=headers, json=payload, timeout=self.timeout)
            
        return resp

    def check_quota_preflight(self, expected_total_samples: int = 0, escalation_rate: float = 0.287) -> Dict[str, Any]:
        """
        Pre-flight diagnostic check:
        1. Tests API connectivity and quota status with a minimal ping test.
        2. Estimates expected LLM calls based on sample count & escalation rate.
        3. Returns status dict with warnings, recommendations, and failover availability.
        """
        estimated_llm_calls = int(expected_total_samples * escalation_rate) if expected_total_samples > 0 else 0
        test_payload = {
            "contents": [{"parts": [{"text": "Reply with strictly JSON: {\"status\": \"ok\"}"}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1},
        }

        primary_status = "Not configured"
        backup_status = "Not configured"
        has_active_key = False
        active_key_name = None

        # Test Primary Key
        if self.api_key:
            try:
                resp = self._send_request(self.api_key, test_payload)
                if resp.status_code == 200:
                    primary_status = "Active (Quota OK)"
                    has_active_key = True
                    active_key_name = "GEMINI_API_KEY (Primary)"
                elif resp.status_code == 429:
                    primary_status = "Exhausted (HTTP 429 Quota Exceeded)"
                else:
                    primary_status = f"Error (HTTP {resp.status_code})"
            except Exception as e:
                primary_status = f"Connection error ({e})"

        # Test Backup Key if present
        if self.backup_api_key:
            try:
                resp = self._send_request(self.backup_api_key, test_payload)
                if resp.status_code == 200:
                    backup_status = "Active (Quota OK)"
                    if not has_active_key:
                        has_active_key = True
                        active_key_name = "GEMINI_API_KEY_BACKUP"
                elif resp.status_code == 429:
                    backup_status = "Exhausted (HTTP 429 Quota Exceeded)"
                else:
                    backup_status = f"Error (HTTP {resp.status_code})"
            except Exception as e:
                backup_status = f"Connection error ({e})"

        # Warning / advice generation
        warning = None
        if not has_active_key:
            warning = "[WARNING] All configured Gemini API keys are currently exhausted (HTTP 429) or unconfigured. Batch will fall back to RoBERTa baseline."
        elif "Exhausted" in primary_status and "Active" in backup_status:
            warning = "[INFO] Primary key is exhausted (429), but GEMINI_API_KEY_BACKUP is active and ready for failover."
        elif estimated_llm_calls > 20 and ("free_tier" in primary_status.lower() or not self.backup_api_key):
            warning = f"[WARNING] Expected ~{estimated_llm_calls} Gemini calls. If using Free-Tier (20 req/day limit), consider enabling billing or setting GEMINI_API_KEY_BACKUP."

        return {
            "has_active_key": has_active_key,
            "active_key_name": active_key_name,
            "primary_key_status": primary_status,
            "backup_key_status": backup_status,
            "has_backup_key_configured": bool(self.backup_api_key),
            "estimated_llm_calls": estimated_llm_calls,
            "warning": warning,
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Send review to Gemini and extract sentiment, confidence, aspects, and rationale.
        Automatically fails over to GEMINI_API_KEY_BACKUP if primary key encounters HTTP 429.
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "No Gemini API key configured in environment.",
                "sentiment": "Neutral",
                "confidence": 0.0,
                "aspects": {},
                "rationale": "",
                "model": self.model,
            }

        prompt = self._build_prompt(text)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        # Determine keys to try (Primary -> Backup)
        keys_to_try = []
        if self.api_key:
            keys_to_try.append(("primary", self.api_key))
        if self.backup_api_key:
            keys_to_try.append(("backup", self.backup_api_key))

        last_error = "Unknown error"

        for key_type, key in keys_to_try:
            try:
                response = self._send_request(key, payload)

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        last_error = "No candidates returned by Gemini API"
                        continue

                    part_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                    parsed = json.loads(part_text)

                    sentiment = str(parsed.get("sentiment", "Neutral")).strip().capitalize()
                    if sentiment not in {"Positive", "Neutral", "Negative"}:
                        if "pos" in sentiment.lower():
                            sentiment = "Positive"
                        elif "neg" in sentiment.lower():
                            sentiment = "Negative"
                        else:
                            sentiment = "Neutral"

                    confidence = float(parsed.get("confidence", 0.85))
                    confidence = max(0.0, min(1.0, confidence))

                    aspects = parsed.get("aspects", {})
                    if not isinstance(aspects, dict):
                        aspects = {}

                    rationale = str(parsed.get("rationale", "")).strip()

                    return {
                        "success": True,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "aspects": aspects,
                        "rationale": rationale,
                        "model": self.model,
                        "key_used": key_type,
                        "error": None,
                    }

                elif response.status_code == 429:
                    last_error = f"HTTP 429 Quota Exceeded on {key_type} key"
                    if key_type == "primary" and self.backup_api_key:
                        logger.warning("Primary Gemini key hit 429 quota. Failing over to GEMINI_API_KEY_BACKUP...")
                        continue
                    else:
                        logger.warning("Gemini key (%s) hit 429 quota limit.", key_type)
                else:
                    last_error = f"API returned status {response.status_code}: {response.text[:150]}"

            except requests.Timeout:
                last_error = f"Request timed out ({self.timeout}s) on {key_type} key"
            except Exception as exc:
                last_error = f"Gemini API call error on {key_type} key: {exc}"

        return {
            "success": False,
            "error": last_error,
            "sentiment": "Neutral",
            "confidence": 0.0,
            "aspects": {},
            "rationale": "",
            "model": self.model,
        }


class GroqClient(BaseLLMClient):
    """
    Groq LLM Client using ultra-fast Llama-3.3-70B-Versatile / Llama-3.1-8B.
    Free Tier provides:
      - 14,400 requests / day (100% Free, NO credit card needed)
      - 30 requests / minute
      - Ultra-fast token generation (~300 tokens/sec)
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 10,
    ):
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.model = os.getenv("GROQ_MODEL", model)
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if GROQ_API_KEY is configured in environment."""
        return bool(self.api_key)

    def _build_prompt(self, text: str) -> str:
        return f"""Analyze the sentiment of the following movie review.
Classify the sentiment strictly as one of: ["Positive", "Neutral", "Negative"].

Review Text:
\"\"\"{text}\"\"\"

Provide your response in strictly valid JSON format matching this exact schema:
{{
  "sentiment": "Positive" | "Neutral" | "Negative",
  "confidence": 0.95,
  "aspects": {{
    "Acting": "Brief assessment of performances",
    "Plot": "Brief assessment of storyline/narrative",
    "Direction/Pacing": "Brief assessment of directing and pacing"
  }},
  "rationale": "1-2 sentence concise summary explaining the sentiment verdict."
}}"""

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Send review to Groq (Llama 3.3 70B) with JSON mode."""
        if not self.is_available():
            return {
                "success": False,
                "error": "GROQ_API_KEY not configured in environment.",
                "sentiment": "Neutral",
                "confidence": 0.0,
                "aspects": {},
                "rationale": "",
                "model": self.model,
            }

        prompt = self._build_prompt(text)
        models_to_try = [self.model]
        if "70b" in self.model:
            models_to_try.append("llama-3.1-8b-instant")
        elif "8b" in self.model:
            models_to_try.append("llama-3.3-70b-versatile")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = "Unknown error"

        for current_model in models_to_try:
            payload = {
                "model": current_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert film critic and sentiment analysis AI. Always output strictly valid JSON conforming to the requested schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 500,
            }

            try:
                resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(raw_content)

                    sentiment = str(parsed.get("sentiment", "Neutral")).strip().capitalize()
                    if sentiment not in {"Positive", "Neutral", "Negative"}:
                        if "pos" in sentiment.lower():
                            sentiment = "Positive"
                        elif "neg" in sentiment.lower():
                            sentiment = "Negative"
                        else:
                            sentiment = "Neutral"

                    confidence = float(parsed.get("confidence", 0.90))
                    confidence = max(0.0, min(1.0, confidence))
                    aspects = parsed.get("aspects", {})
                    if not isinstance(aspects, dict):
                        aspects = {}
                    rationale = str(parsed.get("rationale", "")).strip()

                    return {
                        "success": True,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "aspects": aspects,
                        "rationale": rationale,
                        "model": current_model,
                        "key_used": "groq",
                        "error": None,
                    }
                elif resp.status_code == 429:
                    logger.warning("Groq model %s hit 429 rate limit. Trying fallback model if available...", current_model)
                    last_error = f"Groq API 429 Rate Limit on {current_model}"
                    continue
                else:
                    logger.warning("Groq API error %s: %s", resp.status_code, resp.text)
                    last_error = f"Groq API returned status {resp.status_code}: {resp.text[:120]}"

            except Exception as exc:
                logger.warning("Groq API call failed on %s: %s", current_model, exc)
                last_error = str(exc)

        return {
            "success": False,
            "error": last_error,
            "sentiment": "Neutral",
            "confidence": 0.0,
            "aspects": {},
            "rationale": "",
            "model": self.model,
        }


class LLMFactory:
    """Factory to get the appropriate LLM client (routes all requests to Groq)."""

    _instances = {}

    @classmethod
    def get_client(cls, provider: Optional[str] = None) -> BaseLLMClient:
        # Default provider is always Groq
        if not provider or provider in ("auto", "default", "groq"):
            provider = "groq"

        provider = provider.lower()
        if provider not in cls._instances:
            if provider == "groq":
                cls._instances[provider] = GroqClient()
            elif provider == "gemini":
                # Disabled/Fallback only if explicitly forced
                cls._instances[provider] = GeminiClient()
            else:
                cls._instances[provider] = GroqClient()
        return cls._instances[provider]

