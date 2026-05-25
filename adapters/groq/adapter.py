"""
PRISM — adapters/groq/adapter.py
Groq Adapter

Connects PRISM to Groq's ultra-fast inference API.
Groq runs open-source models (Llama, Mixtral, Gemma)
at speeds 10-100x faster than standard APIs.

Supported models:
  llama-3.3-70b-versatile     ← recommended
  llama-3.1-8b-instant        ← fastest / highest rate limit
  mixtral-8x7b-32768          ← long context
  gemma2-9b-it                ← lightweight

Usage:
    from adapters.groq.adapter import GroqAdapter

    adapter = GroqAdapter(
        api_key = "your-groq-key",
        model   = "llama-3.1-8b-instant"
    )

    response = adapter.complete("What is the strategic risk here?")
    dist     = adapter.get_output_distribution(prompt, hypotheses)
    cot      = adapter.get_chain_of_thought(prompt)

Get your free API key at: https://console.groq.com
"""

from __future__ import annotations
import os
import re
import json
from typing import Dict, List, Optional, Any
from adapters.base.base import BaseAdapter

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class GroqAdapter(BaseAdapter):
    """
    PRISM adapter for Groq inference API.

    Groq provides ultra-low latency inference on
    open-source models — ideal for high-frequency
    EAT event generation in live workflow sessions.

    Implements full BaseAdapter interface:
      - complete()
      - get_output_distribution()
      - get_chain_of_thought()
      - extract_concepts()
      - complete_with_confidence()
    """

    SUPPORTED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str           = "llama-3.1-8b-instant",
        max_tokens:  int           = 1024,
        temperature: float         = 0.7,
        system:      Optional[str] = None
    ):
        if not GROQ_AVAILABLE:
            raise ImportError(
                "groq package not installed. "
                "Run: pip install groq"
            )

        self._api_key    = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.system      = system or (
            "You are a helpful AI assistant participating in a "
            "collaborative research workflow. Be precise, calibrated, "
            "and explicit about your confidence levels."
        )

        self._client = Groq(api_key=self._api_key)

    # ── CORE COMPLETION ───────────────────────────────────────────────────

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a text completion via Groq.
        Ultra-low latency — typical response in < 500ms.
        """
        try:
            response = self._client.chat.completions.create(
                model       = kwargs.get("model", self.model),
                max_tokens  = kwargs.get("max_tokens", self.max_tokens),
                temperature = kwargs.get("temperature", self.temperature),
                messages    = [
                    {"role": "system", "content": self.system},
                    {"role": "user",   "content": prompt}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"[GroqAdapter error: {str(e)}]"

    # ── OUTPUT DISTRIBUTION ───────────────────────────────────────────────

    def get_output_distribution(
        self,
        prompt:     str,
        hypotheses: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Approximate a probability distribution over hypotheses.

        Uses constrained JSON prompting — Groq's speed makes
        this practical for real-time VERDICT scoring.

        Returns: {hypothesis: confidence} summing to 1.0
        """
        if not hypotheses:
            return {}

        hyp_list = "\n".join(
            f"- {h}" for h in hypotheses
        )

        distribution_prompt = f"""You are evaluating how strongly a context supports each hypothesis.

Context:
{prompt}

Hypotheses to evaluate:
{hyp_list}

Instructions:
- Score each hypothesis from 0 to 100
- Higher score = stronger support from context
- Return ONLY a JSON object
- Keys must exactly match the hypothesis names provided
- No explanation, no markdown, no extra text

Required JSON format:
{json.dumps({h: 50 for h in hypotheses})}

Your scores (JSON only):"""

        try:
            response = self._client.chat.completions.create(
                model       = self.model,
                max_tokens  = 256,
                temperature = 0.1,
                messages    = [
                    {
                        "role":    "system",
                        "content": (
                            "You are a precise evaluator. "
                            "Respond only with valid JSON. "
                            "Keys must exactly match the provided hypothesis names."
                        )
                    },
                    {
                        "role":    "user",
                        "content": distribution_prompt
                    }
                ]
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

            # Find JSON object in response
            json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)

            scores_raw = json.loads(raw)

            distribution = {}
            total = 0.0

            for hyp in hypotheses:
                # Try exact match first
                score = scores_raw.get(hyp, 0)
                if score == 0:
                    # Try case-insensitive match
                    for k, v in scores_raw.items():
                        if k.lower() == hyp.lower():
                            score = v
                            break
                distribution[hyp] = float(max(score, 0))
                total += distribution[hyp]

            # Normalize to sum to 1.0
            if total > 0:
                distribution = {
                    k: round(v / total, 4)
                    for k, v in distribution.items()
                }
            else:
                uniform = round(1.0 / len(hypotheses), 4)
                distribution = {h: uniform for h in hypotheses}

            return distribution

        except Exception:
            uniform = round(1.0 / len(hypotheses), 4)
            return {h: uniform for h in hypotheses}

    # ── CHAIN OF THOUGHT ──────────────────────────────────────────────────

    def get_chain_of_thought(
        self,
        prompt: str,
        **kwargs
    ) -> Optional[str]:
        """
        Extract chain-of-thought reasoning via explicit CoT prompting.

        Groq's speed makes multi-step CoT extraction practical
        without significant latency cost.
        """
        cot_prompt = f"""Think through this step by step.
Show your reasoning explicitly before your conclusion.

{prompt}

Format your response exactly like this:
REASONING:
[your step-by-step thinking here]

CONCLUSION:
[your final answer here]"""

        try:
            response = self._client.chat.completions.create(
                model       = self.model,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
                messages    = [
                    {"role": "system", "content": self.system},
                    {"role": "user",   "content": cot_prompt}
                ]
            )
            text = response.choices[0].message.content

            if "REASONING:" in text:
                start = text.find("REASONING:") + len("REASONING:")
                end   = text.find("CONCLUSION:")
                if end > start:
                    return text[start:end].strip()
                return text[start:].strip()

            return text

        except Exception:
            return None

    # ── CALIBRATED COMPLETION ─────────────────────────────────────────────

    def complete_with_confidence(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion with explicit confidence scoring.
        Used by VERDICT calibration tracker.
        """
        confidence_prompt = f"""{prompt}

After your response, on new lines add exactly:
CONFIDENCE: [number 0-100]
REASONING: [one sentence explaining your confidence]"""

        try:
            response = self._client.chat.completions.create(
                model       = self.model,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
                messages    = [
                    {"role": "system", "content": self.system},
                    {"role": "user",   "content": confidence_prompt}
                ]
            )
            text = response.choices[0].message.content

            # Extract confidence score
            confidence = 0.7
            conf_match = re.search(r"CONFIDENCE:\s*(\d+)", text)
            if conf_match:
                confidence = float(conf_match.group(1)) / 100.0
                confidence = max(0.0, min(1.0, confidence))

            # Extract reasoning
            reasoning = ""
            reason_match = re.search(
                r"REASONING:\s*(.+?)(?:\n|$)", text, re.DOTALL
            )
            if reason_match:
                reasoning = reason_match.group(1).strip()

            # Clean response
            clean = re.sub(
                r"\nCONFIDENCE:.*", "", text, flags=re.DOTALL
            ).strip()

            return {
                "response":   clean,
                "confidence": confidence,
                "reasoning":  reasoning
            }

        except Exception as e:
            return {
                "response":   f"[Error: {str(e)}]",
                "confidence": 0.5,
                "reasoning":  "Error occurred"
            }

    # ── CONCEPT EXTRACTION ────────────────────────────────────────────────

    def extract_concepts(
        self,
        text:         str,
        max_concepts: int = 8
    ) -> Dict[str, str]:
        """
        Extract key concepts for C_UPDATE EAT events.

        Returns {concept_id: concept_label}
        concept_id: lowercase_underscore identifier
        concept_label: human readable label
        """
        extract_prompt = f"""Extract the {max_concepts} most important concepts from the text below.

Rules:
- Return ONLY valid JSON
- Keys must be short identifiers using only lowercase letters and underscores
- Values must be human-readable concept labels
- Do NOT use generic keys like "short_id", "concept_1", "key"
- No explanation, no markdown, no extra text

Good example output:
{{"market_risk": "Market Risk", "regulatory_compliance": "Regulatory Compliance", "network_effects": "Network Effects"}}

Text to analyze:
{text[:800]}

JSON only:"""

        try:
            response = self._client.chat.completions.create(
                model       = self.model,
                max_tokens  = 300,
                temperature = 0.1,
                messages    = [
                    {
                        "role":    "system",
                        "content": (
                            "You extract concepts as JSON. "
                            "Keys are lowercase_underscore identifiers. "
                            "Values are readable labels. "
                            "Return only valid JSON."
                        )
                    },
                    {
                        "role":    "user",
                        "content": extract_prompt
                    }
                ]
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

            # Find JSON object
            json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)

            concepts = json.loads(raw)

            # Filter out generic keys
            bad_keys = {
                "short_id", "concept_id", "key", "id",
                "concept_1", "concept_2", "concept_3"
            }
            cleaned = {
                str(k).lower().replace(" ", "_"): str(v)
                for k, v in concepts.items()
                if str(k).lower() not in bad_keys
                and len(str(k)) > 2
            }

            return dict(list(cleaned.items())[:max_concepts])

        except Exception:
            return {}

    # ── ADAPTER METADATA ──────────────────────────────────────────────────

    def supports_cot(self) -> bool:
        return True

    def supports_logprobs(self) -> bool:
        return False

    def provider_name(self) -> str:
        return "groq"

    def model_name(self) -> str:
        return self.model

    def __repr__(self) -> str:
        return f"GroqAdapter(model={self.model})"