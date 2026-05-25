"""
PRISM — adapters/openai/adapter.py
OpenAI GPT Adapter

Connects PRISM to OpenAI's GPT models.
Implements the full BaseAdapter interface.

Supported models:
  gpt-4o               ← recommended
  gpt-4o-mini          ← fastest / lowest cost
  gpt-4-turbo          ← long context
  gpt-3.5-turbo        ← legacy

Usage:
    from adapters.openai.adapter import OpenAIAdapter

    adapter = OpenAIAdapter(
        api_key = "your-openai-key",
        model   = "gpt-4o-mini"
    )

    response = adapter.complete("What is the strategic risk here?")
    dist     = adapter.get_output_distribution(prompt, hypotheses)
    cot      = adapter.get_chain_of_thought(prompt)
"""

from __future__ import annotations
import os
import re
import json
from typing import Dict, List, Optional, Any
from adapters.base.base import BaseAdapter

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIAdapter(BaseAdapter):
    """
    PRISM adapter for OpenAI GPT models.

    Implements full BaseAdapter interface:
      - complete()
      - get_output_distribution()
      - get_chain_of_thought()
      - extract_concepts()
      - complete_with_confidence()
    """

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str           = "gpt-4o-mini",
        max_tokens:  int           = 1024,
        temperature: float         = 0.7,
        system:      Optional[str] = None
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Run: pip install openai"
            )

        self._api_key    = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.system      = system or (
            "You are a helpful AI assistant participating in a "
            "collaborative research workflow. Be precise, calibrated, "
            "and explicit about your confidence levels."
        )

        self._client = OpenAI(api_key=self._api_key)

    # ── CORE COMPLETION ───────────────────────────────────────────────────

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a text completion via OpenAI GPT."""
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
            return f"[OpenAIAdapter error: {str(e)}]"

    # ── OUTPUT DISTRIBUTION ───────────────────────────────────────────────

    def get_output_distribution(
        self,
        prompt:     str,
        hypotheses: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Approximate a probability distribution over hypotheses.
        Uses JSON mode for reliable structured output.
        Returns: {hypothesis: confidence} summing to 1.0
        """
        if not hypotheses:
            return {}

        hyp_list = "\n".join(f"- {h}" for h in hypotheses)

        distribution_prompt = f"""Evaluate how strongly this context supports each hypothesis.

Context:
{prompt}

Hypotheses:
{hyp_list}

Score each hypothesis 0-100. Higher = stronger support.
Return ONLY valid JSON with exact hypothesis names as keys.
Required format: {json.dumps({h: 50 for h in hypotheses})}
JSON only:"""

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
                            "Keys must exactly match hypothesis names."
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

            json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)

            scores_raw = json.loads(raw)

            distribution = {}
            total = 0.0

            for hyp in hypotheses:
                score = scores_raw.get(hyp, 0)
                if score == 0:
                    for k, v in scores_raw.items():
                        if k.lower() == hyp.lower():
                            score = v
                            break
                distribution[hyp] = float(max(score, 0))
                total += distribution[hyp]

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
        """Extract chain-of-thought reasoning via explicit CoT prompting."""
        cot_prompt = f"""Think through this step by step.

{prompt}

Format:
REASONING:
[step-by-step thinking]

CONCLUSION:
[final answer]"""

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

After your response add exactly:
CONFIDENCE: [0-100]
REASONING: [one sentence]"""

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

            confidence  = 0.7
            conf_match  = re.search(r"CONFIDENCE:\s*(\d+)", text)
            if conf_match:
                confidence = float(conf_match.group(1)) / 100.0
                confidence = max(0.0, min(1.0, confidence))

            reasoning   = ""
            reason_match = re.search(
                r"REASONING:\s*(.+?)(?:\n|$)", text, re.DOTALL
            )
            if reason_match:
                reasoning = reason_match.group(1).strip()

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
        """
        extract_prompt = f"""Extract the {max_concepts} most important concepts.

Rules:
- Keys: lowercase_underscore identifiers only
- Values: human-readable labels
- No generic keys like short_id, concept_1, key
- Return ONLY valid JSON

Good example:
{{"market_risk": "Market Risk", "network_effects": "Network Effects"}}

Text:
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
                        "content": "Extract concepts as JSON only."
                    },
                    {
                        "role":    "user",
                        "content": extract_prompt
                    }
                ]
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

            json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)

            concepts = json.loads(raw)

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
        return True  # OpenAI supports logprobs

    def provider_name(self) -> str:
        return "openai"

    def model_name(self) -> str:
        return self.model

    def __repr__(self) -> str:
        return f"OpenAIAdapter(model={self.model})"