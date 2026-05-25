"""
PRISM — adapters/anthropic/adapter.py
Anthropic Claude Adapter

Connects PRISM to Claude via the Anthropic API.
Implements the BaseAdapter interface for full
VERDICT scoring, EAT projection, and SDK instrumentation.

Supports:
  - Text completion (claude-sonnet-4-6, claude-opus-4-6, haiku)
  - Output distribution over hypotheses (for groundedness scoring)
  - Chain-of-thought extraction (for EAT φ_A projection)
  - Streaming (optional)

Usage:
    from adapters.anthropic.adapter import AnthropicAdapter

    adapter = AnthropicAdapter(
        api_key   = "your-key",
        model     = "claude-sonnet-4-6"
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
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ─── ANTHROPIC ADAPTER ────────────────────────────────────────────────────────

class AnthropicAdapter(BaseAdapter):
    """
    PRISM adapter for Anthropic Claude models.

    Wraps the Anthropic Python SDK to provide:
      1. Standard text completion
      2. Hypothesis probability distribution
         (approximated via constrained prompting)
      3. Chain-of-thought extraction
         (via extended thinking or CoT prompting)

    Model strings:
      claude-sonnet-4-6   ← recommended for FDE deployments
      claude-opus-4-6     ← highest capability
      claude-haiku-4-5-20251001   ← fastest, lowest cost
    """

    SUPPORTED_MODELS = [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str           = "claude-sonnet-4-6",
        max_tokens:  int           = 1024,
        temperature: float         = 0.7,
        system:      Optional[str] = None
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )

        self._api_key     = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model        = model
        self.max_tokens   = max_tokens
        self.temperature  = temperature
        self.system       = system or (
            "You are a helpful AI assistant participating in a "
            "collaborative research workflow. Be precise, calibrated, "
            "and explicit about your confidence levels."
        )

        self._client = anthropic.Anthropic(api_key=self._api_key)

    # ── CORE COMPLETION ───────────────────────────────────────────────────

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a text completion for the given prompt.

        Returns the full text response from Claude.
        This is the primary interface for AI agent instrumentation.
        """
        try:
            message = self._client.messages.create(
                model      = kwargs.get("model", self.model),
                max_tokens = kwargs.get("max_tokens", self.max_tokens),
                system     = kwargs.get("system", self.system),
                messages   = [{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        except Exception as e:
            return f"[AnthropicAdapter error: {str(e)}]"

    # ── OUTPUT DISTRIBUTION ───────────────────────────────────────────────

    def get_output_distribution(
        self,
        prompt:     str,
        hypotheses: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Approximate a probability distribution over hypotheses.

        Uses constrained prompting: asks Claude to rate each hypothesis
        on a 0-100 scale, then normalizes to a probability distribution.

        Used by VERDICT groundedness scoring to determine whether
        AI belief updates are anchored in session evidence.

        Returns: {hypothesis: confidence} where values sum ≈ 1.0
        """
        if not hypotheses:
            return {}

        hyp_list = "\n".join(
            f"{i+1}. {h}" for i, h in enumerate(hypotheses)
        )

        distribution_prompt = f"""Given this context:
{prompt}

Rate each hypothesis by how strongly the context supports it.
Use a score from 0 to 100 for each.
Respond ONLY with a JSON object mapping hypothesis to score.
No explanation. No markdown. Just the JSON.

Hypotheses:
{hyp_list}

Example format:
{{"hypothesis_1": 75, "hypothesis_2": 25}}

Your response (JSON only):"""

        try:
            response = self._client.messages.create(
                model      = self.model,
                max_tokens = 256,
                system     = "You are a precise evaluator. Respond only with valid JSON.",
                messages   = [{"role": "user", "content": distribution_prompt}]
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

            scores_raw = json.loads(raw)

            # Map back to original hypotheses
            distribution = {}
            total = 0.0

            for i, hyp in enumerate(hypotheses):
                # Try exact match first, then positional
                key = str(i + 1)
                score = (
                    scores_raw.get(hyp, 0) or
                    scores_raw.get(f"hypothesis_{i+1}", 0) or
                    scores_raw.get(key, 0) or
                    list(scores_raw.values())[i]
                    if i < len(scores_raw) else 0
                )
                distribution[hyp] = float(max(score, 0))
                total += distribution[hyp]

            # Normalize to sum to 1.0
            if total > 0:
                distribution = {k: v / total for k, v in distribution.items()}
            else:
                # Uniform fallback
                uniform = 1.0 / len(hypotheses)
                distribution = {h: uniform for h in hypotheses}

            return distribution

        except Exception as e:
            # Uniform fallback on any error
            uniform = 1.0 / len(hypotheses)
            return {h: uniform for h in hypotheses}

    # ── CHAIN OF THOUGHT ──────────────────────────────────────────────────

    def get_chain_of_thought(
        self,
        prompt: str,
        **kwargs
    ) -> Optional[str]:
        """
        Extract chain-of-thought reasoning from Claude.

        Uses explicit CoT prompting to surface reasoning steps.
        Used by φ_A (AI projection function) to extract C_UPDATE
        events from AI concept formation.

        Returns structured reasoning trace as a string.
        """
        cot_prompt = f"""Think through this step by step before answering.
Show your reasoning explicitly.

{prompt}

Format your response as:
REASONING:
[your step-by-step thinking]

CONCLUSION:
[your final answer]"""

        try:
            response = self._client.messages.create(
                model      = self.model,
                max_tokens = self.max_tokens,
                system     = self.system,
                messages   = [{"role": "user", "content": cot_prompt}]
            )
            full_text = response.content[0].text

            # Extract reasoning section
            if "REASONING:" in full_text:
                reasoning_start = full_text.find("REASONING:") + len("REASONING:")
                conclusion_start = full_text.find("CONCLUSION:")
                if conclusion_start > reasoning_start:
                    return full_text[reasoning_start:conclusion_start].strip()
                return full_text[reasoning_start:].strip()

            return full_text

        except Exception as e:
            return None

    # ── CALIBRATED COMPLETION ─────────────────────────────────────────────

    def complete_with_confidence(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion with explicit confidence scoring.

        Asks Claude to express its confidence level alongside
        its response. Used by VERDICT calibration tracking.

        Returns:
            {
                "response":   str,
                "confidence": float,  # 0.0 - 1.0
                "reasoning":  str
            }
        """
        confidence_prompt = f"""{prompt}

After your response, add:
CONFIDENCE: [a number from 0 to 100 representing how confident you are]
REASONING: [one sentence explaining your confidence level]"""

        try:
            response = self._client.messages.create(
                model      = self.model,
                max_tokens = self.max_tokens,
                system     = self.system,
                messages   = [{"role": "user", "content": confidence_prompt}]
            )
            text = response.content[0].text

            # Extract confidence score
            confidence = 0.7  # default
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

            # Clean response — remove confidence/reasoning lines
            clean_response = re.sub(
                r"\nCONFIDENCE:.*", "", text, flags=re.DOTALL
            ).strip()

            return {
                "response":   clean_response,
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
        text: str,
        max_concepts: int = 10
    ) -> Dict[str, str]:
        """
        Extract key concepts from text for C_UPDATE EAT events.

        Used by φ_A to build conceptual graph updates from
        AI-generated text without white-box activation access.

        Returns: {concept_id: concept_label}
        """
        extract_prompt = f"""Extract the {max_concepts} most important concepts from this text.
Return ONLY a JSON object mapping short_id to concept_label.
No explanation. No markdown.

Text: {text[:1000]}

Example: {{"risk": "Market Risk", "innovation": "Product Innovation"}}

JSON only:"""

        try:
            response = self._client.messages.create(
                model      = self.model,
                max_tokens = 256,
                system     = "Extract concepts. Return only valid JSON.",
                messages   = [{"role": "user", "content": extract_prompt}]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
            concepts = json.loads(raw)
            return {
                str(k).lower().replace(" ", "_"): str(v)
                for k, v in concepts.items()
            }

        except Exception:
            return {}

    # ── ADAPTER METADATA ──────────────────────────────────────────────────

    def supports_cot(self) -> bool:
        return True

    def supports_logprobs(self) -> bool:
        return False  # Anthropic API doesn't expose logprobs directly

    def provider_name(self) -> str:
        return "anthropic"

    def model_name(self) -> str:
        return self.model

    def __repr__(self) -> str:
        return f"AnthropicAdapter(model={self.model})"