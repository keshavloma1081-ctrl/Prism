"""
PRISM — adapters/base/base.py
Abstract Base Adapter Interface

All model adapters implement this interface.
Enables model-agnostic instrumentation across
Anthropic, OpenAI, Cohere, and any other provider.

Adding a new adapter:
  1. Create adapters/your_provider/adapter.py
  2. Subclass BaseAdapter
  3. Implement all abstract methods
  4. Register in adapters/__init__.py
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseAdapter(ABC):
    """
    Abstract base class for all PRISM model adapters.

    Every adapter provides three capabilities:
      1. complete()               → generate text response
      2. get_output_distribution() → belief approximation over hypotheses
      3. get_chain_of_thought()   → CoT trace for EAT projection
    """

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a text completion for the given prompt.
        Primary interface for AI agent instrumentation.
        """
        pass

    @abstractmethod
    def get_output_distribution(
        self,
        prompt:     str,
        hypotheses: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Approximate a probability distribution over hypotheses.

        Used by VERDICT groundedness scoring to determine
        whether AI belief updates are anchored in session evidence.

        Returns: {hypothesis: confidence} where sum ≈ 1.0
        """
        pass

    @abstractmethod
    def get_chain_of_thought(
        self,
        prompt: str,
        **kwargs
    ) -> Optional[str]:
        """
        Return chain-of-thought reasoning trace if available.

        Used by φ_A (AI projection function) to extract
        EAT events from AI reasoning steps.

        Returns None if CoT not available for this model.
        """
        pass

    def provider_name(self) -> str:
        """Return the provider name for this adapter."""
        return self.__class__.__module__.split(".")[1]

    def model_name(self) -> str:
        """Return the model name for this adapter."""
        return getattr(self, "model", "unknown")

    def supports_cot(self) -> bool:
        """Whether this adapter supports chain-of-thought extraction."""
        return False

    def supports_logprobs(self) -> bool:
        """Whether this adapter supports output log probabilities."""
        return False