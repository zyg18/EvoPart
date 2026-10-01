"""
Base LLM interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class FatalLLMError(RuntimeError):
    """An LLM failure that should stop the whole run instead of costing one iteration.

    Raised by a backend when the provider itself is failing (API 5xx, overloaded,
    rate limited) and retries did not help. The parallel controller treats a
    result carrying this error as fatal: it stops submitting iterations, saves a
    checkpoint at the last completed iteration, and re-raises, so an outage
    cannot silently burn through the iteration budget.
    """


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        pass
