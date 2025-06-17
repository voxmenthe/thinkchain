from typing import Dict, Type
from .base import BaseAdapter
from .anthropic_adapter import AnthropicAdapter
from .gemini_adapter import GeminiAdapter

_ADAPTERS: Dict[str, Type[BaseAdapter]] = {
    'anthropic': AnthropicAdapter,
    'gemini': GeminiAdapter,
}

def get_adapter(provider: str) -> BaseAdapter:
    """Factory function to get appropriate adapter"""
    if provider not in _ADAPTERS:
        raise ValueError(f"Unknown provider: {provider}")
    return _ADAPTERS[provider]()