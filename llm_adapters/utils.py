"""Utility helpers shared across LLM adapters.

Provides helper functions for translating between ThinkChain's internal
representations and the provider-agnostic abstractions defined in
`llm_adapters.base`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import Message, Role

__all__ = [
    "convert_transcript_to_messages",
]

def convert_transcript_to_messages(transcript: List[Dict[str, Any]]) -> List[Message]:
    """Convert ThinkChain's *legacy* transcript list into unified `Message` objects.

    Each transcript item is expected to be a mapping with at least the keys
    ``role`` and ``content`` – mirroring Anthropic's REST/SDK schema.  The
    adapters, however, operate on the strongly-typed ``Message`` dataclass.

    Any non-string ``content`` (e.g. rich block structure emitted after
    tool execution) will be *stringified* via ``json.dumps`` so that no
    semantic information is lost while still fitting into the plain-text
    structure accepted by all providers.  This logic can be revisited once
    the other providers achieve feature-parity regarding message blocks.
    """
    result: List[Message] = []

    for entry in transcript:
        role_str: str = str(entry.get("role", "user")).lower()
        try:
            role = Role(role_str)
        except ValueError:
            role = Role.USER  # Fallback – treat unknown roles as user

        content = entry.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)

        result.append(Message(role=role, content=content))

    return result
