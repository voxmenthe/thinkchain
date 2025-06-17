"""
Pytest configuration and shared fixtures for ThinkChain tests.
"""
import os
import pytest
import asyncio

from llm_adapters.base import Message, Role, CompletionConfig
from llm_adapters.anthropic_adapter import AnthropicAdapter
from llm_adapters.gemini_adapter import GeminiAdapter


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set - skipping Anthropic tests")
    return api_key


@pytest.fixture
def google_api_key():
    """Get Google API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set - skipping Gemini tests")
    return api_key


@pytest.fixture
def anthropic_adapter(anthropic_api_key):
    """Create and initialize an Anthropic adapter."""
    adapter = AnthropicAdapter(anthropic_api_key)
    return adapter


@pytest.fixture
def gemini_adapter(google_api_key):
    """Create and initialize a Gemini adapter."""
    adapter = GeminiAdapter(google_api_key)
    return adapter


@pytest.fixture
def test_config():
    """Get test configuration instance."""
    from config import get_test_config
    return get_test_config()


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        Message(role=Role.USER, content="Hello! What's 2 + 2?"),
        Message(role=Role.ASSISTANT, content="Hello! 2 + 2 equals 4."),
        Message(role=Role.USER, content="Thanks! Can you explain why?")
    ]


@pytest.fixture
def basic_completion_config(test_config):
    """Provide a basic completion configuration for testing."""
    provider_config = test_config.get_active_provider_config()
    if provider_config:
        model = provider_config.default_model
    else:
        model = "test-model"
    
    return CompletionConfig(
        model=model,
        max_tokens=test_config.max_tokens,
        temperature=test_config.temperature
    )


@pytest.fixture
def thinking_completion_config(test_config):
    """Provide a completion configuration with thinking enabled."""
    provider_config = test_config.get_active_provider_config()
    if provider_config:
        model = provider_config.default_model
    else:
        model = "test-model"
    
    return CompletionConfig(
        model=model,
        max_tokens=test_config.max_tokens,
        temperature=test_config.temperature,
        thinking_budget=test_config.thinking_budget
    )


@pytest.fixture
def anthropic_basic_config(test_config):
    """Basic config for Anthropic tests."""
    anthropic_config = test_config.providers.get("anthropic")
    if not anthropic_config:
        pytest.skip("Anthropic not configured in test config")
    
    return CompletionConfig(
        model=anthropic_config.default_model,
        max_tokens=anthropic_config.default_max_tokens,
        temperature=anthropic_config.default_temperature
    )


@pytest.fixture
def gemini_basic_config(test_config):
    """Basic config for Gemini tests."""
    gemini_config = test_config.providers.get("gemini")
    if not gemini_config:
        pytest.skip("Gemini not configured in test config")
    
    return CompletionConfig(
        model=gemini_config.default_model,
        max_tokens=gemini_config.default_max_tokens,
        temperature=gemini_config.default_temperature
    )


@pytest.fixture
def anthropic_thinking_config(test_config):
    """Thinking-enabled config for Anthropic tests."""
    anthropic_config = test_config.providers.get("anthropic")
    if not anthropic_config:
        pytest.skip("Anthropic not configured in test config")
    
    # Use a thinking-capable model for Anthropic
    thinking_model = "claude-3-5-sonnet-20241022"
    if thinking_model not in anthropic_config.models:
        thinking_model = anthropic_config.default_model
    
    return CompletionConfig(
        model=thinking_model,
        max_tokens=800,
        temperature=0.1,
        thinking_budget=anthropic_config.default_thinking_budget
    )


@pytest.fixture
def gemini_thinking_config(test_config):
    """Thinking-enabled config for Gemini tests."""
    gemini_config = test_config.providers.get("gemini")
    if not gemini_config:
        pytest.skip("Gemini not configured in test config")
    
    # Use a thinking-capable model for Gemini
    thinking_model = "gemini-2.0-flash-thinking-exp"
    if thinking_model not in gemini_config.models:
        thinking_model = gemini_config.default_model
    
    return CompletionConfig(
        model=thinking_model,
        max_tokens=800,
        temperature=0.1,
        thinking_budget=gemini_config.default_thinking_budget
    )


@pytest.fixture
def simple_math_messages():
    """Simple math question that should trigger thinking."""
    return [
        Message(role=Role.USER, content="Calculate 23 * 47 step by step")
    ]


@pytest.fixture
def complex_reasoning_messages():
    """Complex reasoning question that should trigger thinking."""
    return [
        Message(
            role=Role.USER, 
            content="If I have 5 apples and I give away 2, then buy 3 more, "
                    "then eat 1, how many do I have? Explain your reasoning."
        )
    ] 