# ThinkChain Configuration Guide

This guide explains the centralized configuration system for ThinkChain, which eliminates hardcoded values and provides flexible control over all LLM provider settings.

## Overview

The configuration system consists of:

- **Global Configuration**: Universal settings that apply across all providers
- **Provider-Specific Configuration**: Settings specific to each LLM provider  
- **Model Configuration**: Individual model capabilities and characteristics
- **Test Configuration**: Special settings for testing environments

## Configuration Structure

### Environment Variables

#### Universal Settings
```bash
# Global hyperparameters (apply to all providers unless overridden)
export THINKCHAIN_TEMPERATURE="0.7"
export THINKCHAIN_MAX_TOKENS="4096"
export THINKCHAIN_THINKING_BUDGET="2048"

# UI and behavior
export THINKCHAIN_SHOW_THINKING="true"
export THINKCHAIN_DEBUG="false"
export THINKCHAIN_TEST_MODE="false"

# Provider selection
export THINKCHAIN_PROVIDER="auto"  # or "anthropic" or "gemini"
```

#### Anthropic Configuration
```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Optional - Provider-specific overrides
export ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"
export ANTHROPIC_TEMPERATURE="0.7"
export ANTHROPIC_MAX_TOKENS="4096"
export ANTHROPIC_THINKING_BUDGET="2048"
```

#### Gemini Configuration
```bash
# Option 1: Developer API
export GOOGLE_API_KEY="AIza-your-key-here"

# Option 2: Vertex AI
export GOOGLE_GENAI_USE_VERTEXAI="true"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Optional - Provider-specific overrides
export GEMINI_MODEL="gemini-2.0-flash-exp"
export GEMINI_TEMPERATURE="0.7"
export GEMINI_MAX_TOKENS="4096"
export GEMINI_THINKING_BUDGET="1024"
```

## Available Models

### Anthropic Models

| Model | Context | Thinking | Tools | Vision | Cost (per 1K tokens) |
|-------|---------|----------|-------|--------|--------------------|
| `claude-3-5-sonnet-20241022` | 200K | ✅ | ✅ | ✅ | $3.00/$15.00 |
| `claude-sonnet-4-20250514` | 200K | ✅ | ✅ | ✅ | $3.00/$15.00 |
| `claude-3-haiku-20240307` | 200K | ❌ | ✅ | ❌ | $0.25/$1.25 |

### Gemini Models

| Model | Context | Thinking | Tools | Vision | Cost (per 1K tokens) |
|-------|---------|----------|-------|--------|--------------------|
| `gemini-2.0-flash-thinking-exp` | 1M | ✅ | ✅ | ✅ | $0.075/$0.30 |
| `gemini-2.0-flash-exp` | 1M | ❌ | ✅ | ✅ | $0.075/$0.30 |
| `gemini-2.0-flash-001` | 1M | ❌ | ✅ | ✅ | $0.075/$0.30 |

## Using the Configuration System

### In Application Code

```python
from config import get_config

# Get the global configuration
config = get_config()

# Access provider settings
provider_config = config.get_active_provider_config()
model = provider_config.default_model
temperature = config.get_effective_temperature()
max_tokens = config.get_effective_max_tokens()

# Check model capabilities
supports_thinking = config.supports_thinking("anthropic", "claude-3-5-sonnet-20241022")

# Get available models for a provider
models = config.get_available_models("gemini")
```

### In Adapter Code

```python
from config import get_config

class AnthropicAdapter(BaseAdapter):
    def get_default_model(self) -> str:
        config = get_config()
        provider_config = config.providers.get("anthropic")
        if provider_config:
            return provider_config.default_model
        return "claude-3-5-sonnet-20241022"  # Fallback
    
    def get_token_counting_model(self) -> str:
        config = get_config()
        provider_config = config.providers.get("anthropic")
        # Use cheapest model for token counting
        if provider_config and "claude-3-haiku-20240307" in provider_config.models:
            return "claude-3-haiku-20240307"
        return self.get_default_model()
```

### In Tests

```python
from config import get_test_config

# Use test configuration with safe defaults
def test_something():
    config = get_test_config()
    
    # Test config has lower token limits and deterministic settings
    assert config.max_tokens == 100
    assert config.temperature == 0.1
    assert config.test_mode is True
```

## Configuration Hierarchy

Settings are resolved in this order (highest to lowest priority):

1. **Provider-specific environment variables** (e.g., `ANTHROPIC_TEMPERATURE`)
2. **Global environment variables** (e.g., `THINKCHAIN_TEMPERATURE`)
3. **Provider defaults** (defined in config.py)
4. **System defaults** (fallback values)

Example:
```bash
export THINKCHAIN_TEMPERATURE="0.8"     # Global default
export ANTHROPIC_TEMPERATURE="0.5"     # Anthropic override
export GEMINI_TEMPERATURE="0.9"        # Gemini override
```

Result:
- Anthropic requests use temperature = 0.5
- Gemini requests use temperature = 0.9
- Any other provider would use temperature = 0.8

## Runtime Configuration

### Changing Active Provider

```python
from config import get_config

config = get_config()
config.active_provider = "gemini"  # Switch to Gemini

# Or use environment variable
import os
os.environ["THINKCHAIN_PROVIDER"] = "anthropic"
config = reload_config()  # Reload from environment
```

### Model Validation

```python
from config import get_config

config = get_config()

# Check if model is available for provider
is_valid = config.validate_model_for_provider("anthropic", "claude-3-haiku-20240307")

# Get model capabilities
model_config = config.get_model_config("anthropic", "claude-3-5-sonnet-20241022")
if model_config and model_config.supports_thinking:
    print("Model supports thinking mode")
```

## Testing Configuration

### Running Tests with Different Models

```bash
# Test with specific models
export ANTHROPIC_MODEL="claude-3-haiku-20240307"
export GEMINI_MODEL="gemini-2.0-flash-exp"
python -m pytest test/

# Test in fast mode (lower token limits)
export THINKCHAIN_TEST_MODE="true"
python -m pytest test/
```

### Test Fixtures

Tests automatically use appropriate models based on configuration:

```python
@pytest.fixture
def anthropic_thinking_config(test_config):
    """Get thinking-capable config for Anthropic."""
    anthropic_config = test_config.providers.get("anthropic")
    
    # Automatically selects best thinking model available
    thinking_model = "claude-3-5-sonnet-20241022"
    if thinking_model not in anthropic_config.models:
        thinking_model = anthropic_config.default_model
    
    return CompletionConfig(
        model=thinking_model,
        thinking_budget=anthropic_config.default_thinking_budget
    )
```

## Best Practices

### For Development

1. **Use environment variables** for all configuration
2. **Set provider-specific overrides** for testing different models
3. **Enable debug mode** when troubleshooting: `THINKCHAIN_DEBUG=true`
4. **Use test mode** for faster iterations: `THINKCHAIN_TEST_MODE=true`

### For Production

1. **Set explicit models** rather than relying on auto-selection
2. **Configure thinking budgets** appropriate for your use case
3. **Monitor costs** using the model pricing information
4. **Use cheapest models** for non-critical operations (e.g., token counting)

### For Testing

1. **Use test configuration** to ensure consistent results
2. **Override specific settings** per test when needed
3. **Test with multiple providers** to ensure compatibility
4. **Use model validation** before running expensive tests

## Migration from Hardcoded Values

The old hardcoded approach:
```python
# OLD - Don't do this
response = client.messages.create(
    model="claude-3-haiku-20240307",  # Hardcoded!
    max_tokens=100,                   # Hardcoded!
    temperature=0.1,                  # Hardcoded!
    messages=messages
)
```

The new configuration-driven approach:
```python
# NEW - Use configuration
from config import get_config

config = get_config()
provider_config = config.get_active_provider_config()

response = client.messages.create(
    model=provider_config.default_model,
    max_tokens=config.get_effective_max_tokens(),
    temperature=config.get_effective_temperature(),
    messages=messages
)
```

## Troubleshooting

### Common Issues

1. **No providers configured**: Check that API keys are set
2. **Model not found**: Use `config.get_available_models(provider)` to see options
3. **Thinking not working**: Verify model supports thinking with `config.supports_thinking()`
4. **Tests failing**: Check test configuration with `get_test_config()`

### Debug Commands

```python
from config import get_config

config = get_config()

# Check configuration status
print(f"Active provider: {config.active_provider}")
print(f"Available providers: {list(config.providers.keys())}")

# Check model availability
for provider_name in config.providers:
    models = config.get_available_models(provider_name)
    print(f"{provider_name} models: {models}")

# Check effective settings
print(f"Temperature: {config.get_effective_temperature()}")
print(f"Max tokens: {config.get_effective_max_tokens()}")
print(f"Thinking budget: {config.get_effective_thinking_budget()}")
```

This configuration system provides complete flexibility while maintaining simplicity for common use cases. All hardcoded values have been eliminated, making the system easy to configure and test across different environments. 