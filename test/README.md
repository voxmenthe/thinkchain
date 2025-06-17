# ThinkChain LLM Adapter Tests

This directory contains comprehensive tests for the ThinkChain LLM adapter functionality.

## Test Structure

### Integration Tests (`test_llm_adapters_integration.py`)
These tests make real API calls and require valid API keys:

- **Adapter Initialization**: Tests that adapters initialize correctly with real credentials
- **Message Conversion**: Tests message format conversion between providers
- **Basic Completion**: Tests non-streaming completion functionality
- **Streaming Completion**: Tests streaming response functionality
- **Thinking Mode**: Tests extended thinking capabilities for both providers
- **Token Counting**: Tests token counting functionality
- **Error Handling**: Tests proper error handling for invalid inputs
- **Cross-Provider Consistency**: Tests that both providers behave consistently

### Unit Tests (`test_adapter_units.py`)
These tests don't require API keys and focus on specific functionality:

- **Message Conversion Logic**: Tests message format conversion without API calls
- **Capability Detection**: Tests provider capability reporting
- **Stream Chunk Processing**: Tests chunk processing logic with mocked data
- **Configuration Handling**: Tests configuration parameter handling
- **Edge Cases**: Tests handling of unusual inputs and error conditions

## Running the Tests

### Prerequisites

1. **Install pytest** (if not already installed):
```bash
pip install pytest pytest-asyncio
```

2. **Set up API keys** (for integration tests):
```bash
# For Anthropic tests
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# For Gemini tests  
export GOOGLE_API_KEY="AIza-your-key-here"
```

### Run All Tests
```bash
# From the project root
pytest test/

# With verbose output
pytest test/ -v

# With detailed output including print statements
pytest test/ -v -s
```

### Run Specific Test Categories

```bash
# Run only unit tests (no API calls required)
pytest test/test_adapter_units.py -v

# Run only integration tests (requires API keys)
pytest test/test_llm_adapters_integration.py -v

# Run specific test classes
pytest test/test_llm_adapters_integration.py::TestBasicCompletion -v

# Run specific test methods
pytest test/test_adapter_units.py::TestMessageConversionUnits::test_anthropic_message_conversion_simple -v
```

### Run Tests for Specific Providers

```bash
# Run only Anthropic tests
pytest test/ -k "anthropic" -v

# Run only Gemini tests  
pytest test/ -k "gemini" -v

# Run only thinking mode tests
pytest test/ -k "thinking" -v
```

### Skip Tests When API Keys Are Missing

The tests are designed to automatically skip when API keys are not available:

```bash
# This will skip Anthropic tests if ANTHROPIC_API_KEY is not set
pytest test/test_llm_adapters_integration.py::TestAdapterBasics::test_anthropic_adapter_initialization -v
```

## Test Coverage

### What's Tested

✅ **Adapter Initialization**
- Proper client initialization
- Provider name detection
- Capability reporting

✅ **Message Format Conversion**
- User/assistant/system message handling
- Role mapping between providers
- Content preservation
- Special character handling

✅ **Completion Functionality**
- Basic non-streaming completion
- Streaming completion with chunk assembly
- Response content validation

✅ **Thinking Mode**
- Extended thinking capabilities
- Thinking vs regular content separation
- Thinking budget handling

✅ **Token Counting**
- Accurate token counting
- Proper response format

✅ **Error Handling**
- Invalid model names
- Empty message lists
- Malformed inputs

✅ **Cross-Provider Consistency**
- Consistent behavior between providers
- Equivalent capability reporting

### What's Not Tested Yet

- Tool use functionality (requires tool integration)
- Vision capabilities (requires image inputs)
- Rate limiting and retry logic
- Connection pooling and performance
- Multi-turn conversation state

## Interpreting Test Results

### Successful Test Run
```
test/test_adapter_units.py::TestMessageConversionUnits::test_anthropic_message_conversion_simple PASSED
test/test_llm_adapters_integration.py::TestBasicCompletion::test_anthropic_basic_completion PASSED
```

### Skipped Tests (Missing API Keys)
```
test/test_llm_adapters_integration.py::TestAdapterBasics::test_anthropic_adapter_initialization SKIPPED (ANTHROPIC_API_KEY not set)
```

### Failed Tests
Look for specific assertion failures or API errors. Common issues:
- Invalid API keys
- Network connectivity issues
- Model availability changes
- Rate limiting

## Adding New Tests

When adding new tests:

1. **For unit tests**: Add to `test_adapter_units.py`, use mocks, no API calls
2. **For integration tests**: Add to `test_llm_adapters_integration.py`, use real APIs
3. **Follow naming conventions**: `test_[provider]_[functionality]`
4. **Use appropriate fixtures**: Defined in `conftest.py`
5. **Handle API key requirements**: Use pytest.skip for missing keys

## Debugging Test Failures

1. **Run with verbose output**: `pytest -v -s`
2. **Run single failing test**: `pytest path/to/test::TestClass::test_method -v`
3. **Check API key validity**: Test with simple API calls outside pytest
4. **Verify model availability**: Some models may be deprecated or unavailable
5. **Check network connectivity**: Some tests require internet access

## Performance Considerations

- Integration tests make real API calls and may be slow
- Rate limiting may cause test failures if run too frequently
- Consider running unit tests during development, integration tests before commits
- Use `pytest -x` to stop on first failure for faster debugging 