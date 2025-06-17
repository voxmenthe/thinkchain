# Implementation Details: Universal LLM Provider Support for ThinkChain

## Overview
This document provides a step-by-step implementation guide for adding universal LLM provider support to ThinkChain. The work is broken down into manageable phases that a junior engineer can tackle incrementally.

**Current Status**: ThinkChain has basic adapter infrastructure with Anthropic and partial Gemini support. We need to complete the integration and make the UI provider-agnostic.


**Gemini Implementation Notes**:
The new implementation will use the new google-genai SDK (documentation can be found at https://googleapis.github.io/python-genai/ and also here https://ai.google.dev/gemini-api/docs/text-generation).

Note that the SDK and its documentation has recently changed, so it is **very important to read the current documentation on the web** to understand how to use it, and make sure we are using the latest version - you **CANNOT** use your existing knowledge here but must instead make sure to read the documentation, understand it, and document your findings, and use those in your implementation.

The Anthropic SDK may have changed as well, so probably good to double-check the current documentation for it as well.

---

## Phase 1: Complete Adapter Foundation (4-6 hours)

### Current State Analysis
The codebase already has:
- ✅ `llm_adapters/base.py` - Basic adapter interface
- ✅ `llm_adapters/anthropic_adapter.py` - Working Anthropic adapter
- ✅ `llm_adapters/gemini_adapter.py` - Partial Gemini adapter
- ✅ Tool discovery system in `tool_discovery.py`

### Task 1.1: Fix Anthropic Adapter Issues
**File**: `llm_adapters/anthropic_adapter.py`

**Current Issues to Fix**:
1. Line 33: `parms["tools"]` should be `params["tools"]` (typo)
2. Missing message conversion method `_convert_messages`
3. Incomplete chunk processing

**Implementation**:
```python
# Fix the typo and add missing methods
def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert unified messages to Anthropic format"""
    anthropic_messages = []
    for msg in messages:
        anthropic_msg = {
            "role": msg.role.value,
            "content": msg.content
        }
        anthropic_messages.append(anthropic_msg)
    return anthropic_messages

def _process_chunk(self, chunk) -> StreamChunk:
    """Convert Anthropic stream chunk to unified format."""
    result = StreamChunk()
    
    if chunk.type == 'content_block_delta':
        if hasattr(chunk.delta, 'text'):
            result.delta_text = chunk.delta.text
    elif chunk.type == 'content_block_start':
        if hasattr(chunk.content_block, 'type'):
            if chunk.content_block.type == 'thinking':
                result.thinking_text = ""
            elif chunk.content_block.type == 'tool_use':
                result.tool_use = ToolUse(
                    id=chunk.content_block.id,
                    name=chunk.content_block.name,
                    arguments={}
                )
    elif chunk.type == 'content_block_stop':
        # Handle completion of thinking or tool blocks
        pass
    
    return result
```

### Task 1.2: Complete Gemini Adapter
**File**: `llm_adapters/gemini_adapter.py`

**Missing Implementation**:
1. `_convert_messages` method
2. `count_tokens` method
3. Message format conversion

**Implementation**:
```python
def _convert_messages(self, messages: List[Message]) -> List[types.Content]:
    """Convert unified messages to Gemini format"""
    gemini_contents = []
    for msg in messages:
        # Map roles: assistant -> model for Gemini
        role_mapping = {
            "user": "user",
            "assistant": "model", 
            "system": "user",  # System messages as user messages
            "tool": "user"
        }
        
        gemini_role = role_mapping.get(msg.role.value, "user")
        content = types.Content(
            role=gemini_role,
            parts=[types.Part.from_text(str(msg.content))]
        )
        gemini_contents.append(content)
    
    return gemini_contents

async def count_tokens(self, messages: List[Message]) -> Dict[str, int]:
    """Count tokens for Gemini messages"""
    if not self.client:
        raise ValueError("Client not initialized")
    
    # Convert to Gemini format and count
    gemini_contents = self._convert_messages(messages)
    
    # Use Gemini's token counting if available
    try:
        count_result = await self.client.models.count_tokens(
            model="gemini-2.0-flash-001",  # Default model for counting
            contents=gemini_contents
        )
        return {
            "prompt_tokens": count_result.total_tokens,
            "completion_tokens": 0,
            "total_tokens": count_result.total_tokens
        }
    except:
        # Fallback: rough estimation
        total_text = " ".join(str(msg.content) for msg in messages)
        estimated_tokens = len(total_text.split()) * 1.3  # Rough multiplier
        return {
            "prompt_tokens": int(estimated_tokens),
            "completion_tokens": 0, 
            "total_tokens": int(estimated_tokens)
        }
```

### Task 1.3: Enhance Base Adapter Interface
**File**: `llm_adapters/base.py`

**Add Missing Capabilities**:
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Set

class ProviderCapability(Enum):
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    REASONING_DISPLAY = "reasoning_display"
    SYSTEM_MESSAGES = "system_messages"

@dataclass
class ProviderCapabilities:
    capabilities: Set[ProviderCapability]
    max_context_length: Optional[int] = None
    reasoning_display_name: Optional[str] = None
    
class BaseAdapter(ABC):
    # ... existing methods ...
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities"""
        pass
        
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name"""
        pass
        
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a capability"""
        return capability in self.get_capabilities().capabilities
```

**Implementation in adapters**:
```python
# In AnthropicAdapter
def get_capabilities(self) -> ProviderCapabilities:
    return ProviderCapabilities(
        capabilities={
            ProviderCapability.STREAMING,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.REASONING_DISPLAY,
            ProviderCapability.SYSTEM_MESSAGES,
            ProviderCapability.VISION
        },
        max_context_length=200000,
        reasoning_display_name="thinking"
    )

def get_provider_name(self) -> str:
    return "anthropic"

# In GeminiAdapter  
def get_capabilities(self) -> ProviderCapabilities:
    return ProviderCapabilities(
        capabilities={
            ProviderCapability.STREAMING,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.REASONING_DISPLAY,
            ProviderCapability.VISION
        },
        max_context_length=1000000,
        reasoning_display_name="reasoning"
    )

def get_provider_name(self) -> str:
    return "gemini"
```

---

## Phase 2: Create Configuration System (3-4 hours)

### Task 2.1: Create Configuration Management
**File**: `config.py` (new file)

**Implementation**:
```python
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from llm_adapters.base import ProviderCapability

@dataclass
class ProviderConfig:
    name: str
    default_model: str
    credentials: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class ThinkChainConfig:
    # Active provider selection
    active_provider: str = "auto"
    
    # Provider configurations
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    
    # Universal settings (from current CONFIG dict)
    temperature: float = 0.7
    max_tokens: int = 1024
    thinking_budget: int = 1024
    
    @classmethod
    def from_env(cls) -> 'ThinkChainConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Auto-detect providers
        providers = {}
        
        # Check for Anthropic (already working)
        if os.getenv("ANTHROPIC_API_KEY"):
            providers["anthropic"] = ProviderConfig(
                name="anthropic",
                default_model="claude-sonnet-4-20250514",
                credentials={"api_key": os.getenv("ANTHROPIC_API_KEY")}
            )
        
        # Check for Gemini
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
            creds = {}
            if os.getenv("GOOGLE_API_KEY"):
                creds["api_key"] = os.getenv("GOOGLE_API_KEY")
            providers["gemini"] = ProviderConfig(
                name="gemini", 
                default_model="gemini-2.0-flash-001",
                credentials=creds
            )
        
        config.providers = providers
        
        # Set active provider
        explicit_provider = os.getenv("THINKCHAIN_PROVIDER") 
        if explicit_provider and explicit_provider in providers:
            config.active_provider = explicit_provider
        elif providers:
            config.active_provider = list(providers.keys())[0]
        
        return config
    
    def get_active_provider_config(self) -> Optional[ProviderConfig]:
        if self.active_provider == "auto" and self.providers:
            return list(self.providers.values())[0]
        return self.providers.get(self.active_provider)
```

### Task 2.2: Update Tool Discovery for Multi-Provider
**File**: `tool_discovery.py`

**Current Issue**: Tool discovery is tightly coupled to Claude format. Need to add provider-agnostic layer.

**Add these functions**:
```python
def get_provider_tools(adapter) -> Any:
    """Get tools converted for specific provider"""
    from llm_adapters.base import BaseAdapter
    if not isinstance(adapter, BaseAdapter):
        # Fallback for current usage
        return get_claude_tools()
    
    universal_tools = get_claude_tools()  # These are already universal format
    return [adapter.convert_tool_schema(tool) for tool in universal_tools]

def create_tool_awareness_message(adapter=None) -> str:
    """Create provider-aware tool awareness message"""
    if adapter is None:
        # Current behavior
        return """You have access to various tools that can help you assist users more effectively. These tools allow you to:

- Perform web searches and scrape web content
- Read, create, and edit files
- Check weather information  
- Manage packages and dependencies
- Run linting tools
- Create folders and organize files

When a user asks a question that would benefit from using one of these tools, please use them to provide more accurate and helpful responses. For example:
- Use web search for current information or research
- Use file tools when users want to work with files
- Use weather tools for location-based weather queries

Always explain what you're doing when you use tools, and use the results to provide comprehensive answers."""
    
    # Provider-aware message
    capabilities = adapter.get_capabilities()
    provider_name = adapter.get_provider_name()
    
    base_message = "You have access to various tools for helping users effectively."
    
    if adapter.supports_capability(ProviderCapability.REASONING_DISPLAY):
        reasoning_name = capabilities.reasoning_display_name or "reasoning"
        base_message += f" When using tools, share your {reasoning_name} process."
    
    return base_message + " Use tools when they would help answer questions comprehensively."
```

---

## Phase 3: Refactor UI Layer (6-8 hours)

### Task 3.1: Create Adapter Factory System
**File**: `adapter_factory.py` (new file)

**Implementation**:
```python
from typing import Optional
from llm_adapters.base import BaseAdapter
from llm_adapters.anthropic_adapter import AnthropicAdapter  
from llm_adapters.gemini_adapter import GeminiAdapter
from config import ThinkChainConfig, ProviderConfig

class AdapterFactory:
    """Factory for creating and managing LLM adapters"""
    
    _adapters = {
        'anthropic': AnthropicAdapter,
        'gemini': GeminiAdapter,
    }
    
    @classmethod
    async def create_adapter(cls, provider_name: str, provider_config: ProviderConfig) -> BaseAdapter:
        """Create and initialize an adapter"""
        if provider_name not in cls._adapters:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        adapter_class = cls._adapters[provider_name]
        adapter = adapter_class()
        
        await adapter.initialize(**provider_config.credentials)
        return adapter
    
    @classmethod
    async def create_from_config(cls, config: ThinkChainConfig) -> Optional[BaseAdapter]:
        """Create adapter from configuration"""
        provider_config = config.get_active_provider_config()
        if not provider_config:
            return None
        
        return await cls.create_adapter(provider_config.name, provider_config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return list(cls._adapters.keys())
```

### Task 3.2: Refactor Main UI Functions
**File**: `thinkchain.py`

**Key Changes Needed**:

1. **Replace global CONFIG and client**:
```python
# OLD (lines 38-47)
CONFIG = {
    "model": "claude-sonnet-4-20250514",
    "thinking_budget": 1024,
    "max_tokens": 1024
}
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# NEW
from config import ThinkChainConfig
from adapter_factory import AdapterFactory
from llm_adapters.base import CompletionConfig

# Global state
APP_CONFIG = ThinkChainConfig.from_env()
CURRENT_ADAPTER = None  # Will be initialized
```

2. **Update `stream_once` function** (lines 111-200):
```python
# OLD - Provider-specific streaming
async def stream_once(transcript: list[dict], adapter: BaseAdapter, config: CompletionConfig) -> dict:
    """Universal streaming function"""
    # Convert transcript to Message format
    from llm_adapters.base import Message, Role
    messages = []
    for msg in transcript:
        role = Role(msg["role"])
        messages.append(Message(role=role, content=msg["content"]))
    
    capabilities = adapter.get_capabilities()
    final_content = []
    tool_uses = []
    
    # Stream with unified interface
    async for chunk in adapter.stream_completion(messages, config):
        if chunk.thinking_text and capabilities.reasoning_display_name:
            reasoning_name = capabilities.reasoning_display_name
            ui.print(f"\n💭 [bold blue]{reasoning_name.title()}:[/bold blue] ", end='')
            ui.console.print(chunk.thinking_text, style="italic blue")
        elif chunk.delta_text:
            ui.console.print(chunk.delta_text, end='')
            final_content.append(chunk.delta_text)
        elif chunk.tool_use:
            tool_uses.append(chunk.tool_use)
            ui.print(f"\n🔧 [bold yellow]Tool Use:[/bold yellow] [cyan]{chunk.tool_use.name}[/cyan]")
    
    # Handle tool uses if any
    if tool_uses:
        for tool_use in tool_uses:
            ui.print_json(tool_use.arguments, f"Arguments for {tool_use.name}")
            result = run_tool(tool_use.name, tool_use.arguments)
            
            # Add tool result to transcript
            transcript.append({
                "role": "user",
                "content": f"Tool result from {tool_use.name}: {result}"
            })
    
    return {"role": "assistant", "content": "".join(final_content)}
```

3. **Update initialization** (lines 72-95):
```python
async def initialize_tools_and_adapter():
    """Initialize tools and adapter with progress display"""
    global CURRENT_ADAPTER, TOOLS
    
    steps = [
        "🔍 Discovering local tools...",
        "🌐 Initializing MCP integration...",
        "🤖 Setting up LLM provider...",
        "✅ Finalizing setup..."
    ]
    
    print_initialization_progress(steps)
    
    # Initialize tools (existing logic)
    tools, local_count, mcp_count = initialize_tools_with_progress()
    TOOLS = tools
    
    # Initialize adapter
    ui.print("🤖 Setting up LLM provider...")
    try:
        CURRENT_ADAPTER = await AdapterFactory.create_from_config(APP_CONFIG)
        if CURRENT_ADAPTER:
            provider_name = CURRENT_ADAPTER.get_provider_name()
            ui.print_success(f"Initialized {provider_name} provider")
        else:
            ui.print_error("No LLM provider configured", "Check your API keys")
            return None, 0, 0
    except Exception as e:
        ui.print_error("Provider initialization failed", str(e))
        return None, 0, 0
    
    return CURRENT_ADAPTER, local_count, mcp_count
```

### Task 3.3: Add Provider Management Commands
**File**: `thinkchain.py`

**Add new command handlers**:
```python
def show_providers_command():
    """Show available providers and their status"""
    ui.print_rule("Available Providers", style="blue")
    
    providers_table = Table(title="LLM Providers")
    providers_table.add_column("Provider", style="cyan")
    providers_table.add_column("Status", style="green")
    providers_table.add_column("Model", style="white")
    providers_table.add_column("Capabilities", style="magenta")
    
    for name, provider_config in APP_CONFIG.providers.items():
        status = "✅ Active" if name == APP_CONFIG.active_provider else "⚪ Available"
        
        # Get capabilities if provider is active
        if CURRENT_ADAPTER and CURRENT_ADAPTER.get_provider_name() == name:
            capabilities = CURRENT_ADAPTER.get_capabilities()
            cap_list = [cap.value for cap in capabilities.capabilities]
            cap_str = ", ".join(cap_list[:3]) + ("..." if len(cap_list) > 3 else "")
        else:
            cap_str = "Unknown"
        
        providers_table.add_row(
            name.title(),
            status,
            provider_config.default_model,
            cap_str
        )
    
    ui.console.print(providers_table)

async def handle_provider_command(args):
    """Handle /provider command for switching providers"""
    global CURRENT_ADAPTER
    
    if not args:
        current = APP_CONFIG.active_provider
        available = list(APP_CONFIG.providers.keys())
        ui.print(f"Current provider: {current}")
        ui.print(f"Available: {', '.join(available)}")
        return
    
    new_provider = args[0].lower()
    if new_provider not in APP_CONFIG.providers:
        ui.print_error("Invalid provider", f"Choose from: {', '.join(APP_CONFIG.providers.keys())}")
        return
    
    # Switch provider
    try:
        ui.print(f"Switching to {new_provider}...")
        provider_config = APP_CONFIG.providers[new_provider]
        CURRENT_ADAPTER = await AdapterFactory.create_adapter(new_provider, provider_config)
        APP_CONFIG.active_provider = new_provider
        
        capabilities = CURRENT_ADAPTER.get_capabilities()
        ui.print_success(f"Switched to {new_provider}")
        ui.print(f"Model: {provider_config.default_model}")
        
        # Show capabilities
        features = [cap.value for cap in capabilities.capabilities]
        ui.print(f"Capabilities: {', '.join(features)}")
        
    except Exception as e:
        ui.print_error("Provider switch failed", str(e))
```

### Task 3.4: Update Interactive Chat Loop
**File**: `thinkchain.py` (lines 486+)

**Update command handling**:
```python
def interactive_chat():
    """Enhanced interactive chat with provider management"""
    
    # Initialize everything
    adapter, local_count, mcp_count = await initialize_tools_and_adapter()
    if not adapter:
        ui.print_error("Cannot start chat", "No LLM provider available")
        return
    
    # ... existing setup code ...
    
    # Update command handling in main loop
    elif command == "/providers":
        show_providers_command()
    elif command.startswith("/provider"):
        args = parts[1:] if len(parts) > 1 else []
        await handle_provider_command(args)
    elif command.startswith("/model"):
        # Show available models for current provider
        if CURRENT_ADAPTER:
            provider_name = CURRENT_ADAPTER.get_provider_name()
            ui.print(f"Current provider: {provider_name}")
            # Could add model listing if providers support it
        continue
    
    # Update ask function call
    elif user_input.strip():
        await ask(user_input, chat_history, CURRENT_ADAPTER)
```

---

## Phase 4: Testing and Validation (3-4 hours)

### Task 4.1: Create Test Framework
**File**: `tests/test_adapters.py` (new file)

**Basic test structure**:
```python
import pytest
import asyncio
from unittest.mock import Mock, patch
from llm_adapters.base import Message, Role, CompletionConfig
from llm_adapters.anthropic_adapter import AnthropicAdapter
from llm_adapters.gemini_adapter import GeminiAdapter

@pytest.mark.asyncio
async def test_anthropic_adapter_initialization():
    """Test Anthropic adapter initializes correctly"""
    adapter = AnthropicAdapter()
    
    # Test with valid API key
    await adapter.initialize(api_key="sk-ant-test")
    assert adapter.client is not None
    
    # Test capabilities
    capabilities = adapter.get_capabilities()
    assert ProviderCapability.STREAMING in capabilities.capabilities
    assert ProviderCapability.REASONING_DISPLAY in capabilities.capabilities

@pytest.mark.asyncio  
async def test_gemini_adapter_initialization():
    """Test Gemini adapter initializes correctly"""
    adapter = GeminiAdapter()
    
    with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
        await adapter.initialize()
        assert adapter.client is not None

@pytest.mark.asyncio
async def test_message_conversion():
    """Test message format conversion"""
    messages = [
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="Hi there")
    ]
    
    # Test Anthropic conversion
    anthropic_adapter = AnthropicAdapter()
    anthropic_msgs = anthropic_adapter._convert_messages(messages)
    assert len(anthropic_msgs) == 2
    assert anthropic_msgs[0]["role"] == "user"
    
    # Test Gemini conversion  
    gemini_adapter = GeminiAdapter()
    gemini_contents = gemini_adapter._convert_messages(messages)
    assert len(gemini_contents) == 2
    assert gemini_contents[1].role == "model"  # assistant -> model
```

### Task 4.2: Integration Testing
**File**: `tests/test_integration.py` (new file)

```python
@pytest.mark.integration
async def test_tool_execution_both_providers():
    """Test that tools work with both providers"""
    from tool_discovery import get_claude_tools
    
    tools = get_claude_tools()
    if not tools:
        pytest.skip("No tools available for testing")
    
    test_config = CompletionConfig(
        model="test-model",
        max_tokens=100,
        tools=tools[:1]  # Test with first tool only
    )
    
    # Test with both adapters
    for adapter_class in [AnthropicAdapter, GeminiAdapter]:
        adapter = adapter_class()
        
        # Mock initialization
        adapter.client = Mock()
        
        # Verify tool schema conversion works
        converted_tools = [adapter.convert_tool_schema(tools[0])]
        assert converted_tools[0] is not None
```

### Task 4.3: Manual Testing Checklist

**Create testing checklist**:
1. ✅ Can switch between Anthropic and Gemini using `/provider` command
2. ✅ Tools work identically with both providers
3. ✅ Thinking displays properly for both providers
4. ✅ Configuration loads from environment variables
5. ✅ Error handling works when API keys are missing
6. ✅ Streaming responses work for both providers
7. ✅ Provider capabilities are detected correctly

---

## Phase 5: Documentation and Polish (2-3 hours)

### Task 5.1: Update README.md
**Add provider setup section**:

```markdown
## LLM Provider Setup

ThinkChain supports multiple LLM providers. Set up one or more:

### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
```

### Google Gemini  
```bash
# Option 1: Developer API
export GOOGLE_API_KEY="AIza-your-key"

# Option 2: Vertex AI
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="your-project" 
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### Provider Selection
```bash
# Auto-select first available provider
export THINKCHAIN_PROVIDER=auto

# Or explicitly choose
export THINKCHAIN_PROVIDER=anthropic  # or gemini
```

## Usage

### Provider Management
- `/providers` - List available providers
- `/provider anthropic` - Switch to Anthropic  
- `/provider gemini` - Switch to Gemini
```

### Task 5.2: Create Migration Guide
**File**: `docs/migration_guide.md` (new file)

```markdown
# Migration Guide: Single to Multi-Provider

## For Existing Users

If you're currently using ThinkChain with Claude, no changes needed! 
Your existing `ANTHROPIC_API_KEY` will continue working.

## New Commands Available

- `/providers` - See all configured providers
- `/provider <name>` - Switch between providers
- `/config` - View current configuration

## Configuration Changes

Old single model config:
```python
CONFIG = {"model": "claude-sonnet-4-20250514"}
```

New multi-provider config:
```python
APP_CONFIG = ThinkChainConfig.from_env()
```

The old config is still supported for backward compatibility.
```

---

## Common Pitfalls and Solutions

### Problem: Import Errors
**Solution**: Ensure all new files are properly imported. Add `__init__.py` files if needed.

### Problem: Async/Sync Mixing
**Solution**: Current codebase mixes sync and async. Use `asyncio.run()` or async wrappers where needed.

### Problem: Tool Schema Incompatibility
**Solution**: The `convert_tool_schema` method handles differences. Test with simple tools first.

### Problem: Environment Variable Conflicts
**Solution**: Use consistent naming. Prefix with `THINKCHAIN_` for app-specific settings.

---

## Success Criteria

✅ **Functional Requirements**:
- Both providers stream responses correctly
- Tools execute identically across providers  
- Provider switching works mid-conversation
- Thinking displays properly (where supported)

✅ **Performance Requirements**:
- First token latency <2s for both providers
- Provider switching <1s
- No memory leaks during switches

✅ **Code Quality**:
- All new code follows existing patterns
- Type hints added where missing
- Error handling for all edge cases
- Backward compatibility maintained

---

## Next Steps After Implementation

1. **Add OpenAI Support**: Follow same adapter pattern
2. **Implement Provider Routing**: Smart provider selection based on task type
3. **Add Model Management**: Support multiple models per provider
4. **Enhanced Capabilities**: Vision support, structured outputs
5. **Performance Optimization**: Caching, connection pooling

This implementation maintains the existing ThinkChain experience while adding powerful multi-provider capabilities that users can adopt gradually. 