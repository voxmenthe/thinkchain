from abc import ABC, abstractmethod
from typing import (
    AsyncIterator, List, Dict, Any, Optional, AsyncGenerator, Union
)
from dataclasses import dataclass
import enum

class Role(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class ProviderCapability(enum.Enum):
    """Enum representing different capabilities a provider might support."""
    STREAMING = "streaming"
    TOOL_USE = "tool_use"
    TOKEN_COUNTING = "token_counting"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    SYSTEM_MESSAGES = "system_messages"
    THINKING = "thinking"  # Extended thinking capability

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class StreamChunk:
    """Unified stream chunk for all llm adapters."""
    delta_text: Optional[str] = None
    thinking_text: Optional[str] = None
    tool_use: Optional['ToolUse'] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

# TODO: should ToolUse include tool_output or something to indicate expected output?
@dataclass
class ToolUse:
    # tool_name: str
    # tool_input: Dict[str, Any]
    # tool_output: Optional[str] = None
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class CompletionConfig:
    """Unifiied config for all llm adapters."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    thinking_budget: Optional[int] = None # Anthropic-specific???
    tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    # Provider-specific overrides
    provider_config: Optional[Dict[str, Any]] = None
    thinking: Optional[Union[bool, Dict[str, Any]]] = None
    thinking_budget_tokens: Optional[int] = None
    # Additional parameters for different providers
    top_p: Optional[float] = None
    top_k: Optional[int] = None

@dataclass
class ProviderCapabilities:
    """Class to represent the capabilities of a provider."""
    streaming: bool = False
    tool_use: bool = False
    token_counting: bool = False
    vision: bool = False
    function_calling: bool = False
    system_messages: bool = False
    thinking: bool = False  # Support for thinking/reasoning modes
    
    def has_capability(self, capability: ProviderCapability) -> bool:
        """Check if the provider has a specific capability."""
        return getattr(self, capability.value, False)
    
    def get_supported_capabilities(self) -> List[ProviderCapability]:
        """Get a list of all supported capabilities."""
        supported = []
        for capability in ProviderCapability:
            if self.has_capability(capability):
                supported.append(capability)
        return supported
    
    @property
    def capabilities(self) -> set[ProviderCapability]:
        """Get set of supported capabilities for compatibility."""
        return set(self.get_supported_capabilities())

@dataclass 
class Response:
    """Unified response object for all llm adapters."""
    content: str
    role: str = "assistant"
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None

class BaseAdapter(ABC):
    """Abstract base class for LLM provider adapters"""
    
    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.config = config or {}
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider (e.g., 'anthropic', 'gemini')."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the capabilities supported by this provider."""
        pass
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Response:
        """Complete a conversation with the given messages."""
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion tokens as they arrive."""
        pass
    
    @abstractmethod
    async def count_tokens(self, messages: List[Message]) -> Dict[str, int]:
        """Count the number of tokens in the messages."""
        pass
    
    @abstractmethod
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal Message format to provider-specific format."""
        pass
    
    @abstractmethod
    def _process_chunk(self, chunk: Any) -> StreamChunk:
        """Process a streaming chunk from the provider."""
        pass

    @abstractmethod
    async def initialize(self, api_key: Optional[str] = None) -> None:
        """Initialize the adapter with credentials"""
        pass

    @abstractmethod
    async def stream_completion(
        self, 
        messages: List[Message],
        config: CompletionConfig
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion with unified interface"""
        pass

    @abstractmethod
    def convert_tool_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tool schema to provider-specific format"""
        pass

    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this provider natively supports thinking"""
        pass

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports image inputs"""
        pass