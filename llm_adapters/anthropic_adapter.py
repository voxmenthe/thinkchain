from typing import List, Dict, Any, Optional, AsyncGenerator, AsyncIterator
import anthropic
from .base import (
    BaseAdapter, Message, StreamChunk, CompletionConfig, 
    ProviderCapabilities, Response
)


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic's Claude API with thinking mode support."""
    
    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        return "anthropic"
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the capabilities supported by Anthropic."""
        return ProviderCapabilities(
            streaming=True,
            tool_use=True,
            token_counting=True,
            vision=True,
            function_calling=True,
            system_messages=True,
            thinking=True
        )
    
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        from config import get_config
        config = get_config()
        provider_config = config.providers.get("anthropic")
        if provider_config:
            return provider_config.default_model
        return "claude-3-5-sonnet-20241022"  # Fallback
    
    def get_token_counting_model(self) -> str:
        """Get the fastest/cheapest model for token counting."""
        from config import get_config
        config = get_config()
        provider_config = config.providers.get("anthropic")
        if provider_config and "claude-3-haiku-20240307" in provider_config.models:
            return "claude-3-haiku-20240307"
        return self.get_default_model()
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert unified messages to Anthropic format."""
        anthropic_messages = []
        
        for msg in messages:
            # Skip system messages - they're handled separately
            if msg.role.value == "system":
                continue
                
            anthropic_msg = {
                "role": msg.role.value,
                "content": str(msg.content)
            }
            anthropic_messages.append(anthropic_msg)
        
        return anthropic_messages
    
    def _get_system_message(self, messages: List[Message]) -> Optional[str]:
        """Extract system message from message list."""
        for msg in messages:
            if msg.role.value == "system":
                return str(msg.content)
        return None
    
    async def complete(self, messages: List[Message], 
                      config: CompletionConfig) -> Response:
        """Generate a single completion from Anthropic."""
        anthropic_messages = self._convert_messages(messages)
        system_message = self._get_system_message(messages)
        
        # Prepare request parameters
        request_params = {
            "model": config.model,
            "max_tokens": config.max_tokens or 4096,
            "messages": anthropic_messages
        }
        
        # Add system message if present
        if system_message:
            request_params["system"] = system_message
            
        # Add temperature if specified
        if config.temperature is not None:
            request_params["temperature"] = config.temperature
            
        # Add thinking configuration if specified
        if hasattr(config, 'thinking') and config.thinking:
            thinking_config = config.thinking
            if isinstance(thinking_config, dict):
                request_params["thinking"] = thinking_config
            else:
                # Default thinking configuration
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": getattr(config, 'thinking_budget_tokens', 
                                           config.thinking_budget or 2048)
                }
        
        # Add tools if present
        if config.tools:
            request_params["tools"] = config.tools
            
        response = await self.client.messages.create(**request_params)
        return self._convert_response(response)

    async def stream_complete(self, messages: List[Message], 
                             config: CompletionConfig) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming completion from Anthropic."""
        anthropic_messages = self._convert_messages(messages)
        system_message = self._get_system_message(messages)
        
        # Prepare request parameters
        request_params = {
            "model": config.model,
            "max_tokens": config.max_tokens or 4096,
            "messages": anthropic_messages,
            "stream": True,
        }
        
        # Add system message if present
        if system_message:
            request_params["system"] = system_message
            
        # Add temperature if specified
        if config.temperature is not None:
            request_params["temperature"] = config.temperature
            
        # Add thinking configuration if specified
        if hasattr(config, 'thinking') and config.thinking:
            thinking_config = config.thinking
            if isinstance(thinking_config, dict):
                request_params["thinking"] = thinking_config
            else:
                # Default thinking configuration
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": getattr(config, 'thinking_budget_tokens',
                                           config.thinking_budget or 2048)
                }
        
        # Add tools if present
        if config.tools:
            request_params["tools"] = config.tools

        stream = await self.client.messages.create(**request_params)
        
        async for chunk in stream:
            yield self._process_chunk(chunk)

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
                    from .base import ToolUse
                    result.tool_use = ToolUse(
                        id=chunk.content_block.id,
                        name=chunk.content_block.name,
                        arguments={}
                    )
        elif chunk.type == 'content_block_stop':
            # Handle completion of thinking or tool blocks
            pass
        
        return result
    
    def _convert_response(self, response) -> Response:
        """Convert Anthropic response to unified format."""
        content = ""
        if hasattr(response, 'content') and response.content:
            content = response.content[0].text if response.content else ""
        
        return Response(
            content=content,
            role="assistant"
        )

    async def count_tokens(self, messages: List[Message]) -> Dict[str, int]:
        """Count tokens for Anthropic messages."""
        anthropic_messages = self._convert_messages(messages)
        system_message = self._get_system_message(messages)
        
        # Use the cheapest/fastest model for token counting
        token_model = self.get_token_counting_model()
        
        count_params = {
            "model": token_model,
            "messages": anthropic_messages
        }
        
        if system_message:
            count_params["system"] = system_message
        
        response = await self.client.messages.count_tokens(**count_params)
        
        return {
            "prompt_tokens": response.input_tokens,
            "completion_tokens": 0,
            "total_tokens": response.input_tokens
        }

    async def initialize(self, api_key: Optional[str] = None) -> None:
        """Initialize the adapter with credentials."""
        if api_key:
            self.api_key = api_key
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def stream_completion(
        self, 
        messages: List[Message],
        config: CompletionConfig
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion with unified interface."""
        async for chunk in self.stream_complete(messages, config):
            yield chunk

    def convert_tool_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tool schema to Anthropic format."""
        # Anthropic uses a specific tool schema format
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {})
        }

    @property
    def supports_thinking(self) -> bool:
        """Whether this provider natively supports thinking."""
        return True

    @property
    def supports_vision(self) -> bool:
        """Whether this provider supports image inputs."""
        return True
        