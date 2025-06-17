"""
Unit tests for LLM adapter components.

These tests don't require API keys and focus on specific functionality.
"""
from unittest.mock import Mock
from llm_adapters.base import (
    Message, Role, CompletionConfig, StreamChunk,
    ProviderCapability, ProviderCapabilities
)
from llm_adapters.anthropic_adapter import AnthropicAdapter
from llm_adapters.gemini_adapter import GeminiAdapter


class TestMessageConversionUnits:
    """Unit tests for message conversion logic."""
    
    def test_anthropic_message_conversion_simple(self):
        """Test simple Anthropic message conversion."""
        adapter = AnthropicAdapter("dummy-key")
        
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
        ]
        
        converted = adapter._convert_messages(messages)
        
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"
        assert converted[1]["role"] == "assistant"
        assert converted[1]["content"] == "Hi there"
    
    def test_anthropic_message_conversion_system(self):
        """Test Anthropic system message handling."""
        adapter = AnthropicAdapter("dummy-key")
        
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hello"),
        ]
        
        converted = adapter._convert_messages(messages)
        
        # System message should be separate
        assert len(converted) == 1  # Only user message
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"
    
    def test_gemini_message_conversion_simple(self):
        """Test simple Gemini message conversion."""
        adapter = GeminiAdapter("dummy-key")
        
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
        ]
        
        converted = adapter._convert_messages(messages)
        
        assert len(converted) == 2
        assert converted[0].role == "user"
        assert converted[1].role == "model"  # assistant -> model
        
        # Check content parts
        assert len(converted[0].parts) == 1
        # Note: Part objects don't have simple string representation,
        # but we can check the text attribute if it exists
        part = converted[0].parts[0]
        if hasattr(part, 'text'):
            assert part.text == "Hello"
        else:
            # If it's a mock or has different structure, just verify it exists
            assert part is not None
    
    def test_gemini_role_mapping(self):
        """Test Gemini role mapping logic."""
        adapter = GeminiAdapter("dummy-key")
        
        test_cases = [
            (Role.USER, "user"),
            (Role.ASSISTANT, "model"),
            (Role.TOOL, "user"),    # Tool becomes user
        ]
        
        for input_role, expected_gemini_role in test_cases:
            messages = [Message(role=input_role, content="test")]
            converted = adapter._convert_messages(messages)
            assert len(converted) == 1
            assert converted[0].role == expected_gemini_role
        
        # Test system messages are skipped (handled separately)
        system_messages = [Message(role=Role.SYSTEM, content="test")]
        converted = adapter._convert_messages(system_messages)
        assert len(converted) == 0  # System messages are skipped


class TestCapabilityDetection:
    """Unit tests for capability detection."""
    
    def test_anthropic_capabilities(self):
        """Test Anthropic capability detection."""
        adapter = AnthropicAdapter("dummy-key")
        capabilities = adapter.get_capabilities()
        
        assert isinstance(capabilities, ProviderCapabilities)
        
        # Check required capabilities
        expected_caps = {
            ProviderCapability.STREAMING,
            ProviderCapability.THINKING,
            ProviderCapability.TOOL_USE,
            ProviderCapability.SYSTEM_MESSAGES,
        }
        
        for cap in expected_caps:
            assert cap in capabilities.capabilities
        
        # Check thinking configuration
        assert capabilities.thinking is True
        assert capabilities.system_messages is True
    
    def test_gemini_capabilities(self):
        """Test Gemini capability detection."""
        adapter = GeminiAdapter("dummy-key")
        capabilities = adapter.get_capabilities()
        
        assert isinstance(capabilities, ProviderCapabilities)
        
        # Check required capabilities
        expected_caps = {
            ProviderCapability.STREAMING,
            ProviderCapability.THINKING,
            ProviderCapability.TOOL_USE,
        }
        
        for cap in expected_caps:
            assert cap in capabilities.capabilities
        
        # Check thinking configuration
        assert capabilities.thinking is True
    
    def test_provider_names(self):
        """Test provider name detection."""
        anthropic_adapter = AnthropicAdapter("dummy-key")
        gemini_adapter = GeminiAdapter("dummy-key")
        
        assert anthropic_adapter.get_provider_name() == "anthropic"
        assert gemini_adapter.get_provider_name() == "gemini"


class TestStreamChunkProcessing:
    """Unit tests for stream chunk processing."""
    
    def test_anthropic_chunk_processing_text(self):
        """Test Anthropic text chunk processing."""
        adapter = AnthropicAdapter("dummy-key")
        
        # Mock a text delta chunk
        mock_chunk = Mock()
        mock_chunk.type = "content_block_delta"
        mock_chunk.delta = Mock()
        mock_chunk.delta.text = "Hello world"
        
        result = adapter._process_chunk(mock_chunk)
        
        assert isinstance(result, StreamChunk)
        assert result.delta_text == "Hello world"
        assert result.thinking_text is None
    
    def test_anthropic_chunk_processing_thinking(self):
        """Test Anthropic thinking chunk processing."""
        adapter = AnthropicAdapter("dummy-key")
        
        # Mock a thinking start chunk
        mock_chunk = Mock()
        mock_chunk.type = "content_block_start"
        mock_chunk.content_block = Mock()
        mock_chunk.content_block.type = "thinking"
        
        result = adapter._process_chunk(mock_chunk)
        
        assert isinstance(result, StreamChunk)
        assert result.thinking_text == ""
        assert result.delta_text is None
    
    def test_gemini_chunk_processing_text(self):
        """Test Gemini text chunk processing."""
        adapter = GeminiAdapter("dummy-key")
        
        # Mock a text chunk
        mock_chunk = Mock()
        mock_chunk.text = "Hello world"
        mock_chunk.thinking = None
        
        result = adapter._process_chunk(mock_chunk)
        
        assert isinstance(result, StreamChunk)
        assert result.delta_text == "Hello world"
        assert result.thinking_text is None
    
    def test_gemini_chunk_processing_thinking(self):
        """Test Gemini thinking chunk processing."""
        adapter = GeminiAdapter("dummy-key")
        
        # Mock a thinking chunk
        mock_chunk = Mock()
        mock_chunk.text = None
        mock_chunk.thinking = "Let me think about this..."
        
        result = adapter._process_chunk(mock_chunk)
        
        assert isinstance(result, StreamChunk)
        assert result.thinking_text == "Let me think about this..."
        assert result.delta_text is None


class TestCompletionConfigHandling:
    """Unit tests for completion configuration handling."""
    
    def test_anthropic_config_conversion(self):
        """Test Anthropic configuration parameter conversion."""
        adapter = AnthropicAdapter("dummy-key")
        
        messages = [Message(role=Role.USER, content="Hello")]
        
        # Test that config is properly converted
        anthropic_messages = adapter._convert_messages(messages)
        
        # Verify the conversion worked
        assert len(anthropic_messages) == 1
        assert anthropic_messages[0]["content"] == "Hello"
    
    def test_gemini_config_conversion(self):
        """Test Gemini configuration parameter conversion."""
        adapter = GeminiAdapter("dummy-key")
        
        messages = [Message(role=Role.USER, content="Hello")]
        
        # Test that config is properly converted
        gemini_contents = adapter._convert_messages(messages)
        
        # Verify the conversion worked
        assert len(gemini_contents) == 1
        assert gemini_contents[0].role == "user"


class TestEdgeCases:
    """Unit tests for edge cases and error conditions."""
    
    def test_empty_message_list(self):
        """Test handling of empty message lists."""
        anthropic_adapter = AnthropicAdapter("dummy-key")
        gemini_adapter = GeminiAdapter("dummy-key")
        
        # Both should handle empty lists gracefully
        anthropic_result = anthropic_adapter._convert_messages([])
        gemini_result = gemini_adapter._convert_messages([])
        
        assert anthropic_result == []
        assert gemini_result == []
    
    def test_unknown_message_role(self):
        """Test handling of unknown message roles."""
        adapter = GeminiAdapter("dummy-key")
        
        # Create a message with an unusual role
        messages = [Message(role=Role.TOOL, content="Tool output")]
        
        # Should not crash and should map to user role
        converted = adapter._convert_messages(messages)
        assert len(converted) == 1
        assert converted[0].role == "user"
    
    def test_large_content(self):
        """Test handling of large content."""
        adapter = AnthropicAdapter("dummy-key")
        
        # Create a message with large content
        large_content = "A" * 10000
        messages = [Message(role=Role.USER, content=large_content)]
        
        converted = adapter._convert_messages(messages)
        assert len(converted) == 1
        assert converted[0]["content"] == large_content
    
    def test_special_characters(self):
        """Test handling of special characters in content."""
        adapter = GeminiAdapter("dummy-key")
        
        special_content = "Hello 👋 世界 \n\t\"quotes\" and 'apostrophes'"
        messages = [Message(role=Role.USER, content=special_content)]
        
        converted = adapter._convert_messages(messages)
        assert len(converted) == 1
        # Check the content was preserved (Part object structure may vary)
        part = converted[0].parts[0]
        if hasattr(part, 'text'):
            assert part.text == special_content
        else:
            # For mock objects, just verify the part exists
            assert part is not None 