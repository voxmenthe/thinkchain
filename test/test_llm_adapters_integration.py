"""
Integration tests for LLM adapters.

These tests require real API keys and make actual API calls.
Set ANTHROPIC_API_KEY and/or GOOGLE_API_KEY environment variables.
"""
import pytest
from llm_adapters.base import (
    Message, Role, CompletionConfig,
    ProviderCapability, ProviderCapabilities
)


class TestAdapterBasics:
    """Test basic adapter functionality."""
    
    @pytest.mark.asyncio
    async def test_anthropic_adapter_initialization(self, anthropic_adapter):
        """Test that Anthropic adapter initializes correctly."""
        assert anthropic_adapter is not None
        assert anthropic_adapter.client is not None
        assert anthropic_adapter.get_provider_name() == "anthropic"
        
        # Test capabilities
        capabilities = anthropic_adapter.get_capabilities()
        assert isinstance(capabilities, ProviderCapabilities)
        assert ProviderCapability.STREAMING in capabilities.capabilities
        assert ProviderCapability.THINKING in capabilities.capabilities
        assert capabilities.thinking in [True]
    
    @pytest.mark.asyncio
    async def test_gemini_adapter_initialization(self, gemini_adapter):
        """Test that Gemini adapter initializes correctly."""
        assert gemini_adapter is not None
        assert gemini_adapter.client is not None
        assert gemini_adapter.get_provider_name() == "gemini"
        
        # Test capabilities
        capabilities = gemini_adapter.get_capabilities()
        assert isinstance(capabilities, ProviderCapabilities)
        assert ProviderCapability.STREAMING in capabilities.capabilities
        assert ProviderCapability.THINKING in capabilities.capabilities


class TestMessageConversion:
    """Test message format conversion."""
    
    @pytest.mark.asyncio
    async def test_anthropic_message_conversion(self, anthropic_adapter,
                                                sample_messages):
        """Test Anthropic message conversion."""
        converted = anthropic_adapter._convert_messages(sample_messages)
        
        assert len(converted) == len(sample_messages)
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[2]["role"] == "user"
        
        # Check content is preserved
        assert "2 + 2" in converted[0]["content"]
        assert "equals 4" in converted[1]["content"]
        assert "explain why" in converted[2]["content"]
    
    @pytest.mark.asyncio
    async def test_gemini_message_conversion(self, gemini_adapter,
                                             sample_messages):
        """Test Gemini message conversion."""
        converted = gemini_adapter._convert_messages(sample_messages)
        
        assert len(converted) == len(sample_messages)
        # Gemini uses "model" for assistant role
        assert converted[0].role == "user"
        assert converted[1].role == "model"
        assert converted[2].role == "user"
        
        # Check content structure
        assert len(converted[0].parts) > 0
        assert "2 + 2" in str(converted[0].parts[0])


class TestBasicCompletion:
    """Test basic completion functionality without streaming."""
    
    @pytest.mark.asyncio
    async def test_anthropic_basic_completion(self, anthropic_adapter,
                                              anthropic_basic_config):
        """Test basic completion with Anthropic."""
        messages = [
            Message(role=Role.USER, content="What is 2 + 2? Answer briefly.")
        ]
        
        response = await anthropic_adapter.complete(messages, 
                                                   anthropic_basic_config)
        
        assert response is not None
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0
        # Should contain "4" somewhere in the response
        assert "4" in response.content
    
    @pytest.mark.asyncio
    async def test_gemini_basic_completion(self, gemini_adapter,
                                           gemini_basic_config):
        """Test basic completion with Gemini."""
        messages = [
            Message(role=Role.USER, content="What is 2 + 2? Answer briefly.")
        ]
        
        response = await gemini_adapter.complete(messages, gemini_basic_config)
        
        assert response is not None
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0
        # Should contain "4" somewhere in the response
        assert "4" in response.content


class TestStreamingCompletion:
    """Test streaming completion functionality."""
    
    @pytest.mark.asyncio
    async def test_anthropic_streaming(self, anthropic_adapter,
                                       anthropic_basic_config):
        """Test streaming with Anthropic."""
        messages = [
            Message(role=Role.USER, 
                   content="Count from 1 to 5, one number per line.")
        ]
        
        chunks = []
        content_parts = []
        
        async for chunk in anthropic_adapter.stream_complete(
                messages, anthropic_basic_config):
            chunks.append(chunk)
            if chunk.delta_text:
                content_parts.append(chunk.delta_text)
        
        assert len(chunks) > 0
        assert any(chunk.delta_text for chunk in chunks)
        
        # Reconstruct full content
        full_content = "".join(content_parts)
        assert len(full_content.strip()) > 0
        
        # Should contain numbers 1-5
        for i in range(1, 6):
            assert str(i) in full_content
    
    @pytest.mark.asyncio
    async def test_gemini_streaming(self, gemini_adapter, gemini_basic_config):
        """Test streaming with Gemini."""
        messages = [
            Message(role=Role.USER, 
                   content="Count from 1 to 5, one number per line.")
        ]
        
        chunks = []
        content_parts = []
        
        async for chunk in gemini_adapter.stream_complete(
                messages, gemini_basic_config):
            chunks.append(chunk)
            if chunk.delta_text:
                content_parts.append(chunk.delta_text)
        
        assert len(chunks) > 0
        assert any(chunk.delta_text for chunk in chunks)
        
        # Reconstruct full content
        full_content = "".join(content_parts)
        assert len(full_content.strip()) > 0
        
        # Should contain numbers 1-5
        for i in range(1, 6):
            assert str(i) in full_content


class TestThinkingMode:
    """Test thinking mode functionality."""
    
    @pytest.mark.asyncio
    async def test_anthropic_thinking_mode(self, anthropic_adapter,
                                           anthropic_thinking_config):
        """Test thinking mode with Anthropic."""
        messages = [
            Message(
                role=Role.USER, 
                content="Calculate 17 * 23 step by step. Show your work and provide the final answer."
            )
        ]
        
        chunks = []
        thinking_parts = []
        content_parts = []
        
        async for chunk in anthropic_adapter.stream_complete(
                messages, anthropic_thinking_config):
            chunks.append(chunk)
            if chunk.thinking_text:
                thinking_parts.append(chunk.thinking_text)
            if chunk.delta_text:
                content_parts.append(chunk.delta_text)
        
        assert len(chunks) > 0
        
        # Check if thinking was used (may not always trigger)
        full_thinking = "".join(thinking_parts)
        full_content = "".join(content_parts)
        
        assert len(full_content.strip()) > 0
        
        # Should contain the correct answer (17 * 23 = 391)
        # Check for various possible formats of the answer
        combined_text = full_content + " " + full_thinking
        assert ("391" in combined_text or 
                "= 391" in combined_text or
                "391." in combined_text or
                "answer is 391" in combined_text.lower() or
                "result is 391" in combined_text.lower())
        
        # Should contain calculation elements
        assert ("17" in combined_text and "23" in combined_text)
        
        # If thinking was used, it should contain calculation steps
        if full_thinking:
            assert len(full_thinking.strip()) > 0
            assert any(char.isdigit() for char in full_thinking)
    
    @pytest.mark.asyncio
    async def test_gemini_thinking_mode(self, gemini_adapter,
                                        gemini_thinking_config):
        """Test thinking mode with Gemini."""
        messages = [
            Message(
                role=Role.USER, 
                content="Calculate 17 * 23 step by step. Show your work and provide the final answer."
            )
        ]
        
        chunks = []
        thinking_parts = []
        content_parts = []
        
        async for chunk in gemini_adapter.stream_complete(
                messages, gemini_thinking_config):
            chunks.append(chunk)
            if chunk.thinking_text:
                thinking_parts.append(chunk.thinking_text)
            if chunk.delta_text:
                content_parts.append(chunk.delta_text)
        
        assert len(chunks) > 0
        
        # Check if thinking was used (may not always trigger)
        full_thinking = "".join(thinking_parts)
        full_content = "".join(content_parts)
        
        assert len(full_content.strip()) > 0
        
        # Should contain the correct answer (17 * 23 = 391)
        # Check for various possible formats of the answer
        combined_text = full_content + " " + full_thinking
        assert ("391" in combined_text or 
                "= 391" in combined_text or
                "391." in combined_text or
                "answer is 391" in combined_text.lower() or
                "result is 391" in combined_text.lower())
        
        # Should contain calculation elements
        assert ("17" in combined_text and "23" in combined_text)
        
        # If thinking was used, it should contain reasoning
        if full_thinking:
            assert len(full_thinking.strip()) > 0


class TestTokenCounting:
    """Test token counting functionality."""
    
    @pytest.mark.asyncio
    async def test_anthropic_token_counting(self, anthropic_adapter,
                                            sample_messages):
        """Test token counting with Anthropic."""
        token_count = await anthropic_adapter.count_tokens(sample_messages)
        
        assert isinstance(token_count, dict)
        assert "prompt_tokens" in token_count
        assert "total_tokens" in token_count
        assert token_count["prompt_tokens"] > 0
        assert token_count["total_tokens"] >= token_count["prompt_tokens"]
    
    @pytest.mark.asyncio
    async def test_gemini_token_counting(self, gemini_adapter, 
                                         sample_messages):
        """Test token counting with Gemini."""
        token_count = await gemini_adapter.count_tokens(sample_messages)
        
        assert isinstance(token_count, dict)
        assert "prompt_tokens" in token_count
        assert "total_tokens" in token_count
        assert token_count["prompt_tokens"] > 0
        assert token_count["total_tokens"] >= token_count["prompt_tokens"]


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_anthropic_invalid_model(self, anthropic_adapter):
        """Test handling of invalid model name."""
        messages = [Message(role=Role.USER, content="Hello")]
        
        config = CompletionConfig(
            model="invalid-model-name",
            max_tokens=50
        )
        
        with pytest.raises(Exception):
            await anthropic_adapter.complete(messages, config)
    
    @pytest.mark.asyncio
    async def test_gemini_invalid_model(self, gemini_adapter):
        """Test handling of invalid model name."""
        messages = [Message(role=Role.USER, content="Hello")]
        
        config = CompletionConfig(
            model="invalid-model-name",
            max_tokens=50
        )
        
        with pytest.raises(Exception):
            await gemini_adapter.complete(messages, config)
    
    @pytest.mark.asyncio
    async def test_empty_messages(self, anthropic_adapter,
                                  anthropic_basic_config):
        """Test handling of empty message list."""
        with pytest.raises((ValueError, Exception)):
            await anthropic_adapter.complete([], anthropic_basic_config)


class TestCrossProviderConsistency:
    """Test that both providers behave consistently."""
    
    @pytest.mark.asyncio
    async def test_simple_math_consistency(self, anthropic_adapter,
                                           gemini_adapter,
                                           anthropic_basic_config,
                                           gemini_basic_config):
        """Test that both providers give consistent answers to simple math."""
        messages = [
            Message(role=Role.USER, 
                   content="What is 8 + 7? Give just the number.")
        ]
        
        # Get responses from both providers
        anthropic_response = await anthropic_adapter.complete(
            messages, anthropic_basic_config)
        gemini_response = await gemini_adapter.complete(
            messages, gemini_basic_config)
        
        # Both should contain "15"
        assert "15" in anthropic_response.content
        assert "15" in gemini_response.content
    
    @pytest.mark.asyncio
    async def test_capability_coverage(self, anthropic_adapter, 
                                       gemini_adapter):
        """Test that both providers expose their capabilities correctly."""
        anthropic_caps = anthropic_adapter.get_capabilities()
        gemini_caps = gemini_adapter.get_capabilities()
        
        # Both should support basic capabilities
        required_caps = [
            ProviderCapability.STREAMING,
            ProviderCapability.THINKING
        ]
        
        for cap in required_caps:
            assert cap in anthropic_caps.capabilities
            assert cap in gemini_caps.capabilities
        
        # Check thinking configuration
        assert anthropic_caps.thinking is True
        assert gemini_caps.thinking is True 