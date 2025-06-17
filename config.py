"""
ThinkChain Configuration System

This module provides centralized configuration management for all LLM providers,
models, hyperparameters, and application settings.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    max_context: int
    supports_thinking: bool = False
    supports_tools: bool = True
    supports_vision: bool = False
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    name: str
    default_model: str
    credentials: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # Provider-specific defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    default_thinking_budget: Optional[int] = None
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000


@dataclass
class ThinkChainConfig:
    """Main configuration for ThinkChain application."""
    # Active provider selection
    active_provider: str = "auto"
    
    # Provider configurations
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    
    # Universal hyperparameters (can be overridden per provider)
    temperature: float = 0.7
    max_tokens: int = 4096
    thinking_budget: Optional[int] = None
    
    # UI and behavior settings
    show_thinking: bool = True
    stream_delay_ms: int = 0
    enable_rich_ui: bool = True
    verbose_errors: bool = False
    
    # Testing and development
    enable_debug_logging: bool = False
    test_mode: bool = False
    
    @classmethod
    def get_default_models(cls) -> Dict[str, Dict[str, ModelConfig]]:
        """Get default model configurations for all providers."""
        return {
            "anthropic": {
                "claude-sonnet-4-20250514": ModelConfig(
                    name="claude-sonnet-4-20250514",
                    max_context=200000,
                    supports_thinking=True,
                    supports_tools=True,
                    supports_vision=True,
                    cost_per_1k_input=3.0,
                    cost_per_1k_output=15.0
                ),
                "claude-3-5-haiku-20241022": ModelConfig(
                    name="claude-3-5-haiku-20241022",
                    max_context=200000,
                    supports_thinking=False,
                    supports_tools=True,
                    supports_vision=False,
                    cost_per_1k_input=0.25,
                    cost_per_1k_output=1.25
                ),
                "claude-sonnet-4-20250514": ModelConfig(
                    name="claude-sonnet-4-20250514",
                    max_context=200000,
                    supports_thinking=True,
                    supports_tools=True,
                    supports_vision=True,
                    cost_per_1k_input=3.0,
                    cost_per_1k_output=15.0
                )
            },
            "gemini": {
                "gemini-2.5-flash-preview-05-20": ModelConfig(
                    name="gemini-2.5-flash-preview-05-20",
                    max_context=1000000,
                    supports_thinking=False,
                    supports_tools=True,
                    supports_vision=True,
                    cost_per_1k_input=0.075,
                    cost_per_1k_output=0.3
                ),
                "gemini-2.5-pro-preview-06-05": ModelConfig(
                    name="gemini-2.5-pro-preview-06-05",
                    max_context=1000000,
                    supports_thinking=True,
                    supports_tools=True,
                    supports_vision=True,
                    cost_per_1k_input=0.075,
                    cost_per_1k_output=0.3
                )
            }
        }
    
    @classmethod
    def from_env(cls) -> 'ThinkChainConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Load universal settings from environment
        config.temperature = float(os.getenv("THINKCHAIN_TEMPERATURE", "0.7"))
        config.max_tokens = int(os.getenv("THINKCHAIN_MAX_TOKENS", "4096"))
        config.thinking_budget = (
            int(os.getenv("THINKCHAIN_THINKING_BUDGET"))
            if os.getenv("THINKCHAIN_THINKING_BUDGET")
            else None
        )
        config.show_thinking = os.getenv("THINKCHAIN_SHOW_THINKING", "true").lower() == "true"
        config.enable_debug_logging = os.getenv("THINKCHAIN_DEBUG", "false").lower() == "true"
        config.test_mode = os.getenv("THINKCHAIN_TEST_MODE", "false").lower() == "true"
        
        # Get default model configurations
        default_models = cls.get_default_models()
        
        # Auto-detect and configure providers
        providers = {}
        
        # Configure Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            providers["anthropic"] = ProviderConfig(
                name="anthropic",
                default_model=anthropic_model,
                credentials={"api_key": anthropic_key},
                models=default_models["anthropic"],
                default_temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                default_max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
                default_thinking_budget=int(os.getenv("ANTHROPIC_THINKING_BUDGET", "2048"))
            )
        
        # Configure Gemini
        google_key = os.getenv("GOOGLE_API_KEY")
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
        if google_key or use_vertex:
            gemini_creds = {}
            if google_key:
                gemini_creds["api_key"] = google_key
            if use_vertex:
                gemini_creds["use_vertex"] = True
                gemini_creds["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT")
                gemini_creds["location"] = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            providers["gemini"] = ProviderConfig(
                name="gemini",
                default_model=gemini_model,
                credentials=gemini_creds,
                models=default_models["gemini"],
                default_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                default_max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
                default_thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "1024"))
            )
        
        config.providers = providers
        
        # Set active provider
        explicit_provider = os.getenv("THINKCHAIN_PROVIDER")
        if explicit_provider and explicit_provider in providers:
            config.active_provider = explicit_provider
        elif providers:
            # Auto-select based on preference: Anthropic first, then Gemini
            if "anthropic" in providers:
                config.active_provider = "anthropic"
            else:
                config.active_provider = list(providers.keys())[0]
        
        return config
    
    def get_active_provider_config(self) -> Optional[ProviderConfig]:
        """Get the configuration for the currently active provider."""
        if self.active_provider == "auto" and self.providers:
            return list(self.providers.values())[0]
        return self.providers.get(self.active_provider)
    
    def get_model_config(self, provider: str, model: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        provider_config = self.providers.get(provider)
        if not provider_config:
            return None
        return provider_config.models.get(model)
    
    def get_effective_temperature(self, provider: Optional[str] = None) -> float:
        """Get effective temperature (provider default or global)."""
        if provider and provider in self.providers:
            return self.providers[provider].default_temperature
        return self.temperature
    
    def get_effective_max_tokens(self, provider: Optional[str] = None) -> int:
        """Get effective max tokens (provider default or global)."""
        if provider and provider in self.providers:
            return self.providers[provider].default_max_tokens
        return self.max_tokens
    
    def get_effective_thinking_budget(self, provider: Optional[str] = None) -> Optional[int]:
        """Get effective thinking budget (provider default or global)."""
        if provider and provider in self.providers:
            return self.providers[provider].default_thinking_budget
        return self.thinking_budget
    
    def validate_model_for_provider(self, provider: str, model: str) -> bool:
        """Validate that a model is available for a provider."""
        provider_config = self.providers.get(provider)
        if not provider_config:
            return False
        return model in provider_config.models
    
    def get_available_models(self, provider: str) -> List[str]:
        """Get list of available models for a provider."""
        provider_config = self.providers.get(provider)
        if not provider_config:
            return []
        return list(provider_config.models.keys())
    
    def supports_thinking(self, provider: str, model: str) -> bool:
        """Check if a model supports thinking mode."""
        model_config = self.get_model_config(provider, model)
        return model_config.supports_thinking if model_config else False


# Global configuration instance
_config: Optional[ThinkChainConfig] = None


def get_config() -> ThinkChainConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ThinkChainConfig.from_env()
    return _config


def reload_config() -> ThinkChainConfig:
    """Reload configuration from environment."""
    global _config
    _config = ThinkChainConfig.from_env()
    return _config


def get_test_config() -> ThinkChainConfig:
    """Get configuration for testing (with safe defaults)."""
    config = ThinkChainConfig()
    config.test_mode = True
    config.temperature = 0.1  # Lower temperature for consistent tests
    config.max_tokens = 100   # Lower tokens for faster tests
    config.thinking_budget = 50  # Lower thinking budget for tests
    config.show_thinking = False  # Disable thinking display in tests
    config.enable_debug_logging = True
    
    # Add test model configurations
    default_models = ThinkChainConfig.get_default_models()
    
    # Mock provider configs for testing
    config.providers = {
        "anthropic": ProviderConfig(
            name="anthropic",
            default_model="claude-3-haiku-20240307",  # Fastest for tests
            models=default_models["anthropic"],
            default_temperature=0.1,
            default_max_tokens=100,
            default_thinking_budget=50
        ),
        "gemini": ProviderConfig(
            name="gemini",
            default_model="gemini-2.0-flash-exp",
            models=default_models["gemini"],
            default_temperature=0.1,
            default_max_tokens=100,
            default_thinking_budget=50
        )
    }
    
    return config 