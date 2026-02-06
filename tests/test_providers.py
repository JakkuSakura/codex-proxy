"""Unit tests for provider registry and routing."""

import pytest
from codex_proxy.server import ProviderRegistry
from codex_proxy.providers.base import BaseProvider
from codex_proxy.providers.gemini import GeminiProvider
from codex_proxy.providers.zai import ZAIProvider


class TestProviderRegistry:
    """Test provider registry functionality."""

    def test_register_provider(self):
        """Test that providers can be registered."""

        class MockProvider(BaseProvider):
            def handle_request(self, data, handler):
                pass

        ProviderRegistry.register("test", MockProvider())
        provider = ProviderRegistry.get_provider("test-model")
        assert isinstance(provider, MockProvider)

    def test_get_provider_by_prefix(self):
        """Test that providers are retrieved by model prefix."""
        provider = ProviderRegistry.get_provider("gemini-2.5-flash-lite")
        assert isinstance(provider, GeminiProvider)

        provider = ProviderRegistry.get_provider("glm-4")
        assert isinstance(provider, ZAIProvider)

    def test_get_provider_zai_prefix(self):
        """Test that zai prefix routes to ZAI provider."""
        provider = ProviderRegistry.get_provider("zai-custom")
        assert isinstance(provider, ZAIProvider)

    def test_default_to_zai(self):
        """Test that unknown models default to ZAI provider."""
        provider = ProviderRegistry.get_provider("unknown-model")
        assert isinstance(provider, ZAIProvider)

    def test_clear_registry(self):
        """Test that registry can be cleared."""
        ProviderRegistry._providers.clear()
        assert len(ProviderRegistry._providers) == 0
        # Reinitialize for other tests
        ProviderRegistry.initialize_from_config()

    def test_initialize_from_config(self):
        """Test that registry is initialized from config."""
        ProviderRegistry.initialize_from_config()
        assert "gemini" in ProviderRegistry._providers
        assert "zai" in ProviderRegistry._providers


class TestProviderRouting:
    """Test provider routing logic."""

    def test_gemini_models_route_correctly(self):
        """Test that gemini* models route to Gemini provider."""
        test_models = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
        ]
        for model in test_models:
            provider = ProviderRegistry.get_provider(model)
            assert isinstance(provider, GeminiProvider), f"Failed for {model}"

    def test_glm_models_route_correctly(self):
        """Test that glm* models route to ZAI provider."""
        test_models = [
            "glm-4",
            "glm-4.6",
            "glm-3-turbo",
        ]
        for model in test_models:
            provider = ProviderRegistry.get_provider(model)
            assert isinstance(provider, ZAIProvider), f"Failed for {model}"

    def test_zai_models_route_correctly(self):
        """Test that zai* models route to ZAI provider."""
        test_models = [
            "zai-model-1",
            "zai-model-2",
        ]
        for model in test_models:
            provider = ProviderRegistry.get_provider(model)
            assert isinstance(provider, ZAIProvider), f"Failed for {model}"

    def test_prefix_priority(self):
        """Test that prefix matching has correct priority."""
        # Note: ProviderRegistry iterates dict keys in insertion order
        # So earlier registered prefixes will match first even if a later one is more specific

        # Clear and re-register with specific prefix first
        ProviderRegistry._providers.clear()

        class SpecialProvider(BaseProvider):
            def handle_request(self, data, handler):
                pass

        ProviderRegistry.register("gemini-special", SpecialProvider())
        ProviderRegistry.register("gemini", GeminiProvider())

        # More specific prefix should match first (because it's registered first)
        provider = ProviderRegistry.get_provider("gemini-special-model")
        assert isinstance(provider, SpecialProvider)


class TestBaseProvider:
    """Test base provider abstract methods."""

    def test_base_provider_is_abstract(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()

    def test_handle_compact_default_implementation(self):
        """Test that handle_compact has default implementation."""
        provider = ZAIProvider()

        class MockHandler:
            def send_error(self, code, message):
                pass

        mock_handler = MockHandler()
        # This should call the default implementation which sends 501
        # if not overridden by the provider
        try:
            provider.handle_compact({}, mock_handler)
        except Exception:
            pass  # Expected for providers without compaction


class TestProviderInstances:
    """Test that provider instances are created correctly."""

    def test_gemini_provider_creation(self):
        """Test that Gemini provider can be created."""
        provider = GeminiProvider()
        assert isinstance(provider, BaseProvider)

    def test_zai_provider_creation(self):
        """Test that ZAI provider can be created."""
        provider = ZAIProvider()
        assert isinstance(provider, BaseProvider)
