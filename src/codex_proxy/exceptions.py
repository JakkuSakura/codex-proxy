"""Custom exception hierarchy for codex-proxy."""


class ProxyError(Exception):
    """Base exception for all proxy-related errors."""

    pass


class ProviderError(ProxyError):
    """Exception raised when provider operations fail."""

    pass


class ConfigurationError(ProxyError):
    """Exception raised when configuration is invalid or missing."""

    pass


class AuthenticationError(ProxyError):
    """Exception raised when authentication fails."""

    pass


class ValidationError(ProxyError):
    """Exception raised when input validation fails."""

    pass
