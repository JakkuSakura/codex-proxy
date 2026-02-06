import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from .exceptions import ConfigurationError

# Official gemini-cli credentials
# These are public-safe as per Google's own documentation for installed apps.
GEMINI_CLI_CLIENT_ID = (
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
)
GEMINI_CLI_CLIENT_SECRET = "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"


def _validate_port(port_str: str) -> int:
    """Validate and convert port string to integer."""
    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ConfigurationError(f"Port must be between 1 and 65535, got: {port}")
        return port
    except ValueError as e:
        raise ConfigurationError(f"Invalid port value: {port_str}: {e}")


def _validate_url(url: str, name: str) -> str:
    """Validate URL format."""
    if not url.startswith(("http://", "https://")):
        raise ConfigurationError(
            f"{name} must be a valid URL starting with http:// or https://"
        )
    return url


def _validate_model_prefix(prefix: str) -> str:
    """Validate model prefix format."""
    if not prefix or not prefix.islower():
        raise ConfigurationError(
            f"Model prefix must be lowercase alphanumeric: {prefix}"
        )
    return prefix


@dataclass
class Config:
    # Server
    host: str = "0.0.0.0"
    port: int = field(
        default_factory=lambda: _validate_port(
            os.environ.get("CODEX_PROXY_PORT", "8765")
        )
    )

    # Paths
    gemini_creds_path: str = os.path.expanduser("~/.gemini/oauth_creds.json")
    config_path: str = os.path.expanduser("~/.config/codex-proxy/config.json")

    # APIs
    z_ai_url: str = field(
        default_factory=lambda: _validate_url(
            os.environ.get(
                "CODEX_PROXY_ZAI_URL",
                "https://api.z.ai/api/coding/paas/v4/chat/completions",
            ),
            "Z.AI URL",
        )
    )
    gemini_api_internal: str = field(
        default_factory=lambda: _validate_url(
            os.environ.get(
                "CODEX_PROXY_GEMINI_API_INTERNAL", "https://cloudcode-pa.googleapis.com"
            ),
            "Gemini internal API URL",
        )
    )
    gemini_api_public: str = field(
        default_factory=lambda: _validate_url(
            os.environ.get(
                "CODEX_PROXY_GEMINI_API_PUBLIC",
                "https://generativelanguage.googleapis.com",
            ),
            "Gemini public API URL",
        )
    )

    # Model Configuration (fully configurable)
    models: List[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.environ.get("CODEX_PROXY_MODELS", "").split(",")
            if m.strip()
        ]
    )
    compaction_model: Optional[str] = None
    fallback_models: Dict[str, str] = field(default_factory=dict)

    # Provider Routing (model prefix -> provider key)
    model_prefixes: Dict[str, str] = field(
        default_factory=lambda: {
            _validate_model_prefix("gemini"): "gemini",
            _validate_model_prefix("glm"): "zai",
            _validate_model_prefix("zai"): "zai",
        }
    )

    # Reasoning Configuration
    reasoning_effort: str = "medium"
    reasoning: Dict[str, Any] = field(
        default_factory=lambda: {
            "effort_levels": {
                "none": {"budget": 0, "level": "LOW"},
                "minimal": {"budget": 2048, "level": "LOW"},
                "low": {"budget": 4096, "level": "LOW"},
                "medium": {"budget": 16384, "level": "MEDIUM"},
                "high": {"budget": 32768, "level": "HIGH"},
                "xhigh": {"budget": 65536, "level": "HIGH"},
            },
            "default_effort": "medium",
        }
    )

    # Request timeouts (in seconds)
    request_timeout_connect: int = 10
    request_timeout_read: int = 600
    compaction_temperature: float = 0.1

    # Auth Defaults
    client_id: str = field(
        default_factory=lambda: os.environ.get(
            "CODEX_PROXY_GEMINI_CLIENT_ID", GEMINI_CLI_CLIENT_ID
        )
    )
    client_secret: str = field(
        default_factory=lambda: os.environ.get(
            "CODEX_PROXY_GEMINI_CLIENT_SECRET", GEMINI_CLI_CLIENT_SECRET
        )
    )
    z_ai_api_key: str = field(
        default_factory=lambda: os.environ.get("CODEX_PROXY_ZAI_API_KEY", "")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("CODEX_PROXY_GEMINI_API_KEY", "")
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.environ.get("CODEX_PROXY_LOG_LEVEL", "DEBUG").upper()
    )
    debug_mode: bool = field(
        default_factory=lambda: os.environ.get("CODEX_PROXY_DEBUG", "true").lower()
        == "true"
    )

    def __post_init__(self):
        self._load_from_file()

    def _load_from_file(self):
        """Override defaults from config file if it exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    file_config = json.load(f)

                    if not os.environ.get("CODEX_PROXY_GEMINI_CLIENT_ID"):
                        self.client_id = file_config.get("client_id", self.client_id)
                    if not os.environ.get("CODEX_PROXY_GEMINI_CLIENT_SECRET"):
                        self.client_secret = file_config.get(
                            "client_secret", self.client_secret
                        )
                    if not os.environ.get("CODEX_PROXY_ZAI_API_KEY"):
                        self.z_ai_api_key = file_config.get(
                            "z_ai_api_key", self.z_ai_api_key
                        )
                    if not os.environ.get("CODEX_PROXY_GEMINI_API_KEY"):
                        self.gemini_api_key = file_config.get(
                            "gemini_api_key", self.gemini_api_key
                        )
                    if not os.environ.get("CODEX_PROXY_PORT"):
                        self.port = file_config.get("port", self.port)
                    if not os.environ.get("CODEX_PROXY_LOG_LEVEL"):
                        self.log_level = file_config.get(
                            "log_level", self.log_level
                        ).upper()

                    if not os.environ.get("CODEX_PROXY_MODELS"):
                        self.models = file_config.get("models", self.models)

                    self.compaction_model = file_config.get(
                        "compaction_model", self.compaction_model
                    )
                    self.fallback_models = file_config.get(
                        "fallback_models", self.fallback_models
                    )

                    if "model_prefixes" in file_config:
                        for prefix, provider_key in file_config[
                            "model_prefixes"
                        ].items():
                            self.model_prefixes[_validate_model_prefix(prefix)] = (
                                provider_key
                            )

                    self.reasoning_effort = file_config.get(
                        "reasoning_effort", self.reasoning_effort
                    )
                    if "reasoning" in file_config:
                        self.reasoning.update(file_config["reasoning"])

                    if not os.environ.get("CODEX_PROXY_ZAI_URL"):
                        self.z_ai_url = file_config.get("z_ai_url", self.z_ai_url)
                    if not os.environ.get("CODEX_PROXY_GEMINI_API_INTERNAL"):
                        self.gemini_api_internal = file_config.get(
                            "gemini_api_internal", self.gemini_api_internal
                        )
                    if not os.environ.get("CODEX_PROXY_GEMINI_API_PUBLIC"):
                        self.gemini_api_public = file_config.get(
                            "gemini_api_public", self.gemini_api_public
                        )
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")


# Global Config Instance
config = Config()
