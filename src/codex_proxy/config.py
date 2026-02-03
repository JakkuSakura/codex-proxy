import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --- Constants ---
DEFAULT_GEMINI_MODELS = [
    'gemini-3-flash-preview',
    'gemini-3-pro-preview',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-2.5-flash-lite'
]

# Default Credentials (Must be provided via environment or config file)
DEFAULT_CLIENT_ID = ''
DEFAULT_CLIENT_SECRET = ''

@dataclass
class Config:
    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    
    # Paths
    gemini_creds_path: str = os.path.expanduser('~/.gemini/oauth_creds.json')
    gemini_config_path: str = os.path.expanduser('~/.gemini/proxy_config.json')
    
    # APIs
    z_ai_url: str = "https://api.z.ai/api/coding/paas/v4/chat/completions"
    gemini_api_base: str = "https://cloudcode-pa.googleapis.com"
    
    # Model Configs (from gemini-cli)
    default_thinking_budget: int = 8192
    default_thinking_level: str = "HIGH"
    
    # Models
    gemini_models: List[str] = field(default_factory=lambda: DEFAULT_GEMINI_MODELS)
    default_personality: str = "pragmatic"
    
    # Auth
    client_id: str = field(default_factory=lambda: os.environ.get('GEMINI_CLIENT_ID', DEFAULT_CLIENT_ID))
    client_secret: str = field(default_factory=lambda: os.environ.get('GEMINI_CLIENT_SECRET', DEFAULT_CLIENT_SECRET))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.environ.get('LOG_LEVEL', 'DEBUG').upper())
    debug_mode: bool = field(default_factory=lambda: os.environ.get('DEBUG', 'true').lower() == 'true')

    def __post_init__(self):
        self._load_from_file()

    def _load_from_file(self):
        """Override defaults from config file if it exists."""
        if os.path.exists(self.gemini_config_path):
            try:
                with open(self.gemini_config_path, 'r') as f:
                    file_config = json.load(f)
                    if not os.environ.get('GEMINI_CLIENT_ID'):
                        self.client_id = file_config.get('client_id', self.client_id)
                    if not os.environ.get('GEMINI_CLIENT_SECRET'):
                        self.client_secret = file_config.get('client_secret', self.client_secret)
            except Exception as e:
                logging.warning(f"Failed to load config from {self.gemini_config_path}: {e}")

# Global Config Instance
config = Config()