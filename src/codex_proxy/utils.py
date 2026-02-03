import logging
import sys
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import config

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure structured logging."""
    root = logging.getLogger()
    root.setLevel(config.log_level)
    
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Silence noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def create_session(pool_connections=20, pool_maxsize=100) -> requests.Session:
    """Create a robust HTTP session with retries."""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retries
    )
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s