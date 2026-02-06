import logging
import sys
from .config import config
from .utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def main():
    """Entry point for the codex-proxy server."""
    logger.info("Starting codex-proxy...")
    from .server import run_server

    run_server()


if __name__ == "__main__":
    main()
