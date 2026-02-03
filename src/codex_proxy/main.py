import sys
from .config import config
from .utils import setup_logging
from .server import ThreadedHTTPServer, RequestHandler
import logging

def main():
    setup_logging()
    logger = logging.getLogger("codex_proxy")
    
    server_address = (config.host, config.port)
    server = ThreadedHTTPServer(server_address, RequestHandler)
    
    logger.info(f"High-Performance Codex Proxy running on {config.host}:{config.port}")
    logger.info(f"Loaded {len(config.gemini_models)} Gemini models.")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()
