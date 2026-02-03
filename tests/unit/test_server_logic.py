import json
import pytest
from unittest.mock import MagicMock, patch
from codex_proxy.server import RequestHandler

class MockRequestHandler(RequestHandler):
    def __init__(self, body):
        self.rfile = MagicMock()
        self.rfile.read.return_value = json.dumps(body).encode()
        self.headers = {"Content-Length": "100"}
        self.wfile = MagicMock()
        self.path = "/v1/responses"
        # Do not call super().__init__

def test_routing_gemini():
    # Setup
    handler = MockRequestHandler({"model": "gemini-pro"})
    
    # Patch the class attributes
    with patch.object(RequestHandler, 'gemini_provider') as mock_gemini:
        with patch.object(RequestHandler, 'zai_provider') as mock_zai:
            handler.do_POST()
            
            mock_gemini.handle_request.assert_called_once()
            mock_zai.handle_request.assert_not_called()

def test_routing_zai():
    # Setup
    handler = MockRequestHandler({"model": "glm-4"})
    
    with patch.object(RequestHandler, 'gemini_provider') as mock_gemini:
        with patch.object(RequestHandler, 'zai_provider') as mock_zai:
            handler.do_POST()
            
            mock_zai.handle_request.assert_called_once()
            mock_gemini.handle_request.assert_not_called()
