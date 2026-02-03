from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

class BaseProvider(ABC):
    @abstractmethod
    def handle_request(self, data: Dict[str, Any], handler: BaseHTTPRequestHandler) -> None:
        """Process the request and write to the handler's wfile."""
        pass
