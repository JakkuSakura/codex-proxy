from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict


class BaseProvider(ABC):
    """Abstract base class for AI model providers."""

    @abstractmethod
    def handle_request(
        self, data: Dict[str, Any], handler: BaseHTTPRequestHandler
    ) -> None:
        """Process a standard request and write to the handler's wfile."""
        pass

    def handle_compact(
        self, data: Dict[str, Any], handler: BaseHTTPRequestHandler
    ) -> None:
        """Process a context compaction request (optional)."""
        handler.send_error(501, "Compaction not implemented for this provider.")
