import requests
from codex_proxy.utils import create_session

def test_create_session():
    session = create_session()
    assert isinstance(session, requests.Session)
    assert "https://" in session.adapters
    assert "http://" in session.adapters
    
    adapter = session.adapters["https://"]
    assert adapter.max_retries.total == 3
