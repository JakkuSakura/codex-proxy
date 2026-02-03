import pytest
import responses
import json
import time
import os
from unittest.mock import patch, mock_open
from codex_proxy.auth import GeminiAuth
from codex_proxy.config import config

@pytest.fixture
def auth():
    return GeminiAuth()

def test_get_access_token_from_memory(auth):
    auth._cached_creds = "cached-token"
    auth._creds_expiry = (time.time() + 1000) * 1000
    assert auth.get_access_token() == "cached-token"

@responses.activate
def test_get_access_token_refresh(auth, tmp_path):
    # Mock creds file
    creds_file = tmp_path / "creds.json"
    initial_data = {
        "access_token": "old-token",
        "refresh_token": "ref-token",
        "expiry_date": (time.time() - 1000) * 1000
    }
    creds_file.write_text(json.dumps(initial_data))
    
    with patch("codex_proxy.config.config.gemini_creds_path", str(creds_file)):
        responses.add(
            responses.POST,
            "https://oauth2.googleapis.com/token",
            json={"access_token": "new-token", "expires_in": 3600},
            status=200
        )
        
        token = auth.get_access_token()
        assert token == "new-token"
        assert auth._cached_creds == "new-token"
        
        # Verify file updated
        updated_data = json.loads(creds_file.read_text())
        assert updated_data["access_token"] == "new-token"

@responses.activate
def test_get_project_id_success(auth):
    responses.add(
        responses.POST,
        f"{config.gemini_api_base}/v1internal:loadCodeAssist",
        json={"cloudaicompanionProject": {"id": "test-project"}},
        status=200
    )
    
    pid = auth.get_project_id("token")
    assert pid == "test-project"
    assert auth._cached_project_id == "test-project"

@responses.activate
def test_get_project_id_onboarding(auth):
    """Verifies that if Project ID is missing, we attempt onboarding."""
    
    # 1. loadCodeAssist returns NO project ID, but allowedTiers
    responses.add(
        responses.POST,
        f"{config.gemini_api_base}/v1internal:loadCodeAssist",
        json={
            "cloudaicompanionProject": None,
            "allowedTiers": [{"id": "free-tier", "isDefault": True}]
        },
        status=200
    )
    
    # 2. onboardUser is called
    responses.add(
        responses.POST,
        f"{config.gemini_api_base}/v1internal:onboardUser",
        json={"name": "operations/op-123", "done": False},
        status=200
    )
    
    # 3. Poll Operation (Not done)
    responses.add(
        responses.GET,
        f"{config.gemini_api_base}/v1internal/operations/op-123",
        json={"name": "operations/op-123", "done": False},
        status=200
    )
    
    # 4. Poll Operation (Done with Result)
    responses.add(
        responses.GET,
        f"{config.gemini_api_base}/v1internal/operations/op-123",
        json={
            "name": "operations/op-123", 
            "done": True,
            "response": {
                "cloudaicompanionProject": {"id": "newly-onboarded-project"}
            }
        },
        status=200
    )
    
    with patch("time.sleep", return_value=None): # Speed up polling
        pid = auth.get_project_id("token")
    
    assert pid == "newly-onboarded-project"
    assert auth._cached_project_id == "newly-onboarded-project"
    
    # Verify onboard payload used 'free-tier'
    assert json.loads(responses.calls[1].request.body)["tierId"] == "free-tier"