import json
import time
import logging
import os
import threading
from typing import Optional, Dict, Any

from .config import config
from .utils import create_session

logger = logging.getLogger(__name__)

class AuthError(Exception):
    pass

class GeminiAuth:
    """Manages Gemini OAuth2 credentials."""
    
    def __init__(self):
        self.session = create_session()
        self._cached_creds: Optional[str] = None
        self._creds_expiry: int = 0
        self._lock = threading.Lock()
        self._cached_project_id: Optional[str] = None

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Retrieve a valid access token, refreshing if necessary."""
        with self._lock:
            if not force_refresh and self._cached_creds:
                if time.time() * 1000 < self._creds_expiry - 300000:
                    return self._cached_creds

            if not os.path.exists(config.gemini_creds_path):
                raise AuthError(f"Auth required. File missing: {config.gemini_creds_path}")
            
            try:
                with open(config.gemini_creds_path, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to read creds file: {e}")
                raise AuthError(f"Corrupt credentials file: {e}")

            expiry = data.get('expiry_date', 0)
            
            if not force_refresh and expiry > (time.time() * 1000) + 300000:
                self._cached_creds = data['access_token']
                self._creds_expiry = expiry
                return data['access_token']

            logger.info("Refreshing Access Token...")
            try:
                # Dynamic discovery: must be provided via environment or ~/.gemini/proxy_config.json
                client_id = config.client_id
                client_secret = config.client_secret
                
                if not client_id or not client_secret:
                    raise AuthError("Missing client_id or client_secret in config.")

                resp = self.session.post(
                    'https://oauth2.googleapis.com/token',
                    data={
                        'client_id': client_id,
                        'client_secret': client_secret,
                        'refresh_token': data['refresh_token'],
                        'grant_type': 'refresh_token'
                    },
                    timeout=10
                )
                resp.raise_for_status()
                new_tokens = resp.json()
                
                data['access_token'] = new_tokens['access_token']
                data['expiry_date'] = int((time.time() + new_tokens['expires_in']) * 1000)
                
                with open(config.gemini_creds_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self._cached_creds = data['access_token']
                self._creds_expiry = data['expiry_date']
                return self._cached_creds
                
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                raise AuthError(f"Token refresh failed: {e}")

    def get_project_id(self, token: str) -> str:
        """Retrieve the associated Project ID, onboarding if necessary."""
        if self._cached_project_id:
            return self._cached_project_id
            
        try:
            metadata = {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI"
            }
            
            # 1. Load Code Assist
            resp = self.session.post(
                f"{config.gemini_api_base}/v1internal:loadCodeAssist",
                json={"metadata": metadata},
                headers={
                    'Authorization': f"Bearer {token}",
                    'User-Agent': 'GeminiCLI/0.26.0/gemini-3-pro-preview (linux; x64)'
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            # 2. Check for Project ID
            pid_data = data.get('cloudaicompanionProject')
            pid = None
            if isinstance(pid_data, dict):
                pid = pid_data.get('id')
            elif isinstance(pid_data, str):
                pid = pid_data
                
            if pid:
                self._cached_project_id = pid
                return pid

            # 3. Onboard if missing
            logger.info("Project ID missing. Attempting onboarding...")
            allowed_tiers = data.get('allowedTiers', [])
            tier_id = "free-tier" # Default to free
            for tier in allowed_tiers:
                if tier.get('isDefault'):
                    tier_id = tier.get('id')
                    break
            
            return self._onboard_user(token, tier_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to fetch Project ID: {e}")
            raise AuthError(f"Failed to fetch Project ID: {e}")

    def _onboard_user(self, token: str, tier_id: str, metadata: Dict) -> str:
        """Onboard user to Gemini Code Assist."""
        try:
            onboard_req = {
                "tierId": tier_id,
                "metadata": metadata
            }
            # Note: Do not send cloudaicompanionProject if tier is free
            
            resp = self.session.post(
                f"{config.gemini_api_base}/v1internal:onboardUser",
                json=onboard_req,
                headers={
                    'Authorization': f"Bearer {token}",
                    'User-Agent': 'GeminiCLI/0.26.0 (linux; x64)'
                },
                timeout=10
            )
            resp.raise_for_status()
            lro = resp.json()
            
            # Poll LRO
            op_name = lro.get('name')
            if not op_name:
                 raise AuthError("Onboarding started but no Operation Name returned")

            logger.info(f"Onboarding started: {op_name}")
            
            start_time = time.time()
            while not lro.get('done'):
                if time.time() - start_time > 60:
                    raise AuthError("Onboarding timed out")
                
                time.sleep(2)
                resp = self.session.get(
                    f"{config.gemini_api_base}/v1internal/{op_name}",
                    headers={'Authorization': f"Bearer {token}"},
                    timeout=10
                )
                resp.raise_for_status()
                lro = resp.json()
            
            # Check result
            if 'error' in lro:
                raise AuthError(f"Onboarding failed: {lro['error']}")
                
            pid_data = lro.get('response', {}).get('cloudaicompanionProject', {})
            pid = pid_data.get('id')
            
            if not pid:
                raise AuthError("Onboarding completed but Project ID is still missing")
                
            self._cached_project_id = pid
            return pid

        except Exception as e:
            logger.error(f"Onboarding error: {e}")
            raise AuthError(f"Onboarding error: {e}")