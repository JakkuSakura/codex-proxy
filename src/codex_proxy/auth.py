import json
import time
import logging
import os
import threading
import requests
from typing import Optional, Dict, Any, Tuple

from .config import config
from .utils import create_session

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Custom exception for authentication related errors."""

    pass


class GeminiAuth:
    """Manages Gemini OAuth2 credentials and Project ID mapping with multi-type support."""

    def __init__(self):
        self.session = create_session()
        self._cached_token: Optional[str] = None
        self._token_expiry: int = 0
        self._lock = threading.Lock()
        self._cached_project_id: Optional[str] = None

    def get_auth_context(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Returns the authentication context: either an API key or an access token + project ID.
        """
        # 1. Check for manual API Key override
        api_key = os.environ.get("GEMINI_API_KEY") or config.gemini_api_key
        if api_key:
            return {"api_key": api_key, "type": "public"}

        # 2. Try to get OAuth Access Token
        token = self.get_access_token(force_refresh)
        pid = self.get_project_id(token)
        return {"access_token": token, "project_id": pid, "type": "internal"}

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Retrieve a valid access token, trying multiple discovery methods."""
        with self._lock:
            # 1. Check for manual override via environment
            env_token = os.environ.get("GOOGLE_CLOUD_ACCESS_TOKEN")
            if env_token:
                return env_token

            # 2. Check cache
            if not force_refresh and self._is_cached_token_valid():
                return self._cached_token  # type: ignore

            # 3. Try loading from credentials files (oauth_creds.json or ADC)
            token = self._try_load_from_files(force_refresh)
            if token:
                return token

            # 4. Try Metadata Server (GCP ADC)
            token = self._try_metadata_server()
            if token:
                return token

            raise AuthError(
                "Could not find valid Gemini credentials. Please login using 'gemini login'."
            )

    def _is_cached_token_valid(self) -> bool:
        return bool(self._cached_token and self._is_valid(self._token_expiry))

    def _is_valid(self, expiry_ms: int) -> bool:
        """Check if token is valid for at least another 5 minutes."""
        if expiry_ms == 0:
            return True  # Assume permanent if no expiry provided
        return (time.time() * 1000) < (expiry_ms - 300000)

    def _try_load_from_files(self, force_refresh: bool) -> Optional[str]:
        paths_to_try = [
            config.gemini_creds_path,
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        ]

        for path in filter(None, paths_to_try):
            if not os.path.exists(path):
                continue

            try:
                with open(path, "r") as f:
                    creds_data = json.load(f)

                # Case A: Standard authorized_user or service_account
                if creds_data.get("type") in ("authorized_user", "service_account"):
                    file_token = creds_data.get("access_token")
                    file_expiry = creds_data.get("expiry_date", 0)

                    if not force_refresh and file_token and self._is_valid(file_expiry):
                        self._cached_token = file_token
                        self._token_expiry = file_expiry
                        return file_token

                    # Refresh if possible
                    if "refresh_token" in creds_data:
                        return self._refresh_token(creds_data, path)

            except Exception as e:
                logger.debug(f"Failed to load credentials from {path}: {e}")

        return None

    def _refresh_token(self, creds_data: Dict[str, Any], path: str) -> str:
        """Perform OAuth2 refresh flow."""
        logger.info(f"Refreshing access token from {path}")

        client_id = creds_data.get("client_id") or config.client_id
        client_secret = creds_data.get("client_secret") or config.client_secret
        refresh_token = creds_data.get("refresh_token")

        if not all([client_id, client_secret, refresh_token]):
            raise AuthError("Missing OAuth2 client_id, secret, or refresh_token.")

        try:
            resp = self.session.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )
            resp.raise_for_status()
            new_tokens = resp.json()

            access_token = new_tokens["access_token"]
            expiry_date = int((time.time() + new_tokens["expires_in"]) * 1000)

            # Update local data and save back
            creds_data["access_token"] = access_token
            creds_data["expiry_date"] = expiry_date

            try:
                with open(path, "w") as f:
                    json.dump(creds_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save refreshed tokens to {path}: {e}")

            self._cached_token = access_token
            self._token_expiry = expiry_date
            return access_token

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthError(f"OAuth2 refresh failed: {e}")

    def _try_metadata_server(self) -> Optional[str]:
        """Try to get token from GCP Metadata Server."""
        url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        headers = {"Metadata-Flavor": "Google"}
        try:
            resp = self.session.get(url, headers=headers, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token")
                if token:
                    self._cached_token = token
                    self._token_expiry = int(
                        (time.time() + data.get("expires_in", 3600)) * 1000
                    )
                    return token
        except Exception:
            pass
        return None

    def get_project_id(self, token: str) -> str:
        """Retrieve the associated Google Cloud Project ID."""
        if self._cached_project_id:
            return self._cached_project_id

        env_pid = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT_ID"
        )
        if env_pid:
            self._cached_project_id = env_pid
            return env_pid

        try:
            project_info = self._fetch_project_info(token)
            pid = project_info.get("cloudaicompanionProject")
            if isinstance(pid, dict):
                pid = pid.get("id")

            if pid:
                self._cached_project_id = pid
                return pid

            logger.info("Project ID missing, starting onboarding")
            tier_id = self._determine_tier(project_info)
            return self._onboard_user(token, tier_id)

        except Exception as e:
            logger.debug(f"Failed to resolve Project ID via internal API: {e}")
            raise AuthError(f"Failed to resolve Project ID: {e}")

    def _fetch_project_info(self, token: str) -> Dict[str, Any]:
        resp = self.session.post(
            f"{config.gemini_api_internal}/v1internal:loadCodeAssist",
            json={"metadata": self._get_default_metadata()},
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "GeminiCLI/0.26.0 (linux; x64)",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _determine_tier(self, project_info: Dict[str, Any]) -> str:
        allowed_tiers = project_info.get("allowedTiers", [])
        for tier in allowed_tiers:
            if tier.get("isDefault"):
                return tier.get("id", "free-tier")
        return "free-tier"

    def _get_default_metadata(self) -> Dict[str, str]:
        return {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }

    def _onboard_user(self, token: str, tier_id: str) -> str:
        """Execute onboarding."""
        try:
            resp = self.session.post(
                f"{config.gemini_api_internal}/v1internal:onboardUser",
                json={"tierId": tier_id, "metadata": self._get_default_metadata()},
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "GeminiCLI/0.26.0 (linux; x64)",
                },
                timeout=10,
            )
            resp.raise_for_status()
            operation = resp.json()
            op_name = operation.get("name")
            if not op_name:
                raise AuthError("Onboarding failed: No operation name returned.")

            result = self._poll_operation(token, op_name)
            pid = (
                result.get("response", {}).get("cloudaicompanionProject", {}).get("id")
            )
            if not pid:
                raise AuthError("Onboarding finished but no Project ID found.")

            self._cached_project_id = pid
            return pid
        except Exception as e:
            raise AuthError(f"Onboarding failed: {e}")

    def _poll_operation(
        self, token: str, op_name: str, timeout: int = 60
    ) -> Dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            resp = self.session.get(
                f"{config.gemini_api_internal}/v1internal/{op_name}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            resp.raise_for_status()
            op_data = resp.json()
            if op_data.get("done"):
                if "error" in op_data:
                    raise AuthError(f"Operation failed: {op_data['error']}")
                return op_data
            time.sleep(2)
        raise AuthError("Operation timed out.")
