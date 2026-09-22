"""TypeSafe AI Jev API client.

Handles low-latency communication with TypeSafe AI System One (/v1/systemone).
Supports both typesafe-sdk and direct HTTP with exponential backoff retries.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

from wisprhelp.config import config

logger = logging.getLogger(__name__)


class JevAPIError(Exception):
    """Raised when TypeSafe JEV API returns an error or fails."""
    pass


class JevClient:
    """Client for TypeSafe Jev System One evaluation endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 8.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = (api_key or config.jev_api_key).strip()
        self.api_url = (api_url or config.jev_api_url).strip()
        self.model = (model or config.jev_model).strip()
        self.timeout = timeout
        self.max_retries = max_retries

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluates state against a dictionary of typed questions via TypeSafe JEV.

        Args:
            state: Context string or structured JSON dictionary (user query + OS context).
            questions: Dictionary defining Choice, Score, and Noul questions.

        Returns:
            Dictionary containing model name, answers, and token usage.

        Raises:
            JevAPIError: If API call fails after retries.
        """
        if not self.api_key:
            raise JevAPIError(
                "No TypeSafe JEV API key provided. Set JEV_API_KEY in .env"
            )

        payload = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "WisprHelp/1.0",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.api_url,
                    data=body_bytes,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    status = response.status
                    raw_body = response.read().decode("utf-8")
                    if status != 200:
                        raise JevAPIError(
                            f"TypeSafe API returned non-200 status {status}: {raw_body}"
                        )
                    return json.loads(raw_body)

            except urllib.error.HTTPError as http_err:
                error_body = ""
                try:
                    error_body = http_err.read().decode("utf-8")
                except Exception:
                    pass
                last_error = JevAPIError(
                    f"HTTP {http_err.code} from TypeSafe JEV: {error_body or http_err.reason}"
                )
                # Don't retry on client errors (400, 401, 403, 422)
                if http_err.code in (400, 401, 403, 422):
                    raise last_error from http_err

            except (urllib.error.URLError, TimeoutError) as net_err:
                last_error = JevAPIError(f"Network error contacting TypeSafe JEV: {net_err}")

            except json.JSONDecodeError as json_err:
                raise JevAPIError(f"Malformed JSON response from TypeSafe JEV: {json_err}") from json_err

            if attempt < self.max_retries:
                backoff_time = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying TypeSafe JEV call (attempt %d/%d) in %.2fs due to: %s",
                    attempt,
                    self.max_retries,
                    backoff_time,
                    last_error,
                )
                time.sleep(backoff_time)

        raise JevAPIError(
            f"Failed to communicate with TypeSafe JEV after {self.max_retries} attempts. Last error: {last_error}"
        ) from last_error
