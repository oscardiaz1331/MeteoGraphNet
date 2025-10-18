from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


__all__ = ("RestClient", "RestClientError")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RestClientError(Exception):
    """Generic exception raised by RestClient for request failures."""


class RestClient:
    """
    Small HTTP REST client with retries and sensible defaults.

    Parameters
    ----------
    base_url:
        The base URL / domain for the API (e.g. "https://api.example.com").
        It must not end with a trailing slash — the client will handle joining.
    timeout:
        Default request timeout in seconds (float or tuple). Default is 10s.
    default_headers:
        Optional mapping of headers to include on every request.
    max_retries:
        Number of total retries for idempotent requests (default: 3).
    backoff_factor:
        Retry backoff factor for exponential backoff (default: 0.3).
    status_forcelist:
        HTTP status codes that should trigger a retry.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        *,
        timeout: Union[float, tuple] = 10.0,
        default_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Optional[tuple] = (429, 500, 502, 503, 504),
    ):
        if not base_url:
            raise ValueError("base_url must be provided and non-empty.")
        # normalize base_url: remove trailing slash if present
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers = default_headers or {}
        self.params = {'api_key': auth_token}

        self.session = requests.Session()
        self.session.headers.update(self.default_headers)

        # Configure retries for the session's adapter
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=status_forcelist,
            allowed_methods=("GET", "HEAD", "OPTIONS"),
            backoff_factor=backoff_factor,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # mount adapter to both http and https
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        logger.debug(
            "RestClient created for base_url=%s timeout=%s headers=%s",
            self.base_url,
            self.timeout,
            self.default_headers,
        )

    # -------------------------
    # Core request helper
    # -------------------------
    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, tuple]] = None,
        stream: bool = False,
    ) -> requests.Response:
        """
        Perform an HTTP request.

        - method: "GET", "POST", "PUT", "DELETE", ...
        - endpoint: path relative to base_url, e.g. "/opendata/sh/xxx" or "api/..." (leading slash optional)
        - returns: requests.Response on success; raises RestClientError on failure
        """
        if not endpoint:
            raise ValueError("endpoint must be provided and non-empty")

        # join base + endpoint
        endpoint = endpoint.lstrip("/")  # ensure no double slash
        url = f"{self.base_url}/{endpoint}"

        req_timeout = timeout if timeout is not None else self.timeout
        req_headers = {**(self.default_headers or {}), **(headers or {})}

        logger.debug("Request %s %s params=%s json=%s data=%s headers=%s", method, url, params, bool(json), bool(data), req_headers)

        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json,
                data=data,
                headers=req_headers,
                timeout=req_timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            logger.exception("Network error during HTTP %s %s", method, url)
            raise RestClientError(f"Network error during request to {url}: {exc}") from exc

        # If status code is not in 2xx, raise with helpful message
        if not response.ok:
            # try to include any JSON error payload for debugging
            err_text = ""
            try:
                err_payload = response.json()
                err_text = f" response_json={err_payload!r}"
            except Exception:
                # fallback to text
                err_text = f" response_text={response.text!r}"

            logger.warning("Request to %s failed with %s.%s", url, response.status_code, err_text)
            raise RestClientError(
                f"HTTP {response.status_code} error for {method} {url}.{err_text}"
            )

        logger.debug("Request %s %s -> %s", method, url, response.status_code)
        return response

    # -------------------------
    # Convenience wrappers
    # -------------------------
    def get(
        self,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, tuple]] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Convenience GET request."""
        return self.request(
            "GET", endpoint, params=params, headers=headers, timeout=timeout, stream=stream
        )

    def post(
        self,
        endpoint: str,
        *,
        json: Optional[Any] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, tuple]] = None,
    ) -> requests.Response:
        """Convenience POST request."""
        return self.request(
            "POST",
            endpoint,
            json=json,
            data=data,
            params=params,
            headers=headers,
            timeout=timeout,
        )

    def put(
        self,
        endpoint: str,
        *,
        json: Optional[Any] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, tuple]] = None,
    ) -> requests.Response:
        """Convenience PUT request."""
        return self.request(
            "PUT",
            endpoint,
            json=json,
            data=data,
            params=params,
            headers=headers,
            timeout=timeout,
        )

    def delete(
        self,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, tuple]] = None,
    ) -> requests.Response:
        """Convenience DELETE request."""
        return self.request(
            "DELETE", endpoint, params=params, headers=headers, timeout=timeout
        )

    # -------------------------
    # Helpers / cleanup
    # -------------------------
    def close(self) -> None:
        """Close underlying session (release connections)."""
        try:
            self.session.close()
            logger.debug("Session closed for base_url=%s", self.base_url)
        except Exception:
            logger.exception("Error while closing session for %s", self.base_url)

    def __enter__(self) -> "RestClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
