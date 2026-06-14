from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_NON_RETRY = {400, 401, 402, 403, 404}


class FMPError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class FMPClient:
    BASE = "https://financialmodelingprep.com"

    def __init__(
        self,
        api_key: str,
        cache_dir: str | Path,
        *,
        interval_seconds: float = 0.25,
        max_retries: int = 3,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise FMPError("FMP API key is missing; set ERL_FMP_API_KEY in .env")
        self._api_key = api_key
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._interval = interval_seconds
        self._retries = max_retries
        self._last_call = 0.0
        self._client = httpx.Client(
            base_url=self.BASE,
            timeout=timeout,
            transport=transport,
        )

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        basis = path + "?" + json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None, *, refresh: bool = False) -> Any:
        params = dict(params or {})
        cache_file = self._cache_path(path, params)
        if cache_file.exists() and not refresh:
            with open(cache_file, "r", encoding="utf-8") as handle:
                return json.load(handle)["payload"]

        attempt = 0
        while True:
            self._throttle()
            try:
                response = self._client.get(path, params={**params, "apikey": self._api_key})
            except httpx.HTTPError as exc:
                attempt += 1
                if attempt > self._retries:
                    raise FMPError(f"network error for {path}: {exc}") from exc
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
                continue

            if response.status_code == 200:
                payload = response.json()
                record = {"path": path, "params": params, "fetched_at": time.time(), "payload": payload}
                with open(cache_file, "w", encoding="utf-8") as handle:
                    json.dump(record, handle)
                return payload

            if response.status_code in _NON_RETRY:
                detail = response.text[:200]
                if response.status_code in (402, 403):
                    raise FMPError(
                        f"{path} returned {response.status_code}: this endpoint is not included "
                        f"in your FMP plan ({detail})",
                        status=response.status_code,
                    )
                raise FMPError(
                    f"{path} returned {response.status_code}: {detail}",
                    status=response.status_code,
                )

            attempt += 1
            if attempt > self._retries:
                raise FMPError(
                    f"{path} failed after {self._retries} retries "
                    f"(last status {response.status_code})",
                    status=response.status_code,
                )
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))

    def close(self) -> None:
        self._client.close()
