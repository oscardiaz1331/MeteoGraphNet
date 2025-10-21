from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import requests_cache
from retry_requests import retry

import openmeteo_requests


class OpenMeteoClient:
    """Wrapper around openmeteo_requests.Client with cached + retry session."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        cache_path: str = ".cache",
        expire_after: int = -1,
        retries: int = 5,
        backoff_factor: float = 0.2,
    ) -> None:
        """
        Initialize client.

        Args:
            session: Optional requests.Session (if provided, cache/retry settings
                are ignored and this session is used directly).
            cache_path: Path for requests_cache storage.
            expire_after: Cache expiration (seconds). -1 means never expire.
            retries: Number of retry attempts for transient errors.
            backoff_factor: Backoff factor used by retry wrapper.
        """
        if session is not None:
            self.session = session
        else:
            cache_session = requests_cache.CachedSession(
                cache_name=cache_path, expire_after=expire_after
            )
            # The `retry` wrapper returns a session-like object that retries on errors.
            self.session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)

        # openmeteo_requests expects a session parameter
        self.client = openmeteo_requests.Client(session=self.session)

    def weather_api(self, url: str, params: Dict[str, Any]) -> List[Any]:
        """
        Call the Open-Meteo API and return the parsed responses.

        This delegates to openmeteo_requests.Client.weather_api and returns
        whatever objects that client returns (same API as your example).
        """
        return self.client.weather_api(url, params=params)

    @staticmethod
    def hourly_to_dataframe(response: Any, variable_index: int = 0) -> pd.DataFrame:
        """
        Convert an Open-Meteo hourly block to a pandas DataFrame.

        The function assumes the response objects expose the same methods as in
        your example:
          - response.Hourly()
          - hourly.Time(), hourly.TimeEnd(), hourly.Interval()
          - hourly.Variables(i).ValuesAsNumpy()
          - optional: hourly.Variables(i).Name()

        Args:
            response: One element from client.weather_api(...) result.
            variable_index: Index of the hourly variable requested (0 for first).

        Returns:
            pd.DataFrame with a 'date' column (UTC timestamps) and a column for
            the selected variable (uses variable name if available).
        """
        hourly = response.Hourly()
        # unix timestamps (seconds)
        start_ts = hourly.Time()
        end_ts = hourly.TimeEnd()
        interval_seconds = hourly.Interval()

        values = hourly.Variables(variable_index).ValuesAsNumpy()

        times = pd.date_range(
            start=pd.to_datetime(start_ts, unit="s", utc=True),
            end=pd.to_datetime(end_ts, unit="s", utc=True),
            freq=pd.Timedelta(seconds=interval_seconds),
            inclusive="left",
        )

        try:
            var_name = hourly.Variables(variable_index).Name()
            # sanitize var_name to be a valid column name if needed
            var_name = str(var_name)
        except Exception:
            var_name = f"variable_{variable_index}"

        df = pd.DataFrame({ "date": times, var_name: values })
        return df

    @staticmethod
    def location_info(response: Any) -> Dict[str, Any]:
        """Return basic location metadata from a response object."""
        return {
            "latitude": response.Latitude(),
            "longitude": response.Longitude(),
            "elevation_m": response.Elevation(),
            "utc_offset_seconds": response.UtcOffsetSeconds(),
        }
