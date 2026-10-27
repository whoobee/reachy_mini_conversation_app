"""Weather forecasts from Open-Meteo (free, no API key), with Romanian condition names.

Place names are geocoded with Open-Meteo's geocoding API. A name like
"Neustadt, Hessen, Germany" searches for the first part and prefers results whose
region or country matches the other parts.
"""

from __future__ import annotations
import datetime as dt
from typing import Any, Final

import httpx


GEOCODING_URL: Final[str] = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
MAX_DAYS: Final[int] = 14

_WEEKDAYS_RO: Final[tuple[str, ...]] = ("luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică")
# WMO weather interpretation codes (https://open-meteo.com/en/docs)
_CONDITIONS_RO: Final[dict[int, str]] = {
    0: "senin",
    1: "predominant senin",
    2: "parțial noros",
    3: "înnorat",
    45: "ceață",
    48: "ceață cu chiciură",
    51: "burniță slabă",
    53: "burniță",
    55: "burniță puternică",
    56: "burniță înghețată",
    57: "burniță înghețată puternică",
    61: "ploaie slabă",
    63: "ploaie",
    65: "ploaie puternică",
    66: "ploaie înghețată",
    67: "ploaie înghețată puternică",
    71: "ninsoare slabă",
    73: "ninsoare",
    75: "ninsoare puternică",
    77: "grăunțe de zăpadă",
    80: "averse slabe",
    81: "averse",
    82: "averse puternice",
    85: "averse de ninsoare",
    86: "averse puternice de ninsoare",
    95: "furtună",
    96: "furtună cu grindină",
    99: "furtună puternică cu grindină",
}

TOOL_DESCRIPTION: Final[str] = (
    "Weather for a place: current conditions and a daily forecast for up to 14 days (min/max temperature, "
    "conditions, rain chance, wind). Omit location for the home location. Use days=1 for today, 2 for "
    "today and tomorrow, 7 for the week."
)
TOOL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "Place name, e.g. 'Frankfurt'. Omit for home."},
        "days": {"type": "integer", "minimum": 1, "maximum": MAX_DAYS, "description": "Days of forecast, default 1."},
    },
}


class WeatherService:
    """Geocode places and fetch forecasts from Open-Meteo."""

    def __init__(self, home_location: str | None, http: httpx.AsyncClient | None = None) -> None:
        """Use ``home_location`` when a request names no place."""
        self._home = (home_location or "").strip()
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._geocode_cache: dict[str, dict[str, Any]] = {}

    async def geocode(self, name: str) -> dict[str, Any] | None:
        """Return the best Open-Meteo geocoding result for ``name``, or None."""
        key = name.strip().lower()
        if key in self._geocode_cache:
            return self._geocode_cache[key]
        parts = [part.strip() for part in name.split(",") if part.strip()]
        if not parts:
            return None
        response = await self._http.get(GEOCODING_URL, params={"name": parts[0], "count": 10, "language": "de"})
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        hints = [hint.lower() for hint in parts[1:]]

        def matches(result: dict[str, Any]) -> int:
            fields = [str(result.get(f) or "").lower() for f in ("admin1", "admin2", "admin3", "admin4", "country")]
            fields.append(str(result.get("country_code") or "").lower())

            def hit(hint: str, field: str) -> bool:
                if len(field) <= 2 or len(hint) <= 2:  # country codes: exact only ("de" is inside "Biedenkopf")
                    return hint == field
                return hint in field or field in hint

            return sum(1 for hint in hints if any(hit(hint, field) for field in fields if field))

        best: dict[str, Any] = max(results, key=matches)
        self._geocode_cache[key] = best
        return best

    async def forecast(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Tool handler: current weather plus ``days`` of daily forecast."""
        location = str(arguments.get("location") or "").strip() or self._home
        if not location:
            return {"error": "no location given and no home location configured (WEATHER_LOCATION)"}
        try:
            days = max(1, min(MAX_DAYS, int(arguments.get("days") or 1)))
        except (TypeError, ValueError):
            days = 1
        try:
            place = await self.geocode(location)
            if place is None:
                return {"error": f"place not found: {location}"}
            response = await self._http.get(
                FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                    "precipitation_sum,wind_speed_10m_max",
                    "timezone": "auto",
                    "forecast_days": days,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            return {"error": f"weather service unavailable: {e}"}
        return summarize(place, response.json())


def summarize(place: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Turn an Open-Meteo response into a compact, model-friendly summary."""
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    days = []
    for i, date in enumerate(daily.get("time") or []):
        day = dt.date.fromisoformat(date)
        days.append(
            {
                "date": date,
                "weekday": _WEEKDAYS_RO[day.weekday()],
                "conditions": _CONDITIONS_RO.get(int(daily["weather_code"][i]), "necunoscut"),
                "min_c": round(daily["temperature_2m_min"][i]),
                "max_c": round(daily["temperature_2m_max"][i]),
                "rain_chance_pct": daily["precipitation_probability_max"][i],
                "rain_mm": daily["precipitation_sum"][i],
                "wind_max_kmh": round(daily["wind_speed_10m_max"][i]),
            }
        )
    name = ", ".join(str(place[k]) for k in ("name", "admin3", "country") if place.get(k))
    result: dict[str, Any] = {"location": name, "days": days}
    if current:
        result["now"] = {
            "temperature_c": round(current.get("temperature_2m", 0)),
            "feels_like_c": round(current.get("apparent_temperature", 0)),
            "conditions": _CONDITIONS_RO.get(int(current.get("weather_code", -1)), "necunoscut"),
            "wind_kmh": round(current.get("wind_speed_10m", 0)),
        }
    return result
