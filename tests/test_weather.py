import httpx
import pytest

from reachy_mini_conversation_app.weather import WeatherService


GEOCODE = {
    "results": [
        {
            "name": "Hasselborn",
            "admin1": "Niedersachsen",
            "admin3": "Elsewhere",
            "country": "Deutschland",
            "country_code": "DE",
            "latitude": 52.0,
            "longitude": 9.0,
        },
        {
            "name": "Hasselborn",
            "admin1": "Hessen",
            "admin3": "Waldsolms",
            "country": "Deutschland",
            "country_code": "DE",
            "latitude": 50.41,
            "longitude": 8.49,
        },
    ]
}
FORECAST = {
    "current": {"temperature_2m": 7.1, "apparent_temperature": 4.6, "weather_code": 2, "wind_speed_10m": 10.4},
    "daily": {
        "time": ["2026-10-26", "2026-10-27"],
        "weather_code": [3, 61],
        "temperature_2m_max": [8.4, 10.6],
        "temperature_2m_min": [5.6, 6.2],
        "precipitation_probability_max": [5, 80],
        "precipitation_sum": [0.0, 4.2],
        "wind_speed_10m_max": [14.0, 22.5],
    },
}


def _service(requests: list[httpx.Request], home: str | None = "Hasselborn, Waldsolms") -> WeatherService:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "geocoding" in request.url.host:
            name = request.url.params["name"]
            return httpx.Response(200, json=GEOCODE if name == "Hasselborn" else {})
        return httpx.Response(200, json=FORECAST)

    return WeatherService(home, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.mark.asyncio
async def test_home_forecast_picks_the_matching_place_and_speaks_romanian() -> None:
    """Without a location the home place is used, disambiguated by region, with Romanian names."""
    requests: list[httpx.Request] = []
    result = await _service(requests).forecast({"days": 2})

    forecast_request = requests[-1]
    assert forecast_request.url.params["latitude"] == "50.41"
    assert forecast_request.url.params["forecast_days"] == "2"
    assert result["location"] == "Hasselborn, Waldsolms, Deutschland"
    assert result["now"] == {"temperature_c": 7, "feels_like_c": 5, "conditions": "parțial noros", "wind_kmh": 10}
    assert [(d["weekday"], d["conditions"], d["min_c"], d["max_c"]) for d in result["days"]] == [
        ("luni", "înnorat", 6, 8),
        ("marți", "ploaie slabă", 6, 11),
    ]


@pytest.mark.asyncio
async def test_days_are_clamped_and_geocoding_is_cached() -> None:
    """Out-of-range day counts are clamped to 1..14 and repeated places are geocoded once."""
    requests: list[httpx.Request] = []
    service = _service(requests)

    await service.forecast({"days": 99})
    await service.forecast({"days": 0})

    forecast_days = [r.url.params["forecast_days"] for r in requests if "geocoding" not in r.url.host]
    assert forecast_days == ["14", "1"]
    assert sum(1 for r in requests if "geocoding" in r.url.host) == 1


@pytest.mark.asyncio
async def test_unknown_place_and_missing_home_are_reported() -> None:
    """Unknown places and a missing home location give clear errors instead of guesses."""
    assert await _service([]).forecast({"location": "Nowhereville"}) == {"error": "place not found: Nowhereville"}
    assert "WEATHER_LOCATION" in (await _service([], home=None).forecast({}))["error"]
