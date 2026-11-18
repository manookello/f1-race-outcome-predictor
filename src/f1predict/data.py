"""Download and tidy race data from the Jolpica-F1 API (the community successor to Ergast).

Raw JSON pages are cached under ``data/raw/jolpica`` so the API is only hit once per page.
Requests are throttled (default 1 request / second) and retried on HTTP 429 / 5xx, which keeps
us well inside Jolpica's published limits (4 req/s burst, 500 req/hour).

Usage::

    python -m f1predict.data --start 2014 --end 2026
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

from f1predict.config import DATA_PROCESSED, DATA_RAW, FIRST_SEASON, LAST_SEASON

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100


class JolpicaClient:
    """Tiny polite client: throttle, retry, and on-disk cache."""

    def __init__(
        self,
        cache_dir: Path = DATA_RAW / "jolpica",
        min_interval: float = 1.0,
        max_retries: int = 6,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "f1-race-outcome-predictor/0.1 (open-source research)"
        self._last_request = 0.0

    def _request(self, path: str, params: dict) -> dict:
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2 ** (attempt + 2))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Giving up on {path} after {self.max_retries} attempts")

    def get_all(self, path: str, refresh: bool = False) -> list[dict]:
        """Return every page (``MRData`` dicts) of a paginated endpoint such as ``2024/results``."""
        pages: list[dict] = []
        offset = 0
        stem = path.strip("/").replace("/", "_")
        while True:
            cache_file = self.cache_dir / f"{stem}_{offset}.json"
            if cache_file.exists() and not refresh:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                payload = self._request(
                    f"{path.strip('/')}/", {"limit": PAGE_SIZE, "offset": offset}
                )
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            mr = payload["MRData"]
            pages.append(mr)
            offset += PAGE_SIZE
            if offset >= int(mr["total"]):
                break
        return pages


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------


def _races(pages: list[dict]) -> list[dict]:
    """Flatten races across pages. A race can be split over two pages; we keep both halves."""
    return [race for page in pages for race in page["RaceTable"]["Races"]]


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_races(pages: list[dict]) -> pd.DataFrame:
    rows = {}
    for race in _races(pages):
        key = (int(race["season"]), int(race["round"]))
        loc = race["Circuit"]["Location"]
        rows[key] = {
            "season": key[0],
            "round": key[1],
            "race_name": race["raceName"],
            "date": race["date"],
            "circuit_id": race["Circuit"]["circuitId"],
            "circuit_name": race["Circuit"]["circuitName"],
            "locality": loc.get("locality"),
            "country": loc.get("country"),
            "lat": _to_float(loc.get("lat")),
            "long": _to_float(loc.get("long")),
        }
    return pd.DataFrame(rows.values())


def _driver_cols(entry: dict) -> dict:
    drv, con = entry["Driver"], entry["Constructor"]
    return {
        "driver_id": drv["driverId"],
        "driver_code": drv.get("code"),
        "driver_name": f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip(),
        "constructor_id": con["constructorId"],
        "constructor_name": con["name"],
    }


def parse_results(pages: list[dict], key: str = "Results") -> pd.DataFrame:
    rows = []
    for race in _races(pages):
        for res in race.get(key, []):
            fastest = res.get("FastestLap", {})
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    **_driver_cols(res),
                    "number": _to_int(res.get("number")),
                    "grid": _to_int(res.get("grid")),
                    "position": _to_int(res.get("position")),
                    "position_text": res.get("positionText"),
                    "points": _to_float(res.get("points")) or 0.0,
                    "laps": _to_int(res.get("laps")),
                    "status": res.get("status"),
                    "time_millis": _to_int(res.get("Time", {}).get("millis")),
                    "fastest_lap_rank": _to_int(fastest.get("rank")),
                    "fastest_lap_time": fastest.get("Time", {}).get("time"),
                }
            )
    return pd.DataFrame(rows)


def parse_qualifying(pages: list[dict]) -> pd.DataFrame:
    rows = []
    for race in _races(pages):
        for res in race.get("QualifyingResults", []):
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    **_driver_cols(res),
                    "quali_position": _to_int(res.get("position")),
                    "q1": res.get("Q1"),
                    "q2": res.get("Q2"),
                    "q3": res.get("Q3"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Build the processed tables
# --------------------------------------------------------------------------------------


def build_dataset(
    start: int = FIRST_SEASON,
    end: int = LAST_SEASON,
    client: JolpicaClient | None = None,
    refresh_current: bool = True,
    out_dir: Path = DATA_PROCESSED,
) -> dict[str, pd.DataFrame]:
    """Fetch seasons ``start..end`` and write races/results/qualifying/sprints parquet files."""
    client = client or JolpicaClient()
    current_year = dt.date.today().year
    tables: dict[str, list[pd.DataFrame]] = {
        "races": [],
        "results": [],
        "qualifying": [],
        "sprints": [],
    }

    for season in range(start, end + 1):
        refresh = refresh_current and season >= current_year
        result_pages = client.get_all(f"{season}/results", refresh=refresh)
        quali_pages = client.get_all(f"{season}/qualifying", refresh=refresh)
        tables["races"].append(parse_races(result_pages))
        tables["results"].append(parse_results(result_pages))
        tables["qualifying"].append(parse_qualifying(quali_pages))
        if season >= 2021:  # sprint format introduced in 2021
            sprint_pages = client.get_all(f"{season}/sprint", refresh=refresh)
            sprints = parse_results(sprint_pages, key="SprintResults")
            if not sprints.empty:
                tables["sprints"].append(sprints)
        n_res, n_quali = len(tables["results"][-1]), len(tables["qualifying"][-1])
        print(f"{season}: {n_res} results, {n_quali} quali rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, parts in tables.items():
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True).drop_duplicates()
        if name == "races":
            df = df.drop_duplicates(["season", "round"])
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        frames[name] = df
    return frames


def load_tables(data_dir: Path = DATA_PROCESSED) -> dict[str, pd.DataFrame]:
    """Load the processed parquet tables written by :func:`build_dataset`."""
    names = ["races", "results", "qualifying", "sprints"]
    return {n: pd.read_parquet(data_dir / f"{n}.parquet") for n in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=FIRST_SEASON)
    parser.add_argument("--end", type=int, default=LAST_SEASON)
    parser.add_argument("--no-refresh-current", action="store_true")
    args = parser.parse_args()
    build_dataset(args.start, args.end, refresh_current=not args.no_refresh_current)


if __name__ == "__main__":
    main()
