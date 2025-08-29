from __future__ import annotations

from typing import Tuple
import os
import time
import pathlib
import pandas as pd
from dotenv import load_dotenv

from cfbd.configuration import Configuration
from cfbd.api_client import ApiClient
from cfbd import GamesApi  # type: ignore
from cfbd.models.division_classification import DivisionClassification
from cfbd.rest import ApiException # type: ignore

__all__ = [
    "get_college_football_games",
    "clear_games_cache",
    "_make_games_api",  # exported for reuse/tests if you want it
]

# -------------------------------
# Cache configuration
# -------------------------------
_CACHE_VERSION = "v1"  # bump if you change schema/filters so old files don't mix
_DEFAULT_TTL_HOURS = 24
_DEFAULT_CACHE_DIR = os.getenv("WU_CFB_CACHE_DIR", ".cache/cfbd")


# -------------------------------
# Small cache helpers
# -------------------------------
def _cache_base(cache_dir: str, year: int) -> pathlib.Path:
    """
    Base path for cache; we'll add .parquet or .pkl as an extension.
    Example: .cache/cfbd/games_2025_v1.[parquet|pkl]
    """
    p = pathlib.Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"games_{year}_{_CACHE_VERSION}"


def _is_fresh(base: pathlib.Path, max_age_hours: int) -> bool:
    """Return True if a cache file exists and is newer than the TTL."""
    for ext in (".parquet", ".pkl"):
        path = base.with_suffix(ext)
        if path.exists():
            age_seconds = time.time() - path.stat().st_mtime
            return age_seconds <= max_age_hours * 3600
    return False


def _read_cache(base: pathlib.Path) -> pd.DataFrame | None:
    """Try Parquet, then Pickle; return DataFrame or None."""
    pq = base.with_suffix(".parquet")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    pkl = base.with_suffix(".pkl")
    if pkl.exists():
        try:
            return pd.read_pickle(pkl)
        except Exception:
            pass
    return None


def _write_cache(base: pathlib.Path, df: pd.DataFrame) -> None:
    """Prefer Parquet; fall back to Pickle if pyarrow/fastparquet is missing."""
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_pickle(base.with_suffix(".pkl"))


# -------------------------------
# CFBD client factory
# -------------------------------
def _make_games_api() -> GamesApi:
    """
    Build an authenticated GamesApi client with host + token configured.
    Centralized here to avoid repeating auth/host logic.
    """
    load_dotenv()
    token = os.getenv("CFBD_API_KEY")
    if not token:
        raise RuntimeError("CFBD_API_KEY is not set in the environment")

    cfg = Configuration(host="https://api.collegefootballdata.com")

    # Support both generated-client auth styles
    if hasattr(cfg, "access_token"):
        # Some cfbd client versions expose an access_token attribute
        cfg.access_token = token
    else:
        # Others expect api_key + api_key_prefix to form "Authorization: Bearer <token>"
        if not hasattr(cfg, "api_key"):
            cfg.api_key = {}
        if not hasattr(cfg, "api_key_prefix"):
            cfg.api_key_prefix = {}
        cfg.api_key["Authorization"] = token
        cfg.api_key_prefix["Authorization"] = "Bearer"

    return GamesApi(ApiClient(cfg))


# -------------------------------
# Internal fetcher (REST only)
# -------------------------------
def _fetch_year(api: GamesApi, year: int) -> pd.DataFrame:
    """
    Fetch FBS + FCS for a given year via REST and de-dup by game id.
    """
    fbs = api.get_games(year=year, classification=DivisionClassification("fbs"))
    fcs = api.get_games(year=year, classification=DivisionClassification("fcs"))
    games = list(fbs) + list(fcs)
    if not games:
        return pd.DataFrame()
    df = pd.DataFrame(g.to_dict() for g in games)
    return df.drop_duplicates(subset=["id"]).reset_index(drop=True)


# -------------------------------
# Public API
# -------------------------------
def get_college_football_games(
    year: int = 2024,
    *,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    max_age_hours: int = _DEFAULT_TTL_HOURS,
    force_refresh: bool = False,
    api: GamesApi | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch combined FBS + FCS games for `year` and `year-1`, with a 24h on-disk cache.

    Returns:
        (df, ly_df): DataFrames for the requested year and previous year, each de-duped by game id.

    Parameters:
        year: target season (e.g., 2025).
        cache_dir: where to store cache files (default .cache/cfbd or $WU_CFB_CACHE_DIR).
        max_age_hours: TTL; if cache older than this, we refresh (default 24 hours).
        force_refresh: ignore any cache this call and overwrite it.
        api: optional pre-built GamesApi client (useful for tests). If not provided,
             this function builds and closes its own client.
    """
    # Cache bases
    cur_base = _cache_base(cache_dir, year)
    prev_base = _cache_base(cache_dir, year - 1)

    # Try cache (unless forced)
    df_cur = None if force_refresh or not _is_fresh(cur_base, max_age_hours) else _read_cache(cur_base)
    df_prev = None if force_refresh or not _is_fresh(prev_base, max_age_hours) else _read_cache(prev_base)

    if df_cur is not None and df_prev is not None:
        return df_cur, df_prev

    close_when_done = False
    if api is None:
        api = _make_games_api()
        close_when_done = True

    try:
        if df_cur is None:
            df_cur = _fetch_year(api, year)
            _write_cache(cur_base, df_cur)
        if df_prev is None:
            df_prev = _fetch_year(api, year - 1)
            _write_cache(prev_base, df_prev)
    except ApiException as e:
        raise RuntimeError(
            f"CFBD API error (status={getattr(e, 'status', '?')}): {getattr(e, 'body', e)}"
        ) from e
    finally:
        if close_when_done:
            # Free HTTP resources if we created the client
            try:
                api.api_client.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    return df_cur, df_prev


def clear_games_cache(cache_dir: str = _DEFAULT_CACHE_DIR) -> None:
    """
    Remove all cached game files (handy after bumping _CACHE_VERSION or debugging).
    """
    p = pathlib.Path(cache_dir)
    if not p.exists():
        return
    for f in p.glob("games_*.*"):
        try:
            f.unlink()
        except Exception:
            pass
