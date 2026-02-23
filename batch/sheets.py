"""
batch/sheets.py — Google Sheets writer

Handles authentication and writing the three output DataFrames to their
respective tabs in the target spreadsheet.

Authentication — set ONE of the following in your .env:
    GOOGLE_SERVICE_ACCOUNT_FILE  — path to a service account JSON key file
    GOOGLE_SERVICE_ACCOUNT_JSON  — the full JSON key as a single-line string
                                   (useful for CI/CD environment secrets)

Sheet structure expected:
    'Underwood Shuttle'  — power ratings
    'Worster Shuttle'    — résumé rankings
    'Upcoming Shuttle'   — unplayed games
"""
from __future__ import annotations

import json
import os

import pandas as pd
from dotenv import load_dotenv

# Target spreadsheet — override via GOOGLE_SHEET_ID env var if needed
_DEFAULT_SHEET_ID = "1oLmVWwWZ0YehLhscw6ajfTu5OXO2SsKqd8vyIWshJuo"

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_credentials():
    """Load Google service account credentials from the environment."""
    load_dotenv()
    from google.oauth2 import service_account

    file_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if file_path:
        return service_account.Credentials.from_service_account_file(
            file_path, scopes=_SCOPES
        )

    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if json_str:
        info = json.loads(json_str)
        return service_account.Credentials.from_service_account_info(
            info, scopes=_SCOPES
        )

    raise RuntimeError(
        "Google credentials not found. Set GOOGLE_SERVICE_ACCOUNT_FILE (path to a "
        "service account JSON key file) or GOOGLE_SERVICE_ACCOUNT_JSON (the JSON "
        "content as a string) in your .env or environment."
    )


def _write_tab(sheet, tab_name: str, df: pd.DataFrame) -> None:
    """Clear a worksheet and write a DataFrame to it."""
    ws = sheet.worksheet(tab_name)
    ws.clear()
    rows = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
    ws.update(rows)
    print(f"  {tab_name}: {len(df)} rows written.")


def write_to_sheets(
    underwood: pd.DataFrame,
    worster: pd.DataFrame,
    upcoming: pd.DataFrame,
    sheet_id: str | None = None,
) -> None:
    """
    Write all three output DataFrames to their respective Google Sheets tabs.

    Args:
        underwood: Formatted Underwood power ratings.
        worster:   Formatted Worster résumé rankings.
        upcoming:  Upcoming unplayed games.
        sheet_id:  Google Sheet ID (defaults to the production sheet).
    """
    import gspread

    if sheet_id is None:
        sheet_id = os.getenv("GOOGLE_SHEET_ID", _DEFAULT_SHEET_ID)

    creds = _get_credentials()
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)

    _write_tab(sheet, "Underwood Shuttle", underwood)
    _write_tab(sheet, "Worster Shuttle", worster)
    _write_tab(sheet, "Upcoming Shuttle", upcoming)
