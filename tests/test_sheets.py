from unittest.mock import MagicMock, patch

import pandas as pd

from batch import sheets

DUMMY_DFS = tuple(pd.DataFrame({"a": [1]}) for _ in range(4))


def _write_with_mocked_client():
    """Call write_to_sheets with credentials/gspread mocked out, and return
    the mock client so the test can inspect what sheet ID it was opened with."""
    mock_client = MagicMock()
    with patch("gspread.authorize", return_value=mock_client), \
         patch.object(sheets, "_get_credentials", return_value=object()):
        sheets.write_to_sheets(*DUMMY_DFS)
    return mock_client


def test_unset_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("GOOGLE_SHEET_ID", raising=False)
    client = _write_with_mocked_client()
    client.open_by_key.assert_called_once_with(sheets._DEFAULT_SHEET_ID)


def test_empty_env_var_falls_back_to_default(monkeypatch):
    # This is what GitHub Actions sets when a referenced secret was never
    # configured: the env var exists but is empty, not absent.
    monkeypatch.setenv("GOOGLE_SHEET_ID", "")
    client = _write_with_mocked_client()
    client.open_by_key.assert_called_once_with(sheets._DEFAULT_SHEET_ID)


def test_nonempty_env_var_is_used(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEET_ID", "custom-sheet-id")
    client = _write_with_mocked_client()
    client.open_by_key.assert_called_once_with("custom-sheet-id")


def test_explicit_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEET_ID", "env-sheet-id")
    mock_client = MagicMock()
    with patch("gspread.authorize", return_value=mock_client), \
         patch.object(sheets, "_get_credentials", return_value=object()):
        sheets.write_to_sheets(*DUMMY_DFS, sheet_id="explicit-sheet-id")
    mock_client.open_by_key.assert_called_once_with("explicit-sheet-id")


def test_writes_all_four_tabs():
    client = _write_with_mocked_client()
    sheet = client.open_by_key.return_value
    tab_names = {call.args[0] for call in sheet.worksheet.call_args_list}
    assert tab_names == {
        "WU Shuttle",
        "Underwood Shuttle",
        "Worster Shuttle",
        "Upcoming Shuttle",
    }
