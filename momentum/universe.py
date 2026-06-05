"""S&P 500 historical constituent lists from sp500_ticker_start_end.csv."""
from pathlib import Path

import pandas as pd

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sp500_ticker_start_end.csv"
_CACHED_DF: pd.DataFrame = None


def load_constituent_data() -> pd.DataFrame:
    """Load sp500_ticker_start_end.csv, caching the result in memory."""
    global _CACHED_DF
    if _CACHED_DF is not None:
        return _CACHED_DF
    df = pd.read_csv(_DATA_FILE)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"]).fillna(pd.Timestamp.today())
    _CACHED_DF = df
    return df


def get_constituents(date: pd.Timestamp, df: pd.DataFrame) -> list:
    """Return tickers in the S&P 500 on the given date."""
    mask = (df["start_date"] <= date) & (df["end_date"] >= date)
    return df.loc[mask, "ticker"].tolist()


def get_universe_for_window(
    start: pd.Timestamp, end: pd.Timestamp, df: pd.DataFrame
) -> list:
    """Return all unique tickers that appeared in the S&P 500 between start and end."""
    mask = (df["start_date"] <= end) & (df["end_date"] >= start)
    return df.loc[mask, "ticker"].unique().tolist()
