import requests
import pandas as pd

CHARTS_BASE_URL = "https://api.blockchain.info/charts/"

# chart-name -> output column name
CHARTS = {
    "n-transactions": "onchain_num_tx",
    "hash-rate": "onchain_hash_rate",
    "miners-revenue": "onchain_miners_revenue_usd",
}


def _fetch_single_chart(chart_name, timespan="all"):
    try:
        resp = requests.get(
            f"{CHARTS_BASE_URL}{chart_name}",
            params={"timespan": timespan, "format": "json", "sampled": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        values = resp.json()["values"]
        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["x"], unit="s").dt.date
        return df[["date", "y"]].rename(columns={"y": chart_name})
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"⚠️  onchain_agent failed fetching '{chart_name}' ({e})")
        return None


def fetch_onchain_metrics(timespan="all", charts=None):
    """
    Fetches and merges all configured on-chain charts into one daily-indexed
    DataFrame. Returns None only if EVERY chart fails; partial failures just
    omit that one column (no synthetic default — unlike FNG, there's no
    sane 'neutral' hash rate or tx count to fall back to).
    """
    charts = charts or CHARTS
    merged = None
    for chart_name, col_name in charts.items():
        chart_df = _fetch_single_chart(chart_name, timespan=timespan)
        if chart_df is None:
            continue
        chart_df = chart_df.rename(columns={chart_name: col_name})
        merged = chart_df if merged is None else merged.merge(chart_df, on="date", how="outer")
    return merged


def merge_onchain(df, timestamp_col="timestamp", timespan="all"):
    """
    Daily on-chain values broadcast across all hourly rows on that date.
    If the fetch fails entirely, returns df unchanged — downstream
    create_features() only adds onchain_* derived features when the raw
    columns are present, so a failed fetch just means those features are
    skipped for this run rather than filled with misleading defaults.
    """
    onchain_df = fetch_onchain_metrics(timespan=timespan)
    if onchain_df is None:
        print("⚠️  onchain_agent: all charts failed, skipping on-chain features entirely")
        return df

    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date
    df = df.merge(onchain_df, left_on="_date", right_on="date", how="left")
    return df.drop(columns=["_date", "date"])


if __name__ == "__main__":
    result = fetch_onchain_metrics(timespan="30days")
    print(result if result is not None else "all charts failed")