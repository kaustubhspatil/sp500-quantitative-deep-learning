import os
import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──
OUTPUT_DIR = "data"
START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

# Macroeconomic and Benchmark Tickers
MACRO_TICKERS = {
    "VIX": "^VIX",        # Market Volatility
    "TNX": "^TNX",        # 10-Year Treasury Yield (Interest Rates)
    "SPY": "SPY"          # S&P 500 Benchmark ETF
}

def get_sp500_tickers():
    """Scrape the live S&P 500 ticker list from Wikipedia."""
    print("Fetching live S&P 500 constituents...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Bypass Wikipedia's bot-blocker using urllib
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        html = response.read()
        
    table = pd.read_html(html)[0]
    
    tickers = table['Symbol'].tolist()
    # Clean tickers for yfinance (e.g., BRK.B -> BRK-B)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    print(f"Found {len(tickers)} tickers.")
    return tickers

def flatten_columns(df):
    """Flatten multi-level columns returned by newer yfinance versions."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def download_data(tickers, start, end, is_macro=False):
    """Download OHLCV data in chunks and format it."""
    all_data = []
    failed = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                failed.append(ticker)
                continue
            
            df = flatten_columns(df)
            df["Ticker"] = ticker
            df.index.name = "Date"
            all_data.append(df.reset_index())
        except Exception as e:
            failed.append(ticker)

    if failed:
        print(f"Failed to download {len(failed)} tickers.")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download S&P 500 Universe
    print("\n" + "=" * 60)
    print("STEP 1: Downloading S&P 500 Universe")
    print("=" * 60)
    sp500_tickers = get_sp500_tickers()
    stocks_df = download_data(sp500_tickers, START_DATE, END_DATE)
    
    stocks_path = os.path.join(OUTPUT_DIR, "sp500_universe.parquet")
    stocks_df.to_parquet(stocks_path, engine='pyarrow', index=False)
    print(f"Saved {len(stocks_df):,} rows to {stocks_path}")

    # 2. Download Macro & Benchmark Data
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Macro Indicators")
    print("=" * 60)
    macro_df = download_data(list(MACRO_TICKERS.values()), START_DATE, END_DATE, is_macro=True)
    
    # Map the yfinance tickers back to our clean names
    inv_map = {v: k for k, v in MACRO_TICKERS.items()}
    macro_df['Ticker'] = macro_df['Ticker'].map(inv_map)
    
    macro_path = os.path.join(OUTPUT_DIR, "macro_indicators.parquet")
    macro_df.to_parquet(macro_path, engine='pyarrow', index=False)
    print(f"Saved macro data to {macro_path}")
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE - DATA READY FOR ENGINEERING")
    print("=" * 60)

if __name__ == "__main__":
    main()
