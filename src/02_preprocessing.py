import os
import numpy as np
import pandas as pd
from ta import add_all_ta_features
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"

def load_data():
    """Load parquet datasets."""
    print("Loading datasets...")
    stocks_df = pd.read_parquet(os.path.join(DATA_DIR, "sp500_universe.parquet"))
    macro_df = pd.read_parquet(os.path.join(DATA_DIR, "macro_indicators.parquet"))
    
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    return stocks_df, macro_df

def engineer_features_for_stock(group):
    """Compute technical indicators for a single stock group."""
    group = group.sort_values('Date').reset_index(drop=True)
    if len(group) < 50: # Skip tickers with insufficient history
        return pd.DataFrame()
    
    # Modern pandas syntax for filling missing values
    group = group.ffill().bfill()
    
    try:
        group = add_all_ta_features(
            group, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
    except Exception as e:
        return pd.DataFrame()
        
    return group

def main():
    stocks_df, macro_df = load_data()
    
    print("\nApplying technical indicators across S&P 500 universe (this may take a few minutes)...")
    processed_stocks = []
    
    # Process group by ticker to maintain time-series integrity per stock
    grouped = stocks_df.groupby('Ticker')
    for ticker, group in grouped:
        res = engineer_features_for_stock(group)
        if not res.empty:
            processed_stocks.append(res)
            
    full_df = pd.concat(processed_stocks, ignore_index=True)
    
    # Create Target: Next day log return direction (1 if positive, 0 if negative)
    full_df['Next_Return'] = full_df.groupby('Ticker')['Close'].transform(lambda x: np.log(x.shift(-1) / x))
    full_df['Target'] = (full_df['Next_Return'] > 0).astype(int)
    
    # Drop rows with missing targets
    full_df = full_df.dropna(subset=['Next_Return'])
    
    # Pivot macro data to wide format and merge by Date
    macro_pivot = macro_df.pivot(index='Date', columns='Ticker', values='Close').reset_index()
    macro_pivot.columns.name = None
    
    full_df = pd.merge(full_df, macro_pivot, on='Date', how='left')
    
    # Save processed master dataset
    output_path = os.path.join(DATA_DIR, "processed_features.parquet")
    full_df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\nFeature engineering complete! Saved processed shape: {full_df.shape}")
    print(f"Master features dataset saved to {output_path}")

if __name__ == "__main__":
    main()