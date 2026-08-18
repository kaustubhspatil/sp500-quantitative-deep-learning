import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
SEQ_LEN = 30

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        x = self.relu(out[:, -1, :])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

def main():
    print("Loading dataset for backtesting...")
    df = pd.read_parquet(os.path.join(DATA_DIR, "processed_features.parquet"))
    
    exclude = ['Date', 'Ticker', 'Next_Return', 'Target']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    dates = np.sort(df['Date'].unique())
    train_split_date = dates[int(len(dates) * 0.70)]
    val_split_date = dates[int(len(dates) * 0.85)]
    
    # Isolate Test Set (Data after validation split)
    test_df = df[df['Date'] >= val_split_date].copy()
    top_tickers = test_df['Ticker'].value_counts().head(50).index.tolist()
    test_df = test_df[test_df['Ticker'].isin(top_tickers)].copy()
    
    # Scale features using training params approximation or scaling test directly
    scaler = StandardScaler()
    test_df[feature_cols] = scaler.fit_transform(test_df[feature_cols])
    test_df[feature_cols] = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"Loading trained model weights...")
    model = CNN_LSTM(input_dim=len(feature_cols))
    model.load_state_dict(torch.load("capstone/models/cnn_lstm_model.pt", map_location=torch.device('cpu')))
    model.eval()
    
    print("Running backtest simulation on test set...")
    results = []
    
    for ticker, group in test_df.groupby('Ticker'):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) <= SEQ_LEN:
            continue
            
        values = group[feature_cols].values
        next_returns = group['Next_Return'].values
        actual_dates = group['Date'].values
        
        for i in range(SEQ_LEN, len(group)):
            seq = torch.tensor(values[i-SEQ_LEN:i], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logit = model(seq)
                prob = torch.sigmoid(logit).item()
                
            # Trading Strategy: Go long (1) if model predicts probability > 0.5, else cash/short (0)
            signal = 1 if prob >= 0.5 else 0
            strat_return = signal * next_returns[i]
            
            results.append({
                "Date": actual_dates[i],
                "Ticker": ticker,
                "Probability": prob,
                "Signal": signal,
                "Actual_Return": next_returns[i],
                "Strategy_Return": strat_return
            })
            
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No backtest results generated. Check sequence lengths.")
        return
        
    # Aggregate daily strategy performance
    daily_perf = res_df.groupby('Date')['Strategy_Return'].mean().reset_index()
    daily_perf = daily_perf.sort_values('Date')
    
    # Compute Financial Evaluation Metrics
    strat_returns = daily_perf['Strategy_Return'].values
    ann_return = float(np.mean(strat_returns) * 252)
    ann_vol = float(np.std(strat_returns, ddof=1) * (252 ** 0.5))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    
    print("\n" + "=" * 40)
    print("FINAL BACKTEST PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"Annualized Return: {ann_return * 100:.2f}%")
    print(f"Annualized Volatility: {ann_vol * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()