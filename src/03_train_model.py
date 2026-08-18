import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = "data"
SEQ_LEN = 30

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 1)
        # Note: No Sigmoid here. We output raw logits for numerical stability with BCEWithLogitsLoss.
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        
        x = self.relu(out[:, -1, :])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

def prepare_sequences(df, feature_cols, target_col):
    Xs, ys = [], []
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) <= SEQ_LEN:
            continue
            
        values = group[feature_cols].values
        targets = group[target_col].values
        
        for i in range(SEQ_LEN, len(group)):
            Xs.append(values[i-SEQ_LEN:i])
            ys.append(targets[i])
            
    return np.array(Xs), np.array(ys)

def main():
    print("Loading master processed features dataset...")
    df = pd.read_parquet(os.path.join(DATA_DIR, "processed_features.parquet"))
    
    exclude = ['Date', 'Ticker', 'Next_Return', 'Target']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    dates = np.sort(df['Date'].unique())
    train_split_date = dates[int(len(dates) * 0.70)]
    val_split_date = dates[int(len(dates) * 0.85)]
    
    print(f"Splitting data chronologically...")
    train_df = df[df['Date'] < train_split_date]
    val_df = df[(df['Date'] >= train_split_date) & (df['Date'] < val_split_date)]
    
    top_tickers = train_df['Ticker'].value_counts().head(50).index.tolist()
    train_df = train_df[train_df['Ticker'].isin(top_tickers)]
    val_df = val_df[val_df['Ticker'].isin(top_tickers)]
    
    print(f"Building training sequences...")
    X_train, y_train = prepare_sequences(train_df, feature_cols, 'Target')
    
    print(f"Building validation sequences...")
    X_val, y_val = prepare_sequences(val_df, feature_cols, 'Target')
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    model = CNN_LSTM(input_dim=len(feature_cols)).to(DEVICE)
    
    # Use BCEWithLogitsLoss for numerically stable binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting Model Training...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()
                
                # Apply sigmoid manually on logits to evaluate accuracy
                probs = torch.sigmoid(preds)
                binary_preds = (probs >= 0.5).float()
                correct += (binary_preds == y_batch).sum().item()
                total += y_batch.size(0)
                
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy: {val_acc:.4f}")
        
    torch.save(model.state_dict(), "capstone/models/cnn_lstm_model.pt")
    print("\nTraining complete! Model weights saved to capstone/models/cnn_lstm_model.pt")

if __name__ == "__main__":
    os.makedirs("capstone/models", exist_ok=True)
    main()