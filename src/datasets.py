"""
PyTorch Dataset classes.

Two shapes of data feed the models:

  StockReturnDataset   - one row per (date, ticker): 2D features -> scalar target.
                         Order does not matter, so shuffling is fine.
  StockSequenceDataset - a 60-day trailing window per sample: 3D tensors
                         (seq_len x features) -> scalar target. Windows are
                         built per ticker and NEVER cross ticker boundaries.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class StockReturnDataset(Dataset):
    """Cross-sectional dataset for feedforward networks."""

    def __init__(self, X, y):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequence_arrays(panel, feature_columns, seq_len=60, target_column="target"):
    """
    Materialize sliding windows from a long-format panel sorted by (Ticker, Date).

    Returns:
        X    - float32 array (n_windows, seq_len, n_features)
        y    - float32 array (n_windows,)
        meta - DataFrame with the window-end Date and Ticker per window
               (the prediction is "as of" that date, targeting the next day)

    Windows never cross ticker boundaries. To score a period without losing
    its first `seq_len`-1 days, prepend that many rows of history per ticker
    and pass `meta_start` filtering downstream instead.
    """
    import pandas as pd

    X_list, y_list, meta_rows = [], [], []
    for ticker, group in panel.groupby("Ticker", sort=True):
        F = group[feature_columns].to_numpy(dtype=np.float32)
        t = group[target_column].to_numpy(dtype=np.float32)
        dates = group["Date"].to_numpy()
        for start in range(len(F) - seq_len + 1):
            end = start + seq_len
            X_list.append(F[start:end])
            y_list.append(t[end - 1])
            meta_rows.append((dates[end - 1], ticker))
    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feature_columns)), np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows, columns=["Date", "Ticker"])
    return X, y, meta


class StockSequenceDataset(Dataset):
    """
    Sliding-window dataset for LSTM / 1D CNN.

    Built from a long-format panel already sorted by (Ticker, Date). For each
    ticker, sample i is the window of rows [i, i+seq_len) with the target of
    row i+seq_len-1 (the next-day return following the window's last day).
    """

    def __init__(self, panel, feature_columns, seq_len=60, target_column="target"):
        self.seq_len = seq_len
        self.windows = []   # (array_index, start_row) pairs
        self.X_blocks, self.y_blocks = [], []

        for block_idx, (_, group) in enumerate(panel.groupby("Ticker", sort=True)):
            X = group[feature_columns].to_numpy(dtype=np.float32)
            y = group[target_column].to_numpy(dtype=np.float32)
            self.X_blocks.append(X)
            self.y_blocks.append(y)
            for start in range(len(X) - seq_len + 1):
                self.windows.append((block_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        block_idx, start = self.windows[idx]
        end = start + self.seq_len
        X = self.X_blocks[block_idx][start:end]
        y = self.y_blocks[block_idx][end - 1]
        return torch.from_numpy(X), torch.tensor([y], dtype=torch.float32)
