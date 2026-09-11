"""
Jarvis Capital - Deep Learning Equity Signal Capstone
=====================================================

Reusable modules shared by the capstone notebooks:

    config      - paths, tickers, temporal splits, constants
    plotting    - the project's visual identity (palette, style, chart helpers)
    data        - raw data loading and reshaping
    features    - hand-built technical indicators and feature engineering
    datasets    - PyTorch Dataset classes (tabular and sequence)
    models      - model architectures (FFNN, LSTM, 1D CNN, ResNet transfer)
    training    - the training loop with early stopping and checkpointing
    evaluation  - metrics and walk-forward validation utilities
    backtest    - portfolio backtesting engine with transaction costs
    charts      - candlestick chart image generation for the vision models
"""

__version__ = "1.0.0"
