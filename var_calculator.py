import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
from typing import List, Optional
from .config import DEFAULT_CONFIG


class VaRCalculator:
    def __init__(self, tickers: List[str], start_date: str, end_date: str,
                 window: int = None, confidence: float = None,
                 portfolio_value: float = None, weights: Optional[List[float]] = None):

        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.window = window or DEFAULT_CONFIG['window']
        self.confidence = confidence or DEFAULT_CONFIG['confidence']
        self.portfolio_value = portfolio_value or DEFAULT_CONFIG['portfolio_value']

        # Set weights
        if weights and len(weights) == len(self.tickers) and abs(sum(weights) - 1.0) < 1e-6:
            self.weights = np.array(weights)
        else:
            self.weights = np.array([1.0 / len(self.tickers)] * len(self.tickers))

        # Initialize data containers
        self.returns = None
        self.portfolio_returns = None
        self.rolling_returns = None
        self.historical_var = None
        self.parametric_var = None

    def load_data(self) -> bool:
        try:
            data = yf.download(self.tickers, start=self.start_date,
                               end=self.end_date, progress=False)

            if data.empty:
                return False

            # Extract prices
            if len(self.tickers) == 1:
                prices = data[['Adj Close']] if 'Adj Close' in data.columns else data[['Close']]
                prices.columns = self.tickers
            else:
                prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']

            # Calculate returns
            self.returns = np.log(prices / prices.shift(1)).dropna()
            self.portfolio_returns = (self.returns * self.weights).sum(axis=1)
            self.rolling_returns = self.portfolio_returns.rolling(self.window).sum().dropna()

            return len(self.rolling_returns) > 0

        except Exception:
            return False

    def calculate_var(self) -> bool:
        if not self.load_data():
            return False

        # Historical VaR
        alpha = 1 - self.confidence
        self.historical_var = -np.percentile(self.rolling_returns, alpha * 100) * self.portfolio_value

        # Parametric VaR
        cov_matrix = self.returns.cov() * DEFAULT_CONFIG['trading_days']
        portfolio_vol = np.sqrt(self.weights.T @ cov_matrix @ self.weights)
        period_vol = portfolio_vol * np.sqrt(self.window / DEFAULT_CONFIG['trading_days'])
        z_score = norm.ppf(self.confidence)
        self.parametric_var = abs(z_score) * period_vol * self.portfolio_value

        return True

    def get_results(self) -> dict:
        if self.historical_var is None or self.parametric_var is None:
            return {}

        return {
            'historical_var': self.historical_var,
            'parametric_var': self.parametric_var,
            'portfolio_vol': self.portfolio_returns.std() * np.sqrt(DEFAULT_CONFIG['trading_days']),
            'data_points': len(self.rolling_returns)
        }