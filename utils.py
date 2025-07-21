import pandas as pd
from datetime import datetime
from typing import List, Optional
from .config import VALIDATION_LIMITS


def validate_inputs(tickers: List[str], start_date: str, end_date: str,
                    window: int, confidence: float, portfolio_value: float,
                    weights: Optional[List[float]] = None) -> tuple[bool, str]:

    if not tickers or not all(isinstance(t, str) for t in tickers):
        return False, "Invalid ticker symbols"

    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except:
        return False, "Invalid date format"

    # Validate numeric parameters
    if not (VALIDATION_LIMITS['min_window'] <= window <= VALIDATION_LIMITS['max_window']):
        return False, f"Window must be between {VALIDATION_LIMITS['min_window']} and {VALIDATION_LIMITS['max_window']}"

    if not (VALIDATION_LIMITS['min_confidence'] <= confidence <= VALIDATION_LIMITS['max_confidence']):
        return False, f"Confidence must be between {VALIDATION_LIMITS['min_confidence']} and {VALIDATION_LIMITS['max_confidence']}"

    if portfolio_value < VALIDATION_LIMITS['min_portfolio_value']:
        return False, f"Portfolio value must be at least {VALIDATION_LIMITS['min_portfolio_value']}"

    # Validate weights
    if weights:
        if len(weights) != len(tickers):
            return False, "Number of weights must match number of tickers"
        if any(w < 0 for w in weights):
            return False, "Weights cannot be negative"
        if abs(sum(weights) - 1.0) > 1e-6:
            return False, "Weights must sum to 1.0"

    return True, "Valid inputs"


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    return f"{value:.2%}"


def export_results(calculator, filename: str = None) -> str:
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"var_analysis_{timestamp}.csv"

    if calculator.rolling_returns is None:
        return "No data to export"

    results = pd.DataFrame({
        'Date': calculator.rolling_returns.index,
        'Portfolio_Returns': calculator.rolling_returns,
        'Portfolio_PnL': calculator.rolling_returns * calculator.portfolio_value
    })

    results.to_csv(filename, index=False)
    return filename