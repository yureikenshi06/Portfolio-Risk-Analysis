# Portfolio Risk Analysis


A comprehensive, modular Value at Risk (VaR) calculator for portfolio risk management.

## Features

- **Historical VaR**: Based on actual portfolio return distributions
- **Parametric VaR**: Using normal distribution assumptions
- **Professional Visualizations**: Comprehensive risk analysis charts
- **Flexible Portfolio Management**: Support for custom weights
- **Export Capabilities**: CSV export for further analysis
- **Modular Design**: Clean, maintainable codebase

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from var_calculator import quick_analysis

# Single stock analysis
calculator = quick_analysis(
    tickers='AAPL',
    start_date='2023-01-01',
    portfolio_value=50000
)

# Portfolio analysis
calculator = quick_analysis(
    tickers='AAPL MSFT GOOGL',
    weights=[0.4, 0.3, 0.3],
    start_date='2022-01-01',
    confidence=0.95
)
```

### Command Line Interface

```bash
# Interactive mode
python -m var_calculator.cli --interactive

# Direct analysis
python -m var_calculator.cli -t "AAPL MSFT GOOGL" -w "0.4,0.3,0.3" --confidence 0.95
```

## Project Structure

```
var_calculator/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── var_calculator.py    # Core VaR calculation engine
├── utils.py             # Utility functions
├── visualizer.py        # Visualization module
├── reporter.py          # Report generation
├── cli.py               # Command line interface
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
```

## API Reference

### VaRCalculator Class

```python
calculator = VaRCalculator(
    tickers=['AAPL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    window=20,
    confidence=0.95,
    portfolio_value=100000,
    weights=[0.6, 0.4]
)

# Calculate VaR
success = calculator.calculate_var()

# Get results
results = calculator.get_results()
```

### Quick Analysis Function

```python
from var_calculator import quick_analysis

calculator = quick_analysis(
    tickers='AAPL MSFT GOOGL',
    weights=[0.4, 0.3, 0.3],
    start_date='2022-01-01',
    window=30,
    confidence=0.99,
    portfolio_value=500000,
    show_plots=True,
    export_file='analysis_results.csv'
)
```

### Batch Analysis

```python
from var_calculator import batch_analysis

portfolios = [
    {'tickers': 'AAPL MSFT', 'weights': [0.6, 0.4]},
    {'tickers': 'GOOGL AMZN', 'weights': [0.5, 0.5]}
]

results = batch_analysis(
    portfolios,
    start_date='2023-01-01',
    confidence=0.95
)
```

## Configuration

Default settings can be modified in `config.py`:

```python
DEFAULT_CONFIG = {
    'start_date': '2020-01-01',
    'window': 20,
    'confidence': 0.95,
    'portfolio_value': 100000,
    'trading_days': 252
}
```

## Output

The calculator provides:

1. **Console Report**: Detailed risk metrics and interpretation
2. **Visualizations**: 4-panel chart showing:
   - Historical VaR distribution
   - Parametric VaR with normal overlay
   - Portfolio returns time series
   - VaR methods comparison
3. **CSV Export**: Time series data for further analysis

## Risk Metrics

- **Historical VaR**: Percentile-based loss estimate
- **Parametric VaR**: Normal distribution-based estimate
- **Portfolio Volatility**: Daily and annualized
- **Risk Level Assessment**: Low/Moderate/High classification
