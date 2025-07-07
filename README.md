# Portfolio-Risk-Analysis

# Quantitative Risk Analytics Platform

A comprehensive Value at Risk (VaR) analysis system for professional portfolio risk management. This tool implements multiple VaR methodologies with backtesting, stress testing, and advanced risk metrics calculation.

##  Features

### Core Risk Analysis
- **Multi-Method VaR Calculation**: Historical, Parametric, and Monte Carlo simulation
- **Model Validation**: Backtesting with violation rate analysis
- **Stress Testing**: Scenario analysis for extreme market conditions
- **Risk Metrics**: Sharpe ratio, maximum drawdown, CVaR (Expected Shortfall)
- **Distribution Analysis**: Skewness and kurtosis detection for tail risk assessment

### Advanced Analytics
- **Real-time Data Integration**: Yahoo Finance API for live market data
- **Interactive Visualizations**: 4-panel risk dashboard with charts
- **Automated Risk Assessment**: Risk level classification and recommendations
- **Stop-Loss Recommendations**: Dynamic stop-loss calculations based on VaR
- **Comprehensive Reporting**: Professional risk analysis reports

##  Requirements

```bash
pip install pandas numpy yfinance scipy matplotlib
```

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Financial data retrieval
- **scipy**: Statistical functions
- **matplotlib**: Data visualization

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantitative-risk-analytics.git
cd quantitative-risk-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python var_analyzer.py
```

##  Usage

### Interactive Mode
```bash
python var_analyzer.py
```

Follow the prompts to:
1. Enter stock symbol (e.g., AAPL, TSLA, MSFT)
2. Input portfolio value
3. Select analysis period (1y, 2y, 5y, max)
4. Choose to display visualizations

### Programmatic Usage
```python
from var_analyzer import VaRAnalyzer

# Initialize analyzer
analyzer = VaRAnalyzer()

# Fetch data
analyzer.fetch_data('AAPL', '1y')

# Calculate VaR
var_results = analyzer.calculate_var(confidence=95, horizon=1)
print(var_results)

# Generate comprehensive report
analyzer.generate_report('AAPL', portfolio_value=1000000)

# Create visualizations
analyzer.create_visualization('AAPL')
```

##  Output Examples

### Console Report
```
============================================================
RISK ANALYSIS REPORT - AAPL
============================================================

PORTFOLIO OVERVIEW
Portfolio Value: $1,000,000.00
Analysis Period: 252 trading days
Daily Volatility: 2.34%
Annualized Volatility: 37.15%
Sharpe Ratio: 0.85
Max Drawdown: -28.45%

VALUE AT RISK (95% Confidence)
Historical  : $  23,400 (-2.34%)
Parametric  : $  25,100 (-2.51%)
Monte Carlo : $  24,200 (-2.42%)

STRESS TEST RESULTS
Market Crash   : $  200,000 (-20.0%)
Severe Recession: $  350,000 (-35.0%)
Black Swan     : $  500,000 (-50.0%)
Mild Correction: $  100,000 (-10.0%)

RISK ASSESSMENT
MODERATE RISK - Monitor closely
Recommended Stop Loss: 3.6%
```

### Visualization Dashboard
- **Price Movement**: Historical price chart
- **Returns Distribution**: Histogram with VaR markers
- **VaR Comparison**: Bar chart comparing all three methods
- **Drawdown Analysis**: Underwater equity curve

##  Methodology

### VaR Calculation Methods

1. **Historical VaR**
   - Uses actual historical return distribution
   - No distributional assumptions
   - Reflects actual market behavior

2. **Parametric VaR**
   - Assumes normal distribution
   - Uses mean and standard deviation
   - Faster computation

3. **Monte Carlo VaR**
   - Simulates 10,000 random scenarios
   - Flexible distributional assumptions
   - Robust for complex portfolios

### Risk Metrics

- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **CVaR (Expected Shortfall)**: Average loss beyond VaR threshold
- **Volatility**: Annualized standard deviation
- **Skewness**: Asymmetry of return distribution
- **Kurtosis**: Tail heaviness measure

##  Model Validation

### Backtesting Process
- Uses rolling 252-day windows
- Tests last 60 trading days
- Calculates violation rates
- Compares against expected rates
- Pass/Fail assessment

### Stress Testing Scenarios
- **Market Crash**: -20% shock
- **Severe Recession**: -35% shock
- **Black Swan Event**: -50% shock
- **Mild Correction**: -10% shock

## 📈 Risk Assessment Framework

### Risk Level Classification
- **LOW RISK**: VaR < 2% (Acceptable)
- **MODERATE RISK**: 2% ≤ VaR ≤ 5% (Monitor closely)
- **HIGH RISK**: VaR > 5% (Consider position reduction)

### Stop-Loss Calculation
```
Recommended Stop Loss = Average VaR × 1.5
```

