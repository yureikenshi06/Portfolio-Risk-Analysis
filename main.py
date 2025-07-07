import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class VaRAnalyzer:
    """Professional Value at Risk Analysis Tool"""

    def __init__(self):
        self.returns = None
        self.data = None

    def fetch_data(self, symbol, period='1y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period)

            if self.data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")

            # Calculate returns
            self.data['Returns'] = self.data['Close'].pct_change()
            self.returns = self.data['Returns'].dropna()

            print(f"Fetched {len(self.returns)} days of data for {symbol}")
            print(
                f"Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            return True

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False

    def calculate_var(self, confidence=95, horizon=1):
        """Calculate VaR using three methods"""
        if self.returns is None:
            raise ValueError("No data available")

        alpha = (100 - confidence) / 100

        # Historical VaR
        hist_var = np.percentile(self.returns, alpha * 100) * np.sqrt(horizon)

        # Parametric VaR
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        param_var = (mean_return + norm.ppf(alpha) * std_return) * np.sqrt(horizon)

        # Monte Carlo VaR
        np.random.seed(42)
        mc_returns = np.random.normal(mean_return, std_return, 10000)
        mc_var = np.percentile(mc_returns, alpha * 100) * np.sqrt(horizon)

        return {
            'Historical': hist_var,
            'Parametric': param_var,
            'Monte Carlo': mc_var
        }

    def backtest_var(self, method='Historical', confidence=95, test_days=60):
        """Backtest VaR model performance"""
        if len(self.returns) < test_days + 252:
            return None

        violations = 0
        total_tests = 0

        for i in range(test_days):
            # Use rolling window for estimation
            train_data = self.returns.iloc[i:i + 252]
            test_return = self.returns.iloc[i + 252]

            # Calculate VaR based on method
            if method == 'Historical':
                var = np.percentile(train_data, (100 - confidence) / 100 * 100)
            elif method == 'Parametric':
                var = train_data.mean() + norm.ppf((100 - confidence) / 100) * train_data.std()
            else:  # Monte Carlo
                mc_returns = np.random.normal(train_data.mean(), train_data.std(), 1000)
                var = np.percentile(mc_returns, (100 - confidence) / 100 * 100)

            # Check violation
            if test_return < var:
                violations += 1
            total_tests += 1

        violation_rate = violations / total_tests * 100
        expected_rate = 100 - confidence

        return {
            'Violations': violations,
            'Total Tests': total_tests,
            'Violation Rate': violation_rate,
            'Expected Rate': expected_rate,
            'Accuracy': 'PASS' if abs(violation_rate - expected_rate) < 2 else 'FAIL'
        }

    def stress_test(self, portfolio_value=1000000):
        """Perform stress testing scenarios"""
        scenarios = {
            'Market Crash': -0.20,
            'Severe Recession': -0.35,
            'Black Swan': -0.50,
            'Mild Correction': -0.10
        }

        results = {}
        for name, shock in scenarios.items():
            loss = portfolio_value * abs(shock)
            results[name] = {
                'Shock': f"{shock:.1%}",
                'Loss': f"${loss:,.0f}"
            }

        return results

    def calculate_risk_metrics(self):
        """Calculate additional risk metrics"""
        # Sharpe Ratio (assuming risk-free rate = 0)
        sharpe = (self.returns.mean() / self.returns.std()) * np.sqrt(252)

        # Maximum Drawdown
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        # VaR and CVaR at 95%
        var_95 = np.percentile(self.returns, 5)
        cvar_95 = self.returns[self.returns <= var_95].mean()

        return {
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'VaR 95%': var_95,
            'CVaR 95%': cvar_95,
            'Volatility': self.returns.std() * np.sqrt(252),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis()
        }

    def generate_report(self, symbol, portfolio_value=1000000):
        """Generate comprehensive risk report"""
        print(f"\n{'=' * 60}")
        print(f"RISK ANALYSIS REPORT - {symbol.upper()}")
        print(f"{'=' * 60}")

        # Portfolio Overview
        risk_metrics = self.calculate_risk_metrics()
        print(f"\nPORTFOLIO OVERVIEW")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Analysis Period: {len(self.returns)} trading days")
        print(f"Daily Volatility: {self.returns.std():.2%}")
        print(f"Annualized Volatility: {risk_metrics['Volatility']:.2%}")
        print(f"Sharpe Ratio: {risk_metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {risk_metrics['Max Drawdown']:.2%}")

        # VaR Analysis
        print(f"\nVALUE AT RISK (95% Confidence)")
        var_results = self.calculate_var(95, 1)

        for method, var_pct in var_results.items():
            var_dollar = abs(var_pct) * portfolio_value
            print(f"{method:12s}: ${var_dollar:8,.0f} ({var_pct:.2%})")

        # CVaR (Expected Shortfall)
        print(f"\nCONDITIONAL VAR (Expected Shortfall)")
        cvar_dollar = abs(risk_metrics['CVaR 95%']) * portfolio_value
        print(f"CVaR (95%): ${cvar_dollar:,.0f} ({risk_metrics['CVaR 95%']:.2%})")

        # Backtesting
        print(f"\nMODEL VALIDATION")
        for method in ['Historical', 'Parametric']:
            backtest = self.backtest_var(method, 95, 60)
            if backtest:
                print(f"{method:12s}: {backtest['Violation Rate']:.1f}% violations ({backtest['Accuracy']})")

        # Stress Testing
        print(f"\nSTRESS TEST RESULTS")
        stress_results = self.stress_test(portfolio_value)
        for scenario, result in stress_results.items():
            print(f"{scenario:15s}: {result['Loss']:>12s} ({result['Shock']})")

        # Risk Assessment
        avg_var = np.mean(list(var_results.values()))
        risk_level = abs(avg_var)

        print(f"\nRISK ASSESSMENT")
        if risk_level > 0.05:
            print("HIGH RISK - Consider position reduction")
        elif risk_level > 0.02:
            print("MODERATE RISK - Monitor closely")
        else:
            print("LOW RISK - Acceptable risk level")

        print(f"Recommended Stop Loss: {risk_level * 1.5:.1%}")

        # Distribution Analysis
        print(f"\nDISTRIBUTION ANALYSIS")
        print(f"Skewness: {risk_metrics['Skewness']:.2f}")
        print(f"Kurtosis: {risk_metrics['Kurtosis']:.2f}")

        if risk_metrics['Skewness'] < -0.5:
            print("WARNING: Negative skew - Higher probability of extreme losses")
        if risk_metrics['Kurtosis'] > 3:
            print("WARNING: Fat tails - Higher probability of extreme events")

    def create_visualization(self, symbol):
        """Create risk visualization dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Risk Analysis Dashboard - {symbol.upper()}', fontsize=16, fontweight='bold')

        # Price chart
        ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=1)
        ax1.set_title('Price Movement')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)

        # Returns distribution
        ax2.hist(self.returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(self.returns.mean(), color='red', linestyle='--', label='Mean')
        ax2.axvline(np.percentile(self.returns, 5), color='orange', linestyle='--', label='5% VaR')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # VaR comparison
        var_results = self.calculate_var(95, 1)
        methods = list(var_results.keys())
        var_values = [abs(val) * 100 for val in var_results.values()]

        bars = ax3.bar(methods, var_values, color=['red', 'orange', 'gold'], alpha=0.7)
        ax3.set_title('VaR Comparison (95% Confidence)')
        ax3.set_ylabel('VaR (%)')
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, var_values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.2f}%', ha='center', va='bottom')

        # Drawdown chart
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        ax4.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
        ax4.plot(drawdown.index, drawdown * 100, 'r-', linewidth=1)
        ax4.set_title('Drawdown Analysis')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    print("Professional VaR Analysis Tool")
    print("=" * 40)

    # Get user inputs
    symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
    portfolio_value = float(input("Enter portfolio value ($): "))

    # Period options
    print("\nData period options:")
    print("1. 1 year (1y)")
    print("2. 2 years (2y)")
    print("3. 5 years (5y)")
    print("4. Max available (max)")

    period_map = {'1': '1y', '2': '2y', '3': '5y', '4': 'max'}
    period_choice = input("Select period (1-4): ")
    period = period_map.get(period_choice, '1y')

    # Initialize analyzer
    analyzer = VaRAnalyzer()

    # Fetch data and analyze
    if analyzer.fetch_data(symbol, period):
        analyzer.generate_report(symbol, portfolio_value)

        # Optional visualization
        show_charts = input("\nShow charts? (y/n): ").lower() == 'y'
        if show_charts:
            analyzer.create_visualization(symbol)
    else:
        print("Analysis failed. Please check symbol.")


if __name__ == "__main__":
    main()