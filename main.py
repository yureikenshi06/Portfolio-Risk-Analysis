
import argparse
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class VaRCalculator:
    """Professional Value at Risk Calculator"""

    def __init__(self, tickers: List[str], start_date: str, end_date: str,
                 window: int, confidence: float, portfolio_value: float,
                 weights: Optional[List[float]] = None):
        """Initialize VaR calculator with portfolio parameters"""
        self.tickers = [ticker.upper() for ticker in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.confidence = confidence
        self.portfolio_value = portfolio_value

        # Use provided weights or default to equal weights
        if weights is not None:
            if len(weights) != len(self.tickers):
                raise ValueError("Number of weights must match number of tickers.")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0.")
            self.weights = np.array(weights)
        else:
            self.weights = np.array([1.0 / len(self.tickers)] * len(self.tickers))

        # Results storage
        self.data = None
        self.returns = None
        self.portfolio_returns = None
        self.rolling_returns = None
        self.cov_matrix = None
        self.historical_var = None
        self.parametric_var = None

    def load_data(self) -> bool:
        """Load and process financial data"""
        try:
            print("Loading market data...")

            # Download data with error handling
            data = yf.download(self.tickers, start=self.start_date,
                             end=self.end_date, progress=False)

            if data.empty:
                print("ERROR: No data retrieved. Please check ticker symbols and date range.")
                return False

            # Handle single vs multiple tickers
            if len(self.tickers) == 1:
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].copy()
                    prices.columns = self.tickers
                else:
                    prices = data[['Close']].copy()
                    prices.columns = self.tickers
            else:
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close'].copy()
                else:
                    prices = data['Close'].copy()

            # Check for missing data
            if prices.isna().all().any():
                missing = prices.columns[prices.isna().all()].tolist()
                print(f"ERROR: No data available for: {', '.join(missing)}")
                return False

            # Forward fill missing values
            prices = prices.ffill().dropna()

            if len(prices) < self.window + 1:
                print(f"ERROR: Insufficient data. Need at least {self.window + 1} days.")
                return False

            # Calculate returns
            self.returns = np.log(prices / prices.shift(1)).dropna()

            # Portfolio returns
            self.portfolio_returns = (self.returns * self.weights).sum(axis=1)

            # Rolling window returns
            self.rolling_returns = self.portfolio_returns.rolling(self.window).sum().dropna()

            # Covariance matrix (annualized)
            self.cov_matrix = self.returns.cov() * 252

            print(f"✓ Successfully loaded {len(self.returns)} days of data")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load data - {str(e)}")
            return False

    def calculate_historical_var(self) -> None:
        """Calculate Historical VaR"""
        alpha = 1 - self.confidence
        percentile = alpha * 100

        # Calculate VaR as positive loss
        self.historical_var = -np.percentile(self.rolling_returns, percentile) * self.portfolio_value

    def calculate_parametric_var(self) -> None:
        """Calculate Parametric VaR using normal distribution"""
        # Portfolio volatility (annualized)
        portfolio_vol = np.sqrt(self.weights.T @ self.cov_matrix @ self.weights)

        # Scale to window period
        period_vol = portfolio_vol * np.sqrt(self.window / 252)

        # Z-score for confidence level
        z_score = norm.ppf(self.confidence)

        # VaR calculation
        self.parametric_var = abs(z_score) * period_vol * self.portfolio_value

    def calculate_var(self) -> bool:
        """Calculate both VaR methods"""
        if not self.load_data():
            return False

        print("Calculating VaR metrics...")
        self.calculate_historical_var()
        self.calculate_parametric_var()
        print("✓ VaR calculations completed")
        return True

    def generate_report(self) -> None:
        """Generate comprehensive VaR report"""
        print("\n" + "="*60)
        print("PORTFOLIO VALUE AT RISK REPORT")
        print("="*60)

        print(f"Portfolio Composition:")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {self.weights[i]:.1%}")

        print(f"\nAnalysis Parameters:")
        print(f"  Period: {self.start_date} to {self.end_date}")
        print(f"  Rolling Window: {self.window} days")
        print(f"  Confidence Level: {self.confidence:.1%}")
        print(f"  Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"  Data Points: {len(self.rolling_returns)}")

        print(f"\nRisk Metrics:")
        print(f"  Historical VaR: ${self.historical_var:,.2f}")
        print(f"  Parametric VaR: ${self.parametric_var:,.2f}")
        print(f"  Daily Portfolio Vol: {self.portfolio_returns.std():.2%}")
        print(f"  Annualized Vol: {self.portfolio_returns.std() * np.sqrt(252):.2%}")

        # Risk interpretation
        hist_pct = (self.historical_var / self.portfolio_value) * 100
        para_pct = (self.parametric_var / self.portfolio_value) * 100

        print(f"\nRisk Interpretation:")
        print(f"  Historical VaR: {hist_pct:.2f}% of portfolio value")
        print(f"  Parametric VaR: {para_pct:.2f}% of portfolio value")

        print("="*60)

    def create_plots(self) -> None:
        """Generate professional VaR visualization"""
        returns_dollar = self.rolling_returns * self.portfolio_value

        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Value at Risk Analysis', fontsize=16, fontweight='bold')

        # 1. Historical VaR Distribution
        ax1.hist(returns_dollar, bins=50, density=True, alpha=0.7,
                color='lightblue', edgecolor='black', linewidth=0.5)
        ax1.axvline(-self.historical_var, color='red', linestyle='--', linewidth=2,
                   label=f'Historical VaR: ${self.historical_var:,.0f}')
        ax1.set_title('Historical VaR Distribution')
        ax1.set_xlabel('Portfolio P&L ($)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Parametric VaR with Normal Overlay
        ax2.hist(returns_dollar, bins=50, density=True, alpha=0.7,
                color='lightgreen', edgecolor='black', linewidth=0.5)

        # Normal distribution overlay
        x = np.linspace(returns_dollar.min(), returns_dollar.max(), 100)
        mean = returns_dollar.mean()
        std = returns_dollar.std()
        normal_dist = norm.pdf(x, mean, std)
        ax2.plot(x, normal_dist, 'b-', linewidth=2, label='Normal Distribution')

        ax2.axvline(-self.parametric_var, color='red', linestyle='--', linewidth=2,
                   label=f'Parametric VaR: ${self.parametric_var:,.0f}')
        ax2.set_title('Parametric VaR with Normal Distribution')
        ax2.set_xlabel('Portfolio P&L ($)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Rolling Portfolio Returns
        ax3.plot(self.portfolio_returns.index, self.portfolio_returns * 100,
                linewidth=1, alpha=0.8, color='darkblue')
        ax3.set_title('Daily Portfolio Returns')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Returns (%)')
        ax3.grid(True, alpha=0.3)

        # 4. VaR Comparison
        methods = ['Historical', 'Parametric']
        var_values = [self.historical_var, self.parametric_var]
        colors = ['red', 'orange']

        bars = ax4.bar(methods, var_values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('VaR Method Comparison')
        ax4.set_ylabel('VaR ($)')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, var_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def export_results(self, filename: str = None) -> None:
        """Export results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"var_analysis_{timestamp}.csv"

        results = {
            'Date': self.rolling_returns.index,
            'Portfolio_Returns': self.rolling_returns,
            'Portfolio_PnL': self.rolling_returns * self.portfolio_value
        }

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"✓ Results exported to {filename}")


def get_user_input() -> dict:
    """Get user input interactively"""
    print("\n" + "="*60)
    print("INTERACTIVE VaR CALCULATOR SETUP")
    print("="*60)

    inputs = {}

    # Get tickers
    print("\n1. PORTFOLIO TICKERS")
    print("Enter stock tickers separated by spaces (e.g., AAPL MSFT GOOGL)")
    print("Popular examples: AAPL MSFT GOOGL AMZN TSLA SPY QQQ")
    while True:
        tickers_input = input("Enter tickers: ").strip().upper()
        if tickers_input:
            tickers = tickers_input.split()
            break
        print("Please enter at least one ticker symbol.")
    inputs['tickers'] = tickers

    # Get weights
    print("\n1b. PORTFOLIO WEIGHTS")
    print(f"Enter weights for each ticker (comma-separated, must sum to 1.0):")
    print(f"Example for {len(tickers)} tickers: " + ", ".join(["0.{}"]*len(tickers)))
    while True:
        weights_input = input(f"Enter {len(tickers)} weights: ").strip()
        try:
            weights = [float(w) for w in weights_input.split(',')]
            if len(weights) != len(tickers):
                print(f"Please enter exactly {len(tickers)} weights.")
                continue
            if any(w < 0 for w in weights):
                print("Weights cannot be negative.")
                continue
            if abs(sum(weights) - 1.0) > 1e-6:
                print("Weights must sum to 1.0.")
                continue
            inputs['weights'] = weights
            break
        except Exception:
            print("Please enter valid numbers separated by commas.")

    # Get date range
    print("\n2. DATE RANGE")
    print("Enter the analysis period")

    # Start date
    while True:
        start_date = input("Start date (YYYY-MM-DD) [default: 2020-01-01]: ").strip()
        if not start_date:
            start_date = "2020-01-01"
        try:
            pd.to_datetime(start_date)
            inputs['start_date'] = start_date
            break
        except:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # End date
    while True:
        end_date = input("End date (YYYY-MM-DD) [default: today]: ").strip()
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        try:
            pd.to_datetime(end_date)
            inputs['end_date'] = end_date
            break
        except:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Get rolling window
    print("\n3. ROLLING WINDOW")
    print("Number of days for rolling window calculation")
    print("Common values: 1 (daily), 5 (weekly), 20 (monthly), 252 (annual)")
    while True:
        try:
            window = input("Rolling window in days [default: 20]: ").strip()
            if not window:
                window = 20
            else:
                window = int(window)
            if window > 0:
                inputs['window'] = window
                break
            else:
                print("Window must be positive.")
        except ValueError:
            print("Please enter a valid number.")

    # Get confidence level
    print("\n4. CONFIDENCE LEVEL")
    print("VaR confidence level (common values: 0.95, 0.99)")
    while True:
        try:
            confidence = input("Confidence level [default: 0.95]: ").strip()
            if not confidence:
                confidence = 0.95
            else:
                confidence = float(confidence)
            if 0 < confidence < 1:
                inputs['confidence'] = confidence
                break
            else:
                print("Confidence level must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid number.")

    # Get portfolio value
    print("\n5. PORTFOLIO VALUE")
    print("Total portfolio value in USD")
    while True:
        try:
            portfolio_value = input("Portfolio value in USD [default: 100000]: ").strip()
            if not portfolio_value:
                portfolio_value = 100000
            else:
                portfolio_value = float(portfolio_value)
            if portfolio_value > 0:
                inputs['portfolio_value'] = portfolio_value
                break
            else:
                print("Portfolio value must be positive.")
        except ValueError:
            print("Please enter a valid number.")

    # Get output preferences
    print("\n6. OUTPUT PREFERENCES")
    inputs['show_plots'] = input("Show plots? [Y/n]: ").strip().lower() not in ['n', 'no']

    export_choice = input("Export results to CSV? [y/N]: ").strip().lower()
    if export_choice in ['y', 'yes']:
        filename = input("Enter filename [default: auto-generated]: ").strip()
        inputs['export_file'] = filename if filename else None
    else:
        inputs['export_file'] = None

    return inputs


def display_summary(inputs: dict) -> None:
    """Display summary of user inputs"""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Tickers: {' '.join(inputs['tickers'])}")
    if 'weights' in inputs:
        print("Weights: " + ', '.join(f"{w:.2%}" for w in inputs['weights']))
    print(f"Date Range: {inputs['start_date']} to {inputs['end_date']}")
    print(f"Rolling Window: {inputs['window']} days")
    print(f"Confidence Level: {inputs['confidence']:.1%}")
    print(f"Portfolio Value: ${inputs['portfolio_value']:,.2f}")
    print(f"Show Plots: {'Yes' if inputs['show_plots'] else 'No'}")
    print(f"Export to CSV: {'Yes' if inputs['export_file'] is not None else 'No'}")
    print("="*60)


def confirm_analysis() -> bool:
    """Confirm before running analysis"""
    while True:
        confirm = input("\nProceed with analysis? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Interactive Professional Value at Risk Calculator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--interactive',
                       action='store_true',
                       help='Run in interactive mode with user prompts')

    parser.add_argument('-t', '--tickers',
                       default='AAPL MSFT GOOGL',
                       help='Space-separated ticker symbols')

    parser.add_argument('-wts', '--weights',
                       default=None,
                       help='Comma-separated weights for tickers (must sum to 1.0)')

    parser.add_argument('-s', '--start-date',
                       default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')

    parser.add_argument('-e', '--end-date',
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date (YYYY-MM-DD)')

    parser.add_argument('-w', '--window',
                       type=int, default=20,
                       help='Rolling window in days')

    parser.add_argument('-c', '--confidence',
                       type=float, default=0.95,
                       help='Confidence level (0.0-1.0)')

    parser.add_argument('-v', '--portfolio-value',
                       type=float, default=100000,
                       help='Portfolio value in USD')

    parser.add_argument('--no-plots',
                       action='store_true',
                       help='Skip plot generation')

    parser.add_argument('--export',
                       metavar='FILENAME',
                       help='Export results to CSV file')

    # Handle Jupyter environment
    try:
        return parser.parse_args()
    except SystemExit:
        # If running in Jupyter, default to interactive mode
        class DefaultArgs:
            interactive = True
            tickers = 'AAPL MSFT GOOGL'
            weights = None
            start_date = '2020-01-01'
            end_date = datetime.now().strftime('%Y-%m-%d')
            window = 20
            confidence = 0.95
            portfolio_value = 100000
            no_plots = False
            export = None
        return DefaultArgs()


def validate_inputs(inputs: dict) -> bool:
    """Validate input parameters"""
    try:
        # Validate dates
        pd.to_datetime(inputs['start_date'])
        pd.to_datetime(inputs['end_date'])

        # Validate numeric parameters
        if inputs['window'] <= 0:
            print("ERROR: Window must be positive")
            return False

        if not 0 < inputs['confidence'] < 1:
            print("ERROR: Confidence must be between 0 and 1")
            return False

        if inputs['portfolio_value'] <= 0:
            print("ERROR: Portfolio value must be positive")
            return False

        # Validate weights if provided
        if 'weights' in inputs and inputs['weights'] is not None:
            weights = inputs['weights']
            if len(weights) != len(inputs['tickers']):
                print("ERROR: Number of weights does not match number of tickers.")
                return False
            if any(w < 0 for w in weights):
                print("ERROR: Weights cannot be negative.")
                return False
            if abs(sum(weights) - 1.0) > 1e-6:
                print("ERROR: Weights must sum to 1.0.")
                return False

        return True

    except Exception as e:
        print(f"ERROR: Invalid input - {str(e)}")
        return False


def main():
    """Main execution function"""
    print("Professional Interactive VaR Calculator v1.0")
    print("-" * 50)

    # Parse arguments
    try:
        args = parse_arguments()
    except:
        args = None

    # Determine if interactive mode
    if args is None or args.interactive or not hasattr(args, 'tickers'):
        # Interactive mode
        inputs = get_user_input()
        display_summary(inputs)

        if not confirm_analysis():
            print("Analysis cancelled.")
            return

        # Convert to expected format
        tickers = inputs['tickers']
        weights = inputs.get('weights')
        start_date = inputs['start_date']
        end_date = inputs['end_date']
        window = inputs['window']
        confidence = inputs['confidence']
        portfolio_value = inputs['portfolio_value']
        show_plots = inputs['show_plots']
        export_file = inputs['export_file']

    else:
        # Command line mode
        weights = None
        if args.weights:
            try:
                weights = [float(w) for w in args.weights.split(',')]
            except Exception:
                print("ERROR: Invalid weights format. Use comma-separated floats.")
                return

        inputs = {
            'tickers': args.tickers.split(),
            'weights': weights,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'window': args.window,
            'confidence': args.confidence,
            'portfolio_value': args.portfolio_value,
            'show_plots': not args.no_plots,
            'export_file': args.export
        }

        if not validate_inputs(inputs):
            return

        tickers = inputs['tickers']
        weights = inputs['weights']
        start_date = inputs['start_date']
        end_date = inputs['end_date']
        window = inputs['window']
        confidence = inputs['confidence']
        portfolio_value = inputs['portfolio_value']
        show_plots = inputs['show_plots']
        export_file = inputs['export_file']

    # Initialize calculator
    try:
        calculator = VaRCalculator(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            window=window,
            confidence=confidence,
            portfolio_value=portfolio_value,
            weights=weights
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Calculate VaR
    if not calculator.calculate_var():
        return

    # Generate report
    calculator.generate_report()

    # Generate plots
    if show_plots:
        print("\nGenerating visualizations...")
        calculator.create_plots()

    # Export results
    if export_file:
        calculator.export_results(export_file)

    print("\n✓ Analysis completed successfully")


def run_analysis(tickers='AAPL MSFT GOOGL', start_date='2020-01-01',
                end_date=None, window=20, confidence=0.95,
                portfolio_value=100000, weights=None,
                show_plots=True, export_file=None):

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Handle different ticker input formats
    if isinstance(tickers, str):
        tickers = tickers.split()

    print("Professional VaR Calculator v1.0")
    print("-" * 40)

    # Initialize calculator
    try:
        calculator = VaRCalculator(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            window=window,
            confidence=confidence,
            portfolio_value=portfolio_value,
            weights=weights
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # Calculate VaR
    if not calculator.calculate_var():
        return None

    # Generate report
    calculator.generate_report()

    # Generate plots
    if show_plots:
        print("\nGenerating visualizations...")
        calculator.create_plots()

    # Export results
    if export_file:
        calculator.export_results(export_file)

    print("\n✓ Analysis completed successfully")
    return calculator


def interactive_mode():
    """Run calculator in interactive mode"""
    main()


if __name__ == "__main__":
    main()
