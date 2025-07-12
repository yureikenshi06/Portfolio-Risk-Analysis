"""Main entry point for VaR Calculator"""

from datetime import datetime
from .var_calculator import VaRCalculator
from .visualizer import VaRVisualizer
from .reporter import VaRReporter
from .utils import validate_inputs, export_results


def quick_analysis(tickers='AAPL MSFT GOOGL', start_date='2020-01-01',
                   end_date=None, window=20, confidence=0.95,
                   portfolio_value=100000, weights=None,
                   show_plots=True, export_file=None):
    """Quick VaR analysis function"""

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Handle ticker input
    if isinstance(tickers, str):
        tickers = tickers.split()

    # Validate inputs
    valid, message = validate_inputs(
        tickers, start_date, end_date, window, confidence, portfolio_value, weights
    )

    if not valid:
        print(f"Error: {message}")
        return None

    # Create and run calculator
    calculator = VaRCalculator(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        window=window,
        confidence=confidence,
        portfolio_value=portfolio_value,
        weights=weights
    )

    if not calculator.calculate_var():
        print("Failed to calculate VaR")
        return None

    # Generate outputs
    reporter = VaRReporter(calculator)
    reporter.generate_report()

    if show_plots:
        visualizer = VaRVisualizer(calculator)
        visualizer.create_plots()

    if export_file:
        filename = export_results(calculator, export_file)
        print(f"Results exported to: {filename}")

    return calculator


def batch_analysis(portfolios: list, **kwargs):
    """Run VaR analysis for multiple portfolios"""
    results = []

    for i, portfolio in enumerate(portfolios):
        print(f"\n--- Portfolio {i + 1} ---")

        # Extract portfolio parameters
        tickers = portfolio.get('tickers', 'AAPL MSFT')
        weights = portfolio.get('weights', None)

        # Run analysis
        calculator = quick_analysis(
            tickers=tickers,
            weights=weights,
            show_plots=False,
            **kwargs
        )

        if calculator:
            results.append({
                'portfolio': i + 1,
                'tickers': calculator.tickers,
                'historical_var': calculator.historical_var,
                'parametric_var': calculator.parametric_var
            })

    return results


# Example usage functions
def demo_single_stock():
    """Demo: Single stock analysis"""
    return quick_analysis(
        tickers='AAPL',
        start_date='2023-01-01',
        confidence=0.95,
        portfolio_value=50000
    )


def demo_portfolio():
    """Demo: Multi-stock portfolio"""
    return quick_analysis(
        tickers='AAPL MSFT GOOGL AMZN',
        weights=[0.3, 0.3, 0.2, 0.2],
        start_date='2022-01-01',
        window=30,
        confidence=0.99
    )


def demo_batch():
    """Demo: Batch analysis"""
    portfolios = [
        {'tickers': 'AAPL MSFT', 'weights': [0.6, 0.4]},
        {'tickers': 'GOOGL AMZN TSLA', 'weights': [0.4, 0.3, 0.3]},
        {'tickers': 'SPY QQQ', 'weights': [0.7, 0.3]}
    ]

    return batch_analysis(
        portfolios,
        start_date='2023-01-01',
        confidence=0.95,
        portfolio_value=100000
    )


if __name__ == "__main__":
    # Run demo
    print("Running VaR Calculator Demo...")
    demo_portfolio()