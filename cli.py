"""Command line interface for VaR Calculator"""

import argparse
from datetime import datetime
from .var_calculator import VaRCalculator
from .visualizer import VaRVisualizer
from .reporter import VaRReporter
from .utils import validate_inputs, export_results


def get_user_input() -> dict:
    """Interactive user input collection"""
    print("\n" + "=" * 50)
    print("INTERACTIVE VaR CALCULATOR")
    print("=" * 50)

    # Get tickers
    tickers_input = input("Enter tickers (space-separated): ").strip().upper()
    tickers = tickers_input.split()

    # Get optional weights
    weights_input = input(f"Enter weights for {len(tickers)} tickers (optional): ").strip()
    weights = None
    if weights_input:
        try:
            weights = [float(w) for w in weights_input.split(',')]
        except ValueError:
            print("Invalid weights format, using equal weights")

    # Get other parameters
    start_date = input("Start date (YYYY-MM-DD) [2020-01-01]: ").strip() or "2020-01-01"
    end_date = input("End date (YYYY-MM-DD) [today]: ").strip() or datetime.now().strftime('%Y-%m-%d')

    try:
        window = int(input("Rolling window in days [20]: ").strip() or "20")
        confidence = float(input("Confidence level [0.95]: ").strip() or "0.95")
        portfolio_value = float(input("Portfolio value in USD [100000]: ").strip() or "100000")
    except ValueError:
        print("Using default values for invalid inputs")
        window, confidence, portfolio_value = 20, 0.95, 100000

    return {
        'tickers': tickers,
        'weights': weights,
        'start_date': start_date,
        'end_date': end_date,
        'window': window,
        'confidence': confidence,
        'portfolio_value': portfolio_value
    }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Professional VaR Calculator')

    parser.add_argument('-t', '--tickers', default='AAPL MSFT GOOGL',
                        help='Space-separated ticker symbols')
    parser.add_argument('-w', '--weights',
                        help='Comma-separated weights (must sum to 1.0)')
    parser.add_argument('-s', '--start-date', default='2020-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('-e', '--end-date',
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--window', type=int, default=20,
                        help='Rolling window in days')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level (0.0-1.0)')
    parser.add_argument('--portfolio-value', type=float, default=100000,
                        help='Portfolio value in USD')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--export', metavar='FILENAME',
                        help='Export results to CSV file')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Run in interactive mode')

    return parser.parse_args()


def run_analysis(args):
    """Run VaR analysis with given arguments"""
    # Parse weights
    weights = None
    if args.weights:
        try:
            weights = [float(w) for w in args.weights.split(',')]
        except ValueError:
            print("Invalid weights format")
            return False

    # Validate inputs
    tickers = args.tickers.split()
    valid, message = validate_inputs(
        tickers, args.start_date, args.end_date,
        args.window, args.confidence, args.portfolio_value, weights
    )

    if not valid:
        print(f"Error: {message}")
        return False

    # Create calculator
    calculator = VaRCalculator(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        confidence=args.confidence,
        portfolio_value=args.portfolio_value,
        weights=weights
    )

    # Calculate VaR
    if not calculator.calculate_var():
        print("Failed to calculate VaR")
        return False

    # Generate report
    reporter = VaRReporter(calculator)
    reporter.generate_report()

    # Generate plots
    if not args.no_plots:
        visualizer = VaRVisualizer(calculator)
        visualizer.create_plots()

    # Export results
    if args.export:
        filename = export_results(calculator, args.export)
        print(f"Results exported to: {filename}")

    return True


def main():
    """Main CLI function"""
    try:
        args = parse_arguments()

        if args.interactive:
            inputs = get_user_input()

            # Convert dict to args-like object
            class Args:
                pass

            args = Args()
            for key, value in inputs.items():
                setattr(args, key.replace('_', '_'), value)
            args.tickers = ' '.join(inputs['tickers'])
            args.weights = ','.join(map(str, inputs['weights'])) if inputs['weights'] else None
            args.no_plots = False
            args.export = None

        run_analysis(args)

    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()