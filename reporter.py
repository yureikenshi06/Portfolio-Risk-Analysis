
import numpy as np
from .utils import format_currency, format_percentage


class VaRReporter:
    def __init__(self, calculator):
        self.calculator = calculator

    def generate_report(self):
        print("\n" + "=" * 60)
        print("PORTFOLIO VALUE AT RISK REPORT")
        print("=" * 60)

        self._print_portfolio_composition()
        self._print_analysis_parameters()
        self._print_risk_metrics()
        self._print_risk_interpretation()

        print("=" * 60)

    def _print_portfolio_composition(self):
        print(f"\nPortfolio Composition:")
        for ticker, weight in zip(self.calculator.tickers, self.calculator.weights):
            print(f"  {ticker}: {format_percentage(weight)}")

    def _print_analysis_parameters(self):
        print(f"\nAnalysis Parameters:")
        print(f"  Period: {self.calculator.start_date} to {self.calculator.end_date}")
        print(f"  Rolling Window: {self.calculator.window} days")
        print(f"  Confidence Level: {format_percentage(self.calculator.confidence)}")
        print(f"  Portfolio Value: {format_currency(self.calculator.portfolio_value)}")

        if self.calculator.rolling_returns is not None:
            print(f"  Data Points: {len(self.calculator.rolling_returns)}")

    def _print_risk_metrics(self):
        print(f"\nRisk Metrics:")
        print(f"  Historical VaR: {format_currency(self.calculator.historical_var)}")
        print(f"  Parametric VaR: {format_currency(self.calculator.parametric_var)}")

        if self.calculator.portfolio_returns is not None:
            daily_vol = self.calculator.portfolio_returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            print(f"  Daily Portfolio Vol: {format_percentage(daily_vol)}")
            print(f"  Annualized Vol: {format_percentage(annual_vol)}")

    def _print_risk_interpretation(self):
        hist_pct = (self.calculator.historical_var / self.calculator.portfolio_value) * 100
        para_pct = (self.calculator.parametric_var / self.calculator.portfolio_value) * 100

        print(f"\nRisk Interpretation:")
        print(f"  Historical VaR: {hist_pct:.2f}% of portfolio value")
        print(f"  Parametric VaR: {para_pct:.2f}% of portfolio value")

        # Risk assessment
        if hist_pct > 10:
            risk_level = "High"
        elif hist_pct > 5:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        print(f"  Risk Level: {risk_level}")

    def generate_summary(self) -> dict:
        return {
            'tickers': self.calculator.tickers,
            'weights': self.calculator.weights.tolist(),
            'historical_var': self.calculator.historical_var,
            'parametric_var': self.calculator.parametric_var,
            'portfolio_value': self.calculator.portfolio_value,
            'confidence': self.calculator.confidence,
            'window': self.calculator.window
        }