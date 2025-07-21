
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class VaRVisualizer:
    def __init__(self, calculator):
        self.calculator = calculator

    def create_plots(self):
        """Generate VaR analysis plots"""
        if self.calculator.rolling_returns is None:
            print("No data available for plotting")
            return

        returns_dollar = self.calculator.rolling_returns * self.calculator.portfolio_value

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Value at Risk Analysis', fontsize=14, fontweight='bold')

        # Historical VaR Distribution
        self._plot_historical_var(ax1, returns_dollar)

        # Parametric VaR with Normal Distribution
        self._plot_parametric_var(ax2, returns_dollar)

        # Portfolio Returns Time Series
        self._plot_returns_timeseries(ax3)

        # VaR Comparison
        self._plot_var_comparison(ax4)

        plt.tight_layout()
        plt.show()

    def _plot_historical_var(self, ax, returns_dollar):
        ax.hist(returns_dollar, bins=40, density=True, alpha=0.7,
                color='lightblue', edgecolor='black', linewidth=0.5)
        ax.axvline(-self.calculator.historical_var, color='red', linestyle='--',
                   linewidth=2, label=f'Historical VaR: ${self.calculator.historical_var:,.0f}')
        ax.set_title('Historical VaR Distribution')
        ax.set_xlabel('Portfolio P&L ($)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_parametric_var(self, ax, returns_dollar):
        ax.hist(returns_dollar, bins=40, density=True, alpha=0.7,
                color='lightgreen', edgecolor='black', linewidth=0.5)

        # Normal distribution overlay
        x = np.linspace(returns_dollar.min(), returns_dollar.max(), 100)
        mean, std = returns_dollar.mean(), returns_dollar.std()
        ax.plot(x, norm.pdf(x, mean, std), 'b-', linewidth=2, label='Normal Distribution')

        ax.axvline(-self.calculator.parametric_var, color='red', linestyle='--',
                   linewidth=2, label=f'Parametric VaR: ${self.calculator.parametric_var:,.0f}')
        ax.set_title('Parametric VaR vs Normal Distribution')
        ax.set_xlabel('Portfolio P&L ($)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_returns_timeseries(self, ax):
        """Plot portfolio returns time series"""
        ax.plot(self.calculator.portfolio_returns.index,
                self.calculator.portfolio_returns * 100,
                linewidth=1, alpha=0.8, color='darkblue')
        ax.set_title('Daily Portfolio Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns (%)')
        ax.grid(True, alpha=0.3)

    def _plot_var_comparison(self, ax):
        methods = ['Historical', 'Parametric']
        values = [self.calculator.historical_var, self.calculator.parametric_var]
        colors = ['red', 'orange']

        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('VaR Methods Comparison')
        ax.set_ylabel('VaR ($)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')