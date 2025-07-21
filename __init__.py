"""Professional VaR Calculator Package"""

__version__ = "1.0.0"
__author__ = "VaR Analytics Team"
__description__ = "Professional Value at Risk Calculator"

from .var_calculator import VaRCalculator
from .visualizer import VaRVisualizer
from .reporter import VaRReporter
from .main import quick_analysis, batch_analysis, demo_single_stock, demo_portfolio

# Public API
__all__ = [
    'VaRCalculator',
    'VaRVisualizer',
    'VaRReporter',
    'quick_analysis',
    'batch_analysis',
    'demo_single_stock',
    'demo_portfolio'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'var_calculator',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'requires': [
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'yfinance>=0.1.70',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0'
    ]
}