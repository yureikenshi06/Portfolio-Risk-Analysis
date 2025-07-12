"""Configuration settings for VaR Calculator"""

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Default parameters
DEFAULT_CONFIG = {
    'start_date': '2020-01-01',
    'window': 20,
    'confidence': 0.95,
    'portfolio_value': 100000,
    'trading_days': 252
}

# Validation limits
VALIDATION_LIMITS = {
    'min_confidence': 0.01,
    'max_confidence': 0.99,
    'min_window': 1,
    'max_window': 1000,
    'min_portfolio_value': 1
}