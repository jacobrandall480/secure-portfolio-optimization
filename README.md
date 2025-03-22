# Privacy-Preserving Portfolio Optimization

A privacy-preserving portfolio optimization system integrating Shamir’s Secret Sharing with linear algebra-based methods. This project compares secure optimization approaches to traditional Markowitz optimization using real financial data.

## Overview

This project was developed as part of a university-level applied linear algebra course. It enables institutional investors and asset managers to optimize portfolio allocations collaboratively without revealing sensitive financial data.

The implementation uses Python with real-time financial data pulled from Yahoo Finance (`yfinance`) and includes multiple optimization approaches and risk analysis techniques.

## Features

- Privacy-preserving computation via Shamir’s Secret Sharing
- Multiple optimization methods:
  - Unique solution (exact constraint satisfaction)
  - Overdetermined (least squares)
  - Underdetermined (minimum norm)
  - Traditional Markowitz optimization
- Risk metric analysis:
  - Historical and Monte Carlo VaR / CVaR
  - Annual volatility and Sharpe Ratio
- Data visualizations:
  - Efficient frontier
  - Portfolio composition
  - Return distributions
  - Correlation heatmap

## Project Structure

```
secure-portfolio-optimization/
├── optimization.py
├── MATH2015 Project Report.pdf
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/secure-portfolio-optimization.git
   cd secure-portfolio-optimization
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to test the optimization models and visualize results:

```bash
python optimization.py
```

The script will:
- Solve multiple optimization problems
- Output Sharpe ratios and risk metrics
- Generate visual plots of portfolio performance

## Requirements

- Python 3.7+
- See `requirements.txt` for full list of dependencies

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgments

Developed by Jacob Randall, Janel Perez, and Sean Williams as part of Columbia University's MATH2015 course on Linear Algebra and Probability.

## Authorship Note

This project was completed collaboratively for a course at Columbia University. The majority of the coding, system design, and analysis was completed by Jacob Randall.
