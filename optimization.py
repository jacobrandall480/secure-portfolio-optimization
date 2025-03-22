import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class PortfolioOptimizer:
    def __init__(self, symbols: list, start_date: str = None, end_date: str = None):
        """Initialize portfolio optimizer with stock symbols and date range."""
        self.symbols = symbols
        
        # set default date range if not provided - use current date for end_date
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        # default start date is 5 years ago (5 * 365 days)
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            
        # fetch historical price data and calculate returns
        self.data = self._fetch_data(start_date, end_date)
        # calculate daily returns and remove any rows with missing values
        self.returns = self.data.pct_change().dropna()
        
        # annualize statistics by multiplying by 252 (trading days in a year)
        self.mean_returns = self.returns.mean() * 252  # annualized mean returns
        self.cov_matrix = self.returns.cov() * 252    # annualized covariance matrix
        
    def _fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for all symbols."""
        data = pd.DataFrame()
        
        for symbol in self.symbols:
            try:
                #  reate a Ticker object for each symbol
                stock = yf.Ticker(symbol)
                # get historical closing prices
                hist = stock.history(start=start_date, end=end_date)['Close']
                data[symbol] = hist
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
        return data
    
    def solve_unique_system(self, target_return: float) -> dict:
        """Solve portfolio optimization using matrix inversion."""
        n = len(self.symbols)
        
        # construct system matrix A combining:
        # - covariance matrix for risk equations
        # - mean returns for target return constraint
        # - ones vector for sum-to-one constraint
        A = np.vstack([
            self.cov_matrix,
            self.mean_returns,
            np.ones(n)
        ])
        
        # construct target vector b:
        # - zeros for risk equations
        # - target return value
        # - 1 for sum-to-one constraint
        b = np.array([0] * n + [target_return, 1])
        
        # solve normal equations (A^T A)x = A^T b using matrix inversion
        try:
            weights = np.linalg.solve(A.T @ A, A.T @ b)
            return dict(zip(self.symbols, weights))
        except np.linalg.LinAlgError:
            raise ValueError("No unique solution exists")

    def solve_overdetermined(self, target_return: float, risk_weight: float = 0.5) -> dict:
        """Solve portfolio optimization using least squares projection."""
        n = len(self.symbols)
        
        # construct overdetermined system with:
        # - weighted risk equations from covariance matrix
        # - regularization terms (identity matrix)
        # - return target and sum constraints
        A = np.vstack([
            np.sqrt(risk_weight) * self.cov_matrix, # weight risk terms
            np.sqrt(1-risk_weight) * np.eye(n), # weight regularization terms
            self.mean_returns, # return target constraint
            np.ones(n) # sum-to-one constraint
        ])
        
        # construct target vector
        b = np.array([0] * n + [0] * n + [target_return, 1])
        
        # solve using pseudo-inverse (minimum norm least squares solution)
        weights = np.linalg.pinv(A) @ b
        return dict(zip(self.symbols, weights))

    def solve_underdetermined(self, target_return: float) -> dict:
        """Solve portfolio optimization using minimum norm solution."""
        n = len(self.symbols)
        
        # create minimal constraint matrix with just:
        # - return target constraint
        # - sum-to-one constraint
        A = np.vstack([
            self.mean_returns,
            np.ones(n)
        ])
        
        b = np.array([target_return, 1])
        
        # find minimum norm solution using pseudo-inverse
        # formula: x = A^T (A A^T)^(-1) b
        weights = A.T @ np.linalg.pinv(A @ A.T) @ b
        return dict(zip(self.symbols, weights))
    
    def optimize_portfolio(self, risk_free_rate: float = 0.02, 
                         min_weight: float = 0.0, 
                         max_weight: float = 1.0) -> dict:
        """Optimize portfolio weights to maximize Sharpe ratio."""
        num_assets = len(self.symbols)
        
        def objective(weights):
            # calculate portfolio return as weighted sum of mean returns
            portfolio_return = np.sum(self.mean_returns * weights)
            # calculate portfolio standard deviation using quadratic form
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            # calculate Sharpe ratio (return - risk_free_rate) / std
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio # minimize negative Sharpe ratio (maximize Sharpe ratio)
        
        # constraint: weights must sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # create bounds tuple for each asset weight
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        # start with equal weights as initial guess
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # optimize using Sequential Least Squares Programming
        result = minimize(objective, initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        # calculate portfolio metrics using optimal weights
        optimal_weights = result.x
        portfolio_return = np.sum(self.mean_returns * optimal_weights)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, 
                                     np.dot(self.cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        
        return {
            'weights': dict(zip(self.symbols, optimal_weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def calculate_efficient_frontier(self, num_portfolios: int = 1000) -> dict:
        """Calculate efficient frontier points through Monte Carlo simulation."""
        returns = []
        volatilities = []
        weights_list = []
        
        num_assets = len(self.symbols)
        
        # generate random portfolios
        for _ in range(num_portfolios):
            # generate random weights and normalize to sum to 1
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            weights_list.append(weights)
            
            # calculate portfolio return and standard deviation
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            returns.append(portfolio_return)
            volatilities.append(portfolio_std)
        
        return {
            'returns': returns,
            'volatilities': volatilities,
            'weights': weights_list
        }
    
    def calculate_risk_metrics(self, weights: dict, 
                             confidence_level: float = 0.95,
                             simulation_days: int = 252) -> dict:
        """Calculate VaR and CVaR risk metrics."""
        weights_array = np.array(list(weights.values()))
        
        # calculate historical VaR and CVaR
        portfolio_returns = self.returns.dot(weights_array) # get weighted returns
        # VaR is the negative of the return at the specified percentile
        hist_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        # CVaR is the mean of returns below VaR
        hist_cvar = -portfolio_returns[portfolio_returns <= -hist_var].mean()
        
        # calculate parameters for Monte Carlo simulation
        mean_return = np.sum(self.mean_returns * weights_array) / 252 # daily mean
        portfolio_vol = np.sqrt(np.dot(weights_array.T, 
                                     np.dot(self.cov_matrix, weights_array))) / np.sqrt(252) # daily vol
        
        # generate Monte Carlo simulation of returns
        np.random.seed(42)  # For reproducibility
        sim_returns = np.random.normal(mean_return, portfolio_vol, simulation_days)
        
        # calculate Monte Carlo VaR and CVaR
        mc_var = -np.percentile(sim_returns, (1 - confidence_level) * 100)
        mc_cvar = -sim_returns[sim_returns <= -mc_var].mean()
        
        return {
            'historical_var': hist_var,
            'historical_cvar': hist_cvar,
            'monte_carlo_var': mc_var,
            'monte_carlo_cvar': mc_cvar,
            'annual_volatility': portfolio_vol * np.sqrt(252) # annualize volatility
        }
    
    def plot_analysis(self, optimal_weights: dict = None):
        """Create portfolio analysis visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        
        # plot efficient frontier
        ef_data = self.calculate_efficient_frontier()
        ax1.scatter(ef_data['volatilities'], ef_data['returns'], 
                   c='blue', alpha=0.5, label='Random Portfolios')
        
        # if optimal weights provided, plot optimal portfolio point
        if optimal_weights is not None:
            weights_array = np.array(list(optimal_weights.values()))
            opt_return = np.sum(self.mean_returns * weights_array)
            opt_vol = np.sqrt(np.dot(weights_array.T, 
                                   np.dot(self.cov_matrix, weights_array)))
            ax1.scatter(opt_vol, opt_return, c='red', marker='*', s=200,
                       label='Optimal Portfolio')
            
        ax1.set_title('Efficient Frontier')
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.legend()
    
        # plot correlation matrix heatmap
        sns.heatmap(self.returns.corr(), ax=ax2, cmap='coolwarm', annot=True)
        ax2.set_title('Correlation Matrix')
        
        
        # plot optimal portfolio composition pie chart
        if optimal_weights is not None:
            ax3.pie(optimal_weights.values(), labels=optimal_weights.keys(), 
                   autopct='%1.1f%%')
            ax3.set_title('Optimal Portfolio Composition')
        
        # plot portfolio returns distribution
        if optimal_weights is not None:
            weights_array = np.array(list(optimal_weights.values()))
            portfolio_returns = self.returns.dot(weights_array) # calculate weighted returns
            sns.histplot(portfolio_returns, kde=True, ax=ax4) # plot histogram with density
            ax4.set_title('Portfolio Returns Distribution')
            ax4.set_xlabel('Return')
            ax4.set_ylabel('Frequency')
            
        
        plt.tight_layout()
        plt.show()

# example usage
if __name__ == "__main__":
    # define portfolio parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    target_return = 0.15 # 15% target return
    

    optimizer = PortfolioOptimizer(symbols, start_date, end_date) # initialize portfolio optimizer
    
    # test different linear algebra methods
    print("\nTesting Linear Algebra Methods:")
    print("==============================")
    
    # 1. unique solution
    try:
        unique_weights = optimizer.solve_unique_system(target_return)
        print("\n1. Unique Solution Results:")
        for symbol, weight in unique_weights.items():
            print(f"{symbol}: {weight:.2%}")
            
        # calculate risk metrics for unique solution
        risk_metrics_unique = optimizer.calculate_risk_metrics(unique_weights)
        print("\nRisk Metrics (Unique Solution):")
        for metric, value in risk_metrics_unique.items():
            print(f"{metric}: {value:.2%}")
            
    except ValueError as e:
        print(f"\nUnable to find unique solution: {e}")

    # 2. overdetermined solution
    over_weights = optimizer.solve_overdetermined(target_return)
    print("\n2. Overdetermined Solution Results:")
    for symbol, weight in over_weights.items():
        print(f"{symbol}: {weight:.2%}")
        
    # calculate risk metrics for overdetermined solution
    risk_metrics_over = optimizer.calculate_risk_metrics(over_weights)
    print("\nRisk Metrics (Overdetermined Solution):")
    for metric, value in risk_metrics_over.items():
        print(f"{metric}: {value:.2%}")

    # 3. underdetermined solution
    under_weights = optimizer.solve_underdetermined(target_return)
    print("\n3. Underdetermined Solution Results:")
    for symbol, weight in under_weights.items():
        print(f"{symbol}: {weight:.2%}")
        
    # calculate risk metrics for underdetermined solution
    risk_metrics_under = optimizer.calculate_risk_metrics(under_weights)
    print("\nRisk Metrics (Underdetermined Solution):")
    for metric, value in risk_metrics_under.items():
        print(f"{metric}: {value:.2%}")
    
    # traditional Markowitz optimization
    print("\nPerforming Traditional Markowitz Optimization:")
    print("===========================================")
    
    optimal_portfolio = optimizer.optimize_portfolio(
        risk_free_rate=0.02,
        min_weight=0.05, # minimum 5% in each asset
        max_weight=0.4 # maximum 40% in each asset
    ) 
    
    print("\nOptimal Portfolio Weights:")
    for symbol, weight in optimal_portfolio['weights'].items():
        print(f"{symbol}: {weight:.2%}")

    print(f"\nExpected Annual Return: {optimal_portfolio['expected_return']:.2%}")
    print(f"Annual Volatility: {optimal_portfolio['volatility']:.2%}")
    print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
    
    # calculate risk metrics for Markowitz solution
    risk_metrics = optimizer.calculate_risk_metrics(optimal_portfolio['weights'])
    
    print("\nRisk Metrics (Markowitz Solution):")
    print(f"Historical VaR (95%): {risk_metrics['historical_var']:.2%}")
    print(f"Historical CVaR (95%): {risk_metrics['historical_cvar']:.2%}")
    print(f"Monte Carlo VaR (95%): {risk_metrics['monte_carlo_var']:.2%}")
    print(f"Monte Carlo CVaR (95%): {risk_metrics['monte_carlo_cvar']:.2%}")
    
    # compare all methods
    print("\nComparative Analysis:")
    print("====================")
    
    methods = {
        'Unique': unique_weights if 'unique_weights' in locals() else None,
        'Overdetermined': over_weights,
        'Underdetermined': under_weights,
        'Markowitz': optimal_portfolio['weights']
    }
    
    for method_name, weights in methods.items():
        if weights is not None:
            portfolio_return = np.sum(optimizer.mean_returns * np.array(list(weights.values())))
            portfolio_std = np.sqrt(np.dot(np.array(list(weights.values())).T, 
                                         np.dot(optimizer.cov_matrix, 
                                              np.array(list(weights.values())))))
            
            print(f"\n{method_name} Method:")
            print(f"Expected Return: {portfolio_return:.2%}")
            print(f"Volatility: {portfolio_std:.2%}")
            print(f"Sharpe Ratio: {(portfolio_return - 0.02) / portfolio_std:.2f}")
    
    # cenerate visualizations
    print("\nGenerating visualizations...")
    optimizer.plot_analysis(optimal_portfolio['weights'])
