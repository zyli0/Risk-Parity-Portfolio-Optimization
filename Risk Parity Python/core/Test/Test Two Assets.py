import numpy as np
from core.riskfunction import RiskConcentrationFunction
from core.sco import CoordinateDescentOptimizer
from core.rpp import RiskParityPortfolio

"""
Test Run for Two Assets
"""
array1 = np.array([0.006889, -0.00009279])
array2 = np.array([-0.00009729, 0.00007396])


matrix = np.vstack([array1, array2])

array3 = np.array([0.6, 0.4])

portfolio = RiskParityPortfolio(matrix)
portfolio.design_portfolio()
print("Portfolio Composition is: ")
print(portfolio.weights)
print("Risk Contribution is: ")
print(portfolio.risk_contribution)




