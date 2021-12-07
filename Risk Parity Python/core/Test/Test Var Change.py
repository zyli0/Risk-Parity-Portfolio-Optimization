import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.riskfunction import RiskConcentrationFunction
from core.sco import CoordinateDescentOptimizer
from core.rpp import RiskParityPortfolio

"""
Test Run for Four Assets
"""

covmatrix = pd.read_excel(
    "/Users/lizongyun/Desktop/中信建投/投研/covariance matrix.xls", sheet_name=1, header=0
)

portfolio = RiskParityPortfolio(covmatrix)
portfolio.design_portfolio()
print("Risk Parity Optimization Accomplished")
print("Portfolio Composition is: ")
print(portfolio.weights)
print("Risk Contribution is: ")
print(portfolio.risk_contribution)

