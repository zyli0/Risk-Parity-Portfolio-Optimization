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
    "/Users/lizongyun/Desktop/中信建投/投研/covariance matrix.xls", sheet_name=0, header=0
)

portfolio = RiskParityPortfolio(covmatrix)
portfolio.design_portfolio()
print("Risk Parity Optimization Accomplished")
print("Portfolio Composition is: ")
print(portfolio.weights)
print(type(portfolio.weights))
print("Risk Contribution is: ")
print(portfolio.risk_contribution)

"""
Create a pie chart for the portfolio
"""
plt.rc("font", family='DengXian')
plt.font_manager.list_fonts()
assetnames = list(covmatrix)
portfoliocomp = portfolio.weights
myexplode = [0.1, 0.1, 0.1, 0.1]
mycolors = ['gold', 'skyblue', 'lightcoral', 'green']
plt.pie(portfoliocomp, labels=assetnames, autopct='%.1f%%', explode=myexplode, colors=mycolors)
plt.show()
