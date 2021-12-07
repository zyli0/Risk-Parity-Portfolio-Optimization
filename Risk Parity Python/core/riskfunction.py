import jax.numpy as np
from jax import grad, jit, jacfwd


class RiskConcentrationFunction:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    # returns the jacobian matrix for risk_concentration with respect to weights
    def jacobian_risk_concentration_vector(self):
        return jit(jacfwd(self.risk_concentration_vector))

    # returns an array of distances between real TRCi and ideal TRCi
    def risk_concentration_vector(self, portfolio_weights):
        curTRC = np.multiply((self.portfolio.covariance @ portfolio_weights), portfolio_weights)
        curTRC = curTRC / np.sum(curTRC)
        return curTRC - self.portfolio.budget

    # returns the sum of the squared differences(real TRCi - ideal TRCi)
    # our goal is to minimize this value through coordinate descent
    def evaluate(self):
        return np.sum(np.square(self.risk_concentration_vector(self.portfolio.weights)))


class TotalRiskContribMinusBudget(RiskConcentrationFunction):
    pass






