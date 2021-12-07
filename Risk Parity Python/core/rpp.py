import numpy as np
from core.riskfunction import RiskConcentrationFunction
from core.sco import CoordinateDescentOptimizer
from core.tools import to_array, to_column_matrix


class RiskParityPortfolio:

    def __init__(
        self,
        covariance,
        weights=None,
        budget=None,
        risk_concentration=None,
        risk_contribution=None,
    ):
        self.covariance = covariance
        self.weights = weights
        self.budget = budget
        self.risk_concentration = risk_concentration

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        if value.shape[0] != value.shape[1]:
            raise ValueError("covariance matrix is not a square matrix")
        else:
            try:
                self._covariance = np.atleast_2d(value)
            except Exception as e:
                raise e


    @property
    def num_assets(self):
        return self.covariance.shape[0]

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is None:
            try:
                self._weights = np.atleast_1d(value)
                self._weights = (np.ones(self.num_assets)) / self.num_assets
            except Exception as e:
                raise e
        else:
            self._weights = value

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        if value is None:
            self._budget = np.ones(self.num_assets) / self.num_assets
        else:
            value = np.array(value)
            if value.shape[0] != self.num_assets:
                raise ValueError("risk budget does not match number of assets")
            elif np.sum(value) != 1:
                raise ValueError("sum of risk budget is not 1")
            self._budget = value

    def validate(self):
        if self.covariance.shape[0] != self.budget.shape[0]:
            raise ValueError("covariance matrix and budget does not match")

    def get_variance(self):
        """
        returns assets variance contribution to portfolio
        """
        weights = self.weights
        assetvar = np.multiply((self.covariance @ weights), weights)
        return assetvar

    def get_volatility(self):
        """
        returns the total volatility of the portfolio
        """
        weights = self.weights
        volatility = np.sum(np.multiply((self.covariance @ weights), weights))
        return volatility

    @property
    def risk_contribution(self):
        weights = self.weights
        rc = np.multiply((self.covariance @ weights), weights)
        return rc / np.sum(rc)

    @property
    def risk_concentration(self):
        return self._risk_concentration

    @risk_concentration.setter
    def risk_concentration(self, value):
        if value is None:
            self._risk_concentration = RiskConcentrationFunction(self)

    def design_portfolio(self, **kwargs):
        self.sco = CoordinateDescentOptimizer(self, **kwargs)
        self.sco.solve()



    def volatility(self):
        pass

    def has_variance(self):
        pass

    def has_mean_return(self):
        pass










