import numpy as np
from tqdm import tqdm
import quadprog


class CoordinateDescentOptimizerValidator:
    def __init(self):
        pass

    @property
    def zeta(self):
        return self._zeta

    @zeta.setter
    def zeta(self, value):
        if value < 0:
            raise ValueError(
                "zeta has to be a positive real number"
            )
        else:
            try:
                self._zeta = float(value)
            except Exception as e:
                raise e

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if not (0 < value < 1):
            raise ValueError(
                "theta has to be a real number between 0 and 1"
            )
        else:
            try:
                self._theta = float(value)
            except Exception as e:
                raise e

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        if value < 0:
            raise ValueError(
                "maxiter has to be a positive real number"
            )
        else:
            try:
                self._maxiter = int(value)
            except Exception as e:
                raise e

    @property
    def wtol(self):
        return self._wtol

    @wtol.setter
    def wtol(self, value):
        if value < 0:
            raise ValueError(
                "wtol has to be a positive real number"
            )
        else:
            try:
                self._wtol = float(value)
            except Exception as e:
                raise e

    @property
    def contol(self):
        return self._contol

    @contol.setter
    def contol(self, value):
        if value < 0:
            raise ValueError(
                "contol has to be a positive real number"
            )
        else:
            try:
                self._contol = float(value)
            except Exception as e:
                raise e


class CoordinateDescentOptimizer:

    def __init__(
        self,
        portfolio,
        theta=0.9,
        zeta=1e-5,
        contol=1e-4,
        wtol=1e-4,
        maxiter=1000,
        extreme=0,
        tau=None,
        Cmat=None,
        Dmat=None,
        cvec=None,
        dvec=None,
    ):
        self.portfolio = portfolio
        self.tau = tau or 0.05 * np.sum(np.diag(self.portfolio.covariance)) / (
            2 * self.portfolio.num_assets
        )
        sco_validator = CoordinateDescentOptimizerValidator()
        self.theta = sco_validator.theta = theta
        self.zeta = sco_validator.zeta = zeta
        self.contol = sco_validator.contol = contol
        self.wtol = sco_validator.wtol = wtol
        self.maxiter = sco_validator.maxiter = maxiter
        self.extreme = extreme
        self.rvec = self.portfolio.risk_concentration.risk_concentration_vector
        self.Amat = (self.portfolio.risk_concentration.jacobian_risk_concentration_vector())
        self._tauI = self.tau * np.eye(self.portfolio.num_assets)

        # **kwargs
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.cvec = cvec
        self.dvec = dvec
        self.CCmat = np.vstack((self.Cmat, self.Dmat)).T
        self.bvec = np.concatenate((self.cvec, self.dvec))
        self.meq = self.Cmat.shape[0]
        self._funk = self.get_objective_function_value()
        self.objective_function = [self._funk]

    def get_objective_function_value(self):
        obj = self.portfolio.risk_concentration.evaluate()
        return obj

    @property
    def Cmat(self):
        return self._Cmat

    @Cmat.setter
    def Cmat(self, value):
        if value is None:
            self._Cmat = np.atleast_2d(np.ones(self.portfolio.num_assets))
        elif np.atleast_2d(value).shape[1] == self.portfolio.num_assets:
            self._Cmat = np.atleast_2d(value)
        else:
            raise ValueError(
                "Cmat shape does not agree with number of assets"
            )

    @property
    def Dmat(self):
        return self._Dmat

    @Dmat.setter
    def Dmat(self, value):
        if value is None:
            self._Dmat = np.eye(self.portfolio.num_assets)
        elif np.atleast_2d(value).shape[1] == self.portfolio.num_assets:
            self._Dmat = -np.atleast_2d(value)
        else:
            raise ValueError(
                "Dmat shape does not agree with number of assets"
            )


    @property
    def cvec(self):
        return self._cvec

    @cvec.setter
    def cvec(self, value):
        if value is None:
            self._cvec = np.array([1.0])
        elif len(value) == self.Cmat.shape[0]:
            self._cvec = value
        else:
            raise ValueError(
                "cvec shape does not match Cmat shape"
            )

    @property
    def dvec(self):
        return self._dvec

    @dvec.setter
    def dvec(self, value):
        if value is None:
            self._dvec = np.zeros(self.portfolio.num_assets)
        elif len(value) == self.Dmat.shape[0]:
            self._dvec = -np.atleast_1d(value)
        else:
            raise ValueError(
                "dvec shape does not match with Dmat shape"
            )

    def iterate(self):
        # prepare
        wi = self.portfolio.weights
        contrib = self.portfolio.risk_contribution
        """
        为预防单一资产出现极端风险贡献的代码 
        当一项资产的波动率相对极低从而对于组合的风险贡献为极小数值甚至负数时
        直接介入主动调整该项资产的配比
        （可否增加杠杆）
        This part is to prevent an asset with extreme volatility, this might lead to convergence failure
        when an asset's volatility is extremely low, actively adjust the weight of the asset
        """
        if np.amin(contrib) < 0:
            np.put(wi, np.argmax(contrib), wi[np.argmax(contrib)] / 2)
            np.put(wi, np.argmin(contrib), wi[np.argmin(contrib)] + wi[np.argmax(contrib)])
            self.portfolio.weights = wi
            return True

        r = self.rvec(wi)
        A = np.ascontiguousarray(self.Amat(wi))
        At = np.transpose(A)
        Q = 2 * At @ A + self._tauI
        q = 2 * np.matmul(At, r) - np.matmul(Q, wi)

        # solve the quadratic programming with quadprog
        w_hat = quadprog.solve_qp(Q, -q, C=self.CCmat, b=self.bvec, meq=self.meq)[0]
        # update wi
        self.portfolio.weights = wi + self.theta * (w_hat - wi)
        # calculate the new risk concentration
        fun_new = self.get_objective_function_value()
        self.objective_function.append(fun_new)

        # check whether w and fun converged
        w_converge = (
            np.abs(self.portfolio.weights - wi)
            <= 0.5 * self.wtol * (np.abs(self.portfolio.weights) + np.abs(wi))
        ).all()
        fun_converge = (
            np.abs(self._funk - fun_new)
            <= 0.5 * self.contol * (np.abs(self._funk + np.abs(fun_new)))
        ).all()

        # return false if converged; update fun and return true if not
        if w_converge or fun_converge:
            return False
        self.theta = self.theta * (1 - self.zeta * self.theta)
        self._funk = fun_new
        return True

    def solve(self):
        i = 0
        with tqdm(total=self.maxiter) as pbar:
            while self.iterate() and i < self.maxiter:
                i += 1
                pbar.update()















