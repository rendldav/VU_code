import numpy as np
from scipy.optimize import root_scalar

class NormPrior:
    def __init__(self, q, alpha, beta):
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.fh = self._setup_fh()

    def _mL1(self, x, alpha, beta):
        return alpha * np.ones_like(x) + beta * np.abs(x)

    def _mL0(self, x, u_star, beta_half):
        return beta_half * (x**2) * (np.abs(x) < u_star)

    def _mLn(self, x, u_star, alpha, beta, t, q):
        return np.where(np.abs(x) < t, alpha * np.ones_like(x), beta / (q * np.abs(x)**(q - 1)))

    def _setup_fh(self):
        if self.q == 2:
            return lambda x: self.alpha * np.ones_like(x)
        elif self.q == 1:
            return lambda x: self._mL1(x, self.alpha, self.beta)
        elif self.q == 0:
            u_star = np.sqrt(2 * self.alpha / self.beta)
            return lambda x: self._mL0(x, u_star, self.beta / 2)
        else:  # for 0 < q < 1
            def leftmarker_func(v):
                return -v + self.alpha / self.beta * v**(self.q - 1) * (1 - self.q) * self.q

            def v_star_func(v):
                return -0.5 * v**2 + self.alpha / self.beta * v**self.q * (1 - self.q)

            leftmarker = root_scalar(leftmarker_func, bracket=[np.finfo(float).eps, 10]).root
            v_star = root_scalar(v_star_func, bracket=[leftmarker, 10]).root
            u_star = v_star + self.alpha / self.beta * self.q * v_star**(self.q - 1)
            t = u_star - (self.beta / self.alpha / 2 * u_star**2)**(1 / self.q)

            return lambda x: self._mLn(x, u_star, self.alpha, self.beta, t, self.q)

# Example usage:
# P = NormPrior(q, alpha, beta)
# result = P.fh(np.array([1, 2, 3]))
