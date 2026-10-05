"""Educational Competition Optimizer (ECO) and the paper's RUNU-ECO (Sec. 5.3, Algorithm 1).

ECO is a line-by-line port of the authors' reference MATLAB code (Lian et al., Int. J. Systems
Science 2024, https://github.com/junbolian/ECO, ECO.m v1.0), including its quirks (e.g. stage-1
schools use the scalar mean of their own position vector, as in the MATLAB `mean(X(j,:))`).

RUNU-ECO ("Random Uniform Number Updated" ECO) replaces ECO's uniform random numbers R1, R2,
which choose between the two student-update rules in stages 2 and 3, with the fitness-informed
number of Eq. (8):

        N = (F_current + F_best) / (2 * F_worst)        (minimisation, F > 0)

The extracted equation in the paper is garbled ("N = er+ty / rt2"). We read the denominator as
2*rt, which keeps N in (0, 1] like the uniform number it replaces (best <= current <= worst). Use
`runu_mode="square"` for the literal rt^2 reading. Close to convergence N -> 1, so solutions take
the exploitation-oriented branch. Before that, poorer solutions (larger N) are steered
differently from good ones instead of by coin-flip.
"""
import math

import numpy as np


def levy(dim, rng, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(dim) * sigma
    v = rng.standard_normal(dim)
    return u / np.abs(v) ** (1 / beta)


def logistic_init(n, dim, lb, ub, rng):
    x0 = rng.random((n, dim))
    x = 4 * x0 * (1 - x0)
    return np.clip(lb + (ub - lb) * x, lb, ub)


class ECO:
    name = "ECO"

    def __init__(self, pop=10, iters=50, H=0.5, G1=0.2, G2=0.1, seed=0, verbose=True):
        self.N, self.T, self.H, self.G1, self.G2 = pop, iters, H, G1, G2
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    # hook overridden by RUNU-ECO ---------------------------------------------------------------
    def _rand(self, r_iter, f_j, f_best, f_worst):
        """ECO: the uniform number drawn once per iteration (R1 / R2 in ECO.m)."""
        return r_iter

    def _close(self, t, n_school, X):
        m = X[0]
        for s in range(n_school):
            if np.abs(m - t).sum() > np.abs(X[s] - t).sum():
                m = X[s]
        return m

    def minimize(self, fobj, lb, ub, callback=None):
        lb, ub = np.asarray(lb, float), np.asarray(ub, float)
        dim, N, T, rng = len(lb), self.N, self.T, self.rng
        g1n, g2n = round(N * self.G1), round(N * self.G2)

        X = logistic_init(N, dim, lb, ub, rng)
        fit = np.array([fobj(x) for x in X])
        order = np.argsort(fit)
        X, fit = X[order], fit[order]
        gbest_f, gbest_x = fit[0], X[0].copy()
        curve, avg_curve, history = [], [], []

        for i in range(1, T + 1):
            avg_curve.append(float(fit.mean()))
            P = 4 * rng.standard_normal() * (1 - i / T)
            P = P if abs(P) > 1e-12 else 1e-12  # P = 0 at i = T in ECO.m (division by zero there)
            E = (math.pi * i) / (P * T)
            w = 0.1 * math.log(2 - i / T)
            R1_iter, R2_iter = rng.random(), rng.random()  # ECO draws these once per iteration
            f_best_now, f_worst_now = fit.min(), fit.max()
            Xn, fn = X.copy(), fit.copy()
            for j in range(N):
                R1 = self._rand(R1_iter, fit[j], f_best_now, f_worst_now)
                R2 = self._rand(R2_iter, fit[j], f_best_now, f_worst_now)
                xj = X[j]
                stage = i % 3
                if stage == 1:          # Stage 1: primary school competition
                    if j < g1n:
                        new = xj + w * (xj.mean() - xj) * levy(dim, rng)
                    else:
                        new = xj + w * (self._close(xj, g1n, X) - xj) * rng.standard_normal()
                elif stage == 2:        # Stage 2: middle school competition
                    if j < g2n:
                        new = xj + (gbest_x - X.mean(0)) * math.exp(i / T - 1) * levy(dim, rng)
                    else:
                        c = self._close(xj, g2n, X)
                        if R1 < self.H:
                            new = xj - w * c - P * (E * w * c - xj)
                        else:
                            new = xj - w * c - P * (w * c - xj)
                else:                   # Stage 3: high school competition
                    if j < g2n:
                        new = xj + (gbest_x - xj) * rng.standard_normal() - (gbest_x - xj) * rng.standard_normal()
                    else:
                        if R2 < self.H:
                            new = gbest_x - P * (E * gbest_x - xj)
                        else:
                            new = gbest_x - P * (gbest_x - xj)
                new = np.clip(new, lb, ub)  # boundary control
                f_new = fobj(new)
                if f_new <= fit[j]:         # greedy selection
                    Xn[j], fn[j] = new, f_new
                if fn[j] < gbest_f:
                    gbest_f, gbest_x = fn[j], Xn[j].copy()
            order = np.argsort(fn)
            X, fit = Xn[order], fn[order]
            curve.append(float(gbest_f))
            history.append(X.copy())
            if callback:
                callback(i, gbest_x, gbest_f)
            if self.verbose:
                print(f"[{self.name}] iter {i}/{T} best={gbest_f:.6g} mean={fit.mean():.6g}", flush=True)
        return {"best_x": gbest_x, "best_f": float(gbest_f), "curve": curve, "avg_curve": avg_curve}


class RUNUECO(ECO):
    name = "RUNU-ECO"

    def __init__(self, *a, runu_mode="2x", **kw):
        super().__init__(*a, **kw)
        self.runu_mode = runu_mode

    def _rand(self, r_iter, f_j, f_best, f_worst):
        """Eq. (8): fitness-informed replacement for the uniform random number."""
        if f_worst <= 0 or not np.isfinite(f_worst):
            return r_iter
        den = 2 * f_worst if self.runu_mode == "2x" else f_worst ** 2
        return float(np.clip((f_j + f_best) / den, 0.0, 1.0))


class RandomSearch:
    """Reference baseline with the same evaluation budget (pop * (iters + 1))."""
    name = "RandomSearch"

    def __init__(self, pop=10, iters=50, seed=0, verbose=True, **_):
        self.N, self.T, self.rng, self.verbose = pop, iters, np.random.default_rng(seed), verbose

    def minimize(self, fobj, lb, ub, callback=None):
        lb, ub = np.asarray(lb, float), np.asarray(ub, float)
        best_f, best_x, curve = np.inf, None, []
        X = self.rng.uniform(lb, ub, (self.N, len(lb)))
        for x in X:
            f = fobj(x)
            if f < best_f:
                best_f, best_x = f, x
        for i in range(1, self.T + 1):
            for x in self.rng.uniform(lb, ub, (self.N, len(lb))):
                f = fobj(x)
                if f < best_f:
                    best_f, best_x = f, x
            curve.append(float(best_f))
            if callback:
                callback(i, best_x, best_f)
        return {"best_x": best_x, "best_f": float(best_f), "curve": curve}


OPTIMIZERS = {"eco": ECO, "runu-eco": RUNUECO, "random": RandomSearch}
