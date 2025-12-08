import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy import special

from abc import ABC, abstractmethod


def gbp_pdf(x, a, b, c, d):
    log_num = (
        np.log(c)
        + (a * c - 1) * (np.log(x)- np.log(d))
        - (a + b) * np.log1p((x / d) ** c)
    )
    log_den = np.log(d) + special.betaln(a, b)
    return np.exp(log_num - log_den)


class Family(ABC):
    @abstractmethod
    def W_to_R2(self, w):
        """Convert W values to R^2

        Details are specific to each model family. To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def pdf(self, w):
        """Probability Density Function (PDF) of W.

        Details are specific to each model family. To be implemented by subclasses.
        """
        pass

    def cdf(self, w):
        """Cumulative Distribution Function (CDF) of W

        Computes values of the CDF of W induced by a Beta(a, b) prior on R-squared.
        The strategy is to convert W to R-squared and find the CDF of the corresponding R-squared.
        """
        return stats.beta.cdf(self.W_to_R2(w), a=self.a, b=self.b)

    def ppf(self, p, bounds=None):
        """Quantile Function of W

        Computes the quantiles of W induced by a Beta(a, b) prior on R2.
        It numerically inverts the CDF of W.
        NOTE: It's quite sensitive to `bounds`.
        """
        if bounds is None:
            # Helper to find suitable bonds
            ub_candidates = [1, 10, 100, 1_000, 10_000, 100_000, 500_000, 1_000_000]
            for candidate in ub_candidates:
                if self.cdf(candidate) > 0.99:
                    break
            bounds = (1 / 100_000, candidate)

        log_bounds = (np.log(bounds[0]), np.log(bounds[1]))

        def ppf_scalar(p):
            def distance(logw):
                return (self.cdf(w=np.exp(logw)) - p) ** 2

            result = minimize_scalar(distance, bounds=log_bounds)
            output = result.x

            return np.exp(output)

        return np.array([ppf_scalar(p_i) for p_i in np.atleast_1d(p)])


class PoissonFamily(Family):
    def __init__(self, a, b, intercept):
        self.a = a
        self.b = b
        self.intercept = intercept

    def W_to_R2(self, w):
        return np.expm1(w) / (np.expm1(w) + np.exp(-self.intercept -0.5 * w))

    def pdf(self, w):
        a, b, alpha = self.a, self.b, self.intercept
        f1 = 1 / special.beta(a, b)
        f2_num = np.expm1(w) ** (a - 1) * np.exp(-b * (alpha + w / 2)) * (3 * np.exp(w) - 1)
        f2_den = 2 * (np.expm1(w) + np.exp(-alpha - w / 2)) ** (a + b)
        return f1 * (f2_num / f2_den)


class NegativeBinomialFamily(Family):
    def __init__(self, a, b, intercept, theta):
        self.a = a
        self.b = b
        self.intercept = intercept
        self.theta = theta

    def W_to_R2(self, w):
        return np.expm1(w) / (np.expm1(w) + self.theta * np.exp(-self.intercept -0.5 * w))

    def pdf(self, w):
        a, b, alpha, theta = self.a, self.b, self.intercept, self.theta
        f1 = theta ** b / special.beta(a, b)
        f2_num = np.expm1(w) ** (a - 1) * np.exp(-b * (alpha + w / 2)) * (3 * np.exp(w) - 1)
        f2_den = 2 * (np.expm1(w) + theta * np.exp(-alpha - w / 2)) ** (a + b)
        return f1 * (f2_num / f2_den)


class GaussianFamily(Family):
    def __init__(self, a, b, intercept, sigma):
        self.a = a
        self.b = b
        self.intercept = intercept
        self.sigma = sigma

    def W_to_R2(self, w):
        return w / (w + self.sigma ** 2)

    def pdf(self, w):
        return gbp_pdf(w, self.a, self.b, 1, self.sigma ** 2)


class LogisticFamily(Family):
    def __init__(self, a, b, intercept):
        self.a = a
        self.b = b
        self.intercept = intercept

    def _W_to_R2_scalar(self, w, K=1000):
        # Equation 13 in Yanchenko et al. (2024)
        # We are estimating the integrals via Quasi-Monte Carlo (QMC) integration
        p_grid = np.linspace(1, K - 1, num=K - 1) / K
        eta = stats.norm(loc=self.intercept, scale=w ** 0.5).ppf(p_grid)

        mu = self._mean(eta)
        mu_1 = np.mean(mu).item()
        mu_2 = np.mean(mu ** 2).item()
        sigma_squared = np.mean(self._var(eta)).item()

        # Usage of M and V comes from Equation 3
        M = mu_2 - mu_1 ** 2 # E(mu^2) - E(mu)^2
        V = sigma_squared

        return M / (M + V)

    def W_to_R2(self, w):
        # Vectorized version of Equation 13 in Yanchenko et al. (2024)
        return np.array([self._W_to_R2_scalar(w) for w in np.atleast_1d(w)])

    def pdf(self, w):
        # Density function of W
        # Computes values of the density function of W induced by a Beta(a, b) prior on R2.
        # The computation uses a numeric approxiation to the derivative of the CDF.

        # NOTE: When `w[i] - delta` is <  0 for some index `i`, we'll get a warning and NaN.
        # The following mechanism tries to overcome that.
        delta_candidates = [0.001, 0.0001, 0.00001]
        for delta in delta_candidates:
            if bool(np.all((w - delta) > 0)):
                break

        diff = self.cdf(w=w + delta) - self.cdf(w=w - delta)
        return diff / (2 * delta)

    def _mean(self, eta):
        return special.expit(eta)

    def _var(self, eta):
        mean = self._mean(eta)
        return mean * (1 - mean)



def penalized_divergence(p_true, p_approx, params_current, params_reference, lam=0.25):
    # Penalized Pearson Chi-squared divergence
    integral = np.sum((1 - p_approx / p_true) ** 2)
    penalty = lam * np.sum((params_current - params_reference) ** 2)
    return integral + penalty



def WGBP(family, lam=0.25, x0=np.ones(4), method="Powell"):
    """Compute parameters for the GBP Approximation

    This function finds the closest Generalized Beta Prime (GBP) distribution
    to the true pdf of W as measured by the Pearson Chi-squared divergence.
    """
    a, b = family.a, family.b

    # Quantiles
    p = np.linspace(0.01, 0.99, num=500)

    # Values of 'w' where p_true and p_gbp are evaluated.
    w = family.ppf(p=p).flatten()
    p_true = family.pdf(w=w).flatten()

    # Copied from the R implementation because sometimes p_true has NaNs
    w = w[~np.isnan(p_true)]
    p_true = p_true[~np.isnan(p_true)]

    params_reference = np.array([a, b, 1, 1])
    def divergence(log_params):
        params = np.exp(log_params)
        return penalized_divergence(
            p_true=p_true,
            p_approx=gbp_pdf(w, *params),
            params_current=params,
            params_reference=params_reference,
            lam=lam
        )

    result = minimize(divergence, x0=np.log(x0), method=method)

    if result.success:
        return np.exp(result.x)

    raise Exception("Minimization didn't converge")


def plot_w_approximations(rows_dict, family, **family_kwargs):

    def format_params(params):
        names = ["a", "b", "c", "d"]
        return "$" + ", ".join(f"{n}^*={p:.2f}" for n, p in zip(names, params)) + "$"

    # NOTE: The divergence is _very_ sensible to the range of values for 'w'.
    # When the true density is near zero, it goes up a lot.
    for row in rows_dict:
        # Get parameters
        a, b, intercept = row["a"], row["b"], row["beta_0"]
        params_paper = [row[name] for name in ["a_star", "b_star", "c_star", "d_star"]]
        family_name = "_".join(row["family"].split())


        # Compute approximations
        family_obj = family(a=a, b=b, intercept=intercept, **family_kwargs)

        w_lower = max(family_obj.ppf(0.01).item(), 0.001) # Patch for small lower bounds
        w_upper = min(family_obj.ppf(0.99).item(), 250) # Patch for large upper bounds
        w = np.linspace(w_lower, w_upper, num=500)

        params_own = WGBP(family_obj)
        pdf_own = gbp_pdf(w, *params_own)
        pdf_paper = gbp_pdf(w, *params_paper)
        w_pdf = family_obj.pdf(w=w)

        divergence_own = penalized_divergence(
            p_true=w_pdf,
            p_approx=pdf_own,
            params_current=params_own,
            params_reference=np.array([a, b, 1, 1])
        )

        divergence_paper = penalized_divergence(
            p_true=w_pdf,
            p_approx=pdf_paper,
            params_current=params_paper,
            params_reference=np.array([a, b, 1, 1])
        )

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

        ax.plot(w, pdf_paper, label="Paper")
        ax.plot(w, pdf_own, label="Own")
        ax.plot(w, w_pdf, color="0.3", ls="--", label="True")

        title_left = "\n".join(
            [
                "$\\bf{Approximations}$",
                format_params(params_paper) + " [paper]",
                format_params(params_own) + " [own]",
                "$\\bf{Divergences}$",
                f"{divergence_paper:.2f} [paper] vs. " + f"{divergence_own:.2f} [own]",
            ]
        )
        title_right = "\n".join(
            [
                f"$R^2 \\sim$ Beta({a}, {b})",
                f"$\\alpha = {intercept}$"
            ]
        )

        ax.text(x=0, y=1.025, s=title_left, ha="left", size=11, transform=ax.transAxes)
        ax.text(x=1, y=1.025, s=title_right, ha="right", size=13, transform=ax.transAxes)
        ax.set(xlabel="W", yticks=[])
        ax.legend(loc="upper right")

        file_name = f"imgs/{family_name}/a-{a}_b-{b}_intercept-{intercept}.png"
        fig.savefig(file_name)
        plt.close(fig)

    return None


if __name__ == "__main__":
    df = pl.read_csv("data/paper_approximations.csv")

    families = {
        "poisson": {"class": PoissonFamily, "kwargs": {}},
        "nb": {"class": NegativeBinomialFamily, "kwargs": {"theta": 2}},
        "logistic": {"class": LogisticFamily, "kwargs": {}},
    }

    for family_name, family_dict in families.items():
        for intercept in [-2, 0, 2]:
            print(f"Family: {family_name}, Intercept: {intercept}")

            plot_w_approximations(
                df.filter(
                    pl.col("family") == family_name,
                    pl.col("beta_0") == intercept
                ).to_dicts(),
                family_dict["class"],
                **family_dict["kwargs"]
            );
