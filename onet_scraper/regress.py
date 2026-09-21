"""Regressions: which relationships survive when the variables compete.

Everything else in this repo reports one relationship at a time - a correlation,
a decile table, a quadrant count. Those are honest but they cannot answer "is
that still true holding the obvious alternative explanation fixed", and three of
the headline claims are exactly of that shape. This module runs the multivariate
fits, on stdlib only, with the standard errors the data structure actually
requires.

WHAT CANNOT BE REGRESSED HERE, AND WHY
    susceptibility = 50 + (exposure - anchoring) / 2, and exposure IS
    llm_exposure. So regressing the index on its own rated dimensions is an
    identity with R^2 = 1, not a finding. Every specification below therefore
    puts the index on one side and something the rubric never saw on the other:
    O*NET's human-survey descriptors, BLS wages, published exposure measures,
    O*NET's own task revisions.

THE STANDARD ERRORS ARE NOT THE DEFAULT ONES
    268 O*NET codes collapse to 196 SOC codes: 15-2051.01 and 15-2051.02 are
    two rows, one survey. Classical standard errors would treat them as two
    independent observations, so occupation-level fits cluster by 6-digit SOC.
    The task-level fit has 5,453 rated tasks drawn from 262 occupations and
    clusters there. Heteroskedasticity-robust (HC1) is the floor everywhere else.

WHAT THE DATA STILL CANNOT SUPPORT
    All of it is cross-sectional. Coefficients are conditional means, not
    effects: nothing here identifies what happens to a wage when exposure
    changes, because nothing here observes exposure changing. The churn fit is
    the one place with a time dimension, and it is contaminated by O*NET's
    review cycle, which is why the review lag enters as a control rather than
    being mentioned in a footnote.

RESULTS AS BUILT - and the wage prior below was the one that was wrong:

  1. Against O*NET's independent human surveys (n = 260, R2 = 0.66), the index
     behaves partly as Frey & Osborne's bottleneck logic predicts and partly
     against it. Perception/manipulation (-0.20), social intelligence (-0.18)
     and responsibility for others (-0.15) all load negative, as bottlenecks
     should. Creative intelligence loads POSITIVE (+0.15, t = 4.5): the very
     demand that was supposed to protect an occupation now marks it as more
     exposed. That is not a defect in the index, it is the substantive claim of
     the project surviving a test it could have failed.

     Job Zone is the quiet one. It predicts exposure (+2.1 per zone) and
     anchoring (+2.5 per zone) almost equally, so on net susceptibility it is
     indistinguishable from zero (-0.20, t = -0.3). More training buys more
     exposure and more protection in the same breath.

  2. The index carries information about human expert LLM-exposure ratings that
     the pre-LLM measures do not have. Removing it drops R2 from 0.70 to 0.08;
     removing all three of Frey & Osborne, Felten et al. and SML costs 0.03.
     That asymmetry is what "different phenomenon, not a relabelling" means
     when written as a testable statement instead of a slogan. It holds on the
     wider n = 188 sample that drops the narrowest benchmark.

  3. WAGES DO NOT DISCOUNT EXPOSURE - AND THE PRIOR HERE WAS REFUTED. The
     registered expectation was "near zero on exposure, clearly positive on
     anchoring". Anchoring is indeed positive (+0.019 log points per index
     point, t = 3.7). But exposure is positive too (+0.010, t = 2.4), not zero:
     conditional on how anchored the work is and how much training it needs,
     more exposed work pays MORE. The bivariate correlation of 0.01 that the
     wages module reports is a suppression artifact - exposure and anchoring
     correlate at -0.64, and holding anchoring fixed uncovers the exposure
     gradient the raw correlation hides. Weighted by employment the two swap
     roles: exposure keeps its coefficient (+0.015, t = 2.5) and anchoring
     collapses to nothing (t = 0.9).

  4. "AI TAKES THE BUSYWORK" IS NOT WHAT THE TASK DATA SAYS. Inside the same
     occupation, a task's importance does carry a negative coefficient
     (-0.065, t = -3.6) - but moving a task from the 25th to the 75th
     percentile of importance buys 1.3 susceptibility points against a
     task-level standard deviation of 15.7. Supplemental tasks sit 3.0 points
     below Core. The sign is comforting and the magnitude is not: important
     work is barely safer than peripheral work in the same job. Moving the
     fixed effects up to the 6-digit SOC gives -0.075, so this does not turn
     on the choice of grouping.

"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

log = logging.getLogger(__name__)

COEFFICIENT_COLUMNS = ("fit", "outcome", "term", "coef", "std_error", "t_stat",
                       "p_value", "ci_low", "ci_high", "beta_std", "vif",
                       "n", "se_kind", "clusters")

# Stated from theory and from the bivariate correlations this repo already
# publishes, before any multivariate fit was run - so a refuted prior is a
# result rather than something to reword afterwards. Keyed by fit name.
PRIORS = {
    "validity_descriptors":
        "the three bottlenecks should load NEGATIVE if this index inherits the "
        "pre-LLM theory of what protects work; a positive coefficient on "
        "creative intelligence would contradict that theory, not the index",
    "incremental_validity":
        "our index should retain a large coefficient on human expert ratings "
        "with the pre-LLM measures held fixed; if it does not, it is a "
        "relabelling of measures that already existed",
    "wage_pricing":
        "near zero on exposure and clearly positive on anchoring: the bivariate "
        "wage-susceptibility correlation should decompose into an anchoring "
        "premium rather than an exposure discount",
    "task_within_occupation":
        "close to zero on importance. If exposure concentrated in peripheral "
        "tasks the coefficient would be strongly negative, and the comfortable "
        "reading - AI takes the busywork - would be the correct one",
    "task_new_vs_established":
        "newly added statements should be MORE exposed if O*NET is adding "
        "digital and analytical work; but the flag records when a statement "
        "was written down, not when the work appeared, so a null is easy to "
        "get here for uninteresting reasons",
    "churn_realised":
        "weak at best. The index is a claim about what COULD be automated; "
        "O*NET task revisions are a claim about what survey panels wrote down, "
        "and the review cycle drives most of the variance",
}


# --------------------------------------------------------------------------- #
# distributions - two-sided t and upper-tail F, both from the regularised
# incomplete beta function. Needed because stdlib ships NormalDist and nothing
# else, and treating t(150) as normal would overstate every p-value here.
# --------------------------------------------------------------------------- #

def _betacf(a: float, b: float, x: float, itmax: int = 400,
            eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x)
                     + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, dof: float) -> float:
    """Two-sided p for a t statistic."""
    if dof <= 0:
        return float("nan")
    return betainc(dof / 2.0, 0.5, dof / (dof + t * t))


def f_pvalue(f: float, d1: float, d2: float) -> float:
    """Upper-tail p for an F statistic."""
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return float("nan")
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def t_critical(dof: float, level: float = 0.95) -> float:
    """Two-sided critical value, by bisection on t_pvalue."""
    if dof <= 0:
        return float("nan")
    target = 1.0 - level
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_pvalue(mid, dof) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# linear algebra - k is at most a dozen here, so an explicit inverse of the
# scaled cross-product matrix is both fast enough and what the sandwich
# estimators need anyway.
# --------------------------------------------------------------------------- #

def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan with partial pivoting."""
    k = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(k)]
           for i, row in enumerate(matrix)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError(f"singular design matrix at column {col}")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for row in range(k):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [v - factor * p for v, p in zip(aug[row], aug[col])]
    return [row[k:] for row in aug]


def _solve_scaled(X: Sequence[Sequence[float]], y: Sequence[float]
                  ) -> tuple[list[float], list[list[float]]]:
    """Least squares by normal equations on unit-norm columns.

    Scaling first is what keeps this honest: raw columns here run from a 0/1
    dummy to employment in the millions, and X'X on those spans ~12 orders of
    magnitude before a single arithmetic operation. Scaled, the condition
    number is small enough that the explicit inverse is safe.
    """
    n, k = len(X), len(X[0])
    scale = [math.sqrt(sum(row[j] ** 2 for row in X)) or 1.0 for j in range(k)]
    Z = [[row[j] / scale[j] for j in range(k)] for row in X]
    A = [[sum(Z[i][a] * Z[i][b] for i in range(n)) for b in range(k)]
         for a in range(k)]
    rhs = [sum(Z[i][a] * y[i] for i in range(n)) for a in range(k)]
    Ainv = _inverse(A)
    gamma = [sum(Ainv[a][b] * rhs[b] for b in range(k)) for a in range(k)]
    beta = [gamma[a] / scale[a] for a in range(k)]
    xtx_inv = [[Ainv[a][b] / (scale[a] * scale[b]) for b in range(k)]
               for a in range(k)]
    return beta, xtx_inv


def _ss_resid(y: Sequence[float], X: Sequence[Sequence[float]]) -> float:
    """Residual sum of squares from regressing y on X."""
    beta, _ = _solve_scaled(X, y)
    return sum((y[i] - sum(X[i][j] * beta[j] for j in range(len(beta)))) ** 2
               for i in range(len(y)))


def _total_ss(values: Sequence[float], intercept_col: list[float] | None) -> float:
    """Sum of squares about the mean - or about the WEIGHTED mean.

    Under weights the fit runs on sqrt(w)-transformed data, where the intercept
    column is sqrt(w) rather than 1. Projecting onto that column is what
    centres on the weighted mean; centring on the plain mean of the transformed
    values instead is the classic way to report an R^2 of 0.999 for a weighted
    regression that explains very little.
    """
    if intercept_col is None:
        return sum(v * v for v in values)
    return _ss_resid(values, [[c] for c in intercept_col])


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #

@dataclass
class Fit:
    name: str
    outcome: str
    terms: list[str]
    coef: list[float]
    se: list[float]
    n: int
    dof: float
    se_kind: str
    r2: float
    adj_r2: float
    rmse: float
    clusters: int | None = None
    absorbed: int = 0
    f_stat: float | None = None
    f_p: float | None = None
    beta_std: list[float] = field(default_factory=list)
    vif: list[float | None] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    question: str = ""
    prior: str = ""

    def rows(self) -> list[dict[str, Any]]:
        crit = t_critical(self.dof)
        out = []
        for j, term in enumerate(self.terms):
            t = self.coef[j] / self.se[j] if self.se[j] else float("nan")
            # Three significant figures rather than five decimal places, so a
            # very small p prints as 3.6e-09 instead of rounding to 0.0 and
            # inviting someone to write "p = 0".
            p = t_pvalue(t, self.dof)
            std = self.beta_std[j] if self.beta_std else float("nan")
            out.append({
                "fit": self.name, "outcome": self.outcome, "term": term,
                "coef": round(self.coef[j], 6),
                "std_error": round(self.se[j], 6),
                "t_stat": round(t, 3),
                "p_value": float(f"{p:.3g}") if p == p else None,
                "ci_low": round(self.coef[j] - crit * self.se[j], 6),
                "ci_high": round(self.coef[j] + crit * self.se[j], 6),
                "beta_std": round(std, 4) if std == std else None,
                "vif": (round(self.vif[j], 2)
                        if self.vif and self.vif[j] is not None else None),
                "n": self.n, "se_kind": self.se_kind,
                "clusters": self.clusters,
            })
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name, "question": self.question, "prior": self.prior,
            "outcome": self.outcome, "n": self.n,
            "clusters": self.clusters, "se_kind": self.se_kind,
            "absorbed_fixed_effects": self.absorbed or None,
            "r2": round(self.r2, 4), "adj_r2": round(self.adj_r2, 4),
            "rmse": round(self.rmse, 4), "dof": self.dof,
            "f_stat": round(self.f_stat, 2) if self.f_stat is not None else None,
            "f_p_value": (float(f"{self.f_p:.3g}")
                          if self.f_p is not None and self.f_p == self.f_p
                          else None),
            "coefficients": self.rows(),
            "notes": self.notes,
        }


def ols(y: Sequence[float],
        X: Sequence[Sequence[float]],
        terms: Sequence[str],
        *,
        name: str = "fit",
        outcome: str = "y",
        weights: Sequence[float] | None = None,
        clusters: Sequence[Any] | None = None,
        absorbed: int = 0,
        has_intercept: bool = True,
        notes: Sequence[str] = (),
        question: str = "",
        prior: str = "") -> Fit:
    """Least squares with HC1 or cluster-robust (CR1) standard errors.

    `absorbed` is the number of parameters removed by a within transformation
    before the call (occupation fixed effects, here). They cost degrees of
    freedom even though they are not in X, and ignoring that inflates every
    t statistic in the fixed-effects fit.
    """
    n, k = len(X), len(X[0])
    if n <= k + absorbed:
        raise ValueError(f"{name}: {n} observations cannot support "
                         f"{k + absorbed} parameters")

    # A constant regressor is the failure mode a join actually produces - drop
    # the rows with a missing rating and a category dummy can empty out - so it
    # is worth naming the column rather than raising from inside the inverse.
    for j, term in enumerate(terms):
        if term == "(intercept)":
            continue
        column = {row[j] for row in X}
        if len(column) == 1:
            raise ValueError(
                f"{name}: '{term}' is constant at {column.pop()} across all "
                f"{n} rows that survived the join, so its coefficient is not "
                "identified")

    # WLS is OLS on sqrt(w)-transformed data; every formula below then applies
    # unchanged, including the sandwiches.
    if weights is not None:
        rw = [math.sqrt(w) for w in weights]
        yw = [y[i] * rw[i] for i in range(n)]
        Xw = [[X[i][j] * rw[i] for j in range(k)] for i in range(n)]
    else:
        yw, Xw = list(y), [list(row) for row in X]

    beta, xtx_inv = _solve_scaled(Xw, yw)
    resid = [yw[i] - sum(Xw[i][j] * beta[j] for j in range(k)) for i in range(n)]

    k_total = k + absorbed
    dof_resid = n - k_total

    if clusters is not None:
        groups: dict[Any, list[int]] = {}
        for i, g in enumerate(clusters):
            groups.setdefault(g, []).append(i)
        meat = [[0.0] * k for _ in range(k)]
        for members in groups.values():
            score = [sum(Xw[i][a] * resid[i] for i in members) for a in range(k)]
            for a in range(k):
                for b in range(k):
                    meat[a][b] += score[a] * score[b]
        G = len(groups)
        # CR1: the small-G correction. With G near k this is the difference
        # between a standard error and a fiction.
        adj = (G / (G - 1)) * ((n - 1) / dof_resid) if G > 1 else 1.0
        se_kind, n_clusters = f"cluster-robust (CR1), {G} clusters", G
        dof = G - 1
    else:
        meat = [[sum(Xw[i][a] * Xw[i][b] * resid[i] ** 2 for i in range(n))
                 for b in range(k)] for a in range(k)]
        adj = n / dof_resid          # HC1
        se_kind, n_clusters = "heteroskedasticity-robust (HC1)", None
        dof = dof_resid

    vcov = [[adj * sum(xtx_inv[a][p] * meat[p][q] * xtx_inv[q][b]
                       for p in range(k) for q in range(k))
             for b in range(k)] for a in range(k)]
    se = [math.sqrt(max(vcov[j][j], 0.0)) for j in range(k)]

    # R^2. With fixed effects absorbed this is the WITHIN R^2 - the share of
    # the deviation from each occupation's own mean that the regressors explain
    # - which is the only thing the transformed data can speak to.
    intercept_at = (terms.index("(intercept)") if "(intercept)" in terms
                    else None)
    intercept_col = ([row[intercept_at] for row in Xw]
                     if has_intercept and intercept_at is not None else None)
    ss_res = sum(r * r for r in resid)
    ss_tot = _total_ss(yw, intercept_col)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    adj_r2 = (1.0 - (1.0 - r2) * (n - 1) / dof_resid
              if dof_resid > 0 else float("nan"))

    # Joint test on everything but the intercept, using the robust vcov - so it
    # is a Wald F, not the textbook one that assumes the errors behaved.
    idx = [j for j, t in enumerate(terms) if t != "(intercept)"]
    f_stat = f_p = None
    if idx:
        sub = [[vcov[a][b] for b in idx] for a in idx]
        try:
            sub_inv = _inverse(sub)
            wald = sum(beta[a] * sub_inv[i][j] * beta[b]
                       for i, a in enumerate(idx) for j, b in enumerate(idx))
            f_stat = wald / len(idx)
            f_p = f_pvalue(f_stat, len(idx), dof)
        except ValueError:
            pass

    # Standardised coefficients, on the data as fitted: coef * sd(x) / sd(y),
    # with both spreads taken about the same (possibly weighted) mean as the
    # R^2 above, so the sample-size factor cancels. Comparable across terms
    # measured on different scales; for a 0/1 dummy it is a rescaling of a
    # group difference, which is why the raw coefficient is reported too.
    beta_std = []
    for j, term in enumerate(terms):
        if term == "(intercept)" or not ss_tot:
            beta_std.append(float("nan"))
            continue
        ss_x = _total_ss([row[j] for row in Xw], intercept_col)
        beta_std.append(beta[j] * math.sqrt(ss_x / ss_tot))

    vif = _vifs(Xw, terms, intercept_col)

    return Fit(name=name, outcome=outcome, terms=list(terms), coef=beta, se=se,
               n=n, dof=dof, se_kind=se_kind, r2=r2, adj_r2=adj_r2,
               rmse=math.sqrt(ss_res / dof_resid), clusters=n_clusters,
               absorbed=absorbed, f_stat=f_stat, f_p=f_p, beta_std=beta_std,
               vif=vif, notes=list(notes), question=question, prior=prior)


def _vifs(X: Sequence[Sequence[float]], terms: Sequence[str],
          intercept_col: list[float] | None = None) -> list[float | None]:
    """Variance inflation per regressor, by auxiliary regression.

    VIF = total SS / residual SS of regressing one column on the others, both
    measured about the same mean, which makes this correct for the weighted
    design too - there the intercept column is sqrt(w), not 1.
    """
    others = [j for j, t in enumerate(terms) if t != "(intercept)"]
    out: list[float | None] = []
    for j, term in enumerate(terms):
        rest = [c for c in others if c != j]
        if term == "(intercept)" or not rest:
            out.append(None)
            continue
        target = [row[j] for row in X]
        aux = [([intercept_col[i]] if intercept_col else [])
               + [X[i][c] for c in rest] for i in range(len(X))]
        try:
            ss_res = _ss_resid(target, aux)
        except ValueError:
            out.append(None)
            continue
        ss_tot = _total_ss(target, intercept_col)
        out.append(ss_tot / ss_res if ss_res > ss_tot * 1e-9 else None)
    return out


def nested(y: Sequence[float], X: Sequence[Sequence[float]],
           terms: Sequence[str], drop: Sequence[str], **kwargs: Any
           ) -> dict[str, Any]:
    """Refit without `drop` and report what its presence was worth.

    Two things, because they answer different questions: the change in R^2
    (how much variance those columns account for) and the robust Wald test on
    them jointly (whether the data can distinguish their coefficients from
    zero). A measure can raise R^2 trivially and still be indispensable, or
    carry a significant coefficient and explain nothing.
    """
    keep = [j for j, t in enumerate(terms) if t not in drop]
    full = ols(y, X, terms, **kwargs)
    restricted = ols(y, [[row[j] for j in keep] for row in X],
                     [terms[j] for j in keep], **kwargs)
    return {
        "dropped": list(drop),
        "r2_full": round(full.r2, 4),
        "r2_restricted": round(restricted.r2, 4),
        "r2_gain": round(full.r2 - restricted.r2, 4),
        "adj_r2_full": round(full.adj_r2, 4),
        "adj_r2_restricted": round(restricted.adj_r2, 4),
    }


# --------------------------------------------------------------------------- #
# designs - joining the built tables into (y, X) and saying out loud how many
# rows each join drops. A specification that silently loses a third of the
# sample is a different specification.
# --------------------------------------------------------------------------- #

@dataclass
class Design:
    y: list[float]
    X: list[list[float]]
    terms: list[str]
    clusters: list[Any] | None = None
    weights: list[float] | None = None
    dropped: int = 0
    absorbed: int = 0
    has_intercept: bool = True


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _soc6(code: str) -> str:
    """15-2051.01 -> 15-2051. The unit O*NET actually surveyed."""
    return code.split(".")[0]


def design(rows: Sequence[dict[str, Any]],
           outcome: Callable[[dict[str, Any]], float | None],
           terms: dict[str, Callable[[dict[str, Any]], float | None]],
           *,
           cluster: Callable[[dict[str, Any]], Any] | None = None,
           weight: Callable[[dict[str, Any]], float | None] | None = None,
           intercept: bool = True) -> Design:
    names = (["(intercept)"] if intercept else []) + list(terms)
    y: list[float] = []
    X: list[list[float]] = []
    clusters: list[Any] = []
    weights: list[float] = []
    dropped = 0
    for row in rows:
        yv = outcome(row)
        xs = [terms[t](row) for t in terms]
        wv = weight(row) if weight else 1.0
        if yv is None or any(x is None for x in xs) or wv is None or wv <= 0:
            dropped += 1
            continue
        y.append(yv)
        X.append(([1.0] if intercept else []) + [float(x) for x in xs])
        clusters.append(cluster(row) if cluster else None)
        weights.append(float(wv))
    return Design(y=y, X=X, terms=names,
                  clusters=clusters if cluster else None,
                  weights=weights if weight else None,
                  dropped=dropped, has_intercept=intercept)


def within(d: Design, groups: Sequence[Any]) -> Design:
    """Subtract each group's mean from y and from every regressor.

    This is occupation fixed effects done by hand: what is left is only the
    variation BETWEEN tasks inside the same occupation, so every occupation
    level difference - field, seniority, how the panel writes task statements -
    is differenced out and cannot drive the coefficient.
    """
    cols = len(d.X[0])
    sums: dict[Any, list[float]] = {}
    counts: dict[Any, int] = {}
    for i, g in enumerate(groups):
        acc = sums.setdefault(g, [0.0] * (cols + 1))
        for j in range(cols):
            acc[j] += d.X[i][j]
        acc[cols] += d.y[i]
        counts[g] = counts.get(g, 0) + 1
    keep = [j for j in range(cols) if d.terms[j] != "(intercept)"]
    y_out, X_out = [], []
    for i, g in enumerate(groups):
        n_g = counts[g]
        y_out.append(d.y[i] - sums[g][cols] / n_g)
        X_out.append([d.X[i][j] - sums[g][j] / n_g for j in keep])
    return Design(y=y_out, X=X_out, terms=[d.terms[j] for j in keep],
                  clusters=list(groups), weights=d.weights, dropped=d.dropped,
                  absorbed=len(counts), has_intercept=False)


def fit_design(d: Design, **kwargs: Any) -> Fit:
    return ols(d.y, d.X, d.terms, weights=d.weights, clusters=d.clusters,
               absorbed=d.absorbed, has_intercept=d.has_intercept, **kwargs)


# --------------------------------------------------------------------------- #
# the specifications
# --------------------------------------------------------------------------- #

DESCRIPTORS = ("bottleneck_creative_intelligence",
               "bottleneck_perception_manipulation",
               "bottleneck_social_intelligence",
               "collaboration",
               "responsibility_for_others")


def validity_fits(occ: Sequence[dict[str, Any]],
                  indices: Sequence[dict[str, Any]]) -> list[Fit]:
    """The index against O*NET's own human surveys.

    The rubric scored subtask *sentences*. These five indices are built from
    O*NET's surveys of incumbents and analysts - abilities, work context,
    responsibility - and the model never saw them. So this is the one
    occupation-level fit where the right-hand side is genuinely exogenous to
    the measure being tested, which is what makes a wrong sign here
    informative rather than embarrassing.
    """
    by_code = {r["onet_soc_code"]: r for r in indices}
    joined = []
    for row in occ:
        extra = by_code.get(row["onet_soc_code"])
        if extra:
            joined.append({**row, **{k: extra.get(k) for k in DESCRIPTORS}})
    terms = {k: (lambda k: lambda r: _num(r.get(k)))(k) for k in DESCRIPTORS}
    terms["job_zone"] = lambda r: _num(r.get("job_zone"))

    fits = []
    for outcome, label in (("susceptibility", "validity_descriptors"),
                           ("exposure", "validity_descriptors_exposure"),
                           ("anchoring", "validity_descriptors_anchoring")):
        d = design(joined, (lambda o: lambda r: _num(r.get(o)))(outcome), terms,
                   cluster=lambda r: _soc6(r["onet_soc_code"]))
        note = [f"{d.dropped} of {len(occ)} occupations dropped for missing "
                f"descriptor coverage or job zone",
                "clustered by 6-digit SOC: detail occupations inside one SOC "
                "share a single O*NET survey, so they are not independent draws"]
        if outcome == "susceptibility":
            note.append("job zone enters as the human-capital control - without "
                        "it, 'needs more training' and 'needs more judgment' "
                        "are the same column")
        fits.append(fit_design(
            d, name=label, outcome=outcome, notes=note,
            question=f"does {outcome} line up with what O*NET's human surveys "
                     "say about the same occupations?",
            prior=PRIORS["validity_descriptors"]))
    return fits


def incremental_fits(bench: Sequence[dict[str, Any]]) -> tuple[list[Fit], list[dict[str, Any]]]:
    """Does this index carry information the pre-LLM measures do not?

    The repo already reports r = 0.85 with human expert ratings and r ~ 0
    with Frey & Osborne. Those two facts are compatible with the index being
    a noisy copy of something already published, so the test that matters is
    conditional: with the three pre-LLM measures on the right-hand side, does
    our index still explain human expert judgments of LLM exposure - and do
    they explain anything once it is there?
    """
    pre = ("frey_osborne", "felten_raj_seamans", "brynjolfsson_sml")
    terms = {"our_susceptibility": lambda r: _num(r.get("our_susceptibility"))}
    terms.update({k: (lambda k: lambda r: _num(r.get(k)))(k) for k in pre})
    coverage = {k: sum(1 for r in bench if _num(r.get(k)) is not None)
                for k in pre}
    narrowest = min(coverage, key=lambda k: coverage[k])

    fits, comparisons = [], []
    specs = [("human_gamma", "incremental_validity", terms),
             ("human_beta", "incremental_validity_beta", terms),
             # The same test on the larger sample the narrowest benchmark was
             # costing us. If the conclusion only held on the 150 occupations
             # Frey & Osborne happened to cover, this is where that would show.
             ("human_gamma", "incremental_validity_wider_sample",
              {k: v for k, v in terms.items() if k != narrowest})]
    for outcome, name, spec_terms in specs:
        d = design(bench, (lambda o: lambda r: _num(r.get(o)))(outcome),
                   spec_terms, cluster=lambda r: _soc6(r["onet_soc_code"]))
        notes = [f"complete cases: n = {len(d.y)}, {d.dropped} of {len(bench)} "
                 f"rows dropped",
                 "human ratings are the dependent variable because they are "
                 "the closest thing here to an external ground truth"]
        if narrowest in spec_terms:
            notes.insert(1, f"the binding constraint is {narrowest}, which "
                            f"covers {coverage[narrowest]} of {len(bench)} "
                            "occupations; every drop is attributable to it")
        else:
            notes.insert(1, f"{narrowest} omitted deliberately to recover the "
                            f"occupations it does not cover")
        fits.append(fit_design(
            d, name=name, outcome=outcome, notes=notes,
            question=f"with the pre-LLM measures held fixed, does the index "
                     f"still track {outcome}?",
            prior=PRIORS["incremental_validity"]))
        kw = dict(clusters=d.clusters, name=name, outcome=outcome)
        dropped_pre = [k for k in pre if k in spec_terms]
        comparisons.append({
            "fit": name,
            "n": len(d.y),
            "ours_removed": nested(d.y, d.X, d.terms, ["our_susceptibility"], **kw),
            "pre_llm_removed": nested(d.y, d.X, d.terms, dropped_pre, **kw),
            "reading": "compare the two r2_gain figures: the larger one is the "
                       "measure the other cannot replace",
        })
    return fits, comparisons


def wage_fits(soc: Sequence[dict[str, Any]],
              occ: Sequence[dict[str, Any]]) -> list[Fit]:
    """Is exposure priced into wages - and is it exposure or anchoring?

    The wages module reports wage vs susceptibility at r = -0.18 and calls it
    a composition effect. Susceptibility is exposure minus anchoring, so that
    single number cannot say which half produced it. Here they compete.

    Cross-sectional and descriptive. A negative coefficient would not mean
    exposure lowers wages; it would mean exposed occupations currently pay
    less, which is a fact about 2025 and not a forecast.
    """
    zones: dict[str, list[float]] = {}
    for row in occ:
        z = _num(row.get("job_zone"))
        if z is not None:
            zones.setdefault(_soc6(row["onet_soc_code"]), []).append(z)
    mean_zone = {k: statistics.fmean(v) for k, v in zones.items()}

    terms = {
        "exposure": lambda r: _num(r.get("exposure")),
        "anchoring": lambda r: _num(r.get("anchoring")),
        "job_zone": lambda r: mean_zone.get(r.get("soc_code", "")),
    }
    outcome = lambda r: (math.log(_num(r.get("annual_mean_wage")))
                         if _num(r.get("annual_mean_wage")) else None)

    unweighted = design(soc, outcome, terms)
    weighted = design(soc, outcome, terms,
                      weight=lambda r: _num(r.get("total_employment")))
    shared = ["log wage, so a coefficient reads as a proportional wage "
              "difference per index point",
              "SOC level, so no within-SOC nesting to cluster on; HC1 only",
              "descriptive: cross-sectional, no exposure ever observed changing"]
    return [
        fit_design(unweighted, name="wage_pricing", outcome="log_annual_mean_wage",
                   notes=shared + ["one occupation, one observation - answers "
                                   "'does a more exposed OCCUPATION pay less'"],
                   question="does the labour market already discount exposed work?",
                   prior=PRIORS["wage_pricing"]),
        fit_design(weighted, name="wage_pricing_employment_weighted",
                   outcome="log_annual_mean_wage",
                   notes=shared + ["weighted by employment - answers 'does a more "
                                   "exposed WORKER earn less', a different question "
                                   "with a different answer, as in the decile table"],
                   question="same question, weighted by how many people hold the job",
                   prior=PRIORS["wage_pricing"]),
    ]


def task_fits(tasks: Sequence[dict[str, Any]]) -> list[Fit]:
    """Inside one occupation, is the exposed work the important work?

    This is the fit that tests the comfortable reading directly. If AI lands on
    peripheral admin, importance should carry a clear negative coefficient once
    occupation is differenced out. Pooled, the comparison is contaminated:
    occupations differ in both how exposed they are and how their panels write
    importance ratings. Both are reported, because the gap between them is
    itself the answer to 'is this a within-job or a between-job pattern'.

    One caveat the fixed effects do NOT fix: task scores are propagated from a
    shared subtask vocabulary, so two tasks mapped to the same DWA carry
    correlated scores by construction. Clustering on occupation does not absorb
    that, and it biases the standard errors towards being too small.
    """
    terms = {
        "importance": lambda r: _num(r.get("importance")),
        "is_supplemental": lambda r: 1.0 if r.get("task_category") == "Supplemental" else 0.0,
    }
    outcome = lambda r: _num(r.get("susceptibility"))
    # O*NET publishes no importance rating for a task it has just added: all 52
    # 'New' tasks are unrated, so they cannot enter an importance regression at
    # all. They get their own fit below rather than disappearing into a
    # dropped-rows count.
    rated = [r for r in tasks if r.get("task_category") in ("Core", "Supplemental")]
    pooled = design(rated, outcome, terms, cluster=lambda r: r["onet_soc_code"])
    fe_base = design(rated, outcome, terms, cluster=lambda r: r["onet_soc_code"])
    fe = within(fe_base, fe_base.clusters or [])

    # Same fit, absorbed and clustered one level up. Detail occupations inside
    # a SOC carry different task lists but one survey, so if that shared origin
    # were driving the result, moving the fixed effects to the SOC would move
    # the coefficient. Reported rather than asserted.
    soc_base = design(rated, outcome, terms,
                      cluster=lambda r: _soc6(r["onet_soc_code"]))
    soc_fe = within(soc_base, soc_base.clusters or [])

    labelled = [r for r in tasks
                if r.get("task_category") in ("Core", "Supplemental", "New")]
    new_base = design(
        labelled, outcome,
        {"is_new": lambda r: 1.0 if r.get("task_category") == "New" else 0.0,
         "is_supplemental": lambda r: 1.0 if r.get("task_category") == "Supplemental" else 0.0},
        cluster=lambda r: r["onet_soc_code"])
    new_fe = within(new_base, new_base.clusters or [])

    return [
        fit_design(pooled, name="task_pooled", outcome="susceptibility",
                   notes=[f"{pooled.dropped} tasks dropped for a missing "
                          "importance rating or an unlabelled category",
                          "pooled across occupations: mixes within-job and "
                          "between-job variation"],
                   question="across all STEM tasks, are important tasks less exposed?",
                   prior=PRIORS["task_within_occupation"]),
        fit_design(fe, name="task_within_occupation", outcome="susceptibility",
                   notes=["occupation fixed effects, absorbed by within-group "
                          "demeaning; R2 is therefore a WITHIN R2 - the share "
                          "of the deviation from an occupation's own mean that "
                          "importance explains",
                          "adj R2 charges the full price of every absorbed "
                          "fixed effect and can go negative here; that is the "
                          "penalty for 262 of them, not a broken fit",
                          "subtask-level score propagation correlates tasks that "
                          "share a DWA, which clustering on occupation does not "
                          "absorb - read these standard errors as a floor"],
                   question="inside the SAME occupation, are its important tasks "
                            "less exposed than its peripheral ones?",
                   prior=PRIORS["task_within_occupation"]),
        fit_design(soc_fe, name="task_within_soc", outcome="susceptibility",
                   notes=["robustness: fixed effects and clustering at the "
                          "6-digit SOC instead of the 8-digit occupation",
                          "the coefficient to compare is importance in "
                          "task_within_occupation - if the two disagree, the "
                          "headline fit was picking up within-SOC structure"],
                   question="same question with the fixed effects one level up: "
                            "does the answer depend on that choice?",
                   prior=PRIORS["task_within_occupation"]),
        fit_design(new_fe, name="task_new_vs_established", outcome="susceptibility",
                   notes=["no importance term, which is what lets the 52 'New' "
                          "tasks be used at all",
                          "occupation fixed effects: the comparison is against "
                          "the same occupation's established tasks, not against "
                          "the STEM average",
                          "52 tasks is a small n and O*NET's 'New' flag marks "
                          "when a statement was ADDED, not when the work began"],
                   question="is the work O*NET has only just written down more "
                            "exposed than the work already on the list?",
                   prior=PRIORS["task_new_vs_established"]),
    ]


def churn_fits(churn: Sequence[dict[str, Any]]) -> list[Fit]:
    """Does the index predict task revisions that already happened?

    The churn module documents the confound and handles it by splitting the
    sample. A regression can hold it fixed instead: an occupation reviewed in
    2012 has had no opportunity to show turnover, which is a fact about O*NET's
    schedule and not about the work. The base task count enters because
    turnover is a ratio and small denominators move.

    A null result here is the honest expected outcome, and it is worth writing
    down: the index is a claim about capability, and O*NET task revisions are a
    record of what survey panels got around to rewriting.
    """
    latest = max((_num(r.get("last_reviewed")) or 0) for r in churn) or 2026.0
    terms = {
        "susceptibility": lambda r: _num(r.get("susceptibility")),
        "years_since_review": lambda r: (latest - _num(r.get("last_reviewed"))
                                         if _num(r.get("last_reviewed")) else None),
        "log_tasks_at_first_review": lambda r: (
            math.log(_num(r.get("tasks_first")))
            if (_num(r.get("tasks_first")) or 0) > 0 else None),
    }
    d = design(churn, lambda r: _num(r.get("turnover_rate")), terms,
               cluster=lambda r: _soc6(r["onet_soc_code"]))
    return [fit_design(
        d, name="churn_realised", outcome="task_turnover_rate",
        notes=[f"{d.dropped} of {len(churn)} occupations dropped",
               f"review lag measured against {latest:.0f}, the latest review "
               "year in the data",
               "the outcome is a revision record, not an employment outcome - "
               "no one lost a job because O*NET rewrote a task statement"],
        question="does measured exposure predict the task turnover O*NET has "
                 "already recorded, once the review cycle is held fixed?",
        prior=PRIORS["churn_realised"])]


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def index_reliability(uncertainty: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """How much of the index's between-occupation variance is signal?

    Uses the bootstrap spread this repo already computes per occupation:
    reliability = 1 - mean(within-occupation noise variance) / between-occupation
    variance. It matters for reading the coefficients, and it matters
    differently depending on which side of the equation the index sits.
    """
    sds = [_num(r.get("susceptibility_sd")) for r in uncertainty]
    means = [_num(r.get("susceptibility_mean")) for r in uncertainty]
    pairs = [(s, m) for s, m in zip(sds, means) if s is not None and m is not None]
    if len(pairs) < 10:
        return None
    noise = statistics.fmean([s * s for s, _ in pairs])
    between = statistics.variance([m for _, m in pairs])
    rel = 1.0 - noise / between if between else float("nan")
    return {
        "reliability": round(rel, 4),
        "noise_variance": round(noise, 4),
        "between_occupation_variance": round(between, 2),
        "as_outcome": "sampling noise in the dependent variable inflates the "
                      "standard errors but does not bias the coefficients, so "
                      "the validity fits need no correction",
        "as_regressor": f"classical attenuation shrinks a coefficient by about "
                        f"the reliability factor; dividing by {rel:.4f} moves the "
                        f"wage and churn coefficients by under "
                        f"{abs(1 / rel - 1) * 100:.1f}%, far less than their "
                        "standard errors",
        "what_it_does_not_cover": "this is re-run noise only. It says the model "
                                  "gives the same answer twice, not that the "
                                  "answer is right - calibration error is not "
                                  "something a resample can see",
    }


def build(tables: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Run every specification and return the report."""
    occ = tables.get("occupation_susceptibility") or []
    indices = tables.get("occupation_indices") or []
    bench = tables.get("external_benchmarks") or []
    soc = tables.get("soc_susceptibility") or []
    tasks = tables.get("task_susceptibility") or []
    churn = tables.get("task_churn_occupations") or []

    fits: list[Fit] = []
    comparisons: list[dict[str, Any]] = []
    skipped: list[str] = []

    if occ and indices:
        fits += validity_fits(occ, indices)
    else:
        skipped.append("validity_descriptors: needs occupation_susceptibility "
                       "and occupation_indices (run fetch-descriptors)")
    if bench:
        b_fits, b_cmp = incremental_fits(bench)
        fits += b_fits
        comparisons += b_cmp
    else:
        skipped.append("incremental_validity: needs external_benchmarks "
                       "(run validate-external)")
    if soc:
        fits += wage_fits(soc, occ)
    else:
        skipped.append("wage_pricing: needs soc_susceptibility (run employment)")
    if tasks:
        fits += task_fits(tasks)
    else:
        skipped.append("task_within_occupation: needs task_susceptibility")
    if churn:
        fits += churn_fits(churn)
    else:
        skipped.append("churn_realised: needs task_churn_occupations (run churn)")

    coefficient_rows = [row for f in fits for row in f.rows()]
    tested = len([r for r in coefficient_rows if r["term"] != "(intercept)"])
    return {
        "fits": [f.summary() for f in fits],
        "nested_comparisons": comparisons,
        "index_reliability": index_reliability(tables.get("occupation_uncertainty") or []),
        "coefficient_rows": coefficient_rows,
        "skipped": skipped,
        "caveats": [
            "every fit is cross-sectional. These are conditional means, not "
            "effects; nothing here observes exposure changing",
            f"{len(fits)} specifications, {tested} non-intercept coefficients. "
            "At the 5% level roughly one in twenty would clear the bar by "
            "chance, so read the large standardised coefficients and the "
            "nested comparisons, not the count of asterisks",
            "the index is one model's judgment throughout. A regression can "
            "show it is not redundant with what else exists; it cannot make it "
            "true",
        ],
    }


def log_report(report: dict[str, Any]) -> None:
    """Print the fits the way a reader checks them: prior first, then the table."""
    for fit in report["fits"]:
        log.info("=" * 78)
        log.info("%s   (outcome: %s)", fit["name"], fit["outcome"])
        log.info("  question: %s", fit["question"])
        log.info("  prior:    %s", fit["prior"])
        head = (f"  n={fit['n']}  R2={fit['r2']}  adj={fit['adj_r2']}  "
                f"{fit['se_kind']}")
        if fit["absorbed_fixed_effects"]:
            head += f"  FE absorbed={fit['absorbed_fixed_effects']}"
        log.info("%s", head)
        if fit["f_stat"] is not None:
            log.info("  joint Wald F=%s  p=%s", fit["f_stat"], fit["f_p_value"])
        log.info("  %-36s %10s %9s %8s %10s %6s", "term", "coef", "se", "t",
                 "p", "std")
        for c in fit["coefficients"]:
            log.info("  %-36s %10.4f %9.4f %8.2f %10s %6s",
                     c["term"], c["coef"], c["std_error"], c["t_stat"],
                     "" if c["p_value"] is None else f"{c['p_value']:.3g}",
                     "" if c["beta_std"] is None else f"{c['beta_std']:.2f}")
        for note in fit["notes"]:
            log.info("  note: %s", note)
    for cmp_ in report["nested_comparisons"]:
        log.info("=" * 78)
        log.info("%s - what each side is worth", cmp_["fit"])
        for key in ("ours_removed", "pre_llm_removed"):
            block = cmp_[key]
            log.info("  drop %-28s R2 %s -> %s  (loses %s)",
                     ", ".join(block["dropped"]), block["r2_full"],
                     block["r2_restricted"], block["r2_gain"])
    if report.get("skipped"):
        log.info("=" * 78)
        for s in report["skipped"]:
            log.warning("skipped %s", s)
    log.info("=" * 78)
    for c in report["caveats"]:
        log.info("caveat: %s", c)
