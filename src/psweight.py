"""
psweight.py

Small Python module for propensity score weighting, inspired by the R package
PSweight workflow: estimate propensity scores, construct balancing weights,
check balance, and estimate weighted outcome contrasts.

Binary treatment only in this first version.

New in this version:
- encoding="sklearn", scale_all=True supports OneHotEncoder followed by StandardScaler on all columns.
- encoding="pandas" supports pd.get_dummies() for MAP65-compatible analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Hiragino Sans"
import statsmodels.api as sm
from scipy.stats import norm, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Estimand = Literal["ATO", "OW", "ATE", "IPW", "ATT", "ATC"]
Encoding = Literal["sklearn", "pandas"]


@dataclass
class PSWeightResult:
    df: pd.DataFrame
    ps_col: str
    weight_col: str
    treatment_col: str
    treated_label: object
    control_label: object
    estimand: str
    covariates: list[str]
    model: object
    encoding: str = "sklearn"
    scale_all: bool = False
    design_columns: list[str] | None = None
    complete_col: str = "_complete_ps"

    @property
    def ps(self) -> pd.Series:
        return self.df[self.ps_col]

    @property
    def weights(self) -> pd.Series:
        return self.df[self.weight_col]

    def effective_sample_size(self, by_group: bool = True):
        if by_group:
            return effective_sample_size_by_group(
                self.df, group_col=self.treatment_col, weight_col=self.weight_col
            )
        return effective_sample_size(self.df[self.weight_col])


@dataclass
class WeightedMeanResult:
    group: object
    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    sum_weights: float
    ess: float
    n: int


@dataclass
class WeightedOutcomeResult:
    table: pd.DataFrame
    model: object
    group_results: dict
    diff: float
    diff_ci: tuple[float, float]
    diff_se: float
    p_value: float
    contrast_label: str
    variance_method: str


# ============================================================
# Formatting
# ============================================================

def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_ci(lo: float, hi: float, digits: int = 1) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def format_smd(x: float, digits: int = 3, abs_value: bool = True) -> str:
    if pd.isna(x):
        return ""
    val = abs(x) if abs_value else x
    return f"{val:.{digits}f}"


def median_iqr(x, digits: int = 1) -> str:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if x.empty:
        return ""
    return f"{x.median():.{digits}f} [{x.quantile(0.25):.{digits}f}, {x.quantile(0.75):.{digits}f}]"


# ============================================================
# Weighted statistics
# ============================================================

def _num(x) -> pd.Series:
    return pd.to_numeric(pd.Series(x), errors="coerce")


def _valid_xyw(x, w):
    x = _num(x)
    w = _num(w)
    mask = x.notna() & w.notna() & np.isfinite(w) & (w >= 0)
    return x[mask].to_numpy(float), w[mask].to_numpy(float)


def weighted_mean(x, w) -> float:
    xv, wv = _valid_xyw(x, w)
    if len(xv) == 0 or wv.sum() <= 0:
        return np.nan
    return float(np.average(xv, weights=wv))


def weighted_var(x, w, ddof: int = 0) -> float:
    xv, wv = _valid_xyw(x, w)
    if len(xv) < 2 or wv.sum() <= 0:
        return np.nan
    mu = np.average(xv, weights=wv)
    var = np.average((xv - mu) ** 2, weights=wv)
    if ddof == 0:
        return float(var)
    sw = wv.sum()
    sw2 = np.sum(wv ** 2)
    denom = sw - sw2 / sw
    if denom <= 0:
        return np.nan
    return float(var * sw / denom)


def weighted_sd(x, w, ddof: int = 0) -> float:
    v = weighted_var(x, w, ddof=ddof)
    return float(np.sqrt(v)) if pd.notna(v) else np.nan


def weighted_prop(x, w, level=1) -> float:
    x = pd.Series(x)
    w = _num(w)
    mask = x.notna() & w.notna() & np.isfinite(w) & (w >= 0)
    if mask.sum() == 0 or w[mask].sum() <= 0:
        return np.nan
    return float(np.sum(w[mask].to_numpy()[x[mask].to_numpy() == level]) / w[mask].sum())


def effective_sample_size(w) -> float:
    w = _num(w).dropna()
    w = w[np.isfinite(w) & (w >= 0)]
    if len(w) == 0:
        return np.nan
    denom = float(np.sum(w ** 2))
    if denom <= 0:
        return np.nan
    return float((np.sum(w) ** 2) / denom)


def effective_sample_size_by_group(df: pd.DataFrame, group_col: str, weight_col: str) -> pd.DataFrame:
    rows = []
    for g, d in df.dropna(subset=[group_col]).groupby(group_col, sort=False):
        w = d[weight_col]
        rows.append({
            "Group": g,
            "N": int(w.notna().sum()),
            "Sum of weights": float(w.sum()),
            "ESS": effective_sample_size(w),
        })
    return pd.DataFrame(rows)


def weighted_mean_ci(x, w, alpha: float = 0.05) -> WeightedMeanResult:
    x_s = _num(x)
    w_s = _num(w)
    mask = x_s.notna() & w_s.notna() & np.isfinite(w_s) & (w_s >= 0)
    if mask.sum() == 0:
        return WeightedMeanResult(None, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    mu = weighted_mean(x_s[mask], w_s[mask])
    sd = weighted_sd(x_s[mask], w_s[mask])
    ess = effective_sample_size(w_s[mask])
    se = sd / np.sqrt(ess) if pd.notna(sd) and pd.notna(ess) and ess > 0 else np.nan
    z = norm.ppf(1 - alpha / 2)
    lo = mu - z * se if pd.notna(se) else np.nan
    hi = mu + z * se if pd.notna(se) else np.nan
    return WeightedMeanResult(
        group=None, mean=float(mu), sd=float(sd), se=float(se),
        ci_low=float(lo), ci_high=float(hi), sum_weights=float(w_s[mask].sum()),
        ess=float(ess), n=int(mask.sum())
    )


# ============================================================
# SMD
# ============================================================

def smd_cont(x1, x2) -> float:
    x1 = _num(x1).dropna()
    x2 = _num(x2).dropna()
    if len(x1) == 0 or len(x2) == 0:
        return np.nan
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    n1, n2 = len(x1), len(x2)
    if n1 + n2 - 2 <= 0:
        return np.nan
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0 or np.isnan(pooled):
        return 0.0 if m1 == m2 else np.nan
    return float((m1 - m2) / pooled)


def smd_binary(x1, x2, level=1) -> float:
    x1 = pd.Series(x1).dropna()
    x2 = pd.Series(x2).dropna()
    if len(x1) == 0 or len(x2) == 0:
        return np.nan
    p1 = float((x1 == level).mean())
    p2 = float((x2 == level).mean())
    p = (p1 + p2) / 2
    denom = np.sqrt(p * (1 - p))
    if denom == 0 or np.isnan(denom):
        return 0.0 if p1 == p2 else np.nan
    return float((p1 - p2) / denom)


def weighted_smd_cont(x1, w1, x2, w2) -> float:
    m1, s1 = weighted_mean(x1, w1), weighted_sd(x1, w1)
    m2, s2 = weighted_mean(x2, w2), weighted_sd(x2, w2)
    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0 if m1 == m2 else np.nan
    return float((m1 - m2) / pooled)


def weighted_smd_binary(x1, w1, x2, w2, level=1) -> float:
    p1 = weighted_prop(x1, w1, level)
    p2 = weighted_prop(x2, w2, level)
    p = (p1 + p2) / 2
    denom = np.sqrt(p * (1 - p))
    if denom == 0 or np.isnan(denom):
        return 0.0 if p1 == p2 else np.nan
    return float((p1 - p2) / denom)


# ============================================================
# Propensity score and weights
# ============================================================

def _infer_column_types(df: pd.DataFrame, covariates: Sequence[str], categorical_covars=None, continuous_covars=None):
    if categorical_covars is not None:
        cat = list(categorical_covars)
    else:
        cat = [c for c in covariates if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    if continuous_covars is not None:
        cont = list(continuous_covars)
    else:
        cont = [c for c in covariates if c not in cat]
    return cont, cat


def make_design_matrix_pandas(
    df: pd.DataFrame,
    *,
    continuous_covars: Sequence[str],
    categorical_covars: Sequence[str],
    dummy_prefix: str | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Create a design matrix using pd.get_dummies().

    This reproduces the earlier MAP65 workflow more closely:
    numeric columns + pd.get_dummies(..., drop_first=True), followed by
    StandardScaler on all columns when scale_all=True.
    """
    parts = []

    if continuous_covars:
        parts.append(df[list(continuous_covars)].apply(pd.to_numeric, errors="coerce"))

    for c in categorical_covars:
        prefix = dummy_prefix if (dummy_prefix is not None and len(categorical_covars) == 1) else c
        parts.append(
            pd.get_dummies(
                df[c],
                prefix=prefix,
                drop_first=drop_first,
                dtype=float,
            )
        )

    if not parts:
        raise ValueError("No covariates were supplied.")

    return pd.concat(parts, axis=1)


def compute_balancing_weights(ps, treated, estimand: Estimand = "ATO", stabilize: bool = False, trim=None) -> np.ndarray:
    ps = np.asarray(ps, dtype=float)
    z = np.asarray(treated, dtype=int)
    eps = np.finfo(float).eps
    ps_safe = np.clip(ps, eps, 1 - eps)
    est = estimand.upper()
    if est == "OW":
        est = "ATO"
    if est == "ATO":
        w = np.where(z == 1, 1 - ps_safe, ps_safe)
    elif est in ("ATE", "IPW"):
        w = np.where(z == 1, 1 / ps_safe, 1 / (1 - ps_safe))
        if stabilize:
            pz = np.nanmean(z)
            w = np.where(z == 1, pz / ps_safe, (1 - pz) / (1 - ps_safe))
    elif est == "ATT":
        w = np.where(z == 1, 1.0, ps_safe / (1 - ps_safe))
    elif est == "ATC":
        w = np.where(z == 1, (1 - ps_safe) / ps_safe, 1.0)
    else:
        raise ValueError("estimand must be one of: ATO, OW, ATE, IPW, ATT, ATC")
    if trim is not None:
        lo, hi = trim
        w[(ps < lo) | (ps > hi)] = np.nan
    return w.astype(float)


def estimate_ps_weights(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    treated_label,
    covariates: Sequence[str],
    outcome_col: Optional[str] = None,
    control_label=None,
    categorical_covars: Optional[Sequence[str]] = None,
    continuous_covars: Optional[Sequence[str]] = None,
    estimand: Estimand = "ATO",
    ps_col: str = "PS",
    weight_col: str = "weight",
    complete_col: str = "_complete_ps",
    logistic_C: float = 1.0,
    max_iter: int = 1000,
    trim=None,
    random_state: int = 0,
    encoding: Encoding = "sklearn",
    scale_all: bool = False,
    dummy_prefix: str | None = None,
    drop_first: bool = True,
) -> PSWeightResult:
    missing = [c for c in [treatment_col, *covariates] if c not in df.columns]
    if missing:
        raise ValueError(f"These columns are missing from df: {missing}")
    out = df.copy()
    groups_found = list(out[treatment_col].dropna().unique())
    if control_label is None:
        controls = [g for g in groups_found if g != treated_label]
        if len(controls) != 1:
            raise ValueError("Please provide control_label when treatment_col has more than two values.")
        control_label = controls[0]
    keep_group = out[treatment_col].isin([treated_label, control_label])
    out["_treated"] = np.where(out[treatment_col] == treated_label, 1, 0)
    cont, cat = _infer_column_types(out, covariates, categorical_covars, continuous_covars)
    complete = keep_group.copy()
    for c in covariates:
        complete &= out[c].notna()
    if outcome_col is not None:
        if outcome_col not in out.columns:
            raise ValueError(f"{outcome_col!r} is not in df.columns")
        complete &= out[outcome_col].notna()
    out[complete_col] = complete
    out[ps_col] = np.nan
    out[weight_col] = np.nan
    if complete.sum() < 2:
        raise ValueError("Fewer than 2 complete cases are available for PS estimation.")
    z = out.loc[complete, "_treated"].astype(int)
    if z.nunique() != 2:
        raise ValueError("Complete cases must contain both groups.")

    lr = LogisticRegression(
        C=logistic_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )

    design_columns = None

    if encoding == "sklearn":
        # scale_all=False:
        #   continuous variables are standardized, dummy variables are not.
        # scale_all=True:
        #   numeric + one-hot dummy variables are all standardized after encoding.
        if scale_all:
            pre = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", cont),
                    (
                        "cat",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                        cat,
                    ),
                ],
                remainder="drop",
            )
            pipe = Pipeline([("preprocess", pre), ("scale", StandardScaler()), ("model", lr)])
        else:
            pre = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), cont),
                    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
                ],
                remainder="drop",
            )
            pipe = Pipeline([("preprocess", pre), ("model", lr)])

        X = out.loc[complete, list(covariates)]
        pipe.fit(X, z)
        ps = pipe.predict_proba(X)[:, 1]
        model = pipe
        try:
            design_columns = list(pipe.named_steps["preprocess"].get_feature_names_out())
        except Exception:
            design_columns = None

    elif encoding == "pandas":
        # MAP65-compatible mode: pd.get_dummies + optional StandardScaler on all columns.
        X_df = make_design_matrix_pandas(
            out.loc[complete],
            continuous_covars=cont,
            categorical_covars=cat,
            dummy_prefix=dummy_prefix,
            drop_first=drop_first,
        )
        design_columns = list(X_df.columns)
        X = X_df.to_numpy(dtype=float)
        if scale_all:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            scaler = None
        lr.fit(X, z)
        ps = lr.predict_proba(X)[:, 1]
        model = {"logistic": lr, "scaler": scaler, "design_columns": design_columns}

    else:
        raise ValueError("encoding must be 'sklearn' or 'pandas'.")

    w = compute_balancing_weights(ps, z.to_numpy(), estimand=estimand, trim=trim)
    out.loc[complete, ps_col] = ps
    out.loc[complete, weight_col] = w
    return PSWeightResult(
        out,
        ps_col,
        weight_col,
        treatment_col,
        treated_label,
        control_label,
        estimand.upper(),
        list(covariates),
        model,
        encoding,
        scale_all,
        design_columns,
        complete_col,
    )


def estimate_overlap_weights(
    df: pd.DataFrame,
    *,
    group_col: str = "Group",
    groups: Sequence = ("treated", "control"),
    continuous_covars: Sequence[str],
    categorical_covars: Sequence[str] = (),
    outcome_col: Optional[str] = None,
    ps_col: str = "PS",
    weight_col: str = "OW",
    complete_col: str = "_complete",
    encoding: Encoding = "pandas",
    scale_all: bool = True,
    dummy_prefix: str | None = "dept",
    drop_first: bool = True,
    **kwargs,
) -> PSWeightResult:
    covariates = list(continuous_covars) + list(categorical_covars)
    return estimate_ps_weights(
        df,
        treatment_col=group_col,
        treated_label=groups[0],
        control_label=groups[1],
        covariates=covariates,
        continuous_covars=list(continuous_covars),
        categorical_covars=list(categorical_covars),
        outcome_col=outcome_col,
        estimand="ATO",
        ps_col=ps_col,
        weight_col=weight_col,
        complete_col=complete_col,
        encoding=encoding,
        scale_all=scale_all,
        dummy_prefix=dummy_prefix,
        drop_first=drop_first,
        **kwargs,
    )


# ============================================================
# Balance table
# ============================================================

def build_weighted_table(
    df: pd.DataFrame,
    *,
    continuous: Sequence[str],
    binary_vars: Sequence[str],
    categorical: Sequence[str],
    group_col: str = "Group",
    weight_col: str = "weight",
    groups: Optional[Sequence] = None,
    digits_cont: int = 1,
    digits_pct: int = 1,
    digits_smd: int = 3,
    include_effective_n: bool = True,
    include_sum_weights: bool = True,
) -> pd.DataFrame:
    if groups is None:
        groups = list(df[group_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError("Only two groups are supported.")
    g1, g2 = groups
    sub = df[df[group_col].isin(groups)].copy()
    d1 = sub[sub[group_col] == g1]
    d2 = sub[sub[group_col] == g2]
    w1, w2 = d1[weight_col], d2[weight_col]
    rows = []
    if include_sum_weights:
        rows.append({"Variable": "Sum of weights", g1: f"{w1.sum():.1f}", g2: f"{w2.sum():.1f}", "SMD": ""})
    if include_effective_n:
        rows.append({"Variable": "Effective sample size", g1: f"{effective_sample_size(w1):.1f}", g2: f"{effective_sample_size(w2):.1f}", "SMD": ""})
    for var in continuous:
        m1, s1 = weighted_mean(d1[var], w1), weighted_sd(d1[var], w1)
        m2, s2 = weighted_mean(d2[var], w2), weighted_sd(d2[var], w2)
        smd = weighted_smd_cont(d1[var], w1, d2[var], w2)
        rows.append({"Variable": var, g1: f"{m1:.{digits_cont}f} ({s1:.{digits_cont}f})", g2: f"{m2:.{digits_cont}f} ({s2:.{digits_cont}f})", "SMD": format_smd(smd, digits_smd)})
    for var in binary_vars:
        p1, p2 = weighted_prop(d1[var], w1, 1), weighted_prop(d2[var], w2, 1)
        smd = weighted_smd_binary(d1[var], w1, d2[var], w2, level=1)
        rows.append({"Variable": f"{var}, weighted %", g1: f"{p1 * 100:.{digits_pct}f}%", g2: f"{p2 * 100:.{digits_pct}f}%", "SMD": format_smd(smd, digits_smd)})
    for var in categorical:
        levels = sorted(sub[var].dropna().unique())
        smds = []
        for lv in levels:
            s = weighted_smd_binary((d1[var] == lv).astype(float), w1, (d2[var] == lv).astype(float), w2, level=1)
            if pd.notna(s):
                smds.append(abs(s))
        rows.append({"Variable": var, g1: "", g2: "", "SMD": f"{max(smds):.{digits_smd}f}" if smds else ""})
        for lv in levels:
            p1, p2 = weighted_prop(d1[var], w1, lv), weighted_prop(d2[var], w2, lv)
            smd = weighted_smd_binary((d1[var] == lv).astype(float), w1, (d2[var] == lv).astype(float), w2, level=1)
            rows.append({"Variable": f"  {lv}", g1: f"{p1 * 100:.{digits_pct}f}%", g2: f"{p2 * 100:.{digits_pct}f}%", "SMD": format_smd(smd, digits_smd)})
    return pd.DataFrame(rows)


def build_unweighted_smd_table(df: pd.DataFrame, *, continuous, binary_vars, categorical, group_col="Group", groups=None, digits_smd=3) -> pd.DataFrame:
    if groups is None:
        groups = list(df[group_col].dropna().unique())
    g1, g2 = groups
    sub = df[df[group_col].isin(groups)].copy()
    d1, d2 = sub[sub[group_col] == g1], sub[sub[group_col] == g2]
    rows = []
    for var in continuous:
        rows.append({"Variable": var, "SMD": format_smd(smd_cont(d1[var], d2[var]), digits_smd)})
    for var in binary_vars:
        rows.append({"Variable": var, "SMD": format_smd(smd_binary(d1[var], d2[var]), digits_smd)})
    for var in categorical:
        levels = sorted(sub[var].dropna().unique())
        smds = []
        for lv in levels:
            s = smd_binary((d1[var] == lv).astype(float), (d2[var] == lv).astype(float))
            if pd.notna(s):
                smds.append(abs(s))
        rows.append({"Variable": var, "SMD": f"{max(smds):.{digits_smd}f}" if smds else ""})
        for lv in levels:
            s = smd_binary((d1[var] == lv).astype(float), (d2[var] == lv).astype(float))
            rows.append({"Variable": f"  {lv}", "SMD": format_smd(s, digits_smd)})
    return pd.DataFrame(rows)


# ============================================================
# Outcome analysis
# ============================================================

def weighted_outcome_summary(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    group_col: str = "Group",
    groups: Optional[Sequence] = None,
    weight_col: str = "weight",
    treatment_col: str = "_treated",
    alpha: float = 0.05,
    robust_cov: Optional[str] = "HC3",
    digits: int = 1,
) -> WeightedOutcomeResult:
    if groups is None:
        groups = list(df[group_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError("Only two groups are supported.")
    g1, g2 = groups
    sub = df[df[group_col].isin(groups) & df[outcome_col].notna() & df[weight_col].notna()].copy()
    if sub.empty:
        raise ValueError("No complete cases for weighted outcome analysis.")
    if treatment_col not in sub.columns:
        sub[treatment_col] = np.where(sub[group_col] == g1, 1, 0)
    y = pd.to_numeric(sub[outcome_col], errors="coerce")
    z = pd.to_numeric(sub[treatment_col], errors="coerce")
    w = pd.to_numeric(sub[weight_col], errors="coerce")
    valid = y.notna() & z.notna() & w.notna() & np.isfinite(w) & (w >= 0)
    sub = sub.loc[valid].copy()
    y, z, w = y.loc[valid].to_numpy(float), z.loc[valid].to_numpy(float), w.loc[valid].to_numpy(float)
    X = sm.add_constant(z, has_constant="add")
    fit = sm.WLS(y, X, weights=w).fit()
    variance_method = "model-based"
    fit_inf = fit
    if robust_cov is not None:
        fit_inf = fit.get_robustcov_results(cov_type=robust_cov)
        variance_method = robust_cov
    diff = float(fit_inf.params[1])
    diff_se = float(fit_inf.bse[1])
    ci = np.asarray(fit_inf.conf_int(alpha=alpha))
    diff_ci = (float(ci[1, 0]), float(ci[1, 1]))
    p_value = float(fit_inf.pvalues[1])
    row = {"Analysis": "Overlap-weighted WLS"}
    group_results = {}
    for g in groups:
        d = sub[sub[group_col] == g]
        res = weighted_mean_ci(d[outcome_col], d[weight_col], alpha=alpha)
        res.group = g
        group_results[g] = res
        row[f"{g} weighted mean (SD)"] = f"{res.mean:.{digits}f} ({res.sd:.{digits}f})"
        row[f"{g} weighted mean 95% CI"] = format_ci(res.ci_low, res.ci_high, digits)
        row[f"{g} ESS"] = f"{res.ess:.1f}"
    row[f"Diff ({g1} − {g2})"] = f"{diff:.{digits}f}"
    row["95% CI"] = format_ci(*diff_ci, digits=digits)
    row["p"] = format_p(p_value)
    row["Variance"] = variance_method
    table = pd.DataFrame([row])
    return WeightedOutcomeResult(table, fit_inf, group_results, diff, diff_ci, diff_se, p_value, f"{g1} − {g2}", variance_method)


def unweighted_outcome_row(df: pd.DataFrame, *, outcome_col: str, group_col="Group", groups=None, digits=1) -> pd.DataFrame:
    if groups is None:
        groups = list(df[group_col].dropna().unique())
    g1, g2 = groups
    sub = df[df[group_col].isin(groups) & df[outcome_col].notna()].copy()
    y1 = pd.to_numeric(sub.loc[sub[group_col] == g1, outcome_col], errors="coerce").dropna()
    y2 = pd.to_numeric(sub.loc[sub[group_col] == g2, outcome_col], errors="coerce").dropna()
    p = mannwhitneyu(y1, y2, alternative="two-sided").pvalue if len(y1) and len(y2) else np.nan
    return pd.DataFrame([{ "Analysis": "Unweighted", f"{g1} median [IQR]": median_iqr(y1, digits), f"{g2} median [IQR]": median_iqr(y2, digits), "p": format_p(p)}])


def combined_outcome_table(df: pd.DataFrame, *, outcome_col: str, group_col="Group", groups=None, weight_col="weight", treatment_col="_treated", robust_cov="HC3", digits=1):
    unweighted = unweighted_outcome_row(df, outcome_col=outcome_col, group_col=group_col, groups=groups, digits=digits)
    weighted = weighted_outcome_summary(df, outcome_col=outcome_col, group_col=group_col, groups=groups, weight_col=weight_col, treatment_col=treatment_col, robust_cov=robust_cov, digits=digits)
    return pd.concat([unweighted, weighted.table], ignore_index=True), weighted


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_weighted_outcome(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    treated_label,
    control_label,
    covariates: Sequence[str],
    outcome_col: str,
    categorical_covars: Optional[Sequence[str]] = None,
    continuous_covars: Optional[Sequence[str]] = None,
    estimand: Estimand = "ATO",
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 0,
    logistic_C: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n = len(df)
    mu_t, mu_c, diff = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bdf = df.iloc[idx].reset_index(drop=True)
        try:
            ps = estimate_ps_weights(
                bdf,
                treatment_col=treatment_col,
                treated_label=treated_label,
                control_label=control_label,
                covariates=covariates,
                outcome_col=outcome_col,
                categorical_covars=categorical_covars,
                continuous_covars=continuous_covars,
                estimand=estimand,
                logistic_C=logistic_C,
            )
            out = weighted_outcome_summary(ps.df, outcome_col=outcome_col, group_col=treatment_col, groups=[treated_label, control_label], weight_col="weight", treatment_col="_treated", robust_cov=None)
            mu_t.append(out.group_results[treated_label].mean)
            mu_c.append(out.group_results[control_label].mean)
            diff.append(out.diff)
        except Exception:
            continue
    if len(diff) == 0:
        raise RuntimeError("All bootstrap samples failed.")
    qlo, qhi = alpha / 2, 1 - alpha / 2
    def ci(x):
        return np.nanquantile(np.asarray(x, dtype=float), [qlo, qhi])
    mt_lo, mt_hi = ci(mu_t)
    mc_lo, mc_hi = ci(mu_c)
    d_lo, d_hi = ci(diff)
    return pd.DataFrame([{ 
        "n_boot_success": len(diff),
        f"{treated_label} weighted mean bootstrap CI": f"[{mt_lo:.1f}, {mt_hi:.1f}]",
        f"{control_label} weighted mean bootstrap CI": f"[{mc_lo:.1f}, {mc_hi:.1f}]",
        f"Diff ({treated_label} − {control_label}) bootstrap CI": f"[{d_lo:.1f}, {d_hi:.1f}]",
    }])


# ============================================================
# Love plot
# ============================================================

def love_plot(
    before_table: pd.DataFrame,
    after_table: pd.DataFrame,
    *,
    variable_col="Variable",
    smd_col="SMD",
    exclude_variables: Optional[Iterable[str]] = None,
    threshold=0.1,
    out_path: Optional[str] = None,
    sort_by: str = "table",   # "table" or "smd"
):
    import matplotlib.pyplot as plt

    exclude = set(exclude_variables or [])

    def norm_name(v):
        v = str(v).strip()
        return (
            v.replace(", n (%)", "")
             .replace(", weighted %", "")
             .replace(", n (weighted %)", "")
             .strip()
        )

    def extract(tbl):
        res = {}
        order = []

        for _, row in tbl.iterrows():
            name = norm_name(row[variable_col])

            if name in exclude:
                continue

            try:
                smd = abs(float(str(row[smd_col]).strip()))
            except Exception:
                continue

            res[name] = smd
            order.append(name)

        return res, order

    b, before_order = extract(before_table)
    a, _ = extract(after_table)

    if sort_by == "smd":
        common = sorted(
            [v for v in before_order if v in a],
            key=lambda x: b[x],
            reverse=True,
        )
    elif sort_by == "table":
        common = [v for v in before_order if v in a]
    else:
        raise ValueError("sort_by must be either 'table' or 'smd'")

    bv = [b[v] for v in common]
    av = [a[v] for v in common]

    n = len(common)
    fig_h = max(5, n * 0.30 + 1.5)

    fig, ax = plt.subplots(figsize=(7, fig_h))

    y = np.arange(n)

    for i in range(n):
        ax.plot([bv[i], av[i]], [y[i], y[i]], lw=0.8, zorder=2)

    ax.scatter(bv, y, s=45, zorder=4, label="Before weighting")
    ax.scatter(av, y, s=45, zorder=4, label="After weighting")

    ax.axvline(0, lw=1.0, zorder=3)
    ax.axvline(threshold, lw=1.0, ls="--", alpha=0.6, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(common, fontsize=8)

    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title("Covariate balance before and after weighting")
    ax.set_xlim(left=-0.01)

    ax.grid(axis="x", lw=0.5, zorder=1)
    ax.legend(fontsize=9, frameon=True, loc="lower right")

    # Show first Table 1 variable at the top
    ax.invert_yaxis()

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig

def forest_plot(
    outcome_table: pd.DataFrame,
    *,
    outcome_col="Outcome",
    rd_col="Weighted RD",
    ci_col="95% CI",
    primary_outcome=None,
    xlim=None,
    margin=5,
    symmetric_xlim=False,
    figsize=None,
    out_path=None,
):
    """
    Forest plot of weighted risk differences.

    Parameters
    ----------
    outcome_table : pd.DataFrame
        Table containing outcome names, weighted RD, and 95% CI.

    outcome_col : str
        Column name for outcome labels.

    rd_col : str
        Column name for weighted risk difference.
        Values may be numeric or strings such as "-35.1%".

    ci_col : str
        Column name for confidence intervals.
        Values should look like "[-58.7, -11.6]".

    primary_outcome : str or None
        Outcome label to show in bold.

    xlim : tuple or None
        If None, x-axis limits are determined from the CI range.

    margin : float
        Extra margin in percentage points added to both sides.

    symmetric_xlim : bool
        If True, use symmetric x-axis limits around zero.

    figsize : tuple or None
        If None, figure height is based on number of outcomes.

    out_path : str or Path or None
        If provided, save the figure.
    """

    df = outcome_table.copy()

    # ----------------------------
    # Helper functions
    # ----------------------------
    def parse_percent(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        x = str(x).strip().replace("%", "")
        return float(x)

    def parse_ci(ci):
        if pd.isna(ci):
            return np.nan, np.nan

        nums = re.findall(r"-?\d+\.?\d*", str(ci))

        if len(nums) < 2:
            return np.nan, np.nan

        return float(nums[0]), float(nums[1])

    # ----------------------------
    # Parse RD and CI
    # ----------------------------
    df["RD_plot"] = df[rd_col].apply(parse_percent)

    cis = df[ci_col].apply(parse_ci)
    df["CI_low_plot"] = cis.apply(lambda x: x[0])
    df["CI_high_plot"] = cis.apply(lambda x: x[1])

    df = df.dropna(
        subset=[outcome_col, "RD_plot", "CI_low_plot", "CI_high_plot"]
    ).reset_index(drop=True)

    # ----------------------------
    # Auto x-axis limits
    # ----------------------------
    if xlim is None:
        xmin = np.nanmin(df["CI_low_plot"])
        xmax = np.nanmax(df["CI_high_plot"])

        if symmetric_xlim:
            lim = max(abs(xmin), abs(xmax))
            lim = np.ceil((lim + margin) / 5) * 5
            xlim = (-lim, lim)
        else:
            xmin = np.floor((xmin - margin) / 5) * 5
            xmax = np.ceil((xmax + margin) / 5) * 5

            # Always include zero
            xmin = min(xmin, 0)
            xmax = max(xmax, 0)

            xlim = (xmin, xmax)

    # ----------------------------
    # Figure size
    # ----------------------------
    n = len(df)

    if figsize is None:
        figsize = (8, max(4, n * 0.55 + 1.6))

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(n)[::-1]

    # ----------------------------
    # Plot points and CIs
    # ----------------------------
    for yy, (_, row) in zip(y, df.iterrows()):

        rd = row["RD_plot"]
        lo = row["CI_low_plot"]
        hi = row["CI_high_plot"]

        significant = (hi < 0) or (lo > 0)

        if rd < 0:
            color = "#1b9e77"   # favors FADE
        else:
            color = "#d95f02"   # favors AE

        alpha = 1.0 if significant else 0.65

        ax.plot(
            [lo, hi],
            [yy, yy],
            lw=2.2,
            color=color,
            alpha=alpha,
            zorder=2,
        )

        ax.scatter(
            rd,
            yy,
            s=130,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=alpha,
            zorder=3,
        )

    # ----------------------------
    # Reference line
    # ----------------------------
    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    # ----------------------------
    # Labels
    # ----------------------------
    ax.set_xlim(*xlim)
    ax.set_yticks(y)

    labels = []
    for outcome in df[outcome_col].astype(str):
        if outcome == primary_outcome:
            labels.append(r"$\bf{" + outcome.replace(" ", r"\ ") + "}$")
        else:
            labels.append(outcome)

    ax.set_yticklabels(labels, fontsize=11)

    ax.set_xlabel("Risk difference (%)")
    ax.set_title("Weighted risk differences after overlap weighting")

    # ----------------------------
    # Right-side RD text
    # ----------------------------
    text_x = xlim[1] + 0.06 * (xlim[1] - xlim[0])

    for yy, (_, row) in zip(y, df.iterrows()):

        txt = (
            f"{row['RD_plot']:.1f}% "
            f"[{row['CI_low_plot']:.1f}, {row['CI_high_plot']:.1f}]"
        )

        weight = "bold" if row[outcome_col] == primary_outcome else "normal"

        ax.text(
            text_x,
            yy,
            txt,
            va="center",
            ha="left",
            fontsize=10,
            fontweight=weight,
        )

    # Extend plotting area to make room for right-side text
    ax.set_xlim(xlim[0], xlim[1] + 0.45 * (xlim[1] - xlim[0]))

    # ----------------------------
    # Bottom direction labels
    # ----------------------------
    ax.text(
        xlim[0],
        -0.9,
        "← favors FADE",
        fontsize=11,
        ha="left",
        va="center",
    )

    ax.text(
        xlim[1],
        -0.9,
        "favors AE →",
        fontsize=11,
        ha="right",
        va="center",
    )

    # ----------------------------
    # Style
    # ----------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linewidth=0.5, alpha=0.4)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig