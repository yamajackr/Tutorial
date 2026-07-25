"""
kacr.py

Integrated utility module for Kameda Anesthesiology clinical research.

This file combines the useful functions from my_mod.py and psweight.py while
removing duplicate helper functions. It includes:
- file/data helpers
- drug-event extraction and PK simulation helpers
- Table 1 utilities
- propensity score estimation and weighting: overlap weighting, IPTW, ATT, ATC
- weighted balance/outcome tables
- Love plot and forest plot functions

Group order matters for signed SMDs:
    groups=["FADE", "AE"] means positive SMD = FADE higher.
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.integrate import solve_ivp
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency, norm
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import confint_proportions_2indep

Estimand = Literal["ATO", "OW", "ATE", "IPW", "ATT", "ATC"]
Encoding = Literal["pandas", "statsmodels", "sklearn"]
SMDType = Literal["absolute", "signed"]


# ============================================================
# Shared formatting and unweighted statistics
# ============================================================

def _num(x) -> pd.Series:
    """Convert an array-like object to a numeric pandas Series."""
    return pd.to_numeric(pd.Series(x), errors="coerce")


def format_p(p: float) -> str:
    """Format p values for tables."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_ci(lo: float, hi: float, digits: int = 1) -> str:
    """Format confidence intervals."""
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


@dataclass
class UnweightedRiskDifferenceResult:
    """Crude risk difference and confidence interval for two independent groups."""

    risk_group1: float
    risk_group2: float
    risk_difference: float
    ci_low: float
    ci_high: float
    count_group1: int
    n_group1: int
    count_group2: int
    n_group2: int
    method: str
    alpha: float

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


def unweighted_risk_difference(
    count_group1: int,
    n_group1: int,
    count_group2: int,
    n_group2: int,
    *,
    alpha: float = 0.05,
    method: str = "newcomb",
    correction: bool = False,
) -> UnweightedRiskDifferenceResult:
    """
    Estimate an unadjusted risk difference and its confidence interval.

    The risk difference is defined as:

        risk in group 1 - risk in group 2

    By default, the 95% confidence interval is calculated with the
    Newcombe method based on Wilson score intervals. This method is more
    reliable than the simple Wald interval when samples are small or event
    counts are sparse.

    Parameters
    ----------
    count_group1, count_group2
        Number of patients with the outcome in each group.
    n_group1, n_group2
        Total number of patients with observed outcome data in each group.
    alpha
        Two-sided significance level. alpha=0.05 gives a 95% CI.
    method
        Method accepted by statsmodels.confint_proportions_2indep for
        compare="diff". Recommended: "newcomb".
    correction
        Small-sample correction passed to statsmodels. It is not used by
        every method. False is used for the standard Newcombe interval.

    Returns
    -------
    UnweightedRiskDifferenceResult
        Risks, crude risk difference, and confidence interval.
    """
    values = [count_group1, n_group1, count_group2, n_group2]
    if any(pd.isna(v) for v in values):
        raise ValueError("Counts and sample sizes must not be missing.")

    count_group1 = int(count_group1)
    n_group1 = int(n_group1)
    count_group2 = int(count_group2)
    n_group2 = int(n_group2)

    if n_group1 <= 0 or n_group2 <= 0:
        raise ValueError("Each group must have at least one observation.")
    if not 0 <= count_group1 <= n_group1:
        raise ValueError("count_group1 must be between 0 and n_group1.")
    if not 0 <= count_group2 <= n_group2:
        raise ValueError("count_group2 must be between 0 and n_group2.")

    risk_group1 = count_group1 / n_group1
    risk_group2 = count_group2 / n_group2
    risk_difference = risk_group1 - risk_group2

    ci_low, ci_high = confint_proportions_2indep(
        count1=count_group1,
        nobs1=n_group1,
        count2=count_group2,
        nobs2=n_group2,
        compare="diff",
        method=method,
        alpha=alpha,
        correction=correction,
    )

    return UnweightedRiskDifferenceResult(
        risk_group1=float(risk_group1),
        risk_group2=float(risk_group2),
        risk_difference=float(risk_difference),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        count_group1=count_group1,
        n_group1=n_group1,
        count_group2=count_group2,
        n_group2=n_group2,
        method=method,
        alpha=alpha,
    )


def format_smd(x: float, digits: int = 3, smd_type: SMDType | None = None) -> str:
    """
    Format standardized mean difference.

    smd_type=None preserves the sign. Use smd_type="absolute" when you want
    to display |SMD|. This keeps calculation and display separate.
    """
    if pd.isna(x):
        return ""
    if smd_type == "absolute":
        x = abs(x)
    elif smd_type not in (None, "signed"):
        raise ValueError("smd_type must be None, 'absolute', or 'signed'")
    return f"{x:.{digits}f}"


def median_iqr(x, digits: int | None = 1) -> str:
    """Format median [IQR]. If digits=None, use 2 digits for small values."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if x.empty:
        return ""
    med = x.median()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    d = digits if digits is not None else (2 if abs(med) < 1 else 1)
    return f"{med:.{d}f} [{q1:.{d}f}, {q3:.{d}f}]"


def smd_cont(x1, x2) -> float:
    """Signed SMD for continuous variables. Positive means x1 > x2."""
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
    """Signed SMD for binary/level indicators. Positive means x1 has higher proportion."""
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

def find_concat(pattern, read_func, **kwargs):
    files = glob.glob(pattern)
    files = [
    f for f in files
    if not os.path.basename(f).startswith("~$")
    ]

    if not files:
        raise FileNotFoundError(
            f"No files found for pattern: {pattern}\n"
            "Check:\n"
            " - correct directory\n"
            " - file extensions\n"
            " - files are not inside subfolders"
        )
    dfs = [read_func(file, **kwargs) for file in files]
    return pd.concat(dfs, ignore_index=True)

def _parse_detail(detail: str):
    if detail is None:
        return None, None
    s = str(detail).strip()
    if not s:
        return None, None
    m = re.search(r"([-+]?\d+(\.\d+)?)\s*([^\s]+)?", s)
    if not m:
        return None, None
    try:
        val = float(m.group(1))
    except Exception:
        val = None
    unit = m.group(3) if m.group(3) else None
    return val, unit

def _ensure_datetime_col(df: pd.DataFrame, col="DATETIME") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        return out
    if pd.api.types.is_datetime64_any_dtype(out[col]):
        return out

    s = out[col]
    if pd.api.types.is_numeric_dtype(s):
        s_str = s.astype("Int64").astype(str)
    else:
        s_str = s.astype(str)

    dt1 = pd.to_datetime(s_str, format="%Y%m%d%H%M%S", errors="coerce")
    if dt1.notna().any():
        out[col] = dt1
        return out

    out[col] = pd.to_datetime(s, errors="coerce")
    return out

def _normalize_drug_name(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace(" ", "").replace("　", "")

def _sex_to_int(sex_value) -> int:
    s = str(sex_value).strip().lower()
    if s in ["male", "m", "0", "man"]:
        return 0
    if s in ["female", "f", "1", "woman"]:
        return 1
    try:
        v = int(float(sex_value))
        return 1 if v == 1 else 0
    except Exception:
        return 0

def _high_asa_to_int(asaps_value) -> int:
    # returns 1 if ASA>=3 else 0
    if pd.isna(asaps_value):
        return 0
    s = str(asaps_value).strip().upper()
    digits = "".join([c for c in s if c.isdigit()])
    if digits == "":
        return 0
    try:
        asa = int(digits[0])
        return 1 if asa >= 3 else 0
    except Exception:
        return 0

FLM_MG_PER_ML = 0.5 / 5.0

def flm_volume_ml_to_mg(vol_ml: float) -> float:
    """Convert flumazenil volume (mL) to mg using 0.1 mg/mL."""
    if vol_ml is None or (isinstance(vol_ml, float) and np.isnan(vol_ml)):
        return np.nan
    return float(vol_ml) * FLM_MG_PER_ML

def _ffm_janmahasatian(sex: int, weight_kg: float, height_cm: float) -> float:
    """
    Janmahasatian FFM (adult) used inside Al-Sallami extension in forstudy.html.
    sex: 0 male, 1 female
    """
    h_m = height_cm / 100.0
    bmi = weight_kg / (h_m * h_m)
    if sex == 0:
        return 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    else:
        return 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)

def _ffm_alsallami(sex: int, age_yr: float, weight_kg: float, height_cm: float) -> float:
    """
    Al-Sallami FFM predictor (the same form as forstudy.html).
    """
    ffm_j = _ffm_janmahasatian(sex, weight_kg, height_cm)

    # Maturation extensions differ by sex (as in forstudy.html)
    if sex == 0:
        # male: (0.88 + (1-0.88)/(1+(age/13.4)^(-12.7))) * FFM_j
        return (0.88 + (1.0 - 0.88) / (1.0 + (age_yr / 13.4) ** (-12.7))) * ffm_j
    else:
        # female: (1.11 + (1-1.11)/(1+(age/7.1)^(-1.1))) * FFM_j
        return (1.11 + (1.0 - 1.11) / (1.0 + (age_yr / 7.1) ** (-1.1))) * ffm_j

def _fsigmoid(x: float, b: float, c: float) -> float:
    """Fsigmoid(x,b,c)=x^c/(x^c+b^c)"""
    x = float(x)
    return (x**c) / (x**c + b**c)

def select_events_rmz(drug_event_case: pd.DataFrame) -> pd.DataFrame:
    df = drug_event_case.copy()
    df["_DN"] = df["DRUG_NAME"].map(_normalize_drug_name)
    inc = df["_DN"].str.contains(r"レミマゾラム", na=False)
    return df.loc[inc].drop(columns=["_DN"], errors="ignore")

def select_events_flumazenil(drug_event_case: pd.DataFrame) -> pd.DataFrame:
    df = drug_event_case.copy()
    df["_DN"] = df["DRUG_NAME"].map(_normalize_drug_name)
    inc = df["_DN"].str.contains(r"(フルマゼニル)", na=False, regex=True)
    return df.loc[inc].drop(columns=["_DN"], errors="ignore")

def events_to_bolus_and_infusion_rmz(ev: pd.DataFrame, t0: pd.Timestamp, weight_kg: float):
    bolus = {}
    inf_changes = {}
    if ev.empty or pd.isna(t0):
        return bolus, inf_changes

    x = ev.copy()
    x = x.dropna(subset=["DATETIME"]).sort_values("DATETIME")
    x["t_min"] = (x["DATETIME"] - t0).dt.total_seconds() / 60.0

    inf = x[x["CONTENT"].isin(["開始", "流速変更", "終了"])].copy()
    for _, r in inf.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue
        if str(r["CONTENT"]) == "終了":
            inf_changes[t] = 0.0
            continue

        val = r.get("FLOW", np.nan)
        unit = None
        if pd.isna(val):
            dval, dunit = _parse_detail(r.get("DETAIL", ""))
            val, unit = dval, dunit
        else:
            _, dunit = _parse_detail(r.get("DETAIL", ""))
            unit = dunit

        if val is None or pd.isna(val):
            continue

        if unit is None:
            unit = "mg/kg/h"
        unit = str(unit).lower()

        if "mg/kg/h" in unit or "mg/kg/hr" in unit:
            mg_per_min = float(val) * float(weight_kg) / 60.0
        elif "mg/h" in unit or "mg/hr" in unit:
            mg_per_min = float(val) / 60.0
        elif "mg/min" in unit:
            mg_per_min = float(val)
        else:
            raise ValueError(f"Unsupported RMZ infusion unit: {unit} (DETAIL={r.get('DETAIL','')})")

        inf_changes[t] = mg_per_min

    b = x[x["CONTENT"].isin(["ワンショット"])].copy()
    for _, r in b.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue
        dval, dunit = _parse_detail(r.get("DETAIL", ""))
        if dval is None:
            fv = r.get("FLOW", np.nan)
            if pd.isna(fv):
                continue
            dval = float(fv)
            dunit = dunit or "mg"

        if dunit == "mg":
            pass

        elif dunit == "ml" or dunit == "mL":
            # 0.1 mg/mL → divide by 10
            dval = float(dval) / 10.0

        else:
            raise ValueError(
                f"Unsupported RMZ bolus unit: {dunit} (DETAIL={r.get('DETAIL','')})"
            )

        bolus[t] = bolus.get(t, 0.0) + float(dval)

    return bolus, inf_changes

def events_to_bolus_and_infusion_flumazenil(ev: pd.DataFrame, t0: pd.Timestamp):
    """
    Modified:
    - Accept FLM values recorded in mL and convert to mg using 0.1 mg/mL (0.5 mg = 5 mL).
    - Still accepts mg, mg/h, mg/min.
    - Also supports mL/h, mL/min for infusion if present.
    """
    bolus = {}
    inf_changes = {}
    if ev.empty or pd.isna(t0):
        return bolus, inf_changes

    x = ev.copy()
    x = x.dropna(subset=["DATETIME"]).sort_values("DATETIME")
    x["t_min"] = (x["DATETIME"] - t0).dt.total_seconds() / 60.0

    # ---- Infusion changes ----
    inf = x[x["CONTENT"].isin(["開始", "流速変更", "終了"])].copy()
    for _, r in inf.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue
        if str(r["CONTENT"]) == "終了":
            inf_changes[t] = 0.0
            continue

        val = r.get("FLOW", np.nan)
        unit = None
        if pd.isna(val):
            dval, dunit = _parse_detail(r.get("DETAIL", ""))
            val, unit = dval, dunit
        else:
            _, dunit = _parse_detail(r.get("DETAIL", ""))
            unit = dunit

        if val is None or pd.isna(val):
            continue

        if unit is None:
            unit = "mg/h"
        unit = str(unit).lower()

        # mg-based
        if "mg/h" in unit or "mg/hr" in unit:
            mg_per_min = float(val) / 60.0
        elif "mg/min" in unit:
            mg_per_min = float(val)

        # mL-based (NEW)
        elif "ml/h" in unit or "ml/hr" in unit:
            mg_per_min = flm_volume_ml_to_mg(float(val)) / 60.0
        elif "ml/min" in unit:
            mg_per_min = flm_volume_ml_to_mg(float(val))

        else:
            raise ValueError(f"Unsupported FLM infusion unit: {unit} (DETAIL={r.get('DETAIL','')})")

        inf_changes[t] = mg_per_min

    # ---- Bolus ----
    b = x[x["CONTENT"].isin(["ワンショット"])].copy()
    for _, r in b.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue

        dval, dunit = _parse_detail(r.get("DETAIL", ""))
        if dval is None:
            fv = r.get("FLOW", np.nan)
            if pd.isna(fv):
                continue
            dval = float(fv)
            dunit = dunit or "mg"

        unit = (dunit or "mg").lower()

        # mg bolus
        if unit == "mg":
            dose_mg = float(dval)

        # mL bolus (NEW)
        elif unit in ["ml", "mℓ"]:
            dose_mg = flm_volume_ml_to_mg(float(dval))

        else:
            raise ValueError(f"Unsupported FLM bolus unit: {unit} (DETAIL={r.get('DETAIL','')})")

        bolus[t] = bolus.get(t, 0.0) + float(dose_mg)

    return bolus, inf_changes

def piecewise_rate(t: float, inf_changes: dict) -> float:
    if not inf_changes:
        return 0.0
    keys = [k for k in inf_changes.keys() if k <= t]
    if not keys:
        return 0.0
    return inf_changes[max(keys)]

def select_events_remifentanil(drug_event_case: pd.DataFrame) -> pd.DataFrame:
    df = drug_event_case.copy()
    df["_DN"] = df["DRUG_NAME"].map(_normalize_drug_name)
    # adjust keyword(s) to match your system spelling
    inc = df["_DN"].str.contains(r"(レミフェンタニル|アルチバ)", na=False, regex=True)
    return df.loc[inc].drop(columns=["_DN"], errors="ignore")

def select_events_fentanyl(drug_event_case: pd.DataFrame) -> pd.DataFrame:
    df = drug_event_case.copy()
    df["_DN"] = df["DRUG_NAME"].map(_normalize_drug_name)
    inc = df["_DN"].str.contains(r"(フェンタニル|fentanyl)", na=False, regex=True) & ~df["_DN"].str.contains(r"(レミフェンタニル|remifentanil)", na=False, regex=True)
    return df.loc[inc].drop(columns=["_DN"], errors="ignore")

def select_events_propofol(drug_event_case: pd.DataFrame) -> pd.DataFrame:
    """Select propofol events from the drug event table."""
    df = drug_event_case.copy()
    df["_DN"] = df["DRUG_NAME"].map(_normalize_drug_name)
    inc = df["_DN"].str.contains(r"(プロポフォール|propofol|ディプリバン)", na=False, regex=True)
    return df.loc[inc].drop(columns=["_DN"], errors="ignore")

def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def events_to_bolus_and_infusion_opioid(
    ev: pd.DataFrame,
    t0: pd.Timestamp,
    *,
    weight_kg: float,
    # set defaults per drug
    default_infusion_unit: str,
    default_bolus_unit: str,
    concentration_ug_per_mL: float,
):
    """
    Output units:
      - bolus: mg at time t
      - infusion: mg/min piecewise

    Supported infusion units (case-insensitive):
      - ug/kg/min, mcg/kg/min
      - ug/min, mcg/min
      - mg/h, mg/min
      - ug/h, mcg/h

    Supported bolus units:
      - ug, mcg
      - mg
    """
    bolus = {}
    inf_changes = {}
    if ev.empty or pd.isna(t0):
        return bolus, inf_changes

    x = ev.copy()
    x = x.dropna(subset=["DATETIME"]).sort_values("DATETIME")
    x["t_min"] = (x["DATETIME"] - t0).dt.total_seconds() / 60.0

    # ---- Infusion changes ----
    inf = x[x["CONTENT"].isin(["開始", "流速変更", "終了"])].copy()
    for _, r in inf.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue
        if str(r["CONTENT"]) == "終了":
            inf_changes[t] = 0.0
            continue

        val = r.get("FLOW", np.nan)
        unit = None
        if pd.isna(val):
            dval, dunit = _parse_detail(r.get("DETAIL", ""))
            val, unit = dval, dunit
        else:
            _, dunit = _parse_detail(r.get("DETAIL", ""))
            unit = dunit

        if val is None or pd.isna(val):
            continue

        if unit is None:
            unit = default_infusion_unit
        unit = str(unit).lower().replace("μ", "u").replace("㎍", "ug")

        val = float(val)

        # normalize common spellings
        unit = unit.replace("mcg", "ug").replace("/hr", "/h")

        if "ug/kg/min" in unit:
            mg_per_min = (val * float(weight_kg)) / 1000.0
        elif "ug/kg/h" in unit or "ug/kg/hr" in unit:
            mg_per_min = (val * float(weight_kg)) / 1000.0 / 60.0
        elif "ug/min" in unit:
            mg_per_min = val / 1000.0
        elif "ug/h" in unit or "ug/hr" in unit:
            mg_per_min = val / 1000.0 / 60.0
        elif "mg/min" in unit:
            mg_per_min = val
        elif "mg/h" in unit or "mg/hr" in unit:
            mg_per_min = val / 60.0
        elif "ml/min" in unit:
        # mL/min → ug/min → mg/min
            mg_per_min = (val * concentration_ug_per_mL) / 1000.0

        elif "ml/h" in unit:
            # mL/h → ug/h → ug/min → mg/min
            mg_per_min = (val * concentration_ug_per_mL) / 1000.0 / 60.0
        else:
            raise ValueError(f"Unsupported opioid infusion unit: {unit} (DETAIL={r.get('DETAIL','')})")

        inf_changes[t] = mg_per_min

    # ---- Bolus ----
    b = x[x["CONTENT"].isin(["ワンショット"])].copy()
    for _, r in b.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue

        dval, dunit = _parse_detail(r.get("DETAIL", ""))
        if dval is None:
            fv = r.get("FLOW", np.nan)
            if pd.isna(fv):
                continue
            dval = float(fv)
            dunit = dunit or default_bolus_unit

        unit = (dunit or default_bolus_unit).lower().replace("μ", "u").replace("㎍", "ug")
        unit = unit.replace("mcg", "ug")

        if unit == "mg":
            dose_mg = float(dval)
        elif unit == "ug":
            dose_mg = float(dval) / 1000.0
        elif unit == "ml":
            dose_mg = (float(dval) * concentration_ug_per_mL) / 1000.0
        else:
            raise ValueError(f"Unsupported opioid bolus unit: {unit} (DETAIL={r.get('DETAIL','')})")

        bolus[t] = bolus.get(t, 0.0) + dose_mg

    return bolus, inf_changes

def events_to_bolus_and_infusion_propofol(
    ev: pd.DataFrame,
    t0: pd.Timestamp,
    *,
    weight_kg: float,
    default_infusion_unit: str = "mg/kg/h",
    default_bolus_unit: str = "mg",
    concentration_mg_per_mL: float = 10.0,
):
    """
    Convert propofol drug events to bolus and infusion inputs.

    Output units:
      - bolus: mg
      - infusion: mg/min

    Supported infusion units: mg/kg/h, mg/h, mg/min, mL/h, and mL/min.
    Routine propofol continuous dosing is assumed to be mg/kg/h.
    mL/h is also supported for pump-rate records.
    mg/kg/min is intentionally not supported.
    Supported bolus units: mg and mL.
    """
    bolus = {}
    inf_changes = {}
    if ev.empty or pd.isna(t0):
        return bolus, inf_changes

    x = ev.copy()
    x = x.dropna(subset=["DATETIME"]).sort_values("DATETIME")
    x["t_min"] = (x["DATETIME"] - t0).dt.total_seconds() / 60.0

    inf = x[x["CONTENT"].isin(["開始", "流速変更", "終了"])].copy()
    for _, r in inf.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue
        if str(r["CONTENT"]) == "終了":
            inf_changes[t] = 0.0
            continue

        val = r.get("FLOW", np.nan)
        unit = None
        if pd.isna(val):
            dval, dunit = _parse_detail(r.get("DETAIL", ""))
            val, unit = dval, dunit
        else:
            _, dunit = _parse_detail(r.get("DETAIL", ""))
            unit = dunit

        if val is None or pd.isna(val):
            continue

        unit = str(unit or default_infusion_unit).lower()
        unit = (
            unit.replace("／", "/")
                .replace("/hr", "/h")
                .replace("/hour", "/h")
                .replace("㎎", "mg")
                .replace("ｍｇ", "mg")
                .replace("ｍｌ", "ml")
                .replace("mℓ", "ml")
                .replace(" ", "")
        )
        val = float(val)

        # Continuous propofol: usually mg/kg/h; sometimes mL/h.
        # Do NOT accept mg/kg/min because it is not used in our OR records
        # and would be a high-risk unit misclassification.
        if "mg/kg/min" in unit:
            raise ValueError(
                f"Unsupported propofol infusion unit: {unit}. "
                "Propofol continuous dosing should be mg/kg/h or ml/h in this dataset. "
                f"DETAIL={r.get('DETAIL','')}"
            )
        elif "mg/kg/h" in unit:
            mg_per_min = val * float(weight_kg) / 60.0
        elif "mg/h" in unit:
            mg_per_min = val / 60.0
        elif "mg/min" in unit:
            mg_per_min = val
        elif "ml/h" in unit:
            mg_per_min = val * concentration_mg_per_mL / 60.0
        elif "ml/min" in unit:
            mg_per_min = val * concentration_mg_per_mL
        else:
            raise ValueError(f"Unsupported propofol infusion unit: {unit} (DETAIL={r.get('DETAIL','')})")

        inf_changes[t] = mg_per_min

    b = x[x["CONTENT"].isin(["ワンショット"])].copy()
    for _, r in b.iterrows():
        t = float(r["t_min"])
        if t < 0:
            continue

        dval, dunit = _parse_detail(r.get("DETAIL", ""))
        if dval is None:
            fv = r.get("FLOW", np.nan)
            if pd.isna(fv):
                continue
            dval = float(fv)
            dunit = dunit or default_bolus_unit

        unit = str(dunit or default_bolus_unit).lower()
        unit = (
            unit.replace("㎎", "mg")
                .replace("ｍｇ", "mg")
                .replace("ｍｌ", "ml")
                .replace("mℓ", "ml")
                .replace(" ", "")
        )
        if unit == "mg":
            dose_mg = float(dval)
        elif unit == "ml":
            dose_mg = float(dval) * concentration_mg_per_mL
        else:
            raise ValueError(f"Unsupported propofol bolus unit: {unit} (DETAIL={r.get('DETAIL','')})")

        bolus[t] = bolus.get(t, 0.0) + dose_mg

    return bolus, inf_changes

THETA = {
    "t1": 3.57, "t2": 11.3, "t3": 27.2,
    "t4": 1.03, "t5": 1.10, "t6": 0.401,
    "t8": 0.308, "t9": 0.146, "t10": -0.184
}

def ibw(height_cm: float, sex: int) -> float:
    return 45.4 + 0.89 * (height_cm - 152.4) + 4.5 * (1 - sex)

def abw(tbw_kg: float, height_cm: float, sex: int) -> float:
    _ibw = ibw(height_cm, sex)
    return _ibw + 0.4 * (tbw_kg - _ibw)

def ke0_lookup(sex: int, asa: int) -> float:
    if asa == 0 and sex == 0:
        return 0.22
    if asa == 0 and sex == 1:
        return 0.20
    if asa == 1 and sex == 0:
        return 0.24
    return 0.22

@dataclass
class Masui2022Params:
    V1: float
    V2: float
    V3: float
    CL: float
    Q2: float
    Q3: float
    ke0: float

def masui2022_params(age: float, sex: int, height_cm: float, tbw_kg: float, asa: int) -> Masui2022Params:
    ABW = abw(tbw_kg, height_cm, sex)
    s = ABW / 67.3

    V1 = THETA["t1"] * (s ** 1.0)
    V2 = THETA["t2"] * (s ** 1.0)
    V3 = (THETA["t3"] + THETA["t8"] * (age - 54.0)) * (s ** 1.0)

    CL = (THETA["t4"] + THETA["t9"] * sex + THETA["t10"] * asa) * (s ** 0.75)
    Q2 = THETA["t5"] * (s ** 0.75)
    Q3 = THETA["t6"] * (s ** 0.75)

    ke0 = ke0_lookup(sex, asa)
    return Masui2022Params(V1, V2, V3, CL, Q2, Q3, ke0)

def make_rhs_3cpt_effectsite(V1, V2, V3, CL, Q2, Q3, ke0, u_mg_per_min):
    k10 = CL / V1
    k12 = Q2 / V1
    k21 = Q2 / V2
    k13 = Q3 / V1
    k31 = Q3 / V3

    def rhs(t, y):
        A1, A2, A3, Ce = y
        Cp = A1 / V1
        dA1 = u_mg_per_min - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3
        dA2 = k12 * A1 - k21 * A2
        dA3 = k13 * A1 - k31 * A3
        dCe = ke0 * (Cp - Ce)
        return np.array([dA1, dA2, dA3, dCe], dtype=float)

    return rhs

def simulate_3cpt_case(t1, t2, dt, bolus_mg, inf_changes_mg_per_min,
                       V1, V2, V3, CL, Q2, Q3, ke0, method="RK45"):
    t_grid = np.arange(t1, t2 + 1e-12, dt)

    event_times = set([t1, t2])
    event_times.update(bolus_mg.keys())
    event_times.update(inf_changes_mg_per_min.keys())
    event_times = sorted([t for t in event_times if (t1 <= t <= t2)])

    y = np.zeros(4, dtype=float)  # [A1,A2,A3,Ce]
    if t1 in bolus_mg:
        y[0] += bolus_mg[t1]

    y_out = np.full((4, len(t_grid)), np.nan, dtype=float)

    def grid_idx_in_segment(a, b):
        return np.where((t_grid >= a - 1e-12) & (t_grid <= b + 1e-12))[0]

    for i in range(len(event_times) - 1):
        a = event_times[i]
        b = event_times[i + 1]
        if b <= a:
            if b in bolus_mg:
                y[0] += bolus_mg[b]
            continue

        u = piecewise_rate(a, inf_changes_mg_per_min)
        rhs = make_rhs_3cpt_effectsite(V1, V2, V3, CL, Q2, Q3, ke0, u)

        idx = grid_idx_in_segment(a, b)
        t_eval = t_grid[idx]

        sol = solve_ivp(
            rhs, (a, b), y, method=method, t_eval=t_eval,
            rtol=1e-6, atol=1e-9
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed on segment [{a}, {b}]: {sol.message}")

        y_out[:, idx] = sol.y
        y = sol.y[:, -1].copy()

        if b in bolus_mg:
            y[0] += bolus_mg[b]

    # forward fill if needed
    for k in range(4):
        mask = np.isnan(y_out[k])
        if np.any(mask):
            last = None
            for j in range(len(t_grid)):
                if not np.isnan(y_out[k, j]):
                    last = y_out[k, j]
                elif last is not None:
                    y_out[k, j] = last

    A1 = y_out[0]
    Cp = A1 / V1
    Ce = y_out[3]
    return t_grid, Cp, Ce

@dataclass
class FlumazenilBase70kg:
    # Base (typical adult) parameters at 70 kg
    V1_70: float = 5.6
    V2_70: float = 15.93
    V3_70: float = 39.44
    CL_70: float = 1.12   # L/min
    Q2_70: float = 1.79   # L/min
    Q3_70: float = 0.86   # L/min
    t12_ke0_min: float = 5.0  # minutes

    @property
    def ke0(self) -> float:
        return float(np.log(2) / self.t12_ke0_min)

def flm_params_fixed(base: FlumazenilBase70kg):
    return {
        "V1": base.V1_70, "V2": base.V2_70, "V3": base.V3_70,
        "CL": base.CL_70, "Q2": base.Q2_70, "Q3": base.Q3_70,
        "ke0": base.ke0
    }

def flm_params_allometric(base: FlumazenilBase70kg, weight_kg: float):
    w = float(weight_kg)
    if not np.isfinite(w) or w <= 0:
        w = 70.0
    sV = (w / 70.0) ** 1.0
    sC = (w / 70.0) ** 0.75

    return {
        "V1": base.V1_70 * sV,
        "V2": base.V2_70 * sV,
        "V3": base.V3_70 * sV,
        "CL": base.CL_70 * sC,
        "Q2": base.Q2_70 * sC,
        "Q3": base.Q3_70 * sC,
        "ke0": base.ke0
    }

@dataclass
class RemiEleveld2017Ref:
    V1: float = 5.81
    V2: float = 8.82
    V3: float = 5.03
    CL: float = 2.58
    Q2: float = 1.72
    Q3: float = 0.124
    ke0: float = 1.09

@dataclass
class PropofolEleveld2018Params:
    V1: float
    V2: float
    V3: float
    CL: float
    Q2: float
    Q3: float
    ke0: float

def eleveld2018_propofol_params(
    age_yr: float,
    sex: int,
    height_cm: float,
    weight_kg: float,
    *,
    is_opioid: int = 1,
) -> PropofolEleveld2018Params:
    """
    Propofol Eleveld 2018 PK-PD parameters.

    This follows the JavaScript implementation supplied by the user:
      - maturation terms are ignored
      - presence of opioids is assumed by default (is_opioid=1)
      - sex: 0 male, 1 female

    Units:
      V's: L
      CL,Q's: L/min
      ke0: 1/min
    """
    theta = [
        0, 6.28, 25.5, 273, 1.79, 1.75, 1.11, 0.191, 42.3, 9.06,
        -0.0156, -0.00286, 33.6, -0.0138, 68.3, 2.10, 1.30, 1.42, 0.68
    ]
    theta_pd = [0, 3.08, 0.146, 92.98, 1.47, 8.03, 0.05174, -0.00635, 1.24, 1.89]

    age_yr = float(age_yr)
    sex = int(sex)
    height_cm = float(height_cm)
    weight_kg = float(weight_kg)
    is_opioid = int(is_opioid)

    ffm_as = _ffm_alsallami(sex, age_yr, weight_kg, height_cm)
    ffm_ref = (
        0.88 + (1.0 - 0.88) / (1.0 + (35.0 / 13.4) ** (-12.7))
    ) * (9270.0 * 70.0 / (6680.0 + 216.0 * 70.0 * 10000.0 / (170.0 * 170.0)))

    def fcentral(x):
        return x / (x + theta[12])

    V1 = theta[1] * fcentral(weight_kg) / fcentral(70.0)
    V2 = theta[2] * weight_kg / 70.0 * np.exp(theta[10] * (age_yr - 35.0))
    V3 = theta[3] * ffm_as / ffm_ref * np.exp(theta[13] * age_yr * is_opioid)
    CL = (theta[4] if sex == 0 else theta[15]) * (weight_kg / 70.0) ** 0.75 * np.exp(theta[11] * age_yr * is_opioid)
    Q2 = theta[5] * (weight_kg / 70.0 * np.exp(theta[10] * (age_yr - 35.0))) ** 0.75
    Q3 = theta[6] * (ffm_as / ffm_ref * np.exp(theta[13] * age_yr * is_opioid)) ** 0.75
    ke0 = theta_pd[2] * (weight_kg / 70.0) ** (-0.25)

    return PropofolEleveld2018Params(V1=V1, V2=V2, V3=V3, CL=CL, Q2=Q2, Q3=Q3, ke0=ke0)

def eleveld2017_remifentanil_params(age_yr: float, sex: int, height_cm: float, weight_kg: float) -> RemiEleveld2017Ref:
    """
    Remifentanil Eleveld 2017 parameters as implemented in forstudy.html:
      - Size descriptor: FFM (Al-Sallami)
      - Fsize = FFM_AS / FFMref (ref: male 35y, 70kg, 170cm)
      - Age exponential modifiers
      - Female modifier between ~12 and 45 years
      - Extra weight term on V3
      - ke0 decreases with age
    Units:
      V's: L
      CL,Q's: L/min
      ke0: 1/min
    """
    age_yr = float(age_yr)
    sex = int(sex)
    height_cm = float(height_cm)
    weight_kg = float(weight_kg)

    # Patient FFM (Al-Sallami)
    ffm_as = _ffm_alsallami(sex, age_yr, weight_kg, height_cm)

    # Reference FFMref used in forstudy.html (hard-coded reference individual)
    # ref BMI uses 70kg, 170cm; ref age is 35; male maturation constants.
    ffm_j_ref = _ffm_janmahasatian(0, 70.0, 170.0)
    ffm_ref = (0.88 + (1.0 - 0.88) / (1.0 + (35.0 / 13.4) ** (-12.7))) * ffm_j_ref

    fsize = ffm_as / ffm_ref

    # Female modifier between ~12 and 45 years (smooth step in and out)
    if sex == 0:
        fsex = 1.0
    else:
        fsex = 1.0 + 0.47 * _fsigmoid(age_yr, 12.0, 6.0) * (1.0 - _fsigmoid(age_yr, 45.0, 6.0))

    # Parameters (exact numeric constants from forstudy.html)
    V1 = 5.81 * fsize * np.exp(-0.00554 * (age_yr - 35.0))
    V2 = 8.82 * fsize * np.exp(-0.00327 * (age_yr - 35.0)) * fsex
    V3 = 5.03 * fsize * np.exp(-0.0315 * (age_yr - 35.0)) * np.exp(-0.0260 * (weight_kg - 70.0))

    CL = 2.58 * (fsize ** 0.75) * np.exp(-0.00554 * (age_yr - 35.0)) * fsex
    Q2 = 1.72 * ((V2 / 8.82) ** 0.75) * np.exp(-0.00554 * (age_yr - 35.0)) * fsex
    Q3 = 0.124 * ((V3 / 5.03) ** 0.75) * np.exp(-0.00554 * (age_yr - 35.0))

    ke0 = 1.09 * np.exp(-0.00289 * (age_yr - 35.0))

    return RemiEleveld2017Ref(V1=V1, V2=V2, V3=V3, CL=CL, Q2=Q2, Q3=Q3, ke0=ke0)

@dataclass
class FentBae2020Ref:
    # 70 kg subject (abstract)
    V1_L: float = 10.1
    V2_L: float = 26.5
    V3_L: float = 206.0
    CL_L_per_min: float = 0.704
    Q2_L_per_min: float = 2.38
    Q3_L_per_min: float = 1.49
    # ke0 not provided in abstract; must be assumed for effect-site calculation
    ke0_per_min: float = 0.147

def fent_params_bae2020_allometry(
    ref: FentBae2020Ref,
    weight_kg: float,
    *,
    weight_ref_kg: float = 70.0,
):
    """
    Bae 2020 abstract provides V1/V2/V3/CL/Q’s for 70 kg. :contentReference[oaicite:6]{index=6}
    """
    w = float(weight_kg) if np.isfinite(weight_kg) and weight_kg > 0 else weight_ref_kg
    sV = (w / weight_ref_kg) ** 1.23
    sC = (w / weight_ref_kg) ** 0.313

    return {
        "V1": ref.V1_L * sV,
        "V2": ref.V2_L * sV,
        "V3": ref.V3_L * sV,
        "CL": ref.CL_L_per_min * sC,
        "Q2": ref.Q2_L_per_min * sC,
        "Q3": ref.Q3_L_per_min * sC,
        "ke0": ref.ke0_per_min,
    }

def remimazolam_equivalent_ce_masui2023(
    Ce_remi_mg_per_L: np.ndarray,
    Ce_flum_mg_per_L: np.ndarray,
    PBR_flum: float = 0.314,        # 31.4%
    K_flum_ng_per_mL: float = 2.70  # ng/mL
) -> np.ndarray:
    """
    Masui J Anesth 2023 Eq.(3):
        C_remi_equiv = C_remi / (1 + C_flum*(1-PBR_flum)/K_flum)

    Units:
      - Ce_remi_mg_per_L: mg/L  (numerically = µg/mL)
      - Ce_flum_mg_per_L: mg/L  -> internally converted to ng/mL
      - Output: mg/L (numerically = µg/mL)
    """

    Ce_remi = np.asarray(Ce_remi_mg_per_L, dtype=float)  # mg/L == µg/mL
    Ce_flum_ng_mL = np.asarray(Ce_flum_mg_per_L, dtype=float) * 1e3  # mg/L -> ng/mL

    fu_flum = 1.0 - float(PBR_flum)  # 0.686

    denom = 1.0 + (Ce_flum_ng_mL * fu_flum) / float(K_flum_ng_per_mL)
    # Avoid division by zero if needed:
    denom = np.maximum(denom, 1e-12)

    Ce_equiv = Ce_remi / denom
    return Ce_equiv

def simulate_case_wide_outputs(
    main_row: pd.Series,
    drug_event_case: pd.DataFrame,
    *,
    case_id_col="caseNo",
    start_col="入室日時",
    end_col="EXTUBATION_TIME",
    dt_min=1.0,
    extra_after_end_min=120.0,
    method="RK45",
    flm_base: FlumazenilBase70kg | None = None,
    remi_ref: RemiEleveld2017Ref | None = None,
    fent_ref: FentBae2020Ref | None = None,
    propofol_concentration_mg_per_mL: float = 10.0,
):
    case_id = str(main_row[case_id_col])

    t_start = pd.to_datetime(main_row.get(start_col, pd.NaT), errors="coerce")
    t_end   = pd.to_datetime(main_row.get(end_col, pd.NaT), errors="coerce")

    if pd.isna(t_start):
        return None

    if pd.isna(t_end):
        last_evt = pd.to_datetime(drug_event_case["DATETIME"], errors="coerce").max() if len(drug_event_case) else pd.NaT
        t_end = last_evt
    if pd.isna(t_end) or t_end <= t_start:
        t_end = t_start + pd.Timedelta(minutes=60)

    t2_total_min = (t_end - t_start).total_seconds() / 60.0 + float(extra_after_end_min)
    if not np.isfinite(t2_total_min) or t2_total_min <= 0:
        t2_total_min = 180.0

    # Demographics
    age = float(main_row.get("age", np.nan))
    sex = _sex_to_int(main_row.get("sex", 0))
    height_cm = float(main_row.get("Ht_CIS", np.nan))
    weight_kg = float(main_row.get("Wt_CIS", np.nan))
    asa = _high_asa_to_int(main_row.get("ASAPS", np.nan))

    # IBW for “allometric based on IBW” requirement
    ibw_kg = float(ibw(height_cm, sex)) if np.isfinite(height_cm) else np.nan
    if not np.isfinite(ibw_kg) or ibw_kg <= 0:
        ibw_kg = weight_kg if np.isfinite(weight_kg) and weight_kg > 0 else 70.0

    # Time grid (single common grid)
    t_grid = np.arange(0.0, float(np.ceil(t2_total_min)) + 1e-12, float(dt_min))
    dt_series = t_start + pd.to_timedelta(t_grid, unit="m")

    wide = pd.DataFrame({"CASE_NO": case_id, "DATETIME": dt_series})

    # ---------- RMZ ----------
    rmz_ok = np.isfinite(age) and np.isfinite(height_cm) and np.isfinite(weight_kg)
    if rmz_ok:
        ev_rmz = select_events_rmz(drug_event_case)
        bol_rmz, inf_rmz = events_to_bolus_and_infusion_rmz(ev_rmz, t_start, weight_kg)
        if bol_rmz or inf_rmz:
            p = masui2022_params(age, sex, height_cm, weight_kg, asa)
            tg, Cp, Ce = simulate_3cpt_case(
                0.0, float(np.ceil(t2_total_min)), float(dt_min),
                bol_rmz, inf_rmz,
                p.V1, p.V2, p.V3, p.CL, p.Q2, p.Q3, p.ke0,
                method=method
            )
            wide["rmz_Cp_mg_per_L"] = Cp
            wide["rmz_Ce_mg_per_L"] = Ce

    # ---------- FLM (fixed + IBW-allometry) ----------
    ev_flm = select_events_flumazenil(drug_event_case)
    bol_flm, inf_flm = events_to_bolus_and_infusion_flumazenil(ev_flm, t_start)

    if flm_base is None:
        flm_base = FlumazenilBase70kg()

    if bol_flm or inf_flm:
        # fixed 70kg
        pf = flm_params_fixed(flm_base)
        tg, Cp2, Ce2 = simulate_3cpt_case(
            0.0, float(np.ceil(t2_total_min)), float(dt_min),
            bol_flm, inf_flm,
            pf["V1"], pf["V2"], pf["V3"], pf["CL"], pf["Q2"], pf["Q3"], pf["ke0"],
            method=method
        )
        wide["flm_Cp_fixed_mg_per_L"] = Cp2
        wide["flm_Ce_fixed_mg_per_L"] = Ce2

        # allometric by IBW (your request)
        pa = flm_params_allometric(flm_base, ibw_kg)
        tg, Cp3, Ce3 = simulate_3cpt_case(
            0.0, float(np.ceil(t2_total_min)), float(dt_min),
            bol_flm, inf_flm,
            pa["V1"], pa["V2"], pa["V3"], pa["CL"], pa["Q2"], pa["Q3"], pa["ke0"],
            method=method
        )
        wide["flm_Cp_alloIBW_mg_per_L"] = Cp3
        wide["flm_Ce_alloIBW_mg_per_L"] = Ce3

    # ---------- RMZ-equivalent Ce (Masui 2023 Eq.(3)) ----------
    if ("rmz_Ce_mg_per_L" in wide.columns) and ("flm_Ce_fixed_mg_per_L" in wide.columns):
        wide["rmz_Ce_equiv_fixed_mg_per_L"] = remimazolam_equivalent_ce_masui2023(
            Ce_remi_mg_per_L=wide["rmz_Ce_mg_per_L"].to_numpy(),
            Ce_flum_mg_per_L=wide["flm_Ce_fixed_mg_per_L"].to_numpy(),
        )

    if ("rmz_Ce_mg_per_L" in wide.columns) and ("flm_Ce_alloIBW_mg_per_L" in wide.columns):
        wide["rmz_Ce_equiv_alloIBW_mg_per_L"] = remimazolam_equivalent_ce_masui2023(
            Ce_remi_mg_per_L=wide["rmz_Ce_mg_per_L"].to_numpy(),
            Ce_flum_mg_per_L=wide["flm_Ce_alloIBW_mg_per_L"].to_numpy(),
        )

    # ---------- Propofol (Eleveld 2018) ----------
    ev_prop = select_events_propofol(drug_event_case)
    bol_prop, inf_prop = events_to_bolus_and_infusion_propofol(
        ev_prop, t_start,
        weight_kg=weight_kg if np.isfinite(weight_kg) else 70.0,
        default_infusion_unit="mg/kg/h",
        default_bolus_unit="mg",
        concentration_mg_per_mL=float(propofol_concentration_mg_per_mL),
    )

    prop_ok = np.isfinite(age) and np.isfinite(height_cm) and np.isfinite(weight_kg)
    if prop_ok and (bol_prop or inf_prop):
        p_prop = eleveld2018_propofol_params(age, sex, height_cm, weight_kg, is_opioid=1)
        tg, Cp_prop, Ce_prop = simulate_3cpt_case(
            0.0, float(np.ceil(t2_total_min)), float(dt_min),
            bol_prop, inf_prop,
            p_prop.V1, p_prop.V2, p_prop.V3,
            p_prop.CL, p_prop.Q2, p_prop.Q3,
            p_prop.ke0,
            method=method,
        )
        wide["prop_Cp_mg_per_L"] = Cp_prop
        wide["prop_Ce_mg_per_L"] = Ce_prop

    # ---------- Remifentanil (Eleveld 2017) ----------
    ev_remi = select_events_remifentanil(drug_event_case)
    bol_remi, inf_remi = events_to_bolus_and_infusion_opioid(
        ev_remi, t_start,
        weight_kg=weight_kg if np.isfinite(weight_kg) else 70.0,
        default_infusion_unit="ug/kg/min",
        default_bolus_unit="ug",
        concentration_ug_per_mL=100.0,
    )

    if remi_ref is None:
        remi_ref = RemiEleveld2017Ref()

    if bol_remi or inf_remi:
        
        p_remi = eleveld2017_remifentanil_params(age, sex, height_cm, weight_kg)

        t_grid, Cp_remi, Ce_remi = simulate_3cpt_case(
            0.0, float(np.ceil(t2_total_min)), float(dt_min),
            bol_remi, inf_remi,                 # <- your remifentanil events converted to mg/min
            p_remi.V1, p_remi.V2, p_remi.V3,
            p_remi.CL, p_remi.Q2, p_remi.Q3,
            p_remi.ke0,
            method=method
        )
        # opioids: export as ng/mL for readability
        wide["remi_Cp_ng_per_mL"] = Cp_remi * 1e3
        wide["remi_Ce_ng_per_mL"] = Ce_remi * 1e3

    # ---------- Fentanyl (Bae 2020) ----------
    ev_fent = select_events_fentanyl(drug_event_case)
    bol_fent, inf_fent = events_to_bolus_and_infusion_opioid(
        ev_fent, t_start,
        weight_kg=weight_kg if np.isfinite(weight_kg) else 70.0,
        default_infusion_unit="ug/h",
        default_bolus_unit="ug",
        concentration_ug_per_mL=50.0,
    )

    if fent_ref is None:
        fent_ref = FentBae2020Ref(ke0_per_min=0.147)  # example assumption (literature summary) :contentReference[oaicite:5]{index=5}

    if bol_fent or inf_fent:
        pfent = fent_params_bae2020_allometry(
            fent_ref,
            weight_kg=weight_kg if np.isfinite(weight_kg) else 70.0,
        )
        tg, Cp_mgL, Ce_mgL = simulate_3cpt_case(
            0.0, float(np.ceil(t2_total_min)), float(dt_min),
            bol_fent, inf_fent,
            pfent["V1"], pfent["V2"], pfent["V3"], pfent["CL"], pfent["Q2"], pfent["Q3"], pfent["ke0"],
            method=method
        )
        wide["fent_Cp_ng_per_mL"] = Cp_mgL * 1e3
        wide["fent_Ce_ng_per_mL"] = Ce_mgL * 1e3
    else:
        wide["fent_Cp_ng_per_mL"] = 0
        wide["fent_Ce_ng_per_mL"] = 0

    return wide

def batch_run_one_wide_csv(
    main: pd.DataFrame,
    drug_event: pd.DataFrame,
    *,
    out_dir: str,
    case_id_col="caseNo",
    event_case_col="CASE_NO",
    start_col="入室日時",
    end_col="EXTUBATION_TIME",
    dt_min=1.0,
    extra_after_end_min=120.0,
    method="RK45",
    flm_t12_ke0_min=5.0,
    propofol_concentration_mg_per_mL=10.0,
):
    os.makedirs(out_dir, exist_ok=True)

    de = _ensure_datetime_col(drug_event, col="DATETIME").dropna(subset=["DATETIME"]).copy()
    now = datetime.now().strftime("%Y%m%d%H%M")
    today = datetime.today().strftime("%Y%m%d")
    out_csv = os.path.join(out_dir, f"pkpd_wide_rmz_flm_prop_opioids_{today}.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)

    flm_base = FlumazenilBase70kg(t12_ke0_min=float(flm_t12_ke0_min))
    remi_ref = RemiEleveld2017Ref()
    fent_ref = FentBae2020Ref()

    wrote_header = True
    all_cases = []
    for _, r in main.iterrows():
        case = r.get(case_id_col, None)
        if case is None:
            continue

        de_case = de.loc[de[event_case_col] == case].copy()
        if de_case.empty:
            continue

        wide = simulate_case_wide_outputs(
            r, de_case,
            case_id_col=case_id_col,
            start_col=start_col,
            end_col=end_col,
            dt_min=dt_min,
            extra_after_end_min=extra_after_end_min,
            method=method,
            flm_base=flm_base,
            remi_ref=remi_ref,
            fent_ref=fent_ref,
            propofol_concentration_mg_per_mL=float(propofol_concentration_mg_per_mL),
        )
        if wide is None or wide.empty:
            continue
        all_cases.append(wide)

    if not all_cases:
        raise RuntimeError("No cases produced output.")

    df_all = pd.concat(all_cases, ignore_index=True)

    # Ensure stable column order (optional but recommended)
    base_cols = ["CASE_NO", "DATETIME"]
    other_cols = sorted([c for c in df_all.columns if c not in base_cols])
    df_all = df_all[base_cols + other_cols]

    df_all.to_csv(out_csv, index=False)

    return out_csv if wrote_header else None

def n_pct(count: int, total: int, digits: int = 1) -> str:
    if total == 0:
        return "0 (0.0%)"
    return f"{count} ({100 * count / total:.{digits}f}%)"

def p_cont(x1: pd.Series, x2: pd.Series) -> float:
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    x2 = pd.to_numeric(x2, errors="coerce").dropna()
    if len(x1) == 0 or len(x2) == 0:
        return float("nan")
    return mannwhitneyu(x1, x2, alternative="two-sided").pvalue

def p_cat(tab: pd.DataFrame) -> float:
    if tab.empty or tab.shape[0] == 0 or tab.shape[1] == 0:
        return float("nan")
    if tab.shape == (2, 2):
        return fisher_exact(tab)[1]
    return chi2_contingency(tab)[1]

def _validate_inputs(
    df: pd.DataFrame,
    group_col: str,
    groups: list,
    continuous: list,
    binary_vars: list,
    categorical: list,
) -> None:
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' is not in df.columns")

    missing_groups = [g for g in groups if g not in df[group_col].dropna().unique()]
    if missing_groups:
        raise ValueError(f"These groups were not found in '{group_col}': {missing_groups}")

    all_vars = continuous + binary_vars + categorical
    missing_vars = [v for v in all_vars if v not in df.columns]
    if missing_vars:
        raise ValueError(f"These variables were not found in df.columns: {missing_vars}")

    if len(groups) != 2:
        raise ValueError("Currently this function expects exactly 2 groups.")

def calc_or_ci_ne(tab: pd.DataFrame):
    """
    OR = odds of event in group1 / group2
    OR (95% CI) or 'NE' if not estimable.
    """
    try:
        if tab.shape != (2, 2):
            return ""

        # Ensure correct order: [event=1, no event=0]
        tab = tab[[1, 0]]

        table = tab.values.astype(float)

        if (table == 0).any():
            return "NE"

        t = Table2x2(table)
        or_val = t.oddsratio
        ci_low, ci_high = t.oddsratio_confint()

        return f"{or_val:.2f} ({ci_low:.2f}–{ci_high:.2f})"

    except Exception:
        return ""

def build_table(
    df,
    continuous,
    binary_vars,
    categorical,
    group_col="Group",
    groups=None,
    digits_cont=1,
    digits_pct=1,
    digits_smd=3,
    include_missing=False,
    smd_type: Literal["absolute", "signed"] = "absolute",
):
    """
    Build an unweighted baseline table.

    Parameters
    ----------
    smd_type : {"absolute", "signed"}, default "absolute"
        "absolute":
            Shows absolute SMD. This is the most common format for Table 1
            and Love plots in clinical journals.

        "signed":
            Shows signed SMD. The sign depends on the group order.

            SMD = value in g1 - value in g2

            Therefore:
                positive SMD: higher value or proportion in g1
                negative SMD: higher value or proportion in g2

            Example:
                groups=["FADE", "AE"]
                positive SMD means FADE higher
                negative SMD means AE higher

    Example
    -------
    table1_abs = build_table(
        df=main,
        continuous=continuous,
        binary_vars=binary_vars,
        categorical=categorical,
        group_col="Group",
        groups=["FADE", "AE"],
        smd_type="absolute",
    )

    table1_signed = build_table(
        df=main,
        continuous=continuous,
        binary_vars=binary_vars,
        categorical=categorical,
        group_col="Group",
        groups=["FADE", "AE"],
        smd_type="signed",
    )
    """

    if smd_type not in ["absolute", "signed"]:
        raise ValueError("smd_type must be either 'absolute' or 'signed'")

    def apply_smd_type(smd):
        if pd.isna(smd):
            return smd
        if smd_type == "absolute":
            return abs(smd)
        return smd

    def format_selected_smd(smd):
        smd = apply_smd_type(smd)
        return format_smd(smd, digits_smd)

    if groups is None:
        groups = list(df[group_col].dropna().unique())

    if len(groups) != 2:
        raise ValueError("Only 2 groups supported")

    g1, g2 = groups
    sub = df[df[group_col].isin(groups)].copy()
    rows = []

    # ---------- N ----------
    rows.append({
        "Variable": "N",
        g1: str((sub[group_col] == g1).sum()),
        g2: str((sub[group_col] == g2).sum()),
        "OR (95% CI)": "",
        "p": "",
        "SMD": "",
    })

    # ---------- Continuous ----------
    for var in continuous:
        x1 = sub[sub[group_col] == g1][var]
        x2 = sub[sub[group_col] == g2][var]

        rows.append({
            "Variable": var,
            g1: median_iqr(x1, digits_cont),
            g2: median_iqr(x2, digits_cont),
            "OR (95% CI)": "",
            "p": format_p(p_cont(x1, x2)),
            "SMD": format_selected_smd(smd_cont(x1, x2)),
        })

    # ---------- Binary ----------
    for var in binary_vars:
        tmp = sub[[group_col, var]].copy()
        tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
        tmp = tmp.dropna(subset=[var])

        tab = pd.crosstab(tmp[group_col], tmp[var])
        tab = tab.reindex(index=groups, fill_value=0)
        tab = tab.reindex(columns=[0, 1], fill_value=0)

        row = {"Variable": f"{var}, n (%)"}

        for g in groups:
            s = sub[sub[group_col] == g][var]
            s_num = pd.to_numeric(s, errors="coerce")
            row[g] = n_pct((s_num == 1).sum(), s_num.notna().sum(), digits_pct)

        row["p"] = format_p(p_cat(tab))
        row["OR (95% CI)"] = calc_or_ci_ne(tab)

        row["SMD"] = format_selected_smd(
            smd_binary(
                sub[sub[group_col] == g1][var],
                sub[sub[group_col] == g2][var],
            )
        )

        rows.append(row)

    # ---------- Categorical ----------
    for var in categorical:
        tmp = sub[[group_col, var]].copy()

        header = {"Variable": var}
        for g in groups:
            header[g] = ""

        tmp_nonmiss = tmp.dropna(subset=[var])
        tab = pd.crosstab(tmp_nonmiss[group_col], tmp_nonmiss[var])
        tab = tab.reindex(index=groups, fill_value=0)

        header["p"] = format_p(p_cat(tab))
        header["OR (95% CI)"] = ""

        # For a categorical variable, the header SMD is the level with
        # the largest absolute imbalance.
        # If smd_type="signed", the sign of that largest imbalance is retained.
        smd_list = []
        for level in tab.columns:
            x = (tmp[tmp[group_col] == g1][var] == level).astype(float)
            y = (tmp[tmp[group_col] == g2][var] == level).astype(float)
            smd_val = smd_binary(x, y)
            if not pd.isna(smd_val):
                smd_list.append(smd_val)

        if smd_list:
            if smd_type == "absolute":
                header_smd = max(abs(s) for s in smd_list)
            else:
                header_smd = max(smd_list, key=lambda s: abs(s))

            header["SMD"] = format_smd(header_smd, digits_smd)
        else:
            header["SMD"] = ""

        rows.append(header)

        for level in tab.columns:
            row = {"Variable": f"  {level}"}

            for g in groups:
                s = tmp[tmp[group_col] == g][var]
                row[g] = n_pct((s == level).sum(), s.notna().sum(), digits_pct)

            row["p"] = ""
            row["OR (95% CI)"] = ""

            x = (tmp[tmp[group_col] == g1][var] == level).astype(float)
            y = (tmp[tmp[group_col] == g2][var] == level).astype(float)

            row["SMD"] = format_selected_smd(smd_binary(x, y))

            rows.append(row)

        if include_missing:
            row = {"Variable": "  Missing"}
            for g in groups:
                s = tmp[tmp[group_col] == g][var]
                row[g] = n_pct(s.isna().sum(), len(s), digits_pct)

            row["p"] = ""
            row["OR (95% CI)"] = ""
            row["SMD"] = ""

            rows.append(row)

    return pd.DataFrame(rows)

def logistic_table(df, y_var, x_vars, drop_intercept=False):
    data = df[[y_var] + x_vars].dropna().copy()

    y = pd.to_numeric(data[y_var], errors="coerce")
    X = data[x_vars].apply(pd.to_numeric, errors="coerce")

    valid = y.notna() & X.notna().all(axis=1)
    y = y.loc[valid]
    X = X.loc[valid]

    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X).fit(disp=0)

    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues

    result = pd.DataFrame({
        "Variable": params.index,
        "OR": np.exp(params).values,
        "CI_low": np.exp(conf[0]).values,
        "CI_high": np.exp(conf[1]).values,
        "p": pvals.values,
    })

    result["OR (95% CI)"] = result.apply(
        lambda r: f'{r["OR"]:.2f} ({r["CI_low"]:.2f}–{r["CI_high"]:.2f})',
        axis=1
    )
    result["p"] = result["p"].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")

    result = result[["Variable", "OR (95% CI)", "p"]]

    if drop_intercept:
        result = result[result["Variable"] != "const"].reset_index(drop=True)

    return result.reset_index(drop=True)


# ============================================================
# Propensity score weighting utilities
# ============================================================

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
    encoding: str = "statsmodels"
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


@dataclass
class WeightedRelativeRiskResult:
    """Result object returned by weighted_relative_risk().

    rr is the risk ratio for groups[0] vs groups[1].
    If groups=["FADE", "AE"], rr < 1 means the outcome risk is lower in FADE.
    """
    rr: float
    ci_low: float
    ci_high: float
    p_value: float
    log_rr: float
    log_rr_ci: tuple[float, float]
    risk_treated: float
    risk_control: float
    treatment_label: object
    control_label: object
    model: object | None
    variance_method: str

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


def weighted_relative_risk(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    group_col: str = "Group",
    groups: Optional[Sequence] = None,
    weight_col: str = "weight",
    treatment_col: Optional[str] = None,
    alpha: float = 0.05,
    robust_cov: Optional[str] = "HC0",
    use_var_weights: bool = True,
) -> WeightedRelativeRiskResult:
    """Estimate a weighted risk ratio for a binary outcome.

    This function fits a weighted Poisson regression with a log link and
    robust standard errors, a common approach for estimating risk ratios
    for binary outcomes. It is designed for propensity-score weighted
    analyses such as overlap weighting and stabilized IPTW.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset.

    outcome_col : str
        Binary outcome column. Values should be 0/1.

    group_col : str, default "Group"
        Column containing the two treatment groups.

    groups : sequence, optional
        Two group labels. The first label is treated/exposed and is the
        numerator of the risk ratio. Example: groups=["FADE", "AE"].

    weight_col : str, default "weight"
        Column containing analysis weights.

    treatment_col : str, optional
        Optional precomputed 0/1 treatment indicator. If omitted, it is
        created from group_col and groups[0].

    alpha : float, default 0.05
        Alpha level for confidence intervals.

    robust_cov : str or None, default "HC0"
        Robust covariance type passed to statsmodels. Use None for
        model-based standard errors. HC0 is often more stable than HC3
        for GLM with non-integer propensity-score weights.

    use_var_weights : bool, default True
        If True, pass weights as var_weights. If False, use freq_weights.
        var_weights is usually preferable for non-integer balancing weights.

    Returns
    -------
    WeightedRelativeRiskResult
        Contains RR, 95% CI, p value, weighted risks, and fitted model.
    """
    if groups is None:
        groups = list(df[group_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError("Only two groups are supported.")

    g1, g2 = groups

    needed = [outcome_col, group_col, weight_col]
    if treatment_col is not None:
        needed.append(treatment_col)

    sub = df.loc[df[group_col].isin(groups), needed].copy()
    sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")
    sub[weight_col] = pd.to_numeric(sub[weight_col], errors="coerce")

    if treatment_col is None:
        sub["_treated_for_rr"] = (sub[group_col] == g1).astype(int)
        treatment_col = "_treated_for_rr"
    else:
        sub[treatment_col] = pd.to_numeric(sub[treatment_col], errors="coerce")

    valid = (
        sub[outcome_col].notna()
        & sub[treatment_col].notna()
        & sub[weight_col].notna()
        & np.isfinite(sub[weight_col])
        & (sub[weight_col] >= 0)
    )
    sub = sub.loc[valid].copy()

    if sub.empty:
        return WeightedRelativeRiskResult(
            rr=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan,
            log_rr=np.nan, log_rr_ci=(np.nan, np.nan),
            risk_treated=np.nan, risk_control=np.nan,
            treatment_label=g1, control_label=g2, model=None,
            variance_method=str(robust_cov),
        )

    risk_treated = weighted_prop(
        sub.loc[sub[group_col] == g1, outcome_col],
        sub.loc[sub[group_col] == g1, weight_col],
        1,
    )
    risk_control = weighted_prop(
        sub.loc[sub[group_col] == g2, outcome_col],
        sub.loc[sub[group_col] == g2, weight_col],
        1,
    )

    # The GLM log-link estimate can fail or be non-estimable when one group
    # has no events. In that situation, return the weighted-risk point estimate
    # if possible, but leave the CI and p value as missing.
    rr_point = (
        risk_treated / risk_control
        if pd.notna(risk_treated) and pd.notna(risk_control) and risk_control > 0
        else np.nan
    )

    y = sub[outcome_col].to_numpy(float)
    z = sub[treatment_col].to_numpy(float)
    w = sub[weight_col].to_numpy(float)
    X = sm.add_constant(z, has_constant="add")

    try:
        glm_kwargs = {
            "endog": y,
            "exog": X,
            "family": sm.families.Poisson(),
        }
        if use_var_weights:
            glm_kwargs["var_weights"] = w
        else:
            glm_kwargs["freq_weights"] = w

        model = sm.GLM(**glm_kwargs)
        if robust_cov is None:
            fit = model.fit()
            variance_method = "model-based"
        else:
            fit = model.fit(cov_type=robust_cov)
            variance_method = robust_cov

        log_rr = float(np.asarray(fit.params)[1])
        ci_arr = np.asarray(fit.conf_int(alpha=alpha))
        log_rr_ci = (float(ci_arr[1, 0]), float(ci_arr[1, 1]))
        p_value = float(np.asarray(fit.pvalues)[1])

        return WeightedRelativeRiskResult(
            rr=float(np.exp(log_rr)),
            ci_low=float(np.exp(log_rr_ci[0])),
            ci_high=float(np.exp(log_rr_ci[1])),
            p_value=p_value,
            log_rr=log_rr,
            log_rr_ci=log_rr_ci,
            risk_treated=risk_treated,
            risk_control=risk_control,
            treatment_label=g1,
            control_label=g2,
            model=fit,
            variance_method=variance_method,
        )

    except Exception:
        return WeightedRelativeRiskResult(
            rr=rr_point,
            ci_low=np.nan,
            ci_high=np.nan,
            p_value=np.nan,
            log_rr=float(np.log(rr_point)) if pd.notna(rr_point) and rr_point > 0 else np.nan,
            log_rr_ci=(np.nan, np.nan),
            risk_treated=risk_treated,
            risk_control=risk_control,
            treatment_label=g1,
            control_label=g2,
            model=None,
            variance_method="not estimable",
        )

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

    Continuous columns are converted to numeric and categorical columns are
    expanded with pd.get_dummies(..., drop_first=True). Scaling, when requested,
    is performed later inside estimate_ps_weights without sklearn.
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
    stabilize: bool = False,
    logistic_C: float | None = None,
    max_iter: int = 1000,
    trim=None,
    random_state: int | None = None,
    encoding: Encoding = "statsmodels",
    scale_all: bool = False,
    dummy_prefix: str | None = None,
    drop_first: bool = True,
) -> PSWeightResult:
    """Estimate propensity scores with an unpenalized statsmodels GLM.

    The propensity model is a conventional binomial logistic regression:

        X = sm.add_constant(X, has_constant="add")
        fit = sm.GLM(z, X, family=sm.families.Binomial()).fit()
        ps = fit.predict(X)

    This replaces the previous L2-penalized sklearn LogisticRegression and
    preserves the score-equation balancing property of overlap weighting.

    Notes
    -----
    * ``outcome_col`` is validated when supplied, but outcome completeness is
      deliberately NOT used to select the propensity-score estimation sample.
      One common set of design-stage weights should be estimated independently
      of the outcome being analyzed.
    * ``encoding='sklearn'`` is retained only as a backward-compatible alias;
      all encoding is now performed with ``pd.get_dummies`` and the model is
      fitted with statsmodels.
    * ``logistic_C`` and ``random_state`` are retained only so older calling
      code does not fail. They have no effect on an unpenalized GLM.
    * Scaling is unnecessary for an unpenalized model and defaults to False.
      If ``scale_all=True``, nonconstant design columns are standardized using
      complete-case means and population SDs before fitting.
    """
    missing = [c for c in [treatment_col, *covariates] if c not in df.columns]
    if missing:
        raise ValueError(f"These columns are missing from df: {missing}")

    if outcome_col is not None and outcome_col not in df.columns:
        raise ValueError(f"{outcome_col!r} is not in df.columns")

    out = df.copy()
    groups_found = list(out[treatment_col].dropna().unique())
    if control_label is None:
        controls = [g for g in groups_found if g != treated_label]
        if len(controls) != 1:
            raise ValueError(
                "Please provide control_label when treatment_col has more than two values."
            )
        control_label = controls[0]

    keep_group = out[treatment_col].isin([treated_label, control_label])
    out["_treated"] = np.where(out[treatment_col] == treated_label, 1, 0)

    cont, cat = _infer_column_types(
        out,
        covariates,
        categorical_covars,
        continuous_covars,
    )

    overlap = set(cont).intersection(cat)
    if overlap:
        raise ValueError(
            f"Covariates cannot be both continuous and categorical: {sorted(overlap)}"
        )
    omitted = [c for c in covariates if c not in set(cont).union(cat)]
    if omitted:
        raise ValueError(
            f"These covariates were not assigned a type: {omitted}"
        )

    # Design-stage complete cases depend only on treatment and baseline
    # covariates, not on outcome availability.
    complete = keep_group.copy()
    for c in covariates:
        complete &= out[c].notna()

    out[complete_col] = complete
    out[ps_col] = np.nan
    out[weight_col] = np.nan

    if int(complete.sum()) < 2:
        raise ValueError("Fewer than 2 complete cases are available for PS estimation.")

    z = out.loc[complete, "_treated"].astype(int)
    if z.nunique() != 2:
        raise ValueError("Complete cases must contain both groups.")

    # All former encoding modes now use the same auditable pandas design
    # matrix. 'sklearn' remains accepted as a compatibility alias.
    if encoding not in {"pandas", "statsmodels", "sklearn"}:
        raise ValueError(
            "encoding must be 'statsmodels', 'pandas', or the legacy alias 'sklearn'."
        )

    X_df = make_design_matrix_pandas(
        out.loc[complete],
        continuous_covars=cont,
        categorical_covars=cat,
        dummy_prefix=dummy_prefix,
        drop_first=drop_first,
    ).astype(float)

    # Remove any constant covariate columns before adding the intercept. A
    # constant dummy can arise in a bootstrap sample when one level is absent.
    constant_columns = [
        c for c in X_df.columns
        if X_df[c].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        X_df = X_df.drop(columns=constant_columns)

    if X_df.shape[1] == 0:
        raise ValueError(
            "The propensity-score design matrix has no varying covariates."
        )

    scaling_mean = None
    scaling_sd = None
    if scale_all:
        scaling_mean = X_df.mean(axis=0)
        scaling_sd = X_df.std(axis=0, ddof=0).replace(0, 1.0)
        X_df = (X_df - scaling_mean) / scaling_sd

    X = sm.add_constant(X_df, has_constant="add")

    # Detect exact linear dependence early and provide a clearer message than
    # a low-level linear algebra error from statsmodels.
    matrix_rank = np.linalg.matrix_rank(X.to_numpy(dtype=float))
    if matrix_rank < X.shape[1]:
        raise ValueError(
            "The propensity-score design matrix is rank deficient. "
            "Check redundant covariates, duplicate dummy variables, and "
            "categorical reference levels."
        )

    try:
        ps_model = sm.GLM(
            z.to_numpy(dtype=float),
            X,
            family=sm.families.Binomial(),
        ).fit(maxiter=max_iter, disp=0)
    except Exception as exc:
        raise RuntimeError(
            "The unpenalized propensity-score GLM failed. This may reflect "
            "complete/quasi-complete separation, sparse categorical levels, "
            "or a rank-deficient design matrix."
        ) from exc

    ps = np.asarray(ps_model.predict(X), dtype=float)
    if not np.all(np.isfinite(ps)):
        raise RuntimeError("The propensity-score model produced non-finite predictions.")

    design_columns = list(X.columns)
    w = compute_balancing_weights(
        ps,
        z.to_numpy(),
        estimand=estimand,
        stabilize=stabilize,
        trim=trim,
    )

    out.loc[complete, ps_col] = ps
    out.loc[complete, weight_col] = w

    # Attach preprocessing metadata directly to the fitted statsmodels result
    # so users can reproduce predictions on identically encoded data.
    ps_model.design_columns = design_columns
    ps_model.covariates = list(covariates)
    ps_model.continuous_covars = list(cont)
    ps_model.categorical_covars = list(cat)
    ps_model.constant_columns_removed = constant_columns
    ps_model.drop_first = drop_first
    ps_model.dummy_prefix = dummy_prefix
    ps_model.scale_all = scale_all
    ps_model.scaling_mean = scaling_mean
    ps_model.scaling_sd = scaling_sd
    ps_model.treated_label = treated_label
    ps_model.control_label = control_label

    return PSWeightResult(
        df=out,
        ps_col=ps_col,
        weight_col=weight_col,
        treatment_col=treatment_col,
        treated_label=treated_label,
        control_label=control_label,
        estimand=estimand.upper(),
        covariates=list(covariates),
        model=ps_model,
        encoding="statsmodels",
        scale_all=scale_all,
        design_columns=design_columns,
        complete_col=complete_col,
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
    encoding: Encoding = "statsmodels",
    scale_all: bool = False,
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
    smd_type: Literal["absolute", "signed"] = "absolute",
) -> pd.DataFrame:

    if smd_type not in ["absolute", "signed"]:
        raise ValueError("smd_type must be either 'absolute' or 'signed'")

    def apply_smd_type(smd):
        if pd.isna(smd):
            return smd
        if smd_type == "absolute":
            return abs(smd)
        return smd

    if groups is None:
        groups = list(df[group_col].dropna().unique())

    if len(groups) != 2:
        raise ValueError("Only two groups are supported.")

    g1, g2 = groups

    sub = df[df[group_col].isin(groups)].copy()
    d1 = sub[sub[group_col] == g1]
    d2 = sub[sub[group_col] == g2]

    w1 = d1[weight_col]
    w2 = d2[weight_col]

    rows = []

    if include_sum_weights:
        rows.append({
            "Variable": "Sum of weights",
            g1: f"{w1.sum():.1f}",
            g2: f"{w2.sum():.1f}",
            "SMD": "",
        })

    if include_effective_n:
        rows.append({
            "Variable": "Effective sample size",
            g1: f"{effective_sample_size(w1):.1f}",
            g2: f"{effective_sample_size(w2):.1f}",
            "SMD": "",
        })

    for var in continuous:
        m1 = weighted_mean(d1[var], w1)
        s1 = weighted_sd(d1[var], w1)
        m2 = weighted_mean(d2[var], w2)
        s2 = weighted_sd(d2[var], w2)

        smd = weighted_smd_cont(d1[var], w1, d2[var], w2)
        smd = apply_smd_type(smd)

        rows.append({
            "Variable": var,
            g1: f"{m1:.{digits_cont}f} ({s1:.{digits_cont}f})",
            g2: f"{m2:.{digits_cont}f} ({s2:.{digits_cont}f})",
            "SMD": format_smd(smd, digits_smd),
        })

    for var in binary_vars:
        p1 = weighted_prop(d1[var], w1, 1)
        p2 = weighted_prop(d2[var], w2, 1)

        smd = weighted_smd_binary(d1[var], w1, d2[var], w2, level=1)
        smd = apply_smd_type(smd)

        rows.append({
            "Variable": f"{var}, weighted %",
            g1: f"{p1 * 100:.{digits_pct}f}%",
            g2: f"{p2 * 100:.{digits_pct}f}%",
            "SMD": format_smd(smd, digits_smd),
        })

    for var in categorical:
        levels = sorted(sub[var].dropna().unique())

        smds = []
        for lv in levels:
            smd = weighted_smd_binary(
                (d1[var] == lv).astype(float), w1,
                (d2[var] == lv).astype(float), w2,
                level=1,
            )
            if pd.notna(smd):
                smds.append(smd)

        if smds:
            if smd_type == "absolute":
                overall_smd = max(abs(s) for s in smds)
            else:
                overall_smd = max(smds, key=lambda x: abs(x))
            overall_smd_txt = format_smd(overall_smd, digits_smd)
        else:
            overall_smd_txt = ""

        rows.append({
            "Variable": var,
            g1: "",
            g2: "",
            "SMD": overall_smd_txt,
        })

        for lv in levels:
            p1 = weighted_prop(d1[var], w1, lv)
            p2 = weighted_prop(d2[var], w2, lv)

            smd = weighted_smd_binary(
                (d1[var] == lv).astype(float), w1,
                (d2[var] == lv).astype(float), w2,
                level=1,
            )
            smd = apply_smd_type(smd)

            rows.append({
                "Variable": f"  {lv}",
                g1: f"{p1 * 100:.{digits_pct}f}%",
                g2: f"{p2 * 100:.{digits_pct}f}%",
                "SMD": format_smd(smd, digits_smd),
            })

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
    n_boot: int = 5000,
    alpha: float = 0.05,
    random_state: int = 0,
    logistic_C: float | None = None,
    ci_method: Literal["percentile", "basic"] = "percentile",
    min_success_fraction: float = 0.80,
    return_replicates: bool = False,
):
    """Bootstrap PS-weighted binary-outcome effects with PS refitting.

    This is the preferred Python-only inference method when the goal is to
    approximate PSweight's treatment of propensity-score estimation
    uncertainty without reproducing its full M-estimation sandwich variance.

    Each bootstrap replicate:
      1. resamples patients with replacement;
      2. refits the unpenalized binomial-logit propensity model;
      3. recalculates the requested balancing weights;
      4. recalculates weighted risks, risk difference, and risk ratio.

    The risk ratio is calculated directly from the two weighted risks rather
    than from a fixed-weight modified Poisson model. Its confidence interval
    is constructed on the log-RR scale and exponentiated, which guarantees a
    positive interval when both weighted risks are positive.

    Parameters
    ----------
    ci_method : {"percentile", "basic"}
        "percentile" uses empirical bootstrap quantiles.
        "basic" reflects the bootstrap quantiles around the original estimate.
    min_success_fraction : float
        Raise an error when too many replicates fail, commonly because a
        resample contains only one treatment group, a sparse factor level,
        separation, or zero weighted risk in one group.
    return_replicates : bool
        If True, return ``(summary, replicates)``. Otherwise return summary.

    Notes
    -----
    ``logistic_C`` is retained only for backward compatibility and is ignored
    because the propensity model is now an unpenalized statsmodels GLM.
    """
    if ci_method not in {"percentile", "basic"}:
        raise ValueError("ci_method must be 'percentile' or 'basic'.")
    if not 0 < min_success_fraction <= 1:
        raise ValueError("min_success_fraction must be in (0, 1].")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")

    analysis = df.copy().reset_index(drop=True)

    # Estimate the original-sample effects with the same PS and weighting
    # pipeline used in every bootstrap replicate.
    original_ps = estimate_ps_weights(
        analysis,
        treatment_col=treatment_col,
        treated_label=treated_label,
        control_label=control_label,
        covariates=covariates,
        outcome_col=None,  # PS fitting must not depend on outcome availability
        categorical_covars=categorical_covars,
        continuous_covars=continuous_covars,
        estimand=estimand,
        encoding="statsmodels",
        scale_all=False,
    )

    original = original_ps.df.loc[
        original_ps.df[outcome_col].notna() & original_ps.df["weight"].notna()
    ].copy()
    original[outcome_col] = pd.to_numeric(original[outcome_col], errors="coerce")
    original["weight"] = pd.to_numeric(original["weight"], errors="coerce")
    original = original.loc[
        original[outcome_col].notna()
        & np.isfinite(original["weight"])
        & (original["weight"] >= 0)
    ]

    def _effects(d: pd.DataFrame) -> dict[str, float]:
        treated = d.loc[d[treatment_col] == treated_label]
        control = d.loc[d[treatment_col] == control_label]
        if treated.empty or control.empty:
            raise ValueError("Both treatment groups are required.")

        risk_t = weighted_mean(treated[outcome_col], treated["weight"])
        risk_c = weighted_mean(control[outcome_col], control["weight"])
        if not np.isfinite(risk_t) or not np.isfinite(risk_c):
            raise ValueError("Weighted risks are not finite.")

        rd = risk_t - risk_c
        rr = risk_t / risk_c if risk_c > 0 else np.nan
        log_rr = np.log(rr) if np.isfinite(rr) and rr > 0 else np.nan
        return {
            "risk_treated": float(risk_t),
            "risk_control": float(risk_c),
            "risk_difference": float(rd),
            "risk_ratio": float(rr) if np.isfinite(rr) else np.nan,
            "log_risk_ratio": float(log_rr) if np.isfinite(log_rr) else np.nan,
        }

    original_effects = _effects(original)

    rng = np.random.default_rng(random_state)
    n = len(analysis)
    records: list[dict[str, float]] = []

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bdf = analysis.iloc[idx].reset_index(drop=True)
        try:
            ps = estimate_ps_weights(
                bdf,
                treatment_col=treatment_col,
                treated_label=treated_label,
                control_label=control_label,
                covariates=covariates,
                outcome_col=None,
                categorical_covars=categorical_covars,
                continuous_covars=continuous_covars,
                estimand=estimand,
                encoding="statsmodels",
                scale_all=False,
            )

            bd = ps.df.loc[
                ps.df[outcome_col].notna() & ps.df["weight"].notna()
            ].copy()
            bd[outcome_col] = pd.to_numeric(bd[outcome_col], errors="coerce")
            bd["weight"] = pd.to_numeric(bd["weight"], errors="coerce")
            bd = bd.loc[
                bd[outcome_col].notna()
                & np.isfinite(bd["weight"])
                & (bd["weight"] >= 0)
            ]

            effect = _effects(bd)
            effect["replicate"] = b
            records.append(effect)
        except Exception:
            continue

    replicates = pd.DataFrame.from_records(records)
    n_success = len(replicates)
    required_success = int(np.ceil(n_boot * min_success_fraction))
    if n_success < required_success:
        raise RuntimeError(
            f"Only {n_success}/{n_boot} bootstrap replicates succeeded; "
            f"at least {required_success} were required. Check sparse levels, "
            "separation, positivity, or reduce min_success_fraction deliberately."
        )

    qlo, qhi = alpha / 2, 1 - alpha / 2

    def _bootstrap_ci(values, estimate):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan, np.nan
        lo_q, hi_q = np.quantile(arr, [qlo, qhi])
        if ci_method == "percentile":
            lo, hi = lo_q, hi_q
        else:
            lo, hi = 2 * estimate - hi_q, 2 * estimate - lo_q
        se = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
        return float(lo), float(hi), se

    rt_lo, rt_hi, rt_se = _bootstrap_ci(
        replicates["risk_treated"], original_effects["risk_treated"]
    )
    rc_lo, rc_hi, rc_se = _bootstrap_ci(
        replicates["risk_control"], original_effects["risk_control"]
    )
    rd_lo, rd_hi, rd_se = _bootstrap_ci(
        replicates["risk_difference"], original_effects["risk_difference"]
    )

    # Construct RR inference on the log scale. Replicates with zero risk in
    # either group cannot contribute to a finite log-RR interval and are
    # excluded only from RR inference, not from RD inference.
    finite_log_rr = replicates["log_risk_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if np.isfinite(original_effects["log_risk_ratio"]) and len(finite_log_rr) > 0:
        lrr_lo, lrr_hi, lrr_se = _bootstrap_ci(
            finite_log_rr,
            original_effects["log_risk_ratio"],
        )
        rr_lo, rr_hi = np.exp(lrr_lo), np.exp(lrr_hi)
    else:
        lrr_lo = lrr_hi = lrr_se = rr_lo = rr_hi = np.nan

    # Two-sided bootstrap p values based on the empirical sign distribution.
    # These are descriptive bootstrap tests; the CIs should remain primary.
    rd_vals = replicates["risk_difference"].to_numpy(float)
    rd_p = 2 * min(np.mean(rd_vals <= 0), np.mean(rd_vals >= 0))
    rd_p = float(min(1.0, rd_p))

    if len(finite_log_rr) > 0:
        lrr_vals = finite_log_rr.to_numpy(float)
        rr_p = 2 * min(np.mean(lrr_vals <= 0), np.mean(lrr_vals >= 0))
        rr_p = float(min(1.0, rr_p))
    else:
        rr_p = np.nan

    summary = pd.DataFrame([{
        "Estimand": str(estimand),
        "CI method": f"refitted PS bootstrap ({ci_method})",
        "n_boot_requested": int(n_boot),
        "n_boot_success_RD": int(n_success),
        "n_boot_success_RR": int(len(finite_log_rr)),
        f"{treated_label} weighted risk": original_effects["risk_treated"],
        f"{treated_label} risk SE": rt_se,
        f"{treated_label} risk CI low": rt_lo,
        f"{treated_label} risk CI high": rt_hi,
        f"{control_label} weighted risk": original_effects["risk_control"],
        f"{control_label} risk SE": rc_se,
        f"{control_label} risk CI low": rc_lo,
        f"{control_label} risk CI high": rc_hi,
        "Risk difference": original_effects["risk_difference"],
        "RD SE": rd_se,
        "RD CI low": rd_lo,
        "RD CI high": rd_hi,
        "RD bootstrap p": rd_p,
        "Risk ratio": original_effects["risk_ratio"],
        "log(RR) SE": lrr_se,
        "RR CI low": rr_lo,
        "RR CI high": rr_hi,
        "RR bootstrap p": rr_p,
    }])

    if return_replicates:
        return summary, replicates
    return summary

def love_plot(
    before_table: pd.DataFrame,
    after_table: pd.DataFrame,
    *,
    variable_col: str = "Variable",
    smd_col: str = "SMD",
    exclude_variables: Optional[Iterable[str]] = None,
    threshold: float = 0.1,
    out_path: Optional[str] = None,
    sort_by: Literal["table", "smd"] = "table",
    smd_type: Literal["absolute", "signed"] = "absolute",
    color_mode: Literal["color", "monotone"] = "monotone",
    positive_label: str = "FADE higher",
    negative_label: str = "AE higher",
    title: str = "Covariate balance before and after weighting",
    figsize_width: float = 7,
    dpi: int = 600,
):
    """
    Create a Love plot before and after weighting.

    Parameters
    ----------
    smd_type:
        "absolute" : plot absolute SMD. This is the most common journal style.
        "signed"   : plot signed SMD. Positive values indicate higher values
                     in the group specified by positive_label.

    color_mode:
        "monotone" : black/gray journal-style figure.
        "color"    : color figure.

    sort_by:
        "table" : keep the order from before_table.
        "smd"   : sort by the magnitude of SMD before weighting.

    Notes
    -----
    If smd_type="signed", the SMD column must contain signed SMD values.
    """

    import matplotlib.pyplot as plt

    if smd_type not in ["absolute", "signed"]:
        raise ValueError("smd_type must be either 'absolute' or 'signed'")

    if color_mode not in ["color", "monotone"]:
        raise ValueError("color_mode must be either 'color' or 'monotone'")

    if sort_by not in ["table", "smd"]:
        raise ValueError("sort_by must be either 'table' or 'smd'")

    exclude = set(exclude_variables or [])

    def norm_name(v):
        v = str(v).strip()
        return (
            v.replace(", n (%)", "")
             .replace(", weighted %", "")
             .replace(", n (weighted %)", "")
             .strip()
        )

    def parse_smd(x):
        val = float(str(x).strip())
        if smd_type == "absolute":
            val = abs(val)
        return val

    def extract(tbl):
        res = {}
        order = []

        for _, row in tbl.iterrows():
            name = norm_name(row[variable_col])

            if name in exclude:
                continue

            try:
                smd = parse_smd(row[smd_col])
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
            key=lambda x: abs(b[x]),
            reverse=True,
        )
    else:
        common = [v for v in before_order if v in a]

    bv = [b[v] for v in common]
    av = [a[v] for v in common]

    n = len(common)
    fig_h = max(5, n * 0.30 + 1.5)

    fig, ax = plt.subplots(figsize=(figsize_width, fig_h))
    y = np.arange(n)

    if color_mode == "monotone":
        line_color = "0.65"
        before_face = "black"
        before_edge = "black"
        after_face = "white"
        after_edge = "black"
        grid_color = "0.85"
    else:
        line_color = "0.70"
        before_face = "tab:blue"
        before_edge = "tab:blue"
        after_face = "tab:orange"
        after_edge = "tab:orange"
        grid_color = "0.85"

    for i in range(n):
        ax.plot(
            [bv[i], av[i]],
            [y[i], y[i]],
            color=line_color,
            lw=0.8,
            zorder=2,
        )

    ax.scatter(
        bv,
        y,
        s=40,
        marker="o",
        facecolors=before_face,
        edgecolors=before_edge,
        linewidths=0.8,
        zorder=4,
        label="Before weighting",
    )

    ax.scatter(
        av,
        y,
        s=40,
        marker="o",
        facecolors=after_face,
        edgecolors=after_edge,
        linewidths=0.8,
        zorder=5,
        label="After weighting",
    )

    ax.axvline(0, color="black", lw=1.0, zorder=3)

    if smd_type == "absolute":
        ax.axvline(threshold, color="black", lw=1.0, ls="--", alpha=0.7, zorder=3)
        ax.set_xlabel("|Standardized mean difference|")
        ax.set_xlim(left=-0.01)

    else:
        ax.axvline(threshold, color="black", lw=1.0, ls="--", alpha=0.7, zorder=3)
        ax.axvline(-threshold, color="black", lw=1.0, ls="--", alpha=0.7, zorder=3)

        xmax = max(
            abs(np.nanmax(bv)) if len(bv) else threshold,
            abs(np.nanmin(bv)) if len(bv) else threshold,
            abs(np.nanmax(av)) if len(av) else threshold,
            abs(np.nanmin(av)) if len(av) else threshold,
            threshold,
        )
        xmax = max(0.30, np.ceil((xmax + 0.05) * 10) / 10)

        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel("Standardized mean difference")

        ax.text(
            -xmax,
            -0.9,
            negative_label,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            xmax,
            -0.9,
            positive_label,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(common, fontsize=8)

    ax.set_title(title)
    ax.grid(axis="x", color=grid_color, lw=0.6, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=9, frameon=False, loc="lower right")

    # Show first Table 1 variable at the top
    ax.invert_yaxis()

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

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
    color_mode="color",  # "color" or "monotone"
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


    color_mode:
        "color"    : green/orange by direction.
        "monotone" : black/gray journal-style plot.
    """

    if color_mode not in ["color", "monotone"]:
        raise ValueError("color_mode must be either 'color' or 'monotone'")

    df = outcome_table.copy()

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

    df["RD_plot"] = df[rd_col].apply(parse_percent)

    cis = df[ci_col].apply(parse_ci)
    df["CI_low_plot"] = cis.apply(lambda x: x[0])
    df["CI_high_plot"] = cis.apply(lambda x: x[1])

    df = df.dropna(
        subset=[outcome_col, "RD_plot", "CI_low_plot", "CI_high_plot"]
    ).reset_index(drop=True)

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
            xmin = min(xmin, 0)
            xmax = max(xmax, 0)
            xlim = (xmin, xmax)

    n = len(df)

    if figsize is None:
        figsize = (8, max(4, n * 0.55 + 1.6))

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(n)[::-1]

    for yy, (_, row) in zip(y, df.iterrows()):
        rd = row["RD_plot"]
        lo = row["CI_low_plot"]
        hi = row["CI_high_plot"]

        significant = (hi < 0) or (lo > 0)

        if color_mode == "monotone":
            color = "black" if significant else "0.45"
            marker_face = "black" if significant else "white"
            marker_edge = "black"
            alpha = 1.0
        else:
            if rd < 0:
                color = "#1b9e77"   # favors FADE
            else:
                color = "#d95f02"   # favors AE
            marker_face = color
            marker_edge = "white"
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
            facecolor=marker_face,
            edgecolor=marker_edge,
            linewidth=0.8,
            alpha=alpha,
            zorder=3,
        )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

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

    ax.set_xlim(xlim[0], xlim[1] + 0.45 * (xlim[1] - xlim[0]))

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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linewidth=0.5, alpha=0.4)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig

# ============================================================
# Convenience wrappers
# ============================================================

def estimate_iptw_weights(
    df: pd.DataFrame,
    *,
    group_col: str = "Group",
    groups: Sequence = ("treated", "control"),
    continuous_covars: Sequence[str],
    categorical_covars: Sequence[str] = (),
    outcome_col: Optional[str] = None,
    ps_col: str = "PS",
    weight_col: str = "IPTW",
    stabilized: bool = True,
    complete_col: str = "_complete",
    encoding: Encoding = "statsmodels",
    scale_all: bool = False,
    dummy_prefix: str | None = "dept",
    drop_first: bool = True,
    **kwargs,
) -> PSWeightResult:
    """
    Estimate stabilized or unstabilized IPTW weights.

    Example
    -------
    iptw = estimate_iptw_weights(
        main,
        group_col="Group",
        groups=["FADE", "AE"],
        continuous_covars=["Age", "BMI"],
        categorical_covars=["Procedure type"],
        stabilized=True,
    )
    main_iptw = iptw.df

    Notes
    -----
    With groups=["FADE", "AE"], the propensity score is P(FADE | covariates).
    Stabilized IPTW uses P(FADE)/PS for FADE and P(AE)/(1-PS) for AE.
    """
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
        estimand="ATE",
        ps_col=ps_col,
        weight_col=weight_col,
        complete_col=complete_col,
        encoding=encoding,
        scale_all=scale_all,
        dummy_prefix=dummy_prefix,
        drop_first=drop_first,
        stabilize=stabilized,
        **kwargs,
    )

# ============================================================
# EEG / DSA pipeline dependencies
# ============================================================
from datetime import date

import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

try:
    from mne.time_frequency import psd_array_multitaper
except Exception:
    psd_array_multitaper = None

try:
    from asrpy import asr_calibrate, asr_process, clean_windows
    ASRPY_AVAILABLE = True
except Exception:
    ASRPY_AVAILABLE = False


# ============================================================
# Integrated EEG + OR pipeline
# ============================================================


# -----------------------------------------------------------------------------
# EEG/OR integration note
# The following section is integrated into kacr.py. It uses the PK simulation
# functions defined above and therefore does not require a separate my_mod.py.
# -----------------------------------------------------------------------------
# --------------------------
# Basic signal processing
# --------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Zero-phase Butterworth band-pass filter."""
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=0)


def notch_filter(data, fs, notch_hz=50.0, q=30.0):
    """Zero-phase notch filter. Use 50 Hz in Japan and 60 Hz in the US."""
    b, a = iirnotch(notch_hz, Q=q, fs=fs)
    return filtfilt(b, a, data, axis=0)


def apply_asr_2ch(data_2ch, fs=250, cutoff=20):
    """Apply ASR to 2-channel EEG. Falls back outside this function if ASR fails."""
    if not ASRPY_AVAILABLE:
        raise ImportError("asrpy is not installed.")

    eeg_array = data_2ch.T  # (n_channels, n_samples)
    pre_cleaned, _ = clean_windows(eeg_array, fs, max_bad_chans=0.1)
    M, T = asr_calibrate(pre_cleaned, fs, cutoff=cutoff)
    clean_array = asr_process(eeg_array, fs, M, T)
    return clean_array.T


def safe_datetime_parse(ts_series):
    ts = pd.to_datetime(ts_series, errors="coerce")
    if ts.isna().any():
        ts = pd.to_datetime(
            ts_series,
            format="%Y/%m/%d %H:%M:%S:%f",
            errors="coerce",
        )
    return ts


def sliding_windows_1d(x, fs, window_sec=3.0, step_sec=0.5):
    """Return overlapping windows and centre sample indices."""
    x = np.asarray(x, dtype=float)
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))

    if len(x) < win:
        raise ValueError("EEG signal is shorter than one analysis window.")

    starts = np.arange(0, len(x) - win + 1, step, dtype=int)
    windows = np.stack([x[s:s + win] for s in starts], axis=0)
    centers = starts + win // 2
    return windows, centers


def reject_artifact_windows(windows, amplitude_uv=200.0, flat_sd_uv=0.5):
    """
    Simple epoch rejection.
    Reject windows with large amplitudes or near-flat signal.
    Adjust amplitude_uv depending on the scale of exported BIS raw EEG.
    """
    max_abs = np.nanmax(np.abs(windows), axis=1)
    sd = np.nanstd(windows, axis=1)

    bad = (
        (max_abs > amplitude_uv)
        | (sd < flat_sd_uv)
        | ~np.isfinite(max_abs)
        | ~np.isfinite(sd)
    )
    return bad


# --------------------------
# PSD methods
# --------------------------
def compute_psd_windows(
    windows,
    fs,
    method="multitaper",
    fmin=0.5,
    fmax=40.0,
    bandwidth=2.0,
    adaptive=True,
):
    """Compute PSD using either multitaper or Welch."""
    method = method.lower()

    if method == "multitaper":
        P, f = psd_array_multitaper(
            windows,
            sfreq=fs,
            fmin=fmin,
            fmax=fmax,
            bandwidth=bandwidth,
            adaptive=adaptive,
            normalization="full",
            verbose=False,
        )
        return np.asarray(P, dtype=float), f

    if method == "welch":
        psd_list = []
        f_ref = None

        for w in windows:
            f, p = welch(
                w,
                fs=fs,
                nperseg=len(w),
                noverlap=0,
                scaling="density",
            )
            mask = (f >= fmin) & (f <= fmax)

            if f_ref is None:
                f_ref = f[mask]

            psd_list.append(p[mask])

        return np.asarray(psd_list, dtype=float), f_ref

    raise ValueError("method must be 'multitaper' or 'welch'.")


def compute_dsa(
    x_1d,
    fs,
    timestamps=None,
    window_sec=3.0,
    step_sec=0.5,
    method="multitaper",
    fmin=0.5,
    fmax=40.0,
    bandwidth=2.0,
    adaptive=True,
    artifact_amplitude_uv=200.0,
    flat_sd_uv=0.5,
):
    """
    Compute a DSA/spectrogram using either multitaper or Welch PSD.

    Returns
    -------
    f : ndarray
        Frequency grid in Hz.
    P : ndarray, shape (n_windows, n_freqs)
        Linear PSD. Artefact windows are set to NaN.
    t : list or ndarray
        Window-centre times.
    bad : ndarray
        Boolean artefact mask.
    """
    windows, centers = sliding_windows_1d(
        x_1d,
        fs,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    windows = windows - np.nanmean(windows, axis=1, keepdims=True)

    bad = reject_artifact_windows(
        windows,
        amplitude_uv=artifact_amplitude_uv,
        flat_sd_uv=flat_sd_uv,
    )

    good_windows = windows.copy()
    good_windows[bad, :] = 0.0

    P, f = compute_psd_windows(
        good_windows,
        fs=fs,
        method=method,
        fmin=fmin,
        fmax=fmax,
        bandwidth=bandwidth,
        adaptive=adaptive,
    )

    P = np.asarray(P, dtype=float)
    P[bad, :] = np.nan

    if timestamps is not None:
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
        t = ts.iloc[centers].to_list()
    else:
        t = centers / fs

    return f, P, t, bad


# --------------------------
# Analysis helpers
# --------------------------
def psd_to_db(P_lin):
    return 10.0 * np.log10(np.maximum(P_lin, 1e-20))


def normalize_spectrum(P_lin, f_hz, mode="relative_db"):
    """
    Optional normalization for ML or secondary plots.
    This is not used to overwrite raw PSD outputs.
    """
    eps = 1e-20
    P = np.maximum(P_lin, eps)

    if mode == "relative_db":
        denom = np.trapezoid(P, f_hz, axis=1).reshape(-1, 1) + eps
        return 10.0 * np.log10(P / denom + eps)

    if mode == "zscore_db":
        P_db = psd_to_db(P)
        mu = np.nanmean(P_db, axis=1, keepdims=True)
        sd = np.nanstd(P_db, axis=1, keepdims=True) + eps
        return (P_db - mu) / sd

    raise ValueError("mode must be 'relative_db' or 'zscore_db'.")


def robust_limits(arr_list, low=5, high=95):
    vals = []

    for A in arr_list:
        x = np.asarray(A).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)

    if not vals:
        return -40.0, 20.0

    x = np.concatenate(vals)
    vmin = float(np.percentile(x, low))
    vmax = float(np.percentile(x, high))

    if vmax <= vmin:
        vmax = vmin + 1.0

    return vmin, vmax


def _time_extent(times, f):
    if len(times) == 0:
        raise ValueError("No time points to plot.")

    if isinstance(times[0], pd.Timestamp) or np.issubdtype(np.asarray(times).dtype, np.datetime64):
        x0 = mdates.date2num(pd.to_datetime(times[0]))
        x1 = mdates.date2num(pd.to_datetime(times[-1]))
    else:
        x0 = float(times[0]) / 60.0
        x1 = float(times[-1]) / 60.0

    return [x0, x1, float(f[0]), float(f[-1])]


def bandpower(P_lin, f_hz, lo, hi):
    m = (f_hz >= lo) & (f_hz < hi)
    if not np.any(m):
        return np.full(P_lin.shape[0], np.nan)
    return np.trapezoid(P_lin[:, m], f_hz[m], axis=1)


def compute_relative_bandpower(P_lin, f_hz):
    """Return raw relative bandpower. Smoothing is applied only during plotting."""
    total = bandpower(P_lin, f_hz, f_hz[0], f_hz[-1]) + 1e-20

    bands = {
        "delta": bandpower(P_lin, f_hz, 0.5, 4) / total,
        "theta": bandpower(P_lin, f_hz, 4, 8) / total,
        "alpha": bandpower(P_lin, f_hz, 8, 12) / total,
        "beta": bandpower(P_lin, f_hz, 12, 30) / total,
    }

    if f_hz[-1] >= 40:
        bands["gamma"] = bandpower(P_lin, f_hz, 30, 40) / total

    return bands


def smooth_bandpower_for_plot(bands, smooth=True, sigma=8):
    """Smooth bandpower for visualization only."""
    if not smooth:
        return bands

    out = {}
    for k, v in bands.items():
        if np.all(~np.isfinite(v)):
            out[k] = v
        else:
            # Replace isolated NaN for plotting, then restore all-NaN-free smooth curve.
            s = pd.Series(v).interpolate(limit_direction="both").to_numpy()
            out[k] = gaussian_filter1d(s, sigma=sigma)
    return out


# --------------------------
# Plot functions
# --------------------------
def plot_dsa_with_relative_bandpower(
    f_hz,
    P_lin,
    times,
    channel_name="ch1",
    psd_method="multitaper",
    output_path=None,
    cmap="jet",
    smooth_bandpower=True,
    smooth_sigma=8,
    vmin=None,
    vmax=None,
):
    """
    Plot absolute DSA and relative band power aligned vertically.
    Relative band power is smoothed only for visualization.
    """
    P_db = psd_to_db(P_lin)

    if vmin is None or vmax is None:
        vmin, vmax = robust_limits([P_db], low=5, high=95)

    bands_raw = compute_relative_bandpower(P_lin, f_hz)
    bands_plot = smooth_bandpower_for_plot(
        bands_raw,
        smooth=smooth_bandpower,
        sigma=smooth_sigma,
    )

    fig = plt.figure(figsize=(14, 6))

    gs = GridSpec(
        2,
        2,
        width_ratios=[30, 0.8],
        height_ratios=[2.2, 1.4],
        hspace=0.08,
        wspace=0.05,
        figure=fig,
    )

    ax_dsa = fig.add_subplot(gs[0, 0])
    ax_rel = fig.add_subplot(gs[1, 0], sharex=ax_dsa)
    cax = fig.add_subplot(gs[0, 1])

    ext = _time_extent(times, f_hz)

    im = ax_dsa.imshow(
        P_db.T,
        origin="lower",
        aspect="auto",
        extent=ext,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax_dsa.set_title(f"{psd_method.capitalize()} DSA - {channel_name}")
    ax_dsa.set_ylabel("Freq (Hz)")
    ax_dsa.set_ylim([float(f_hz[0]), float(f_hz[-1])])

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Power (dB)")

    line_order = ["delta", "theta", "alpha", "beta", "gamma"]
    labels = {
        "delta": "delta 0.5-4 Hz",
        "theta": "theta 4-8 Hz",
        "alpha": "alpha 8-12 Hz",
        "beta": "beta 12-30 Hz",
        "gamma": "gamma 30-40 Hz",
    }

    for name in line_order:
        if name in bands_plot:
            lw = 1.8 if name == "alpha" else 1.4
            ax_rel.plot(times, bands_plot[name], linewidth=lw, label=labels[name])

    ax_rel.set_ylim([0, 1])
    ax_rel.set_ylabel("Relative\npower")
    ax_rel.set_xlabel("Time")
    ax_rel.legend(loc="upper right")

    if len(times) and (
        isinstance(times[0], pd.Timestamp)
        or np.issubdtype(np.asarray(times).dtype, np.datetime64)
    ):
        ax_dsa.xaxis_date()
        ax_rel.xaxis_date()
        ax_rel.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.setp(ax_dsa.get_xticklabels(), visible=False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_dsa_relpower_two_channels_aligned(
    f_hz,
    P1_lin,
    t1,
    P2_lin,
    t2,
    psd_method="multitaper",
    output_path=None,
    cmap="jet",
    smooth_bandpower=True,
    smooth_sigma=8,
):
    """
    Four-panel aligned figure:
      1. ch1 absolute DSA
      2. ch1 relative band power
      3. ch2 absolute DSA
      4. ch2 relative band power
    Color bar has its own column, preserving x-axis alignment.
    """
    P1_db = psd_to_db(P1_lin)
    P2_db = psd_to_db(P2_lin)
    vmin, vmax = robust_limits([P1_db, P2_db], low=5, high=95)

    bands1 = smooth_bandpower_for_plot(
        compute_relative_bandpower(P1_lin, f_hz),
        smooth=smooth_bandpower,
        sigma=smooth_sigma,
    )
    bands2 = smooth_bandpower_for_plot(
        compute_relative_bandpower(P2_lin, f_hz),
        smooth=smooth_bandpower,
        sigma=smooth_sigma,
    )

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(
        4,
        2,
        width_ratios=[30, 0.8],
        height_ratios=[2.2, 1.3, 2.2, 1.3],
        hspace=0.12,
        wspace=0.05,
        figure=fig,
    )

    ax_dsa1 = fig.add_subplot(gs[0, 0])
    ax_rel1 = fig.add_subplot(gs[1, 0], sharex=ax_dsa1)
    ax_dsa2 = fig.add_subplot(gs[2, 0], sharex=ax_dsa1)
    ax_rel2 = fig.add_subplot(gs[3, 0], sharex=ax_dsa1)
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[2, 1])

    ext1 = _time_extent(t1, f_hz)
    ext2 = _time_extent(t2, f_hz)

    im1 = ax_dsa1.imshow(
        P1_db.T,
        origin="lower",
        aspect="auto",
        extent=ext1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_dsa1.set_title(f"{psd_method.capitalize()} DSA - ch1 (BIS_1)")
    ax_dsa1.set_ylabel("Freq (Hz)")
    ax_dsa1.set_ylim([float(f_hz[0]), float(f_hz[-1])])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label("Power (dB)")

    im2 = ax_dsa2.imshow(
        P2_db.T,
        origin="lower",
        aspect="auto",
        extent=ext2,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_dsa2.set_title(f"{psd_method.capitalize()} DSA - ch2 (BIS_2)")
    ax_dsa2.set_ylabel("Freq (Hz)")
    ax_dsa2.set_ylim([float(f_hz[0]), float(f_hz[-1])])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label("Power (dB)")

    def _plot_bands(ax, times, bands):
        labels = {
            "delta": "delta",
            "theta": "theta",
            "alpha": "alpha",
            "beta": "beta",
            "gamma": "gamma",
        }
        for name in ["delta", "theta", "alpha", "beta", "gamma"]:
            if name in bands:
                lw = 1.8 if name == "alpha" else 1.3
                ax.plot(times, bands[name], linewidth=lw, label=labels[name])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Relative\npower")
        ax.legend(loc="upper right", ncol=5, fontsize=9)

    _plot_bands(ax_rel1, t1, bands1)
    _plot_bands(ax_rel2, t2, bands2)

    for ax in [ax_dsa1, ax_rel1, ax_dsa2, ax_rel2]:
        if len(t1) and (
            isinstance(t1[0], pd.Timestamp)
            or np.issubdtype(np.asarray(t1).dtype, np.datetime64)
        ):
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.setp(ax_dsa1.get_xticklabels(), visible=False)
    plt.setp(ax_rel1.get_xticklabels(), visible=False)
    plt.setp(ax_dsa2.get_xticklabels(), visible=False)
    ax_rel2.set_xlabel("Time")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_psd_summary(
    f,
    P_epochs,
    times,
    channel_name,
    psd_method="multitaper",
    output_path=None,
    smooth_bandpower=True,
    smooth_sigma=8,
):
    """PSD summary + relative band power. Downstream data remain unsmoothed."""
    P_db = psd_to_db(P_epochs)
    mean_db = np.nanmean(P_db, axis=0)
    p25 = np.nanpercentile(P_db, 25, axis=0)
    p75 = np.nanpercentile(P_db, 75, axis=0)

    bands = compute_relative_bandpower(P_epochs, f)
    bands_plot = smooth_bandpower_for_plot(bands, smooth=smooth_bandpower, sigma=smooth_sigma)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    method_label = psd_method.capitalize()

    axs[0].plot(f, mean_db, linewidth=1.5)
    axs[0].fill_between(f, p25, p75, alpha=0.2)
    axs[0].set_title(f"{method_label} PSD summary - {channel_name}")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Power (dB)")
    axs[0].set_xlim([f[0], f[-1]])

    label_map = {
        "delta": "delta 0.5-4 Hz",
        "theta": "theta 4-8 Hz",
        "alpha": "alpha 8-12 Hz",
        "beta": "beta 12-30 Hz",
        "gamma": "gamma 30-40 Hz",
    }
    for name in ["delta", "theta", "alpha", "beta", "gamma"]:
        if name in bands_plot:
            lw = 1.8 if name == "alpha" else 1.3
            axs[1].plot(times, bands_plot[name], linewidth=lw, label=label_map[name])

    axs[1].set_title(f"Relative band power - {channel_name}")
    axs[1].set_ylabel("Relative power")
    axs[1].set_ylim([0, 1])
    axs[1].legend(loc="upper right")

    if len(times) and (
        isinstance(times[0], pd.Timestamp)
        or np.issubdtype(np.asarray(times).dtype, np.datetime64)
    ):
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    axs[1].set_xlabel("Time")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_hemo(meta, output_path=None):
    """Plot hemodynamic parameters (HR, sART as lines; sNIBP as scatter)."""
    m = meta.copy()
    if "timestamp" not in m.columns:
        m = m.reset_index()
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")
    m = m.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(14, 3))

    for col, label in {"HR bpm": "HR", "ART(S) mmHg": "sART"}.items():
        if col in m.columns:
            ax.plot(m["timestamp"], pd.to_numeric(m[col], errors="coerce"),
                    linewidth=1.2, label=label)

    if "NIBP(収縮期血圧) mmHg" in m.columns:
        nibp = m.dropna(subset=["NIBP(収縮期血圧) mmHg"])
        ax.scatter(
            nibp["timestamp"],
            pd.to_numeric(nibp["NIBP(収縮期血圧) mmHg"], errors="coerce"),
            s=18, color="orange", label="sNIBP", zorder=3,
        )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("Hemodynamics")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_eeg_metrics(meta, output_path=None):
    """Plot EEG monitor metrics: BIS, SEF95, SQI, EMG."""
    m = meta.copy()
    if "timestamp" not in m.columns:
        m = m.reset_index()
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")
    m = m.dropna(subset=["timestamp"]).sort_values("timestamp")

    eeg_cols = {
        "BIS": "BIS",
        "SEF95(bis) Hz": "SEF95",
        "SQI(bis) %": "SQI",
        "EMG(bis) dB": "EMG",
    }

    fig, ax = plt.subplots(figsize=(14, 3))

    for col, label in eeg_cols.items():
        if col in m.columns:
            ax.plot(m["timestamp"], pd.to_numeric(m[col], errors="coerce"),
                    linewidth=1.2, label=label)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("EEG metrics")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# --------------------------
# Loader matching your file format
# --------------------------
def load_eeg_and_meta(input_root, ID):
    input_dir = os.path.join(input_root, ID)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    dfs = []
    meta = None

    for file in os.listdir(input_dir):
        if file.startswith("実波形") and file.endswith(".csv"):
            dfw = pd.read_csv(
                os.path.join(input_dir, file),
                skiprows=2,
                encoding="shift_jis",
                header=None,
            )
            dfs.append(dfw)

        elif file.startswith("数値") and file.endswith(".csv"):
            meta = pd.read_csv(
                os.path.join(input_dir, file),
                skiprows=1,
                encoding="shift_jis",
            )
            meta = meta.rename(columns={"Unnamed: 0": "timestamp", "BIS ": "BIS"})

    if len(dfs) == 0:
        raise FileNotFoundError("No waveform files found (実波形*.csv).")

    if meta is None:
        raise FileNotFoundError("No meta file found (数値*.csv).")

    comb_df = pd.concat(dfs, ignore_index=True)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(
        comb_df[0],
        format="%Y/%m/%d %H:%M:%S:%f",
        errors="coerce",
    )

    if out["timestamp"].isna().any():
        out["timestamp"] = safe_datetime_parse(comb_df[0])

    out["BIS_1"] = pd.to_numeric(comb_df[1], errors="coerce")
    out["BIS_2"] = pd.to_numeric(comb_df[2], errors="coerce")

    out = (
        out.dropna(subset=["timestamp", "BIS_1", "BIS_2"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    meta["timestamp"] = pd.to_datetime(
        meta["timestamp"],
        errors="coerce",
    )

    meta = (
        meta.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .sort_index()
    )

    return out, meta, input_dir


# --------------------------
# Main pipeline
# --------------------------
def run_dsa_pipeline(
    ID,
    input_root,
    output_root,
    sampling_freq=250,
    window_sec=3.0,
    step_sec=0.5,
    psd_method="multitaper",  # "multitaper" or "welch"
    use_asr=True,
    use_notch=True,
    notch_hz=50.0,
    bandpass=(0.5, 40.0),
    psd_fmin=0.5,
    psd_fmax=40.0,
    multitaper_bandwidth=2.0,
    multitaper_adaptive=True,
    artifact_amplitude_uv=200.0,
    flat_sd_uv=0.5,
    show_summary=True,
    show_aligned_two_channel=True,
    show_single_channel_aligned=False,
    show_normalized_csv=True,
    show_hemo=True,
    show_eeg_metrics=True,
    norm_mode="relative_db",
    smooth_bandpower=True,
    smooth_sigma=8,
    cmap="jet",
    return_analysis=False,
):
    psd_method = psd_method.lower()

    if psd_method not in ["multitaper", "welch"]:
        raise ValueError("psd_method must be 'multitaper' or 'welch'.")

    df_raw, meta, _ = load_eeg_and_meta(input_root, ID)

    output_dir = os.path.join(output_root, ID)
    os.makedirs(output_dir, exist_ok=True)

    today = date.today()
    meta.to_csv(os.path.join(output_dir, f"meta_{today}.csv"))

    raw = df_raw[["BIS_1", "BIS_2"]].to_numpy(dtype=float)

    if use_notch:
        raw = notch_filter(raw, fs=sampling_freq, notch_hz=notch_hz)

    raw_bp = bandpass_filter(raw, bandpass[0], bandpass[1], sampling_freq)

    if use_asr:
        try:
            raw_clean = apply_asr_2ch(raw_bp, fs=sampling_freq, cutoff=20)
        except Exception as e:
            print(f"[WARN] ASR failed or unavailable: {e}. Continue without ASR.")
            raw_clean = raw_bp
    else:
        raw_clean = raw_bp

    df_clean = pd.DataFrame(raw_clean, columns=["BIS_1", "BIS_2"])
    df_clean["timestamp"] = df_raw["timestamp"].iloc[:len(df_clean)].values
    df_clean = (
        df_clean.dropna(subset=["timestamp", "BIS_1", "BIS_2"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    f1, P1, t1, bad1 = compute_dsa(
        df_clean["BIS_1"].to_numpy(dtype=float),
        fs=sampling_freq,
        timestamps=df_clean["timestamp"],
        window_sec=window_sec,
        step_sec=step_sec,
        method=psd_method,
        fmin=psd_fmin,
        fmax=psd_fmax,
        bandwidth=multitaper_bandwidth,
        adaptive=multitaper_adaptive,
        artifact_amplitude_uv=artifact_amplitude_uv,
        flat_sd_uv=flat_sd_uv,
    )

    f2, P2, t2, bad2 = compute_dsa(
        df_clean["BIS_2"].to_numpy(dtype=float),
        fs=sampling_freq,
        timestamps=df_clean["timestamp"],
        window_sec=window_sec,
        step_sec=step_sec,
        method=psd_method,
        fmin=psd_fmin,
        fmax=psd_fmax,
        bandwidth=multitaper_bandwidth,
        adaptive=multitaper_adaptive,
        artifact_amplitude_uv=artifact_amplitude_uv,
        flat_sd_uv=flat_sd_uv,
    )

    # Align frequency grids if needed.
    if len(f1) != len(f2) or np.any(np.abs(f1 - f2) > 1e-9):
        P2_use = np.vstack([np.interp(f1, f2, row) for row in P2])
        f_map = f1
    else:
        P2_use = P2
        f_map = f1

    prefix = psd_method

    # Save raw linear PSD and dB PSD. These are not smoothed.
    pd.DataFrame({
        "timestamp": t1,
        "artifact_rejected": bad1,
        **{f"{x:.2f}Hz": P1[:, j] for j, x in enumerate(f_map)},
    }).to_csv(
        os.path.join(output_dir, f"{prefix}_psd_linear_ch1_{today}.csv"),
        index=False,
    )

    pd.DataFrame({
        "timestamp": t2,
        "artifact_rejected": bad2,
        **{f"{x:.2f}Hz": P2_use[:, j] for j, x in enumerate(f_map)},
    }).to_csv(
        os.path.join(output_dir, f"{prefix}_psd_linear_ch2_{today}.csv"),
        index=False,
    )

    pd.DataFrame({
        "timestamp": t1,
        "artifact_rejected": bad1,
        **{f"{x:.2f}Hz": psd_to_db(P1)[:, j] for j, x in enumerate(f_map)},
    }).to_csv(
        os.path.join(output_dir, f"{prefix}_psd_db_ch1_{today}.csv"),
        index=False,
    )

    pd.DataFrame({
        "timestamp": t2,
        "artifact_rejected": bad2,
        **{f"{x:.2f}Hz": psd_to_db(P2_use)[:, j] for j, x in enumerate(f_map)},
    }).to_csv(
        os.path.join(output_dir, f"{prefix}_psd_db_ch2_{today}.csv"),
        index=False,
    )

    # Save raw relative bandpower for downstream analysis.
    bands1_raw = compute_relative_bandpower(P1, f_map)
    bands2_raw = compute_relative_bandpower(P2_use, f_map)

    pd.DataFrame({
        "timestamp": t1,
        "artifact_rejected": bad1,
        **{f"rel_{k}": v for k, v in bands1_raw.items()},
    }).to_csv(
        os.path.join(output_dir, f"{prefix}_relative_bandpower_raw_ch1_{today}.csv"),
        index=False,
    )

    pd.DataFrame({
        "timestamp": t2,
        "artifact_rejected": bad2,
        **{f"rel_{k}": v for k, v in bands2_raw.items()},
    }).to_csv(
        os.path.join(output_dir, f"{prefix}_relative_bandpower_raw_ch2_{today}.csv"),
        index=False,
    )

    if show_normalized_csv:
        N1 = normalize_spectrum(P1, f_map, mode=norm_mode)
        N2 = normalize_spectrum(P2_use, f_map, mode=norm_mode)

        pd.DataFrame({
            "timestamp": t1,
            "artifact_rejected": bad1,
            **{f"{x:.2f}Hz": N1[:, j] for j, x in enumerate(f_map)},
        }).to_csv(
            os.path.join(output_dir, f"{prefix}_normalized_{norm_mode}_ch1_{today}.csv"),
            index=False,
        )

        pd.DataFrame({
            "timestamp": t2,
            "artifact_rejected": bad2,
            **{f"{x:.2f}Hz": N2[:, j] for j, x in enumerate(f_map)},
        }).to_csv(
            os.path.join(output_dir, f"{prefix}_normalized_{norm_mode}_ch2_{today}.csv"),
            index=False,
        )

    if show_summary:
        plot_psd_summary(
            f_map,
            P1,
            t1,
            "ch1 (BIS_1)",
            psd_method=psd_method,
            output_path=os.path.join(output_dir, f"{prefix}_psd_summary_ch1_{today}.jpg"),
            smooth_bandpower=smooth_bandpower,
            smooth_sigma=smooth_sigma,
        )

        plot_psd_summary(
            f_map,
            P2_use,
            t2,
            "ch2 (BIS_2)",
            psd_method=psd_method,
            output_path=os.path.join(output_dir, f"{prefix}_psd_summary_ch2_{today}.jpg"),
            smooth_bandpower=smooth_bandpower,
            smooth_sigma=smooth_sigma,
        )

    if show_single_channel_aligned:
        P1_db = psd_to_db(P1)
        P2_db = psd_to_db(P2_use)
        vmin, vmax = robust_limits([P1_db, P2_db], low=5, high=95)

        plot_dsa_with_relative_bandpower(
            f_hz=f_map,
            P_lin=P1,
            times=t1,
            channel_name="ch1 (BIS_1)",
            psd_method=psd_method,
            output_path=os.path.join(output_dir, f"{prefix}_dsa_relpower_ch1_{today}.jpg"),
            cmap=cmap,
            smooth_bandpower=smooth_bandpower,
            smooth_sigma=smooth_sigma,
            vmin=vmin,
            vmax=vmax,
        )

        plot_dsa_with_relative_bandpower(
            f_hz=f_map,
            P_lin=P2_use,
            times=t2,
            channel_name="ch2 (BIS_2)",
            psd_method=psd_method,
            output_path=os.path.join(output_dir, f"{prefix}_dsa_relpower_ch2_{today}.jpg"),
            cmap=cmap,
            smooth_bandpower=smooth_bandpower,
            smooth_sigma=smooth_sigma,
            vmin=vmin,
            vmax=vmax,
        )

    if show_aligned_two_channel:
        plot_dsa_relpower_two_channels_aligned(
            f_hz=f_map,
            P1_lin=P1,
            t1=t1,
            P2_lin=P2_use,
            t2=t2,
            psd_method=psd_method,
            output_path=os.path.join(output_dir, f"{prefix}_dsa_relpower_aligned_two_channels_{today}.jpg"),
            cmap=cmap,
            smooth_bandpower=smooth_bandpower,
            smooth_sigma=smooth_sigma,
        )

    if show_hemo:
        plot_hemo(
            meta,
            output_path=os.path.join(output_dir, f"hemo_{today}.jpg"),
        )

    if show_eeg_metrics:
        plot_eeg_metrics(
            meta,
            output_path=os.path.join(output_dir, f"eeg_metrics_{today}.jpg"),
        )

    df_clean.to_csv(os.path.join(output_dir, f"eeg_cleaned_{today}.csv"), index=False)

    print(f"PSD method: {psd_method}")
    print(f"Rejected windows: ch1={bad1.mean() * 100:.1f}%, ch2={bad2.mean() * 100:.1f}%")
    print("Saved outputs to:", output_dir)

    analysis = {
        "frequency_hz": f_map,
        "psd_ch1": P1,
        "psd_ch2": P2_use,
        "times_ch1": t1,
        "times_ch2": t2,
        "artifact_ch1": bad1,
        "artifact_ch2": bad2,
        "relative_bandpower_ch1": bands1_raw,
        "relative_bandpower_ch2": bands2_raw,
    }
    if return_analysis:
        return df_clean, meta, output_dir, analysis
    return df_clean, meta, output_dir

try:
    import my_mod as _my_mod
    MY_MOD_AVAILABLE = True
except Exception:
    MY_MOD_AVAILABLE = False


# ============================================================
# OR data loading
# ============================================================

def _load_or_data_for_patient(or_data_root, patient_id: str):
    """
    Search or_data_root (a single directory, or a list of directories) for
    OR_* subdirs matching patient_id.
    Returns (or_case_dir, basic_df, attr_df, drug_event_merged_df, vital_df) or None.
    Drug events are merged with 使用薬剤情報 to add DRUG_NAME.
    """
    roots = [or_data_root] if isinstance(or_data_root, str) else list(or_data_root)
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "OR_*"))):
            basic_path = os.path.join(d, "基本情報.csv")
            if not os.path.exists(basic_path):
                continue
            try:
                basic = pd.read_csv(basic_path, encoding="cp932")
                if str(basic["PATIENT_ID"].iloc[0]).lstrip("0") != str(patient_id).lstrip("0"):
                    continue

                attr       = pd.read_csv(os.path.join(d, "患者属性情報.csv"),    encoding="cp932")
                drug_info  = pd.read_csv(os.path.join(d, "使用薬剤情報.csv"),    encoding="cp932")
                drug_event = pd.read_csv(os.path.join(d, "薬剤投与イベント情報.csv"), encoding="cp932")
                vital      = pd.read_csv(os.path.join(d, "バイタルデータ.csv"),   encoding="cp932")

                drug_event = drug_event[["CASE_NO", "DOSAGE_NO", "DATETIME", "CONTENT", "FLOW", "DETAIL"]]
                drug_event = drug_event.merge(
                    drug_info[["DOSAGE_NO", "DRUG_NAME"]], on="DOSAGE_NO", how="left"
                )
                return d, basic, attr, drug_event, vital
            except Exception:
                continue
    return None


def _load_extubation_time(or_dir: str, case_no) -> pd.Timestamp:
    """
    Read STARTED_AT of the 抜管 (extubation) row in 手術イベント記録.csv for case_no.
    Returns pd.NaT if the file or event is missing.
    """
    ev_path = os.path.join(or_dir, "手術イベント記録.csv")
    if not os.path.exists(ev_path):
        return pd.NaT
    try:
        ev = pd.read_csv(ev_path, encoding="cp932")
        ev = ev[ev["CASE_NO"] == case_no]
        m = ev[(ev["NAME"] == "手術イベント") & (ev["VALUE"] == "抜管")]
        if m.empty:
            return pd.NaT
        return pd.to_datetime(str(int(m["STARTED_AT"].iloc[0])), format="%Y%m%d%H%M", errors="coerce")
    except Exception:
        return pd.NaT


def _load_or_data_for_case(or_data_root, case_no):
    """
    Search or_data_root (a single directory, or a list of directories) for the
    OR_* dir matching case_no exactly (via CASE_NO in 基本情報.csv).
    Preferred over _load_or_data_for_patient when case_no is known: a patient
    can have multiple visits/case numbers, so matching on PATIENT_ID alone can
    silently pick the wrong visit's OR data. Tries the zero-padded directory
    name directly first (fast path), falling back to a full scan.
    Returns (or_case_dir, basic_df, attr_df, drug_event_merged_df, vital_df) or None.
    """
    roots = [or_data_root] if isinstance(or_data_root, str) else list(or_data_root)
    case_no_str = str(int(case_no))
    for root in roots:
        candidate = os.path.join(root, f"OR_{case_no_str.zfill(20)}")
        dirs_to_check = [candidate] if os.path.isdir(candidate) else sorted(glob.glob(os.path.join(root, "OR_*")))
        for d in dirs_to_check:
            basic_path = os.path.join(d, "基本情報.csv")
            if not os.path.exists(basic_path):
                continue
            try:
                basic = pd.read_csv(basic_path, encoding="cp932")
                if str(int(basic["CASE_NO"].iloc[0])) != case_no_str:
                    continue

                attr       = pd.read_csv(os.path.join(d, "患者属性情報.csv"),    encoding="cp932")
                drug_info  = pd.read_csv(os.path.join(d, "使用薬剤情報.csv"),    encoding="cp932")
                drug_event = pd.read_csv(os.path.join(d, "薬剤投与イベント情報.csv"), encoding="cp932")
                vital      = pd.read_csv(os.path.join(d, "バイタルデータ.csv"),   encoding="cp932")

                drug_event = drug_event[["CASE_NO", "DOSAGE_NO", "DATETIME", "CONTENT", "FLOW", "DETAIL"]]
                drug_event = drug_event.merge(
                    drug_info[["DOSAGE_NO", "DRUG_NAME"]], on="DOSAGE_NO", how="left"
                )
                return d, basic, attr, drug_event, vital
            except Exception:
                continue
    return None


def _build_main_row(basic_df: pd.DataFrame, attr_df: pd.DataFrame, or_dir: str | None = None) -> pd.Series:
    """Build a pd.Series compatible with simulate_case_wide_outputs."""
    attr_num = dict(zip(attr_df["ITEM_CODE"], attr_df["NUMERICAL_VALUE"]))
    attr_val = dict(zip(attr_df["ITEM_CODE"], attr_df["VALUE"]))

    case_no = int(basic_df["CASE_NO"].iloc[0])
    entrance_raw  = str(int(basic_df["ENTRANCE_AT"].iloc[0]))
    entrance_time = pd.to_datetime(entrance_raw, format="%Y%m%d%H%M", errors="coerce")

    extubation_time = _load_extubation_time(or_dir, case_no) if or_dir else pd.NaT

    return pd.Series({
        "caseNo":          case_no,
        "ID":              str(basic_df["PATIENT_ID"].iloc[0]),
        "入室日時":         entrance_time,
        "EXTUBATION_TIME": extubation_time,
        "age":             float(attr_num.get("AGE",  np.nan)),
        "sex":             str(attr_val.get("SEX",    "M")),
        "Ht_CIS":          float(attr_num.get("STAT", np.nan)),
        "Wt_CIS":          float(attr_num.get("WEIT", np.nan)),
        "ASAPS":           np.nan,
    })


def _load_endtidal_agent_timeseries(
    vital_df: pd.DataFrame,
    *,
    vital_name: str,
    output_column: str,
) -> pd.DataFrame:
    """Load an end-tidal volatile-agent series from the OR vital table."""
    out = vital_df[vital_df["NAME"] == vital_name].copy()
    out["time"] = pd.to_datetime(
        out["STARTED_AT"].astype(str).str.zfill(12),
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    out["NUMERICAL_VALUE"] = pd.to_numeric(out["NUMERICAL_VALUE"], errors="coerce")
    out = out.dropna(subset=["time", "NUMERICAL_VALUE"]).sort_values("time")
    return out[["time", "NUMERICAL_VALUE"]].rename(
        columns={"NUMERICAL_VALUE": output_column}
    )


def _load_etsev_timeseries(vital_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame [time, etsev (%)] for *exp.SEV."""
    return _load_endtidal_agent_timeseries(
        vital_df, vital_name="*exp.SEV", output_column="etsev"
    )


def _load_etdes_timeseries(vital_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame [time, etdes (%)] for *exp.DES."""
    return _load_endtidal_agent_timeseries(
        vital_df, vital_name="*exp.DES", output_column="etdes"
    )


def _simulate_ce(main_row: pd.Series, drug_event: pd.DataFrame) -> pd.DataFrame | None:
    """
    Simulate remifentanil, fentanyl, remimazolam, and other supported Ce
    profiles using the PK functions defined earlier in this kacr module.
    """
    try:
        case_no = int(main_row["caseNo"])
        de_case = drug_event[drug_event["CASE_NO"] == case_no].copy()
        de_case = _ensure_datetime_col(de_case, col="DATETIME")
        de_case = de_case.dropna(subset=["DATETIME"])
        return simulate_case_wide_outputs(
            main_row,
            de_case,
            case_id_col="caseNo",
            start_col="入室日時",
            end_col="EXTUBATION_TIME",
            dt_min=1.0,
            extra_after_end_min=30.0,
        )
    except Exception as e:
        print(f"[WARN] Ce simulation failed: {e}")
        return None



def plot_effectsite_and_volatile_agents(
    ce_wide: pd.DataFrame | None,
    etsev_df: pd.DataFrame | None = None,
    etdes_df: pd.DataFrame | None = None,
    *,
    output_path: str | None = None,
    title: str | None = None,
):
    """Plot drug effect-site concentrations and volatile agents without EEG.

    Panels
    ------
    1. Remimazolam and propofol Ce (mg/L, numerically equal to µg/mL)
    2. Remifentanil and fentanyl Ce (ng/mL)
    3. End-tidal sevoflurane and desflurane (%)
    """
    has_ce = ce_wide is not None and not ce_wide.empty
    has_sev = etsev_df is not None and not etsev_df.empty
    has_des = etdes_df is not None and not etdes_df.empty
    if not (has_ce or has_sev or has_des):
        raise ValueError("No Ce, etSEV, or etDES data are available to plot.")

    panels = []
    if has_ce and any(c in ce_wide.columns for c in ["rmz_Ce_mg_per_L", "prop_Ce_mg_per_L"]):
        panels.append("hypnotics")
    if has_ce and any(c in ce_wide.columns for c in ["remi_Ce_ng_per_mL", "fent_Ce_ng_per_mL"]):
        panels.append("opioids")
    if has_sev or has_des:
        panels.append("volatile")

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(14, 3.2 * len(panels)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    ce_t = pd.to_datetime(ce_wide["DATETIME"], errors="coerce") if has_ce else None

    for ax, panel in zip(axes, panels):
        if panel == "hypnotics":
            if "rmz_Ce_mg_per_L" in ce_wide.columns:
                ax.plot(ce_t, ce_wide["rmz_Ce_mg_per_L"], linewidth=1.7, label="Remimazolam Ce")
            if "prop_Ce_mg_per_L" in ce_wide.columns:
                prop = pd.to_numeric(ce_wide["prop_Ce_mg_per_L"], errors="coerce")
                if prop.notna().any() and np.nanmax(prop.to_numpy()) > 0:
                    ax.plot(ce_t, prop, linewidth=1.7, label="Propofol Ce")
            ax.set_ylabel("Ce (µg/mL)")
            ax.set_title("Hypnotic effect-site concentrations")

        elif panel == "opioids":
            if "remi_Ce_ng_per_mL" in ce_wide.columns:
                ax.plot(ce_t, ce_wide["remi_Ce_ng_per_mL"], linewidth=1.7, label="Remifentanil Ce")
            if "fent_Ce_ng_per_mL" in ce_wide.columns:
                fent = pd.to_numeric(ce_wide["fent_Ce_ng_per_mL"], errors="coerce")
                if fent.notna().any() and np.nanmax(fent.to_numpy()) > 0:
                    ax.plot(ce_t, fent, linewidth=1.7, label="Fentanyl Ce")
            ax.set_ylabel("Ce (ng/mL)")
            ax.set_title("Opioid effect-site concentrations")

        elif panel == "volatile":
            if has_sev:
                ax.plot(etsev_df["time"], etsev_df["etsev"], linewidth=1.7, label="etSev")
            if has_des:
                ax.plot(etdes_df["time"], etdes_df["etdes"], linewidth=1.7, label="etDes")
            ax.set_ylabel("End-tidal (%)")
            ax.set_title("Volatile anesthetic concentrations")

        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    axes[-1].set_xlabel("Time")
    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_or_effectsite_pipeline(
    or_data_root,
    *,
    case_no=None,
    patient_id=None,
    output_path: str | None = None,
    extra_after_end_min: float = 30.0,
):
    """Generate Ce, etSev, and etDes graphs directly from OR data, without EEG.

    Matching uses CASE_NO when supplied; otherwise it searches by PATIENT_ID.
    Returns a dictionary containing the matched OR directory and plotted data.
    """
    if case_no is None and patient_id is None:
        raise ValueError("Specify case_no or patient_id.")

    result = (
        _load_or_data_for_case(or_data_root, case_no)
        if case_no is not None
        else _load_or_data_for_patient(or_data_root, str(patient_id))
    )
    if result is None:
        raise FileNotFoundError(
            f"No OR data found for case_no={case_no}, patient_id={patient_id}."
        )

    or_dir, basic_df, attr_df, drug_event, vital_df = result
    main_row = _build_main_row(basic_df, attr_df, or_dir=or_dir)
    case_no_matched = int(main_row["caseNo"])
    de_case = drug_event[drug_event["CASE_NO"] == case_no_matched].copy()
    de_case = _ensure_datetime_col(de_case, col="DATETIME").dropna(subset=["DATETIME"])
    ce_wide = simulate_case_wide_outputs(
        main_row, de_case, case_id_col="caseNo", start_col="入室日時",
        end_col="EXTUBATION_TIME", dt_min=1.0,
        extra_after_end_min=extra_after_end_min,
    )
    etsev_df = _load_etsev_timeseries(vital_df)
    etdes_df = _load_etdes_timeseries(vital_df)

    if output_path is None:
        output_path = os.path.join(
            or_dir, f"effectsite_etvolatile_{case_no_matched}_{date.today()}.jpg"
        )
    plot_effectsite_and_volatile_agents(
        ce_wide, etsev_df, etdes_df, output_path=output_path,
        title=f"CASE_NO {case_no_matched}",
    )
    return {
        "or_dir": or_dir,
        "case_no": case_no_matched,
        "ce_wide": ce_wide,
        "etsev": etsev_df,
        "etdes": etdes_df,
        "output_path": output_path,
    }

# ============================================================
# Combined EEG + Ce figure
# ============================================================

def plot_dsa_relpower_two_channels_with_ce(
    f_hz,
    P1_lin,
    t1,
    P2_lin,
    t2,
    ce_wide=None,
    etsev_df=None,
    meta=None,
    include_hemo=True,
    include_eeg_metrics=True,
    psd_method="multitaper",
    output_path=None,
    cmap="jet",
    smooth_bandpower=True,
    smooth_sigma=8,
):
    """
    Combined figure sharing the time axis:
      1. ch1 DSA
      2. ch1 relative band power
      3. ch2 DSA
      4. ch2 relative band power
      5. Hemodynamics (HR, sART, sNIBP)        — if meta provided
      6. EEG metrics (BIS, SEF95, SQI, EMG)    — if meta provided
      7. Opioids / Remimazolam Ce               — if ce_wide provided
      8. End-tidal Sevoflurane                  — if etsev_df provided
    """
    has_ce   = ce_wide  is not None and not ce_wide.empty
    has_sevo = etsev_df is not None and not etsev_df.empty

    # Prepare meta DataFrame (timestamp as column, numeric values)
    _meta = None
    if meta is not None:
        _meta = meta.copy()
        if "timestamp" not in _meta.columns:
            _meta = _meta.reset_index()
        _meta["timestamp"] = pd.to_datetime(_meta["timestamp"], errors="coerce")
        for c in _meta.columns.drop("timestamp", errors="ignore"):
            _meta[c] = pd.to_numeric(_meta[c], errors="coerce")
        _meta = _meta.dropna(subset=["timestamp"]).sort_values("timestamp")
        if _meta.empty:
            _meta = None

    has_hemo = _meta is not None and include_hemo
    has_eeg_meta = _meta is not None and include_eeg_metrics

    P1_db = psd_to_db(P1_lin)
    P2_db = psd_to_db(P2_lin)
    vmin, vmax = robust_limits([P1_db, P2_db], low=5, high=95)

    bands1 = smooth_bandpower_for_plot(compute_relative_bandpower(P1_lin, f_hz),
                                       smooth=smooth_bandpower, sigma=smooth_sigma)
    bands2 = smooth_bandpower_for_plot(compute_relative_bandpower(P2_lin, f_hz),
                                       smooth=smooth_bandpower, sigma=smooth_sigma)

    eeg_heights  = [2.5, 1.2, 2.5, 1.2]
    extra_heights = (
        ([1.2] if has_hemo     else []) +
        ([1.2] if has_eeg_meta else []) +
        ([1.5] if has_ce       else []) +
        ([0.9] if has_sevo     else [])
    )
    height_ratios = eeg_heights + extra_heights
    n_rows = len(height_ratios)
    total_height = sum(h * 2.8 for h in height_ratios)

    fig = plt.figure(figsize=(15, total_height))
    gs  = GridSpec(
        n_rows, 2,
        width_ratios=[30, 0.8],
        height_ratios=height_ratios,
        hspace=0.10,
        wspace=0.05,
        figure=fig,
    )

    ax_dsa1 = fig.add_subplot(gs[0, 0])
    ax_rel1 = fig.add_subplot(gs[1, 0], sharex=ax_dsa1)
    ax_dsa2 = fig.add_subplot(gs[2, 0], sharex=ax_dsa1)
    ax_rel2 = fig.add_subplot(gs[3, 0], sharex=ax_dsa1)
    cax1    = fig.add_subplot(gs[0, 1])
    cax2    = fig.add_subplot(gs[2, 1])
    extra_axes = [fig.add_subplot(gs[4 + i, 0], sharex=ax_dsa1)
                  for i in range(len(extra_heights))]

    ext1 = _time_extent(t1, f_hz)
    ext2 = _time_extent(t2, f_hz)

    im1 = ax_dsa1.imshow(P1_db.T, origin="lower", aspect="auto", extent=ext1,
                         cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_dsa1.set_title(f"{psd_method.capitalize()} DSA – ch1 (BIS_1)")
    ax_dsa1.set_ylabel("Freq (Hz)")
    ax_dsa1.set_ylim([float(f_hz[0]), float(f_hz[-1])])
    fig.colorbar(im1, cax=cax1).set_label("Power (dB)")

    im2 = ax_dsa2.imshow(P2_db.T, origin="lower", aspect="auto", extent=ext2,
                         cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_dsa2.set_title(f"{psd_method.capitalize()} DSA – ch2 (BIS_2)")
    ax_dsa2.set_ylabel("Freq (Hz)")
    ax_dsa2.set_ylim([float(f_hz[0]), float(f_hz[-1])])
    fig.colorbar(im2, cax=cax2).set_label("Power (dB)")

    def _plot_bands(ax, times, bands):
        for name in ["delta", "theta", "alpha", "beta", "gamma"]:
            if name in bands:
                lw = 1.8 if name == "alpha" else 1.3
                ax.plot(times, bands[name], linewidth=lw, label=name)
        ax.set_ylim([0, 1])
        ax.set_ylabel("Rel.\npower")
        ax.legend(loc="upper right", ncol=5, fontsize=8)

    _plot_bands(ax_rel1, t1, bands1)
    _plot_bands(ax_rel2, t2, bands2)

    ax_idx = 0

    if has_hemo:
        ax_hemo = extra_axes[ax_idx]; ax_idx += 1
        for col, label in {"HR bpm": "HR", "ART(S) mmHg": "sART"}.items():
            if col in _meta.columns:
                ax_hemo.plot(_meta["timestamp"], _meta[col],
                             linewidth=1.2, label=label)
        if "NIBP(収縮期血圧) mmHg" in _meta.columns:
            _nibp = _meta.dropna(subset=["NIBP(収縮期血圧) mmHg"])
            ax_hemo.scatter(_nibp["timestamp"], _nibp["NIBP(収縮期血圧) mmHg"],
                            s=18, color="orange", label="sNIBP", zorder=3)
        ax_hemo.set_ylabel("Hemodynamics")
        ax_hemo.legend(loc="upper right", ncol=3, fontsize=8)
        ax_hemo.grid(True, alpha=0.25)

    if has_eeg_meta:
        ax_eeg = extra_axes[ax_idx]; ax_idx += 1
        for col, label in {"BIS": "BIS", "SEF95(bis) Hz": "SEF95",
                           "SQI(bis) %": "SQI", "EMG(bis) dB": "EMG"}.items():
            if col in _meta.columns:
                ax_eeg.plot(_meta["timestamp"], _meta[col],
                            linewidth=1.2, label=label)
        ax_eeg.set_ylabel("EEG metrics")
        ax_eeg.legend(loc="upper right", ncol=4, fontsize=8)
        ax_eeg.grid(True, alpha=0.25)

    if has_ce:
        ax_ce = extra_axes[ax_idx]; ax_idx += 1
        ce_t  = pd.to_datetime(ce_wide["DATETIME"])

        if "remi_Ce_ng_per_mL" in ce_wide.columns:
            ax_ce.plot(ce_t, ce_wide["remi_Ce_ng_per_mL"],
                       color="#0072B2", linewidth=1.5, label="Remifentanil Ce (ng/mL)")
        if "fent_Ce_ng_per_mL" in ce_wide.columns:
            fent = ce_wide["fent_Ce_ng_per_mL"].replace(0, np.nan)
            if fent.notna().any():
                ax_ce.plot(ce_t, fent,
                           color="#56B4E9", linewidth=1.5, label="Fentanyl Ce (ng/mL)")
        ax_ce.set_ylabel("Ce (ng/mL)", color="#0072B2")
        ax_ce.tick_params(axis="y", labelcolor="#0072B2")
        ax_ce.set_ylim(bottom=0)
        ax_ce.set_title("Effect-site concentrations")

        if "rmz_Ce_mg_per_L" in ce_wide.columns:
            ax_rmz = ax_ce.twinx()
            ax_rmz.plot(ce_t, ce_wide["rmz_Ce_mg_per_L"],
                        color="#E69F00", linewidth=1.5, label="Remimazolam Ce (µg/mL)")
            ax_rmz.set_ylabel("Ce (µg/mL)", color="#E69F00")
            ax_rmz.tick_params(axis="y", labelcolor="#E69F00")
            ax_rmz.set_ylim(bottom=0)
            lines1, labels1 = ax_ce.get_legend_handles_labels()
            lines2, labels2 = ax_rmz.get_legend_handles_labels()
            ax_ce.legend(lines1 + lines2, labels1 + labels2,
                         loc="upper right", fontsize=8, ncol=3)
        else:
            ax_ce.legend(loc="upper right", fontsize=8)

    if has_sevo:
        ax_sev = extra_axes[ax_idx]
        ax_sev.plot(etsev_df["time"], etsev_df["etsev"],
                    color="#009E73", linewidth=1.5, label="etSEV (%)")
        ax_sev.set_ylabel("etSEV (%)", color="#009E73")
        ax_sev.tick_params(axis="y", labelcolor="#009E73")
        ax_sev.set_ylim(bottom=0)
        ax_sev.legend(loc="upper right", fontsize=8)

    all_axes = [ax_dsa1, ax_rel1, ax_dsa2, ax_rel2] + extra_axes
    use_datetime = len(t1) and (
        isinstance(t1[0], pd.Timestamp)
        or np.issubdtype(np.asarray(t1).dtype, np.datetime64)
    )
    for ax in all_axes:
        if use_datetime:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in all_axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    all_axes[-1].set_xlabel("Time")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ============================================================
# Integrated EEG + optional OR/Ce pipeline
# ============================================================
def run_eeg_pipeline(
    ID,
    input_root,
    output_root,
    sampling_freq=250,
    epoch_len_sec=12.0,
    step_sec=None,
    use_asr=True,
    use_notch=True,
    notch_hz=50.0,
    bandpass=(0.5, 45.0),
    psd_fmin=0.5,
    psd_fmax=45.0,
    bandwidth=4.0,
    adaptive=True,
    artifact_amplitude_uv=200.0,
    flat_sd_uv=0.5,
    map_norm_mode="rel_db",
    cmap="jet",
    or_data_root=None,
    or_patient_id=None,
    or_case_no=None,
    smooth_bandpower=True,
    smooth_sigma=8,
    psd_method="multitaper",
    show_summary=True,
    show_aligned_two_channel=False,
    show_single_channel_aligned=False,
    show_normalized_csv=True,
    show_hemo=True,
    show_eeg_metrics=True,
    show_combined_figure=True,
):
    """Run EEG processing once, then optionally add OR-derived Ce and etSEV.

    This is the preferred entry point. It replaces the former two-script
    workflow and avoids recomputing DSA/PSD for the combined figure.
    """
    psd_method = psd_method.lower()
    if step_sec is None:
        step_sec = epoch_len_sec / 6.0

    norm_alias = {"rel_db": "relative_db", "zscore": "zscore_db"}
    norm_mode = norm_alias.get(map_norm_mode, map_norm_mode)

    df_clean, meta, output_dir, eeg = run_dsa_pipeline(
        ID=ID,
        input_root=input_root,
        output_root=output_root,
        sampling_freq=sampling_freq,
        window_sec=epoch_len_sec,
        step_sec=step_sec,
        psd_method=psd_method,
        use_asr=use_asr,
        use_notch=use_notch,
        notch_hz=notch_hz,
        bandpass=bandpass,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        multitaper_bandwidth=bandwidth,
        multitaper_adaptive=adaptive,
        artifact_amplitude_uv=artifact_amplitude_uv,
        flat_sd_uv=flat_sd_uv,
        show_summary=show_summary,
        show_aligned_two_channel=show_aligned_two_channel,
        show_single_channel_aligned=show_single_channel_aligned,
        show_normalized_csv=show_normalized_csv,
        show_hemo=False,
        show_eeg_metrics=False,
        norm_mode=norm_mode,
        smooth_bandpower=smooth_bandpower,
        smooth_sigma=smooth_sigma,
        cmap=cmap,
        return_analysis=True,
    )

    ce_wide = None
    etsev_df = None
    or_dir = None

    if or_data_root is not None:
        case_no_guess = (
            or_case_no
            if or_case_no is not None
            else (ID if str(ID).isdigit() else None)
        )
        result = (
            _load_or_data_for_case(or_data_root, case_no_guess)
            if case_no_guess is not None
            else None
        )
        if result is not None:
            print(f"OR match by case number: {case_no_guess}")
        else:
            lookup_id = str(or_patient_id) if or_patient_id is not None else str(ID)
            result = _load_or_data_for_patient(or_data_root, lookup_id)

        if result is None:
            print(
                f"[WARN] No OR data found for case {case_no_guess} "
                f"/ patient {ID} under {or_data_root}"
            )
        else:
            or_dir, basic_df, attr_df, drug_event, vital_df = result
            main_row = _build_main_row(basic_df, attr_df, or_dir=or_dir)
            ce_wide = _simulate_ce(main_row, drug_event)
            etsev_df = _load_etsev_timeseries(vital_df)
            print(f"OR case: {or_dir}")
            if ce_wide is not None:
                print(f"Ce columns: {[c for c in ce_wide.columns if 'Ce' in c]}")
            print(f"etSEV rows: {len(etsev_df) if etsev_df is not None else 0}")

    if show_combined_figure:
        today = date.today()
        out_fig = os.path.join(
            output_dir,
            f"{psd_method}_dsa_with_or_data_{today}.jpg",
        )
        plot_dsa_relpower_two_channels_with_ce(
            f_hz=eeg["frequency_hz"],
            P1_lin=eeg["psd_ch1"],
            t1=eeg["times_ch1"],
            P2_lin=eeg["psd_ch2"],
            t2=eeg["times_ch2"],
            ce_wide=ce_wide,
            etsev_df=etsev_df,
            meta=meta if (show_hemo or show_eeg_metrics) else None,
            include_hemo=show_hemo,
            include_eeg_metrics=show_eeg_metrics,
            psd_method=psd_method,
            output_path=out_fig,
            cmap=cmap,
            smooth_bandpower=smooth_bandpower,
            smooth_sigma=smooth_sigma,
        )
    else:
        if show_hemo:
            plot_hemo(meta, os.path.join(output_dir, f"hemo_{date.today()}.jpg"))
        if show_eeg_metrics:
            plot_eeg_metrics(meta, os.path.join(output_dir, f"eeg_metrics_{date.today()}.jpg"))

    eeg["ce_wide"] = ce_wide
    eeg["etsev"] = etsev_df
    eeg["or_dir"] = or_dir
    return df_clean, meta, output_dir, eeg



