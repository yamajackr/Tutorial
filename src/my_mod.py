import os
import re
import numpy as np
import pandas as pd
import glob
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from datetime import datetime

# ============================================================
# 0) Helpers
# ============================================================

def find_concat(pattern, read_func, **kwargs):
    files = glob.glob(pattern)
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

# --- NEW: flumazenil concentration conversion ---
# User requirement: 0.5 mg = 5 mL  => 0.1 mg/mL
FLM_MG_PER_ML = 0.5 / 5.0  # 0.1 mg/mL

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

# ============================================================
# 1) Drug selection
# ============================================================

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

# ============================================================
# 2) Convert events -> bolus + infusion changes
# ============================================================

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


# ============================================================
# 3) Remimazolam model (Masui 2022)
# ============================================================

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

# ============================================================
# 4) Flumazenil parameters (fixed and allometric)
# ============================================================

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

# ============================================================
# 4) Opioid parameters (fixed and allometric)
# ============================================================

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
    ke0_per_min: float = 0.147  # example assumption (literature summary) :contentReference[oaicite:5]{index=5}


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

# ============================================================
# 5) Case simulation: RMZ + FLM fixed + FLM allometric
# ============================================================
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

# ============================================================
# 6) Batch runner -> CSV
# ============================================================

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



#######Table 1

# my_module.py

import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency


def median_iqr(x: pd.Series, digits: int = None) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return ""

    med = x.median()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)

    d = digits if digits is not None else (2 if abs(med) < 1 else 1)
    return f"{med:.{d}f} [{q1:.{d}f}, {q3:.{d}f}]"


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


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


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

def smd_cont(x, y):
    """
    Standardized mean difference for continuous variables.
    Uses pooled SD.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    nx, ny = len(x), len(y)

    if nx + ny - 2 <= 0:
        return np.nan

    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0 if mx == my else np.nan

    return (mx - my) / pooled_sd


def smd_binary(x, y):
    """
    Standardized mean difference for binary variables.
    Treats binary variable as 0/1 numeric.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    px = (x == 1).mean()
    py = (y == 1).mean()

    p = (px + py) / 2
    denom = np.sqrt(p * (1 - p))

    if denom == 0 or np.isnan(denom):
        return 0.0 if px == py else np.nan

    return (px - py) / denom


def format_smd(x, digits=3, abs_value=True):
    if pd.isna(x):
        return ""
    return f"{abs(x) if abs_value else x:.{digits}f}"

from statsmodels.stats.contingency_tables import Table2x2


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
):

    if groups is None:
        groups = list(df[group_col].dropna().unique())

    if len(groups) != 2:
        raise ValueError("Only 2 groups supported")

    g1, g2 = groups
    sub = df[df[group_col].isin(groups)].copy()
    rows = []

    # N
    rows.append({
        "Variable": "N",
        g1: str((sub[group_col] == g1).sum()),
        g2: str((sub[group_col] == g2).sum()),
        "OR (95% CI)": "",
        "p": "",
        "SMD": ""
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
            "SMD": format_smd(smd_cont(x1, x2), digits_smd)
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

        row["SMD"] = format_smd(
            smd_binary(
                sub[sub[group_col] == g1][var],
                sub[sub[group_col] == g2][var]
            ),
            digits_smd
        )

        rows.append(row)

    # ---------- Categorical ----------
    for var in categorical:
        tmp = sub[[group_col, var]].copy()

        # header row
        header = {"Variable": var}
        for g in groups:
            header[g] = ""

        # global p
        tmp_nonmiss = tmp.dropna(subset=[var])
        tab = pd.crosstab(tmp_nonmiss[group_col], tmp_nonmiss[var])
        tab = tab.reindex(index=groups, fill_value=0)

        header["p"] = format_p(p_cat(tab))
        header["OR (95% CI)"] = ""

        # SMD (max across levels)
        smd_list = []
        for level in tab.columns:
            x = (tmp[tmp[group_col] == g1][var] == level).astype(float)
            y = (tmp[tmp[group_col] == g2][var] == level).astype(float)
            smd_val = smd_binary(x, y)
            if not pd.isna(smd_val):
                smd_list.append(abs(smd_val))

        header["SMD"] = f"{max(smd_list):.{digits_smd}f}" if smd_list else ""
        rows.append(header)

        # levels
        for level in tab.columns:
            row = {"Variable": f"  {level}"}

            for g in groups:
                s = tmp[tmp[group_col] == g][var]
                row[g] = n_pct((s == level).sum(), s.notna().sum(), digits_pct)

            row["p"] = ""
            row["OR (95% CI)"] = ""

            x = (tmp[tmp[group_col] == g1][var] == level).astype(float)
            y = (tmp[tmp[group_col] == g2][var] == level).astype(float)
            row["SMD"] = format_smd(smd_binary(x, y), digits_smd)

            rows.append(row)

        # missing row
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


#####Logistic regression table
import statsmodels.api as sm

import numpy as np
import pandas as pd
import statsmodels.api as sm

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