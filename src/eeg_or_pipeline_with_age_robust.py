# ============================================================
# EEG + OR data pipeline
#
# Wraps the core DSA pipeline (dsa_pipeline_welch_multitaper_aligned)
# with OR-record loading, PK/PD Ce simulation (my_mod), and a
# combined EEG + Effect-site concentration figure.
#
# Main entry point:
#   run_eeg_pipeline(ID, input_root, output_root, ..., or_data_root=...)
# ============================================================

import glob
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Core EEG processing lives in the sibling module
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dsa_pipeline_welch_multitaper_aligned import (
    run_dsa_pipeline,
    compute_dsa,
    compute_relative_bandpower,
    smooth_bandpower_for_plot,
    psd_to_db,
    robust_limits,
    _time_extent,
    plot_hemo,
    plot_eeg_metrics,
)

try:
    import my_mod as _my_mod
    MY_MOD_AVAILABLE = True
except Exception:
    MY_MOD_AVAILABLE = False


# ============================================================
# OR data loading
# ============================================================

def _load_or_data_for_patient(or_data_root: str, patient_id: str):
    """
    Search or_data_root for OR_* subdirs matching patient_id.
    Returns (or_case_dir, basic_df, attr_df, drug_event_merged_df, vital_df) or None.
    Drug events are merged with 使用薬剤情報 to add DRUG_NAME.
    """
    for d in sorted(glob.glob(os.path.join(or_data_root, "OR_*"))):
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


def _build_main_row(basic_df: pd.DataFrame, attr_df: pd.DataFrame) -> pd.Series:
    """Build a pd.Series compatible with simulate_case_wide_outputs."""
    attr_num = dict(zip(attr_df["ITEM_CODE"], attr_df["NUMERICAL_VALUE"]))
    attr_val = dict(zip(attr_df["ITEM_CODE"], attr_df["VALUE"]))

    entrance_raw  = str(int(basic_df["ENTRANCE_AT"].iloc[0]))
    entrance_time = pd.to_datetime(entrance_raw, format="%Y%m%d%H%M", errors="coerce")

    return pd.Series({
        "caseNo":          int(basic_df["CASE_NO"].iloc[0]),
        "ID":              str(basic_df["PATIENT_ID"].iloc[0]),
        "入室日時":         entrance_time,
        "EXTUBATION_TIME": pd.NaT,
        "age":             float(attr_num.get("AGE",  np.nan)),
        "sex":             str(attr_val.get("SEX",    "M")),
        "Ht_CIS":          float(attr_num.get("STAT", np.nan)),
        "Wt_CIS":          float(attr_num.get("WEIT", np.nan)),
        "ASAPS":           np.nan,
    })


def _load_etsev_timeseries(vital_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame [time (Timestamp), etsev (%)] for *exp.SEV."""
    sevo = vital_df[vital_df["NAME"] == "*exp.SEV"].copy()
    sevo["time"] = pd.to_datetime(
        sevo["STARTED_AT"].astype(str).str.zfill(12),
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    sevo = sevo.dropna(subset=["time", "NUMERICAL_VALUE"]).sort_values("time")
    return sevo[["time", "NUMERICAL_VALUE"]].rename(columns={"NUMERICAL_VALUE": "etsev"})


def _simulate_ce(main_row: pd.Series, drug_event: pd.DataFrame) -> pd.DataFrame | None:
    """
    Run 3-compartment PK simulation via my_mod for remifentanil (Eleveld 2017),
    fentanyl (Bae 2020), and remimazolam (Masui 2022).
    Returns a wide DataFrame with DATETIME and Ce columns, or None on failure.
    """
    if not MY_MOD_AVAILABLE:
        print("[WARN] my_mod not available — Ce simulation skipped.")
        return None
    try:
        case_no = int(main_row["caseNo"])
        de_case = drug_event[drug_event["CASE_NO"] == case_no].copy()
        de_case = _my_mod._ensure_datetime_col(de_case, col="DATETIME")
        de_case = de_case.dropna(subset=["DATETIME"])
        wide = _my_mod.simulate_case_wide_outputs(
            main_row,
            de_case,
            case_id_col="caseNo",
            start_col="入室日時",
            end_col="EXTUBATION_TIME",
            dt_min=1.0,
            extra_after_end_min=30.0,
        )
        return wide
    except Exception as e:
        print(f"[WARN] Ce simulation failed: {e}")
        return None


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
    psd_method="multitaper",
    output_path=None,
    cmap="jet",
    smooth_bandpower=True,
    smooth_sigma=8,
    case_info=None,
):
    """
    Combined figure sharing the EEG time axis.

    This version is deliberately conservative:
    - DSA image extents are calculated directly instead of relying on _time_extent.
    - All external OR/meta/Ce timestamps are aligned to the EEG reference date only
      when their calendar date does not overlap the EEG timestamps.
    - The x-axis is forced to the EEG DSA range so OR data cannot expand the axis
      over several days/months and make the EEG look blank.
    - The figure is saved before plt.show().
    """
    has_ce   = ce_wide  is not None and not ce_wide.empty
    has_sevo = etsev_df is not None and not etsev_df.empty
    _ce_x_arr = None  # will be set inside has_ce block for xlim extension

    # ---------- helper functions ----------
    def _as_datetime_series(x):
        return pd.to_datetime(pd.Series(x), errors="coerce")

    def _is_datetime_like(x):
        try:
            s = _as_datetime_series(x)
            return s.notna().any()
        except Exception:
            return False

    def _align_to_eeg_day_if_needed(times, eeg_times):
        """Keep time-of-day; align date only if the ranges do not overlap."""
        s = _as_datetime_series(times)
        ref = _as_datetime_series(eeg_times)
        if s.dropna().empty or ref.dropna().empty:
            return s

        ref_min, ref_max = ref.min(), ref.max()
        s_min, s_max = s.min(), s.max()
        overlaps = (s_min <= ref_max) and (s_max >= ref_min)
        if overlaps:
            return s

        ref_date = ref.dropna().iloc[0].date()
        return s.map(
            lambda x: pd.Timestamp.combine(ref_date, x.time()) if pd.notna(x) else pd.NaT
        )

    def _to_plot_x(times, eeg_times=None):
        """Return x values for plotting and whether they are matplotlib date numbers."""
        if _is_datetime_like(times):
            s = _as_datetime_series(times)
            if eeg_times is not None:
                s = _align_to_eeg_day_if_needed(s, eeg_times)
            return mdates.date2num(s.dt.to_pydatetime()), True
        arr = np.asarray(times, dtype=float)
        return arr, False

    def _image_extent_from_times(times, freqs):
        """Safe extent for imshow: [xmin, xmax, ymin, ymax]."""
        x, is_dt = _to_plot_x(times)
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("No valid DSA timestamps were available for plotting.")

        if x.size >= 2:
            diffs = np.diff(np.sort(np.unique(x)))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            dx = float(np.median(diffs)) if diffs.size else (1.0 / 1440.0 if is_dt else 1.0)
        else:
            dx = 1.0 / 1440.0 if is_dt else 1.0

        xmin = float(np.nanmin(x) - dx / 2)
        xmax = float(np.nanmax(x) + dx / 2)
        if xmin == xmax:
            xmin -= dx / 2
            xmax += dx / 2
        return [xmin, xmax, float(freqs[0]), float(freqs[-1])], is_dt

    # ---------- meta preparation ----------
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

    has_hemo     = _meta is not None
    has_eeg_meta = _meta is not None

    # ---------- DSA and bandpower ----------
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

    ext1, use_datetime = _image_extent_from_times(t1, f_hz)
    ext2, _ = _image_extent_from_times(t2, f_hz)
    t1_x, _ = _to_plot_x(t1)
    t2_x, _ = _to_plot_x(t2, t1)

    # ---------- figure layout ----------
    eeg_heights = [2.5, 1.2, 2.5, 1.2]
    extra_heights = (
        ([1.2] if has_hemo     else []) +
        ([1.2] if has_eeg_meta else []) +
        ([1.5] if has_ce       else []) +
        ([0.9] if has_sevo     else [])
    )
    height_ratios = eeg_heights + extra_heights
    n_rows = len(height_ratios)
    total_height = max(8, sum(h * 2.8 for h in height_ratios))

    fig = plt.figure(figsize=(15, total_height))
    gs = GridSpec(
        n_rows, 2,
        width_ratios=[30, 0.8],
        height_ratios=height_ratios,
        hspace=0.06,
        wspace=0.05,
        figure=fig,
    )

    ax_dsa1 = fig.add_subplot(gs[0, 0])
    ax_rel1 = fig.add_subplot(gs[1, 0], sharex=ax_dsa1)
    ax_dsa2 = fig.add_subplot(gs[2, 0], sharex=ax_dsa1)
    ax_rel2 = fig.add_subplot(gs[3, 0], sharex=ax_dsa1)
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[2, 1])
    extra_axes = [
        fig.add_subplot(gs[4 + i, 0], sharex=ax_dsa1)
        for i in range(len(extra_heights))
    ]

    title1 = f"{psd_method.capitalize()} DSA – ch1 (BIS_1)"
    if case_info is not None:
        title1 = f"{case_info}\n{title1}"
    ax_dsa1.set_title(title1, fontsize=10, pad=2)

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
    ax_dsa1.set_ylabel("Freq (Hz)")
    ax_dsa1.set_ylim([float(f_hz[0]), float(f_hz[-1])])
    fig.colorbar(im1, cax=cax1).set_label("Power (dB)")

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
    ax_dsa2.set_title(f"{psd_method.capitalize()} DSA – ch2 (BIS_2)", fontsize=10, pad=2)
    ax_dsa2.set_ylabel("Freq (Hz)")
    ax_dsa2.set_ylim([float(f_hz[0]), float(f_hz[-1])])
    fig.colorbar(im2, cax=cax2).set_label("Power (dB)")

    def _plot_bands(ax, x, bands):
        for name in ["delta", "theta", "alpha", "beta", "gamma"]:
            if name in bands:
                lw = 1.8 if name == "alpha" else 1.3
                ax.plot(x, bands[name], linewidth=lw, label=name)
        ax.set_ylim([0, 1])
        ax.set_ylabel("Rel.\npower")
        ax.legend(loc="upper right", ncol=5, fontsize=8)

    _plot_bands(ax_rel1, t1_x, bands1)
    _plot_bands(ax_rel2, t2_x, bands2)

    ax_idx = 0

    if has_hemo:
        ax_hemo = extra_axes[ax_idx]; ax_idx += 1
        meta_x, _ = _to_plot_x(_meta["timestamp"], t1)
        for col, label in {"HR bpm": "HR", "ART(S) mmHg": "sART"}.items():
            if col in _meta.columns:
                ax_hemo.plot(meta_x, _meta[col], linewidth=1.2, label=label)
        if "NIBP(収縮期血圧) mmHg" in _meta.columns:
            ok = _meta["NIBP(収縮期血圧) mmHg"].notna()
            ax_hemo.scatter(
                np.asarray(meta_x)[ok],
                _meta.loc[ok, "NIBP(収縮期血圧) mmHg"],
                s=18,
                color="orange",
                label="sNIBP",
                zorder=3,
            )
        ax_hemo.set_ylabel("Hemodynamics")
        ax_hemo.legend(loc="upper right", ncol=3, fontsize=8)
        ax_hemo.grid(True, alpha=0.25)

    if has_eeg_meta:
        ax_eeg = extra_axes[ax_idx]; ax_idx += 1
        meta_x, _ = _to_plot_x(_meta["timestamp"], t1)
        for col, label in {"BIS": "BIS", "SEF95(bis) Hz": "SEF95",
                           "SQI(bis) %": "SQI", "EMG(bis) dB": "EMG"}.items():
            if col in _meta.columns:
                ax_eeg.plot(meta_x, _meta[col], linewidth=1.2, label=label)
        ax_eeg.set_ylabel("EEG metrics")
        ax_eeg.legend(loc="upper right", ncol=4, fontsize=8)
        ax_eeg.grid(True, alpha=0.25)

    if has_ce:
        ax_ce = extra_axes[ax_idx]; ax_idx += 1
        ce_x, _ = _to_plot_x(ce_wide["DATETIME"], t1)
        _ce_x_arr = ce_x

        if "remi_Ce_ng_per_mL" in ce_wide.columns:
            ax_ce.plot(ce_x, ce_wide["remi_Ce_ng_per_mL"],
                       color="#0072B2", linewidth=1.5, label="Remifentanil Ce (ng/mL)")
        if "fent_Ce_ng_per_mL" in ce_wide.columns:
            fent = ce_wide["fent_Ce_ng_per_mL"].replace(0, np.nan)
            if fent.notna().any():
                ax_ce.plot(ce_x, fent,
                           color="#56B4E9", linewidth=1.5, label="Fentanyl Ce (ng/mL)")
        ax_ce.set_ylabel("Ce (ng/mL)", color="#0072B2")
        ax_ce.tick_params(axis="y", labelcolor="#0072B2")
        ax_ce.set_ylim(bottom=0)
        ax_ce.set_title("Effect-site concentrations", fontsize=10, pad=2)

        _right_cols = ["rmz_Ce_mg_per_L", "rmz_Ce_equiv_alloIBW_mg_per_L", "prop_Ce_mg_per_L"]
        has_right = any(c in ce_wide.columns for c in _right_cols)
        if has_right:
            ax_rmz = ax_ce.twinx()
            if "rmz_Ce_mg_per_L" in ce_wide.columns:
                ax_rmz.plot(ce_x, ce_wide["rmz_Ce_mg_per_L"],
                            color="#E69F00", linewidth=1.5, label="Remimazolam Ce (µg/mL)")
            if "rmz_Ce_equiv_alloIBW_mg_per_L" in ce_wide.columns:
                rmz_equiv = ce_wide["rmz_Ce_equiv_alloIBW_mg_per_L"].replace(0, np.nan)
                if rmz_equiv.notna().any():
                    ax_rmz.plot(ce_x, rmz_equiv, color="#D55E00", linewidth=1.5,
                                linestyle="--", label="RMZ equiv Ce IBW (µg/mL)")
            if "prop_Ce_mg_per_L" in ce_wide.columns:
                prop_ce = ce_wide["prop_Ce_mg_per_L"].replace(0, np.nan)
                if prop_ce.notna().any():
                    ax_rmz.plot(ce_x, prop_ce, color="#CC79A7", linewidth=1.5,
                                label="Propofol Ce (µg/mL)")
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
        sev_x, _ = _to_plot_x(etsev_df["time"], t1)
        ax_sev.plot(sev_x, etsev_df["etsev"],
                    color="#009E73", linewidth=1.5, label="etSEV (%)")
        ax_sev.set_ylabel("etSEV (%)", color="#009E73")
        ax_sev.tick_params(axis="y", labelcolor="#009E73")
        ax_sev.set_ylim(bottom=0)
        ax_sev.legend(loc="upper right", fontsize=8)

    # Extend left boundary to include Ce start (e.g. propofol induction before EEG).
    # The date alignment above already ensures Ce timestamps are on the EEG calendar day,
    # so this cannot stretch the axis across multiple days.
    x_lim_left = ext1[0]
    if _ce_x_arr is not None:
        _finite = _ce_x_arr[np.isfinite(_ce_x_arr)]
        if _finite.size > 0:
            x_lim_left = min(x_lim_left, float(_finite.min()))

    all_axes = [ax_dsa1, ax_rel1, ax_dsa2, ax_rel2] + extra_axes
    for ax in all_axes:
        ax.set_xlim(x_lim_left, ext1[1])
        if use_datetime:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in all_axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    all_axes[-1].set_xlabel("Time")

    fig.subplots_adjust(
        top=0.985,
        bottom=0.04,
        left=0.06,
        right=0.95,
        hspace=0.06,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white")
        print(f"Saved combined figure: {output_path}")

    plt.close(fig)


# ============================================================
# Main entry point
# ============================================================

def run_eeg_pipeline(
    ID,
    input_root,
    output_root,
    sampling_freq=250,
    epoch_len_sec=12,
    use_asr=True,
    bandpass=(0.5, 45),
    psd_fmin=0.5,
    psd_fmax=45,
    bandwidth=4.0,
    map_norm_mode="rel_db",
    cmap="jet",
    or_data_root=None,
    smooth_bandpower=True,
    smooth_sigma=8,
    psd_method="multitaper",
    show_hemo=True,
    show_eeg_metrics=True,
):
    """
    DSA pipeline (multitaper or Welch) with optional Effect-site Concentration overlay.

    Parameters
    ----------
    ID : str
        Patient / EEG folder ID (PATIENT_ID).
    input_root : str
        Parent directory of the ID folder containing 実波形*.csv / 数値*.csv.
    output_root : str
        Where to save figures and CSV outputs.
    epoch_len_sec : float
        Analysis window length in seconds.
    psd_method : str
        'multitaper' (default) or 'welch'.
    bandwidth : float
        Multitaper half-bandwidth in Hz (ignored for Welch).
    map_norm_mode : str
        Normalisation mode for relative-power CSV: 'rel_db' or 'zscore_db'.
    or_data_root : str or None
        Parent directory of OR_* subdirectories (e.g. '../data/EEG_2026May12').
        When provided the matching case is found via PATIENT_ID, Ce is simulated,
        and Ce + etSEV panels are added to the output figure.
    """
    psd_method = psd_method.lower()
    _norm_alias = {"rel_db": "relative_db", "zscore": "zscore_db"}
    norm_mode = _norm_alias.get(map_norm_mode, map_norm_mode)

    df_clean, meta, output_dir = run_dsa_pipeline(
        ID=ID,
        input_root=input_root,
        output_root=output_root,
        sampling_freq=sampling_freq,
        window_sec=epoch_len_sec,
        step_sec=epoch_len_sec / 6,
        psd_method=psd_method,
        use_asr=use_asr,
        use_notch=True,
        notch_hz=50.0,
        bandpass=bandpass,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        multitaper_bandwidth=bandwidth,
        multitaper_adaptive=True,
        show_summary=True,
        show_aligned_two_channel=False,
        show_single_channel_aligned=False,
        show_normalized_csv=True,
        show_hemo=False,
        show_eeg_metrics=False,
        norm_mode=norm_mode,
        smooth_bandpower=smooth_bandpower,
        smooth_sigma=smooth_sigma,
        cmap=cmap,
    )

    # Re-compute DSA for the combined figure
    f1, P1, t1, _ = compute_dsa(
        df_clean["BIS_1"].to_numpy(dtype=float),
        fs=sampling_freq,
        timestamps=df_clean["timestamp"],
        window_sec=epoch_len_sec,
        step_sec=epoch_len_sec / 6,
        method=psd_method,
        fmin=psd_fmin,
        fmax=psd_fmax,
        bandwidth=bandwidth,
    )
    f2, P2, t2, _ = compute_dsa(
        df_clean["BIS_2"].to_numpy(dtype=float),
        fs=sampling_freq,
        timestamps=df_clean["timestamp"],
        window_sec=epoch_len_sec,
        step_sec=epoch_len_sec / 6,
        method=psd_method,
        fmin=psd_fmin,
        fmax=psd_fmax,
        bandwidth=bandwidth,
    )
    if len(f1) != len(f2) or np.any(np.abs(f1 - f2) > 1e-9):
        P2 = np.vstack([np.interp(f1, f2, row) for row in P2])

    # ---- OR data and Ce simulation ----
    ce_wide  = None
    etsev_df = None
    case_info = None

    if or_data_root is not None:
        result = _load_or_data_for_patient(or_data_root, str(ID))
        if result is None:
            print(f"[WARN] No OR data found for patient {ID} under {or_data_root}")
        else:
            or_dir, basic_df, attr_df, drug_event, vital_df = result
            main_row = _build_main_row(basic_df, attr_df)

            age = main_row.get("age", np.nan)
            age_text = "NA" if pd.isna(age) else f"{age:.0f} years"
            sex = main_row.get("sex", "NA")
            case_no = main_row.get("caseNo", "NA")
            case_info = f"ID {ID} | Case {case_no} | Age {age_text} | Sex {sex}"

            ce_wide  = _simulate_ce(main_row, drug_event)
            etsev_df = _load_etsev_timeseries(vital_df)
            print(f"OR case: {or_dir}")
            if ce_wide is not None:
                print(f"Ce columns: {[c for c in ce_wide.columns if 'Ce' in c]}")
            print(f"etSEV rows: {len(etsev_df) if etsev_df is not None else 0}")

    # ---- Combined figure ----
    today   = date.today()
    out_fig = os.path.join(output_dir, f"{psd_method}_dsa_with_ce_{today}.jpg")

    plot_dsa_relpower_two_channels_with_ce(
        f_hz=f1, P1_lin=P1, t1=t1, P2_lin=P2, t2=t2,
        ce_wide=ce_wide, etsev_df=etsev_df,
        meta=meta if (show_hemo or show_eeg_metrics) else None,
        psd_method=psd_method, output_path=out_fig,
        cmap=cmap, smooth_bandpower=smooth_bandpower, smooth_sigma=smooth_sigma,
        case_info=case_info,
    )

    return df_clean, meta, output_dir


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    run_eeg_pipeline(
        ID="10299775",
        input_root="../data",
        output_root="../output",
        sampling_freq=250,
        epoch_len_sec=12,
        use_asr=True,
        bandpass=(0.5, 45),
        psd_fmin=0.5,
        psd_fmax=45,
        bandwidth=4.0,
        map_norm_mode="rel_db",
        cmap="jet",
        or_data_root="../data/EEG_2026May12",
        psd_method="multitaper",  # or "welch"
    )
