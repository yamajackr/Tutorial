# ============================================================
# Integrated EEG + OR + DSA + PCA/UMAP pipeline
#
# This file merges the previous dsa_pipeline.py and eeg_or_pipeline.py
# and adds epoch-level PCA/UMAP visualisation.
#
# Main entry point:
#   run_eeg_pipeline(case_no, input_root, output_root, ...)
#
# Important folder rule:
#   - case_no is used for OR-data matching by CASE_NO.
#   - eeg_folder_name can be supplied when the EEG folder is named by patient ID
#     or another identifier. If omitted, eeg_folder_name = case_no.
# ============================================================

# ============================================================
# DSA pipeline for anesthesia EEG
# Choose PSD method:
#   psd_method="multitaper" or psd_method="welch"
#
# Features:
#   - Band-pass and optional notch filtering
#   - Optional ASR with safe fallback
#   - Sliding-window PSD
#   - Absolute DSA in dB
#   - Relative band power computed from raw PSD
#   - Smoothed relative band power for visualization only
#   - DSA and relative band power vertically aligned with shared time axis
# ============================================================

import os
import glob
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from scipy.signal import (
    butter,
    sosfiltfilt,
    iirnotch,
    filtfilt,
    welch,
)
from scipy.ndimage import gaussian_filter1d

try:
    from mne.time_frequency import psd_array_multitaper
    MNE_AVAILABLE = True
except Exception:
    psd_array_multitaper = None
    MNE_AVAILABLE = False

try:
    from asrpy import asr_calibrate, asr_process, clean_windows
    ASRPY_AVAILABLE = True
except Exception:
    ASRPY_AVAILABLE = False



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
        if psd_array_multitaper is None:
            print("[WARN] MNE is not available. Falling back to Welch PSD.")
            return compute_psd_windows(windows, fs, method="welch", fmin=fmin, fmax=fmax, bandwidth=bandwidth, adaptive=adaptive)
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
def load_eeg_and_meta(input_root, case_no, eeg_folder_name=None):
    folder_name = str(eeg_folder_name) if eeg_folder_name is not None else str(case_no)
    input_dir = os.path.join(input_root, folder_name)

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
    case_no,
    input_root,
    output_root,
    eeg_folder_name=None,
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
):
    psd_method = psd_method.lower()

    if psd_method not in ["multitaper", "welch"]:
        raise ValueError("psd_method must be 'multitaper' or 'welch'.")

    df_raw, meta, _ = load_eeg_and_meta(input_root, case_no, eeg_folder_name=eeg_folder_name)

    output_dir = os.path.join(output_root, str(case_no))
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

    return df_clean, meta, output_dir



# Optional PK/PD module for effect-site concentration simulation
try:
    import my_mod as _my_mod
    MY_MOD_AVAILABLE = True
except Exception:
    _my_mod = None
    MY_MOD_AVAILABLE = False

# ============================================================
# OR data loading
# ============================================================

def _load_or_data_for_case(or_data_root: str, case_no: str):
    """
    Search or_data_root for OR_* subdirs matching case_no.
    Returns (or_case_dir, basic_df, attr_df, drug_event_merged_df, vital_df) or None.
    Drug events are merged with 使用薬剤情報 to add DRUG_NAME.
    """
    for d in sorted(glob.glob(os.path.join(or_data_root, "OR_*"))):
        basic_path = os.path.join(d, "基本情報.csv")
        if not os.path.exists(basic_path):
            continue
        try:
            basic = pd.read_csv(basic_path, encoding="cp932")
            if str(basic["CASE_NO"].iloc[0]).lstrip("0") != str(case_no).lstrip("0"):
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
    case_no,
    input_root,
    output_root,
    eeg_folder_name=None,
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
    run_dim_reduction=False,
    dim_feature_set="all",
    dim_pca_components=10,
    dim_umap_neighbors=10,
    dim_umap_min_dist=0.3,
):
    """
    DSA pipeline (multitaper or Welch) with optional Effect-site Concentration overlay.

    Parameters
    ----------
    case_no : str
        Case number used as the EEG folder name.
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
        When provided the matching case is found via CASE_NO, Ce is simulated,
        and Ce + etSEV panels are added to the output figure.
    """
    psd_method = psd_method.lower()
    _norm_alias = {"rel_db": "relative_db", "zscore": "zscore_db"}
    norm_mode = _norm_alias.get(map_norm_mode, map_norm_mode)

    df_clean, meta, output_dir = run_dsa_pipeline(
        case_no=case_no,
        input_root=input_root,
        eeg_folder_name=eeg_folder_name,
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
        result = _load_or_data_for_case(or_data_root, str(case_no))
        if result is None:
            print(f"[WARN] No OR data found for case {case_no} under {or_data_root}")
        else:
            or_dir, basic_df, attr_df, drug_event, vital_df = result
            main_row = _build_main_row(basic_df, attr_df)

            age = main_row.get("age", np.nan)
            age_text = "NA" if pd.isna(age) else f"{age:.0f} years"
            sex = main_row.get("sex", "NA")
            patient_id = main_row.get("ID", "NA")
            case_info = f"ID {patient_id} | Case {case_no} | Age {age_text} | Sex {sex}"

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

    dim_result = None
    if run_dim_reduction:
        dim_result = run_dimensionality_reduction(
            df_clean=df_clean,
            meta=meta,
            output_dir=output_dir,
            sampling_freq=sampling_freq,
            epoch_len_sec=epoch_len_sec,
            feature_set=dim_feature_set,
            pca_components=dim_pca_components,
            umap_neighbors=dim_umap_neighbors,
            umap_min_dist=dim_umap_min_dist,
            cmap=cmap,
        )

    return df_clean, meta, output_dir, dim_result



# ============================================================
# Dimensionality-reduction features and visualisation
# ============================================================

def create_nonoverlapping_epochs(data, epoch_len_sec, fs):
    """Create non-overlapping epochs from a 2D array (samples x channels)."""
    data = np.asarray(data, dtype=float)
    epoch_len = int(round(epoch_len_sec * fs))
    if epoch_len <= 0:
        raise ValueError("epoch_len_sec must be positive.")
    n_epochs = data.shape[0] // epoch_len
    if n_epochs < 2:
        raise ValueError("Not enough EEG data for at least two epochs.")
    trimmed = data[: n_epochs * epoch_len]
    return trimmed.reshape(n_epochs, epoch_len, data.shape[1])


def permutation_entropy(time_series, order=3, delay=1):
    """Compute permutation entropy for one time series."""
    import math
    x = np.asarray(time_series, dtype=float)
    n = len(x)
    if n < delay * (order - 1) + 2:
        return np.nan
    patterns = {}
    for i in range(n - delay * (order - 1)):
        pattern = tuple(np.argsort(x[i : i + delay * order : delay]))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def hjorth_params(signal):
    """Return Hjorth activity, mobility, and complexity."""
    x = np.asarray(signal, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    if var_x <= 0 or var_dx <= 0:
        return np.nan, np.nan, np.nan
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / mobility if mobility > 0 else np.nan
    return float(var_x), float(mobility), float(complexity)


def compute_epoch_psd_features(epoch, fs, fmin=0.5, fmax=45.0):
    """Compute Welch PSD features for all channels in one epoch."""
    values = []
    names = []
    nperseg = min(int(fs * 2), epoch.shape[0])
    for ch_idx in range(epoch.shape[1]):
        f, pxx = welch(epoch[:, ch_idx], fs=fs, nperseg=nperseg)
        mask = (f >= fmin) & (f <= fmax)
        values.extend(pxx[mask])
        names.extend([f"BIS_{ch_idx + 1}_{freq:.1f}Hz" for freq in f[mask]])
    return np.asarray(values, dtype=float), names


def extract_time_domain_features(epoch, fs):
    """Extract compact time-domain features from a 2-channel epoch."""
    from scipy.stats import skew, kurtosis
    from scipy.signal import coherence, hilbert

    features = {}
    for ch_idx, label in enumerate(["BIS_1", "BIS_2"][: epoch.shape[1]]):
        x = np.asarray(epoch[:, ch_idx], dtype=float)
        features[f"{label}_mean"] = np.nanmean(x)
        features[f"{label}_std"] = np.nanstd(x)
        features[f"{label}_skew"] = skew(x, nan_policy="omit")
        features[f"{label}_kurtosis"] = kurtosis(x, nan_policy="omit")
        features[f"{label}_line_length"] = np.nansum(np.abs(np.diff(x)))
        features[f"{label}_zero_crossing"] = int(np.sum(x[:-1] * x[1:] < 0))
        features[f"{label}_perm_entropy"] = permutation_entropy(x)
        act, mob, comp = hjorth_params(x)
        features[f"{label}_hjorth_activity"] = act
        features[f"{label}_hjorth_mobility"] = mob
        features[f"{label}_hjorth_complexity"] = comp

    if epoch.shape[1] >= 2:
        ch1 = epoch[:, 0]
        ch2 = epoch[:, 1]
        features["suppression_ratio"] = np.mean(np.abs(ch1) < 5)
        try:
            f, cxy = coherence(ch1, ch2, fs=fs, nperseg=min(int(fs * 2), len(ch1)))
            alpha = (f >= 8) & (f <= 12)
            features["coherence_alpha"] = np.nanmean(cxy[alpha]) if np.any(alpha) else np.nan
        except Exception:
            features["coherence_alpha"] = np.nan
        try:
            env1 = np.abs(hilbert(ch1))
            env2 = np.abs(hilbert(ch2))
            features["amplitude_envelope_corr"] = np.corrcoef(env1, env2)[0, 1]
        except Exception:
            features["amplitude_envelope_corr"] = np.nan
    return features


def extract_dimensionality_features(df_clean, fs=250, epoch_len_sec=12, feature_set="all", fmin=0.5, fmax=45.0):
    """
    Build epoch-level EEG features for PCA/UMAP.

    feature_set: 'all', 'frequency', or 'time'.
    """
    data = df_clean[["BIS_1", "BIS_2"]].to_numpy(dtype=float)
    epochs = create_nonoverlapping_epochs(data, epoch_len_sec, fs)
    epoch_len = int(round(epoch_len_sec * fs))
    epoch_timestamps = pd.to_datetime(df_clean["timestamp"].iloc[::epoch_len].iloc[: len(epochs)]).reset_index(drop=True)

    feature_rows = []
    psd_names = None
    for epoch in epochs:
        row = {}
        if feature_set in ("all", "frequency"):
            psd_values, names = compute_epoch_psd_features(epoch, fs=fs, fmin=fmin, fmax=fmax)
            # Scale each epoch's PSD profile by channel/frequency shape, as in the older script.
            if np.nanstd(psd_values) > 0:
                psd_values = (psd_values - np.nanmean(psd_values)) / np.nanstd(psd_values)
            row.update(dict(zip(names, psd_values)))
            psd_names = names
        if feature_set in ("all", "time"):
            row.update(extract_time_domain_features(epoch, fs=fs))
        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    features.insert(0, "timestamp", epoch_timestamps)
    return features


def summarize_meta_by_epoch(meta, epoch_timestamps, epoch_len_sec):
    """Compute mean BIS/SEF/SQI/EMG/SR for each feature epoch."""
    if meta is None or meta.empty:
        return pd.DataFrame({"timestamp": epoch_timestamps})
    m = meta.copy()
    if "timestamp" not in m.columns:
        m = m.reset_index()
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")
    m = m.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    cols = {
        "BIS": "BIS_mean",
        "SEF95(bis) Hz": "SEF95_mean",
        "EMG(bis) dB": "EMG_mean",
        "SQI(bis) %": "SQI_mean",
        "SR(bis) %": "SR_mean",
    }
    rows = []
    for ts in pd.to_datetime(epoch_timestamps):
        seg = m.loc[ts : ts + pd.Timedelta(seconds=epoch_len_sec)]
        row = {"timestamp": ts}
        for source, target in cols.items():
            row[target] = pd.to_numeric(seg[source], errors="coerce").mean() if source in seg.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_umap_dynamics(embedding, timestamps):
    """Calculate simple trajectory metrics in the reduced 2D space."""
    emb = np.asarray(embedding, dtype=float)
    velocity = np.concatenate([[0.0], np.linalg.norm(np.diff(emb, axis=0), axis=1)])
    acceleration = np.diff(velocity, prepend=velocity[0])
    ts = pd.to_datetime(timestamps)
    total_minutes = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60.0, 1e-9)
    return pd.DataFrame({
        "timestamp": ts,
        "umap_velocity": velocity,
        "umap_acceleration": acceleration,
        "trajectory_distance_per_min": np.sum(velocity) / total_minutes,
    })


def _remove_axes(ax):
    for pos in ["right", "top", "bottom", "left"]:
        ax.spines[pos].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def plot_embedding_by_time(embedding, timestamps, output_path, title="UMAP colored by time", cmap="jet"):
    times = pd.to_datetime(timestamps)
    time_numeric = mdates.date2num(times)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=time_numeric, cmap=cmap, s=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time")
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(title)
    _remove_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_by_metadata(embedding, epoch_meta, output_path, cmap="jet"):
    """Plot UMAP panels coloured by epoch-level metadata."""
    if epoch_meta is None or epoch_meta.empty:
        return
    plot_cols = [c for c in epoch_meta.columns if c != "timestamp"]
    if not plot_cols:
        return
    n_panels = len(plot_cols) + 1
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.asarray(axes).ravel()
    for i, col in enumerate(plot_cols):
        data = pd.to_numeric(epoch_meta[col], errors="coerce")
        mask = data.notna().to_numpy()
        if mask.any():
            sc = axes[i].scatter(embedding[mask, 0], embedding[mask, 1], c=data[mask], cmap=cmap, s=8)
            fig.colorbar(sc, ax=axes[i])
        if (~mask).any():
            axes[i].scatter(embedding[~mask, 0], embedding[~mask, 1], s=8, alpha=0.3)
        axes[i].set_title(f"UMAP by {col}")
        _remove_axes(axes[i])
    tnum = mdates.date2num(pd.to_datetime(epoch_meta["timestamp"]))
    sc = axes[len(plot_cols)].scatter(embedding[:, 0], embedding[:, 1], c=tnum, cmap=cmap, s=8)
    axes[len(plot_cols)].set_title("UMAP by time")
    fig.colorbar(sc, ax=axes[len(plot_cols)]).ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _remove_axes(axes[len(plot_cols)])
    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_dimensionality_reduction(
    df_clean,
    meta,
    output_dir,
    sampling_freq=250,
    epoch_len_sec=12,
    feature_set="all",
    pca_components=10,
    umap_neighbors=10,
    umap_min_dist=0.3,
    random_state=42,
    cmap="jet",
):
    """
    Run PCA and UMAP on epoch-level EEG features and save CSV/figures.

    Returns a dict with features, metadata summaries, PCA coordinates, UMAP coordinates,
    and trajectory metrics. UMAP is skipped if umap-learn is not installed.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = extract_dimensionality_features(
        df_clean,
        fs=sampling_freq,
        epoch_len_sec=epoch_len_sec,
        feature_set=feature_set,
    )
    today = date.today()
    os.makedirs(output_dir, exist_ok=True)
    features.to_csv(os.path.join(output_dir, f"dim_features_{feature_set}_{today}.csv"), index=False)

    feature_matrix = features.drop(columns=["timestamp"]).replace([np.inf, -np.inf], np.nan)
    keep = feature_matrix.notna().all(axis=1)
    if keep.sum() < 3:
        raise ValueError("Too few complete epochs for PCA/UMAP after removing NaN values.")
    X = feature_matrix.loc[keep].to_numpy(dtype=float)
    kept_timestamps = pd.to_datetime(features.loc[keep, "timestamp"]).reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = int(min(pca_components, X_scaled.shape[0], X_scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=random_state)
    pcs = pca.fit_transform(X_scaled)

    pca_cols = [f"PC{i + 1}" for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, columns=pca_cols)
    pca_df.insert(0, "timestamp", kept_timestamps)
    pca_df.to_csv(os.path.join(output_dir, f"pca_scores_{feature_set}_{today}.csv"), index=False)

    result = {"features": features, "pca_scores": pca_df, "umap_scores": None, "epoch_meta": None, "dynamics": None}

    try:
        import umap
    except Exception as e:
        print(f"[WARN] umap-learn is not installed or failed to import: {e}. PCA was saved, UMAP skipped.")
        return result

    reducer = umap.UMAP(
        n_neighbors=min(umap_neighbors, max(2, len(pcs) - 1)),
        n_components=2,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(pcs)
    umap_df = pd.DataFrame({"timestamp": kept_timestamps, "UMAP1": embedding[:, 0], "UMAP2": embedding[:, 1]})
    umap_df.to_csv(os.path.join(output_dir, f"umap_scores_{feature_set}_{today}.csv"), index=False)
    np.save(os.path.join(output_dir, f"umap_embedding_{feature_set}_{today}.npy"), embedding)

    epoch_meta = summarize_meta_by_epoch(meta, kept_timestamps, epoch_len_sec)
    epoch_meta.to_csv(os.path.join(output_dir, f"epoch_meta_mean_{feature_set}_{today}.csv"), index=False)

    dynamics = calculate_umap_dynamics(embedding, kept_timestamps)
    dynamics.to_csv(os.path.join(output_dir, f"umap_dynamics_{feature_set}_{today}.csv"), index=False)

    plot_embedding_by_time(
        embedding,
        kept_timestamps,
        os.path.join(output_dir, f"umap_time_{feature_set}_{today}.jpg"),
        cmap=cmap,
    )
    plot_embedding_by_metadata(
        embedding,
        epoch_meta,
        os.path.join(output_dir, f"umap_metadata_{feature_set}_{today}.jpg"),
        cmap=cmap,
    )

    result.update({"umap_scores": umap_df, "epoch_meta": epoch_meta, "dynamics": dynamics})
    return result
