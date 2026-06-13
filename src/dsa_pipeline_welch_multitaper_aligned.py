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

from mne.time_frequency import psd_array_multitaper

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

    return df_clean, meta, output_dir


# ============================================================
# Example usage
# ============================================================

# ============================================================
# Example usage (original)
# ============================================================
if __name__ == "__main__":
    df_clean, meta, output_dir = run_dsa_pipeline(
        ID="11111373",
        input_root="data",
        output_root="output",
        sampling_freq=250,
        window_sec=3.0,
        step_sec=0.5,

        # Choose one:
        psd_method="multitaper",
        # psd_method="welch",

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

        # Figures and outputs
        show_summary=True,
        show_aligned_two_channel=True,
        show_single_channel_aligned=False,
        show_normalized_csv=True,
        norm_mode="relative_db",

        # Visualization only. Raw relative bandpower is still saved unsmoothed.
        smooth_bandpower=True,
        smooth_sigma=8,
        cmap="jet",
    )
