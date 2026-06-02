import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        r"""
        # MAP≤65 AUC hands-on seminar

        This notebook reproduces the analysis in `map65_2026May18.py` step by step.

        Main flow:

        1. Load and prepare the dataset
        2. Estimate the propensity score
        3. Apply overlap weighting
        4. Check covariate balance with SMD and a love plot
        5. Compare MAP≤65 AUC before and after weighting
        6. Export tables and figures

        The unweighted Table 1 functions are imported from `my_mod.py` or `my_mod_2026May.py`.
        Weighted helper functions are defined here because they are specific to this overlap-weighting exercise.
        """
    )
    return (mo,)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from scipy.stats import mannwhitneyu
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    try:
        plt.rcParams["font.family"] = "Hiragino Sans"
    except Exception:
        pass
    return (
        LogisticRegression,
        Path,
        StandardScaler,
        mannwhitneyu,
        np,
        os,
        pd,
        plt,
        sm,
        sys,
    )


@app.cell
def _(Path, os, sys):
    # Change this path when needed.
    # The default follows the original script layout:
    #   project_root/output/map65/main3_2026-05-18.xlsx
    os.chdir('/Users/jack/Desktop/claude')

    PROJECT_ROOT = Path(os.getcwd())  # Adjust if this notebook is not in the "src" directory.')
    NOTEBOOK_DIR = PROJECT_ROOT / "src"
    DATA_PATH = PROJECT_ROOT / "output" / "map65" / "main3_2026-05-18.xlsx"
    OUT_DIR = PROJECT_ROOT / "output" / "map65_hands_on"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Add likely module locations to Python path.
    for p in [NOTEBOOK_DIR, PROJECT_ROOT, PROJECT_ROOT / "scripts"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    GROUPS = ["専攻医", "PAN/PAME"]

    print(f"DATA_PATH = {DATA_PATH}")
    print(f"OUT_DIR   = {OUT_DIR}")
    return DATA_PATH, GROUPS, OUT_DIR


@app.cell
def _():
    # Import existing project functions from my_mod.
    # This avoids duplicating your Table 1 and formatting functions.
    try:
        from my_mod import build_table, median_iqr, format_p, format_smd
        MOD_NAME = "my_mod"
    except ImportError:
        from my_mod_2026May import build_table, median_iqr, format_p, format_smd
        MOD_NAME = "my_mod_2026May"

    print(f"Imported helper functions from: {MOD_NAME}")
    return build_table, format_p, format_smd, median_iqr


@app.cell
def _(GROUPS, np, pd):
    def asa_numeric(x):
        """Extract the first digit from ASAPS and return it as a number."""
        if pd.isna(x):
            return np.nan
        digits = [c for c in str(x) if c.isdigit()]
        return int(digits[0]) if digits else np.nan


    def load_data(path, groups=GROUPS):
        df = pd.read_excel(path, engine="openpyxl")
        df = df[df["Group"].isin(groups)].copy().reset_index(drop=True)

        df["BMI"] = df["Wt_CIS"] / (df["Ht_CIS"] / 100) ** 2
        df["ASAPS_num"] = df["ASAPS"].apply(asa_numeric)
        df["HighASA"] = (df["ASAPS_num"] >= 3).astype(float)
        df["treated"] = (df["Group"] == groups[0]).astype(int)

        return df

    return (load_data,)


@app.cell
def _(DATA_PATH, load_data):
    df_raw = load_data(DATA_PATH)

    print(f"Total cases: {len(df_raw)}")
    print(df_raw["Group"].value_counts())
    df_raw.head()
    return (df_raw,)


@app.cell
def _(GROUPS, LogisticRegression, StandardScaler, np, pd):
    def compute_overlap_weights(df, groups=GROUPS):
        """
        Fit propensity score model on age, BMI, department, and ASAPS.

        Treated group: 専攻医
        Control group: PAN/PAME

        Overlap weights:
          - treated: 1 - PS
          - control: PS
        """
        df = df.copy()

        dept_dummies = pd.get_dummies(
            df["診療科"], prefix="dept", drop_first=True, dtype=float
        )

        covars = pd.concat(
            [
                df[["age", "BMI", "ASAPS_num"]].apply(pd.to_numeric, errors="coerce"),
                dept_dummies,
            ],
            axis=1,
        )

        complete = covars.notna().all(axis=1) & df["MAP_AUC_below65"].notna()
        df["_complete"] = complete

        X = covars.loc[complete].values.astype(float)
        y = df.loc[complete, "treated"].values

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        lr.fit(X_sc, y)

        df["PS"] = np.nan
        df.loc[complete, "PS"] = lr.predict_proba(X_sc)[:, 1]

        df["OW"] = np.where(
            df["treated"] == 1,
            1.0 - df["PS"],
            df["PS"],
        )

        return df, covars, lr, scaler

    return (compute_overlap_weights,)


@app.cell
def _(compute_overlap_weights, df_raw):
    df, covars, ps_model, scaler = compute_overlap_weights(df_raw)

    complete = df["_complete"]
    print(f"Complete cases used for PS: {complete.sum()} / {len(df)}")
    print(df.loc[complete, ["Group", "PS", "OW"]].groupby("Group").describe())

    df.loc[complete, ["Group", "age", "BMI", "ASAPS_num", "MAP_AUC_below65", "PS", "OW"]].head()
    return complete, df


@app.cell
def _(GROUPS, build_table, complete, df):
    t1_before = build_table(
        df[complete],
        continuous=["age", "BMI"],
        binary_vars=["HighASA"],
        categorical=["診療科"],
        group_col="Group",
        groups=GROUPS,
        digits_cont=1,
        digits_pct=1,
    )

    t1_before
    return (t1_before,)


@app.cell
def _(GROUPS, format_smd, np, pd):
    def _wmean(x, w):
        x = pd.to_numeric(x, errors="coerce")
        mask = x.notna() & w.notna()
        if mask.sum() == 0:
            return np.nan
        return float(np.average(x[mask], weights=w[mask]))


    def _wstd(x, w):
        x = pd.to_numeric(x, errors="coerce")
        mask = x.notna() & w.notna()
        if mask.sum() < 2:
            return np.nan
        xv = x[mask].values
        wv = w[mask].values
        mu = np.average(xv, weights=wv)
        var = np.average((xv - mu) ** 2, weights=wv)
        return float(np.sqrt(var))


    def _wprop(x, w, level):
        x_num = pd.to_numeric(x, errors="coerce") if level in (0, 1) else x
        mask = x_num.notna() & w.notna()
        if mask.sum() == 0:
            return np.nan
        wv = w[mask].values
        return float(np.sum(wv[x_num[mask].values == level]) / wv.sum())


    def _wsmd_cont(x1, w1, x2, w2):
        m1, s1 = _wmean(x1, w1), _wstd(x1, w1)
        m2, s2 = _wmean(x2, w2), _wstd(x2, w2)
        pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
        if pooled == 0 or np.isnan(pooled):
            return 0.0 if m1 == m2 else np.nan
        return (m1 - m2) / pooled


    def _wsmd_binary(x1, w1, x2, w2, level=1):
        p1 = _wprop(x1, w1, level)
        p2 = _wprop(x2, w2, level)
        p_avg = (p1 + p2) / 2
        denom = np.sqrt(p_avg * (1 - p_avg))
        if denom == 0 or np.isnan(denom):
            return 0.0 if p1 == p2 else np.nan
        return (p1 - p2) / denom


    def build_weighted_table(
        df,
        continuous,
        binary_vars,
        categorical,
        group_col="Group",
        weight_col="OW",
        groups=None,
        digits_cont=1,
        digits_pct=1,
        digits_smd=3,
    ):
        if groups is None:
            groups = GROUPS

        g1, g2 = groups
        sub = df[df[group_col].isin(groups)].copy()
        w1 = sub.loc[sub[group_col] == g1, weight_col]
        w2 = sub.loc[sub[group_col] == g2, weight_col]
        rows = []

        rows.append(
            {
                "Variable": "Effective N (sum of weights)",
                g1: f"{w1.sum():.1f}" if w1.notna().any() else "—",
                g2: f"{w2.sum():.1f}" if w2.notna().any() else "—",
                "SMD": "",
            }
        )

        for var in continuous:
            x1 = sub.loc[sub[group_col] == g1, var]
            x2 = sub.loc[sub[group_col] == g2, var]
            m1, s1 = _wmean(x1, w1), _wstd(x1, w1)
            m2, s2 = _wmean(x2, w2), _wstd(x2, w2)
            smd_val = _wsmd_cont(x1, w1, x2, w2)
            rows.append(
                {
                    "Variable": var,
                    g1: f"{m1:.{digits_cont}f} ± {s1:.{digits_cont}f}",
                    g2: f"{m2:.{digits_cont}f} ± {s2:.{digits_cont}f}",
                    "SMD": format_smd(smd_val, digits_smd),
                }
            )

        for var in binary_vars:
            x1 = sub.loc[sub[group_col] == g1, var]
            x2 = sub.loc[sub[group_col] == g2, var]
            p1 = _wprop(x1, w1, 1)
            p2 = _wprop(x2, w2, 1)
            smd_val = _wsmd_binary(x1, w1, x2, w2)
            rows.append(
                {
                    "Variable": f"{var}, n (weighted %)",
                    g1: f"{p1 * 100:.{digits_pct}f}%",
                    g2: f"{p2 * 100:.{digits_pct}f}%",
                    "SMD": format_smd(smd_val, digits_smd),
                }
            )

        for var in categorical:
            tmp = sub[[group_col, var, weight_col]].copy()
            levels = tmp[var].dropna().unique()
            smd_list = []

            for lv in levels:
                x1v = (sub.loc[sub[group_col] == g1, var] == lv).astype(float)
                x2v = (sub.loc[sub[group_col] == g2, var] == lv).astype(float)
                sv = _wsmd_binary(x1v, w1, x2v, w2, level=1.0)
                if not np.isnan(sv):
                    smd_list.append(abs(sv))

            rows.append(
                {
                    "Variable": var,
                    g1: "",
                    g2: "",
                    "SMD": f"{max(smd_list):.{digits_smd}f}" if smd_list else "",
                }
            )

            for lv in sorted(levels):
                x1v = sub.loc[sub[group_col] == g1, var]
                x2v = sub.loc[sub[group_col] == g2, var]
                p1 = _wprop(x1v, w1, lv)
                p2 = _wprop(x2v, w2, lv)
                x1b = (x1v == lv).astype(float)
                x2b = (x2v == lv).astype(float)
                smd_val = _wsmd_binary(x1b, w1, x2b, level=1.0)
                rows.append(
                    {
                        "Variable": f"  {lv}",
                        g1: f"{p1 * 100:.{digits_pct}f}%",
                        g2: f"{p2 * 100:.{digits_pct}f}%",
                        "SMD": format_smd(smd_val, digits_smd),
                    }
                )

        return pd.DataFrame(rows)

    return (build_weighted_table,)


@app.cell
def _(GROUPS, build_weighted_table, complete, df):
    t1_after = build_weighted_table(
        df[complete],
        continuous=["age", "BMI"],
        binary_vars=["HighASA"],
        categorical=["診療科"],
        group_col="Group",
        weight_col="OW",
        groups=GROUPS,
        digits_cont=1,
        digits_pct=1,
    )

    t1_after
    return (t1_after,)


@app.cell
def _(pd, t1_after, t1_before):
    def max_smd(tbl):
        smds = pd.to_numeric(tbl["SMD"], errors="coerce").dropna()
        return smds.abs().max() if not smds.empty else float("nan")

    balance_summary = pd.DataFrame(
        {
            "Timing": ["Before weighting", "After overlap weighting"],
            "Max |SMD|": [max_smd(t1_before), max_smd(t1_after)],
        }
    )

    balance_summary
    return (balance_summary,)


@app.cell
def _(np, plt):
    def love_plot(t1_before, t1_after, out_path=None):
        def _norm(v):
            v = str(v).strip()
            v = v.replace(", n (%)", "").replace(", n (weighted %)", "")
            return v.strip()

        def _extract(tbl):
            result = {}
            for _, row in tbl.iterrows():
                try:
                    val = abs(float(str(row["SMD"]).strip()))
                except (ValueError, KeyError):
                    continue
                result[_norm(row["Variable"])] = val
            return result

        before = _extract(t1_before)
        after = _extract(t1_after)
        common = [v for v in before if v in after and v != "診療科"]
        common_sorted = sorted(common, key=lambda v: before[v])

        bv = [before[v] for v in common_sorted]
        av = [after[v] for v in common_sorted]
        n = len(common_sorted)

        fig_h = max(5, n * 0.30 + 1.5)
        fig, ax = plt.subplots(figsize=(7, fig_h))
        y = np.arange(n)

        for i in range(n):
            ax.plot([bv[i], av[i]], [y[i], y[i]], lw=0.8, zorder=2)

        ax.scatter(bv, y, s=45, zorder=4, label="Before weighting")
        ax.scatter(av, y, s=45, zorder=4, label="After overlap weighting")

        ax.axvline(0, lw=1.0, zorder=3)
        ax.axvline(0.1, lw=1.0, ls="--", alpha=0.6, zorder=3)

        top_vars = {"age", "BMI", "HighASA"}
        labels = [v if v in top_vars else f"  {v}" for v in common_sorted]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("|Standardized Mean Difference|")
        ax.set_title("Love plot: 専攻医 vs PAN/PAME\nOverlap weighting")
        ax.set_xlim(left=-0.01)
        ax.grid(axis="x", lw=0.5, zorder=1)
        ax.legend(fontsize=9, frameon=True, loc="lower right")

        plt.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    return (love_plot,)


@app.cell
def _(OUT_DIR, love_plot, t1_after, t1_before):
    love_plot_path = OUT_DIR / "love_plot_map65_overlap_weighting.png"
    fig_love = love_plot(t1_before, t1_after, out_path=love_plot_path)
    print(f"Saved: {love_plot_path}")
    fig_love
    return (love_plot_path,)


@app.cell
def _(GROUPS, format_p, mannwhitneyu, median_iqr, np, pd, sm):
    def outcome_analysis(df, groups=GROUPS):
        sub = df[df["Group"].isin(groups) & df["MAP_AUC_below65"].notna()].copy()
        g1_mask = sub["Group"] == groups[0]
        g2_mask = sub["Group"] == groups[1]

        y1 = sub.loc[g1_mask, "MAP_AUC_below65"]
        y2 = sub.loc[g2_mask, "MAP_AUC_below65"]
        _, uw_p = mannwhitneyu(y1, y2, alternative="two-sided")

        sub_c = sub.dropna(subset=["OW"]).copy()
        g1c = sub_c["Group"] == groups[0]
        g2c = sub_c["Group"] == groups[1]

        y_all = sub_c["MAP_AUC_below65"].values
        X_all = sm.add_constant(sub_c["treated"].values.astype(float))
        w_all = sub_c["OW"].values
        wls = sm.WLS(y_all, X_all, weights=w_all).fit()

        coef = float(wls.params[1])
        ci = np.asarray(wls.conf_int())
        ci_lo = float(ci[1, 0])
        ci_hi = float(ci[1, 1])
        p_wls = float(wls.pvalues[1])

        def wm_sd(mask):
            y = sub_c.loc[mask, "MAP_AUC_below65"]
            w = sub_c.loc[mask, "OW"]
            return f"{_wmean(y, w):.1f} ({_wstd(y, w):.1f})"

        unweighted = {
            f"{groups[0]} median [IQR]": median_iqr(y1),
            f"{groups[1]} median [IQR]": median_iqr(y2),
            "Mann-Whitney p": format_p(uw_p),
        }

        weighted = {
            f"{groups[0]} weighted mean (SD)": wm_sd(g1c),
            f"{groups[1]} weighted mean (SD)": wm_sd(g2c),
            f"Diff ({groups[0]} − {groups[1]})": f"{coef:.1f}",
            "95% CI": f"[{ci_lo:.1f}, {ci_hi:.1f}]",
            "WLS p": format_p(p_wls),
        }

        outcome_table = pd.DataFrame(
            [{"Analysis": "Unweighted", **unweighted}, {"Analysis": "Overlap-weighted WLS", **weighted}]
        )

        return {
            "unweighted": unweighted,
            "weighted": weighted,
            "table": outcome_table,
            "model": wls,
        }

    return (outcome_analysis,)


@app.cell
def _(df, outcome_analysis):
    outcome = outcome_analysis(df)
    outcome["table"]
    return (outcome,)


@app.cell
def _(GROUPS, df, plt):
    def weighted_boxplot(df, groups=GROUPS):
        """Simple outcome plot for teaching. Boxplots are unweighted; dots show weighted means."""
        sub = df[df["_complete"] & df["MAP_AUC_below65"].notna()].copy()
        data = [sub.loc[sub["Group"] == g, "MAP_AUC_below65"].values for g in groups]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(data, labels=groups, showfliers=False)

        for i, g in enumerate(groups, start=1):
            y = sub.loc[sub["Group"] == g, "MAP_AUC_below65"]
            w = sub.loc[sub["Group"] == g, "OW"]
            wm = _wmean(y, w)
            ax.scatter([i], [wm], s=60, zorder=4)
            ax.text(i + 0.05, wm, f"weighted mean: {wm:.1f}", va="center")

        ax.set_ylabel("MAP≤65 AUC (mmHg·min)")
        ax.set_title("MAP≤65 AUC: crude distribution and overlap-weighted mean")
        ax.grid(axis="y", lw=0.5)
        plt.tight_layout()
        return fig

    fig_outcome = weighted_boxplot(df)
    fig_outcome
    return


@app.cell
def _(OUT_DIR, balance_summary, df, outcome, pd, t1_after, t1_before):
    out_xlsx = OUT_DIR / "map65_overlap_weighting_hands_on_results.xlsx"
    out_df = OUT_DIR / "map65_with_propensity_score_and_overlap_weights.csv"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        t1_before.to_excel(xw, sheet_name="Table1_Before", index=False)
        t1_after.to_excel(xw, sheet_name="Table1_After", index=False)
        balance_summary.to_excel(xw, sheet_name="SMD_Summary", index=False)
        outcome["table"].to_excel(xw, sheet_name="Outcome", index=False)

    df.to_csv(out_df, index=False)

    print(f"Saved table file: {out_xlsx}")
    print(f"Saved analysis dataset: {out_df}")
    return out_df, out_xlsx


@app.cell
def _(love_plot_path, mo, out_df, out_xlsx):
    mo.md(
        f"""
        ## Exported files

        - Results table: `{out_xlsx}`
        - Analysis dataset with PS and OW: `{out_df}`
        - Love plot: `{love_plot_path}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
