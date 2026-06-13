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
        Propensity-score weighting functions are imported from `psweight.py`, so the notebook stays short and reusable.
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
    return Path, np, os, pd, plt, sys


@app.cell
def _(Path, os, sys):
    # Change this path when needed.
    # The default follows the original script layout:
    #   project_root/output/map65/main3_2026-05-18.xlsx
    os.chdir('/Users/jack/Library/CloudStorage/OneDrive-医療法人鉄蕉会/亀田総合病院麻酔科 - 麻酔科スタッフ専用チャネル（後期研修医・PAN含む） - チーム抜管/Tutorial') # change path to fit your environment

    PROJECT_ROOT = Path(os.getcwd()) 
    NOTEBOOK_DIR = PROJECT_ROOT / "src"
    DATA_PATH = PROJECT_ROOT /'data' / "main3_2026-05-18.xlsx"
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
    # Standard Table 1 functions remain in my_mod.
    # Propensity-score weighting functions are kept in psweight.py.

    from my_mod import build_table, median_iqr, format_p, format_smd
    from psweight import (
        estimate_overlap_weights,
        build_weighted_table,
        combined_outcome_table,
        weighted_mean,
    )

    return (
        build_table,
        build_weighted_table,
        combined_outcome_table,
        estimate_overlap_weights,
        weighted_mean,
    )


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
def _(mo):
    mo.md(r"""
    ## 図で理解する：Propensity Score と Overlap Weight

    この解析では、まず患者背景から **「専攻医担当になりやすい確率」** を推定します。
    これが Propensity Score (PS) です。

    ```text
    患者背景
    age / BMI / ASAPS / 診療科
             ↓
    ロジスティック回帰
             ↓
    Propensity Score
    = 専攻医担当になる推定確率
             ↓
    Overlap Weight
    = 比較しやすい症例に大きな重み
             ↓
    重み付き比較
    MAP≤65 AUC の群間差
    ```

    ポイントは、**専攻医にもPAN/PAMEにもなり得た患者** を重視することです。
    逆に、明らかに一方の群に割り当てられそうな患者は重みを小さくします。
    """)
    return


@app.cell
def _(GROUPS, df_raw, estimate_overlap_weights):
    ps_result = estimate_overlap_weights(
            df  = df_raw,
            group_col="Group",
            groups=GROUPS,
            continuous_covars=["age", "BMI", "ASAPS_num"],
            categorical_covars=["診療科"],
            outcome_col="MAP_AUC_below65",
            ps_col="PS",
            weight_col="OW",
            complete_col="_complete",
        )
    df = ps_result.df

    complete = df["_complete"]
    print(f"Complete cases used for PS: {complete.sum()} / {len(df)}")
    print(df.loc[complete, ["Group", "PS", "OW"]].groupby("Group").describe())
    print("\nEffective sample size:")
    print(ps_result.effective_sample_size())

    df.loc[complete, ["Group", "age", "BMI", "ASAPS_num", "MAP_AUC_below65", "PS", "OW"]].head()
    return complete, df


@app.cell
def _(np, pd, plt):
    def plot_overlap_weight_concept():
        """PSとOverlap Weightの関係を示す教育用の図。"""
        ps = np.linspace(0.001, 0.999, 500)
        concept_df = pd.DataFrame(
            {
                "PS": ps,
                "専攻医の重み = 1 - PS": 1 - ps,
                "PAN/PAMEの重み = PS": ps,
            }
        )

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(concept_df["PS"], concept_df["専攻医の重み = 1 - PS"], label="専攻医: 1 - PS")
        ax.plot(concept_df["PS"], concept_df["PAN/PAMEの重み = PS"], label="PAN/PAME: PS")
        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.text(0.5, 0.52, "PS=0.5\nどちらの群にも入り得る", ha="center", va="bottom")
        ax.text(0.08, 0.92, "PAN/PAMEらしい症例", ha="left", va="center")
        ax.text(0.74, 0.92, "専攻医らしい症例", ha="left", va="center")
        ax.set_xlabel("Propensity Score：専攻医担当になる推定確率")
        ax.set_ylabel("Overlap Weight")
        ax.set_title("Overlap Weightの考え方")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linewidth=0.5, alpha=0.6)
        ax.legend(loc="lower center")
        plt.tight_layout()
        return fig

    fig_weight_concept = plot_overlap_weight_concept()
    fig_weight_concept
    return


@app.cell
def _(GROUPS, df, plt):
    def plot_ps_distribution(df, groups=GROUPS):
        """実データでPS分布を確認する図。"""
        sub = df[df["_complete"] & df["PS"].notna()].copy()

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for g in groups:
            x = sub.loc[sub["Group"] == g, "PS"]
            ax.hist(x, bins=20, alpha=0.55, label=g)

        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("Propensity Score：専攻医担当になる推定確率")
        ax.set_ylabel("症例数")
        ax.set_title("実データにおけるPropensity Score分布")
        ax.legend()
        ax.grid(axis="y", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        return fig

    fig_ps_distribution = plot_ps_distribution(df)
    fig_ps_distribution
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 図の読み方

    - PSが0に近い症例：PAN/PAMEに割り当てられやすい症例です。
    - PSが1に近い症例：専攻医に割り当てられやすい症例です。
    - PSが0.5前後の症例：どちらの群にも入り得るため、比較対象として重要です。

    Overlap Weightでは、専攻医群は `1 - PS`、PAN/PAME群は `PS` を重みにします。
    そのため、極端な症例の影響を弱め、背景が重なる症例を中心にMAP≤65 AUCを比較します。
    """)
    return


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
def _(GROUPS, combined_outcome_table):
    def outcome_analysis(df, groups=GROUPS):
        """Run crude and overlap-weighted outcome analysis using psweight.py."""
        outcome_table, weighted_result = combined_outcome_table(
            df,
            outcome_col="MAP_AUC_below65",
            group_col="Group",
            groups=groups,
            weight_col="OW",
            treatment_col="_treated",
            robust_cov="HC3",
            digits=1,
        )

        return {
            "table": outcome_table,
            "model": weighted_result.model,
            "weighted_result": weighted_result,
        }

    return (outcome_analysis,)


@app.cell
def _(df, outcome_analysis):
    outcome = outcome_analysis(df)
    outcome["table"]
    return (outcome,)


@app.cell
def _(GROUPS, df, plt, weighted_mean):
    def weighted_boxplot(df, groups=GROUPS):
        """Simple outcome plot for teaching. Boxplots are unweighted; dots show weighted means."""
        sub = df[df["_complete"] & df["MAP_AUC_below65"].notna()].copy()
        data = [sub.loc[sub["Group"] == g, "MAP_AUC_below65"].values for g in groups]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(data, labels=groups, showfliers=False)

        for i, g in enumerate(groups, start=1):
            y = sub.loc[sub["Group"] == g, "MAP_AUC_below65"]
            w = sub.loc[sub["Group"] == g, "OW"]
            wm = weighted_mean(y, w)
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
    mo.md(f"""
    ## Exported files

    - Results table: `{out_xlsx}`
    - Analysis dataset with PS and OW: `{out_df}`
    - Love plot: `{love_plot_path}`
    """)
    return


if __name__ == "__main__":
    app.run()
