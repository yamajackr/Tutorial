import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import os
    from datetime import date
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf

    # Change this path if needed
    os.chdir('/Users/jack/Desktop/claude')
    return date, np, os, pd, plt, smf


@app.cell
def _(os):
    input_dir = os.path.join(os.getcwd(), 'data/map65')
    output_dir = os.path.join(os.getcwd(), 'output/map65')
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir


@app.cell
def _(input_dir, np, os, pd):
    import my_mod_2026May as mm

    # Main Excel files
    pattern = os.path.join(input_dir, '*/*.xlsx')
    main = mm.find_concat(pattern, pd.read_excel)

    # OR vital sign CSV files
    pattern = os.path.join(input_dir, '*/OR_*', 'バイタルデータ.csv')
    vs = mm.find_concat(pattern, pd.read_csv, encoding='cp932')

    # OR basic info CSV files
    pattern = os.path.join(input_dir, '*/OR_*', '手術基本情報.csv')
    basic = mm.find_concat(pattern, pd.read_csv, encoding='cp932')

    # Anesthesiologists
    pattern = os.path.join(input_dir, '*/OR_*', '手術確定情報.csv')

    # Combine all DataFrames into one
    gen = mm.find_concat(pattern, pd.read_csv, encoding='cp932')
    Anes = gen[gen['NAME']=='麻酔科医'][['CASE_NO','VALUE']].rename(columns={"VALUE": "Anesthesiologist"})
    # Combine multiple anesthesiologists per case into a single comma-separated string
    Anes_combined = (
        Anes.groupby("CASE_NO", as_index=False)
            .agg({"Anesthesiologist": lambda x: ", ".join(sorted(set(x.dropna())))})
    )
    main = pd.merge(main, Anes_combined,left_on='caseNo',right_on='CASE_NO', how='left').drop(columns=['CASE_NO'])

    # Clean column names
    main.columns = main.columns.str.strip()
    basic.columns = basic.columns.str.strip()

    # Standardize case numbers
    main["caseNo_str"] = main["caseNo"].astype(str).str.zfill(8)
    basic["caseNo_str"] = basic["CASE_NO"].astype(str).str.zfill(8)

    # Extract anesthesia start / end from basic
    tmp1 = (
        basic[basic["NAME"] == "麻酔開始日時(最初)"]
        [["caseNo_str", "VALUE"]]
        .rename(columns={"VALUE": "Anes_start"})
    )

    tmp2 = (
        basic[basic["NAME"] == "麻酔終了日時(最後)"]
        [["caseNo_str", "VALUE"]]
        .rename(columns={"VALUE": "Anes_end"})
    )

    # Merge into main
    main = main.merge(tmp1, on="caseNo_str", how="left")
    main = main.merge(tmp2, on="caseNo_str", how="left")

    # Convert to datetime
    main["Anes_start"] = pd.to_datetime(main["Anes_start"], format="%Y%m%d%H%M", errors="coerce")
    main["Anes_end"] = pd.to_datetime(main["Anes_end"], format="%Y%m%d%H%M", errors="coerce")
    ### Drug Data
    # Define the pattern for the CSV files
    pattern = os.path.join(input_dir, '*/OR_*', '使用薬剤情報.csv')

    # Combine all DataFrames into one
    drug = mm.find_concat(pattern, pd.read_csv, encoding='cp932')

    # Build boolean masks (NO regex)
    name = drug['DRUG_NAME'].fillna('')

    masks = {
        'Remifentanil': name.str.contains('アルチバ') | name.str.contains('レミフェンタニル'),
        'Fentanyl':     name.str.contains('フェンタニル') & ~name.str.contains('レミフェンタニル'),
        'Propofol':     name.str.contains('プロポフォール'),
        'Remimazolam':  name.str.contains('レミマゾラム'),
        'Flumazenil':   name.str.contains('フルマゼニル'),
        'Sevoflurane':   name.str.contains('ｾﾎﾞﾌﾙ'),
        'Desflurane':   name.str.contains('ﾃﾞｽﾌﾙ'),
    }

    for col_name, mask in masks.items():
        tmp = (
            drug.loc[mask, ['CASE_NO', 'VALUE']]
                .groupby('CASE_NO', as_index=False)['VALUE']
                .sum()
                .rename(columns={'VALUE': col_name})
        )

        main = (
            main.merge(tmp, left_on='caseNo', right_on='CASE_NO', how='left')
                .drop(columns='CASE_NO')
        )

        main[col_name] = main[col_name].fillna(0)

    # rule: if Remifentanil < 20 → mg → convert to mcg
    mask_mg = (main['Remifentanil'] > 0) & (main['Remifentanil'] < 10)
    main.loc[mask_mg, 'Remifentanil'] = main.loc[mask_mg, 'Remifentanil'] * 1000

    # Ensure numeric (in case of object dtype)
    for col in ["Remimazolam", "Propofol", "Sevoflurane", "Desflurane"]:
        main[col] = pd.to_numeric(main[col], errors="coerce")

    # Define conditions
    conditions = [
        (main["Remimazolam"] > 15) & (main["Sevoflurane"] > 1),   # RS
        (main["Remimazolam"] > 15),   # Remimazolam
        (main["Propofol"] > 200),                               # Propofol
        (main["Sevoflurane"] > main["Desflurane"]),             # Sevo > Des
        (main["Sevoflurane"] < main["Desflurane"])              # Des > Sevo
    ]

    choices = ["RS", "Remimazolam", "Propofol", "Sevoflurane", "Desflurane"]

    # Create Group column
    main["Anes_type"] = np.select(conditions, choices, default='Other')
    main
    return main, vs


@app.cell
def _(main, vs):
    # ---------- Normalize case numbers ----------
    # This avoids mismatches such as 248819 vs 00248819 or int vs str.

    main['caseNo_str'] = main['caseNo'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)
    vs['CASE_NO_str'] = vs['CASE_NO'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)

    print('main caseNo example:')
    print(main[['caseNo', 'caseNo_str']].head())
    print('\nvs CASE_NO example:')
    print(vs[['CASE_NO', 'CASE_NO_str']].head())
    return


@app.cell
def _(pd, vs):
    # Build MAP pivot table from vs using ART(MEAN) and NIBP(MEAN)
    from six import b
    vs_map_names = ['*ART(MEAN)', '*NIBP(MEAN)']
    tmp_map = vs[vs['NAME'].isin(vs_map_names)].copy()

    tmp_map['STARTED_AT'] = pd.to_datetime(
        tmp_map['STARTED_AT'], format='%Y%m%d%H%M', errors='coerce'
    )

    tmp_map['NUMERICAL_VALUE'] = pd.to_numeric(tmp_map['NUMERICAL_VALUE'], errors='coerce')

    pivot_map = tmp_map.pivot_table(
        index=['CASE_NO_str', 'STARTED_AT'],
        columns='NAME',
        values='NUMERICAL_VALUE',
        aggfunc='first',
    )

    print('pivot_map columns:', list(pivot_map.columns))
    print('number of cases in pivot_map:', pivot_map.index.get_level_values('CASE_NO_str').nunique())
    pivot_map.head(20)
    return (pivot_map,)


@app.cell
def _(date, main, np, os, output_dir, pd, pivot_map):
    # MAP < 65 mmHg: visualization and AUC calculation
    threshold = 65

    # output_dir is already output/map65, so do not add another map65 folder
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    map65_list = []
    skipped = []

    for case_no in pivot_map.index.get_level_values('CASE_NO_str').unique():
        try:
            cd = pivot_map.xs(case_no, level='CASE_NO_str').copy()
            cd = cd.apply(pd.to_numeric, errors='coerce')

            # Use ART(MEAN); fall back to NIBP(MEAN) where ART is missing
            if '*ART(MEAN)' in cd.columns:
                map_series = cd['*ART(MEAN)']
                if '*NIBP(MEAN)' in cd.columns:
                    map_series = map_series.combine_first(cd['*NIBP(MEAN)'])
            elif '*NIBP(MEAN)' in cd.columns:
                map_series = cd['*NIBP(MEAN)']
            else:
                skipped.append((case_no, 'No ART(MEAN) or NIBP(MEAN) column'))
                continue

            # Clamp to physiological range
            map_series = map_series.where((map_series >= 30) & (map_series <= 200))
            map_series.index = pd.to_datetime(map_series.index)
            map_series = map_series.dropna().sort_index()

            if map_series.empty:
                skipped.append((case_no, 'No valid MAP after cleaning'))
                continue

            # Match anesthesia timing using normalized case number
            row = main[main['caseNo_str'] == str(case_no).zfill(8)]
            if row.empty:
                skipped.append((case_no, 'No matching caseNo in main'))
                continue

            anes_start = pd.to_datetime(row['Anes_start'].values[0], format='%Y%m%d%H%M', errors='coerce')
            anes_end = pd.to_datetime(row['Anes_end'].values[0], format='%Y%m%d%H%M', errors='coerce')

            if pd.isna(anes_start) or pd.isna(anes_end):
                skipped.append((case_no, 'Anes_start or Anes_end could not be parsed'))
                continue

            # Restrict to anesthesia period
            map_series = map_series[(map_series.index >= anes_start) & (map_series.index <= anes_end)]
            if len(map_series) < 2:
                skipped.append((case_no, 'Fewer than 2 MAP points during anesthesia period'))
                continue

            t_min = np.array((map_series.index - map_series.index[0]).total_seconds()) / 60.0

            # AUC below 65 (mmHg·min) via trapezoidal integration
            deficit = np.where(map_series.values < threshold, threshold - map_series.values, 0.0)
            auc = float(np.trapezoid(deficit, x=t_min))

            # Approximate continuous time below 65
            below = map_series.values < threshold
            dt = np.diff(t_min)
            time_below = float(np.sum(dt[below[:-1] & below[1:]]))

            # Time-weighted average MAP
            total_t = float(t_min[-1] - t_min[0])
            twa_map = float(np.trapezoid(map_series.values, x=t_min)) / total_t if total_t > 0 else np.nan

            # # --- Plot ---
            # fig, ax = plt.subplots(figsize=(12, 4))
            # ax.plot(map_series.index, map_series.values, linewidth=1.2, label='MAP')
            # ax.axhline(y=threshold, linestyle='--', linewidth=1.0, label=f'{threshold} mmHg')
            # ax.fill_between(
            #     map_series.index,
            #     map_series.values,
            #     threshold,
            #     where=map_series.values < threshold,
            #     interpolate=True,
            #     alpha=0.35,
            #     label=f'AUC<65: {auc:.1f} mmHg·min',
            # )
            # ax.axvline(x=anes_start, linestyle='--', linewidth=1.5, label='Anes start')
            # ax.axvline(x=anes_end, linestyle='--', linewidth=1.5, label='Anes end')
            # ax.set_title(
            #     f'MAP  Case {case_no}  |  AUC<65 = {auc:.1f} mmHg·min  |  '
            #     f'Time<65 = {time_below:.1f} min  |  TWA-MAP = {twa_map:.1f} mmHg',
            #     fontsize=11,
            # )
            # ax.set_xlabel('Time')
            # ax.set_ylabel('MAP (mmHg)')
            # ax.set_ylim(bottom=0)
            # ax.legend(loc='upper right', fontsize=8, ncol=4)
            # ax.grid(True, alpha=0.25)
            # plt.tight_layout()

            # fig_path = os.path.join(out_dir, f'map65_{case_no}_{date.today()}.jpg')
            # plt.savefig(fig_path, bbox_inches='tight', dpi=300)
            # # plt.show()
            # plt.close(fig)

            map65_list.append({
                'caseNo': case_no,
                'MAP_AUC_below65': round(auc, 2),
                'MAP_time_below65_min': round(time_below, 2),
                'MAP_TWA_mmHg': round(twa_map, 2),
                'n_MAP_points': int(len(map_series)),
                # 'figure_path': fig_path,
            })

        except Exception as e:
            skipped.append((case_no, f'Error: {e}'))

    map65_df = pd.DataFrame(map65_list)
    out_xlsx = os.path.join(output_dir, f'map65_auc_{date.today()}.xlsx')
    map65_df.to_excel(out_xlsx, index=False)

    print('\nSaved summary:', out_xlsx)
    print('\nCompleted cases:')
    print(map65_df)

    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=['caseNo', 'reason'])
        skipped_xlsx = os.path.join(output_dir, f'map65_skipped_{date.today()}.xlsx')
        skipped_df.to_excel(skipped_xlsx, index=False)
        print('\nSkipped cases:')
        print(skipped_df)
        print('\nSaved skipped-case log:', skipped_xlsx)
    return (map65_df,)


@app.cell
def _(main, map65_df, pd):
    main2 = (
        main.drop(columns=['caseNo'])
        .merge(
            map65_df,
            left_on='caseNo_str',
            right_on='caseNo',
            how='left'
        )
    )

    # keep age ≥18
    main2 = main2[
        pd.to_numeric(main2["age"], errors="coerce") >= 18
    ]

    main2
    return (main2,)


@app.cell
def _(main2):
    main3 = main2[main2['人工心肺開始日時(最初)'].isna()].copy()
    main3
    return (main3,)


@app.cell
def _(main3, plt):
    from datetime import datetime

    plt.rcParams['font.family'] = 'Hiragino Sans'
    # Group by 診療科
    dept_rank = (
        main3
        .dropna(subset=["診療科", "MAP_AUC_below65"])
        .groupby("診療科")
        .agg(
            n_cases=("MAP_AUC_below65", "count"),
            mean_auc65=("MAP_AUC_below65", "mean"),
            median_auc65=("MAP_AUC_below65", "median"),
            mean_time65=("MAP_time_below65_min", "mean"),
            median_time65=("MAP_time_below65_min", "median"),
        )
        .reset_index()
    )

    # Optional: exclude small sample sizes
    dept_rank = dept_rank[dept_rank["n_cases"] >= 5]

    # Sort by mean AUC
    dept_rank = dept_rank.sort_values("mean_auc65")

    plot_df1 = dept_rank.sort_values("mean_auc65")

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.barh(
        plot_df1["診療科"],
        plot_df1["mean_auc65"]
    )

    ax1.set_xlabel("Mean MAP AUC below 65 (mmHg·min)")
    ax1.set_ylabel("診療科")
    ax1.set_title("Mean MAP AUC < 65 by Department")

    # Add values
    for i1, v1 in enumerate(plot_df1["mean_auc65"]):
        ax1.text(v1 + 2, i1, f"{v1:.1f}", va='center')

    today = datetime.today().strftime("%Y%m%d")

    plt.tight_layout()

    plt.savefig(
        f"output/map65/mean_auc65_by_department_{today}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Group by ASAPS

    asaps_rank = (
        main3
        .dropna(subset=["ASAPS", "MAP_AUC_below65"])
        .groupby("ASAPS")
        .agg(
            n_cases=("MAP_AUC_below65", "count"),
            mean_auc65=("MAP_AUC_below65", "mean"),
            median_auc65=("MAP_AUC_below65", "median"),
            mean_time65=("MAP_time_below65_min", "mean"),
            median_time65=("MAP_time_below65_min", "median"),
        )
        .reset_index()
    )

    # Optional: exclude small sample sizes
    asaps_rank = asaps_rank[asaps_rank["n_cases"] >= 5]

    # Sort by mean AUC
    plot_df2 = asaps_rank.sort_values("mean_auc65")

    # Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.barh(
        plot_df2["ASAPS"].astype(str),
        plot_df2["mean_auc65"]
    )

    ax2.set_xlabel("Mean MAP AUC below 65 (mmHg·min)")
    ax2.set_ylabel("ASAPS")
    ax2.set_title("Mean MAP AUC < 65 by ASAPS")

    # Add values and case counts
    for i2, (v2, n2) in enumerate(
        zip(plot_df2["mean_auc65"], plot_df2["n_cases"])
    ):
        ax2.text(
            v2 + 2,
            i2,
            f"{v2:.1f} (n={n2})",
            va='center'
        )

    today = datetime.today().strftime("%Y%m%d")

    plt.tight_layout()

    plt.savefig(
        f"output/map65/mean_auc65_by_ASAPS_{today}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    return (today,)


@app.cell
def _(main3, pd):
    # Load anesthesiologist-position table
    staff = pd.read_excel("data/anesthesiologist_positions_2026-05-17.xlsx")

    # Clean spaces
    staff["Anesthesiologist_clean"] = (
        staff["Anesthesiologist"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("　", "", regex=False)
    )

    main3["Anesthesiologist_clean"] = (
        main3["Anesthesiologist"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("　", "", regex=False)
    )

    # Name → Position dictionary
    name_to_position = dict(
        zip(
            staff["Anesthesiologist_clean"],
            staff["Position"]
        )
    )

    priority = [
        "専攻医",
        "PAN/PAME",
        "初期研修医",
        "International Fellow",
        "指導医"
    ]

    def determine_group(text):
        text = str(text)

        found_positions = []

        for name, position in name_to_position.items():
            if name in text:
                found_positions.append(position)

        # Apply priority
        for p in priority:
            if p in found_positions:
                return p

        return "Unknown"

    main3["Group"] = main3["Anesthesiologist_clean"].apply(determine_group)
    print(main3[["Anesthesiologist", "Group"]].head())
    return (staff,)


@app.cell
def _(main3):
    print(main3["Group"].value_counts())
    main3
    return


@app.cell
def _(main3):
    # Keep only needed columns
    rank_df = main3[
        [
            "Anesthesiologist",
            "MAP_AUC_below65",
            "MAP_time_below65_min",
        ]
    ].copy()

    # Remove missing anesthesiologist rows
    rank_df = rank_df.dropna(subset=["Anesthesiologist"])

    # Split combined names
    rank_df["Anesthesiologist"] = (
        rank_df["Anesthesiologist"]
        .astype(str)
        .str.split(",")
    )

    # Expand into one row per person
    rank_df = rank_df.explode("Anesthesiologist")

    # Remove whitespace
    rank_df["Anesthesiologist"] = (
        rank_df["Anesthesiologist"]
        .str.strip()
    )

    # Group by anesthesiologist
    anes_rank = (
        rank_df
        .groupby("Anesthesiologist")
        .agg(
            n_cases=("MAP_AUC_below65", "count"),
            mean_auc65=("MAP_AUC_below65", "mean"),
            median_auc65=("MAP_AUC_below65", "median"),
            mean_time65=("MAP_time_below65_min", "mean"),
            median_time65=("MAP_time_below65_min", "median"),
        )
        .reset_index()
    )

    # Optional: exclude low-volume anesthesiologists
    anes_rank = anes_rank[anes_rank["n_cases"] >= 5]

    # Rank by lowest hypotension burden
    anes_rank = anes_rank.sort_values("mean_auc65")

    # Sort by mean_auc65
    plot_df = anes_rank.sort_values("mean_auc65").reset_index(drop=True)


    return (plot_df,)


@app.cell
def _(date, os, output_dir, plot_df, staff):
    print(plot_df["Anesthesiologist"])
    staff
    plot_df.merge(staff, on='Anesthesiologist', how='left')[['Anesthesiologist', 'Position']].to_excel(os.path.join(output_dir, f'anesthesiologist_positions_{date.today()}.xlsx'), index=False)    
    return


@app.cell
def _(plot_df):
    # keep n_cases ≥ 100
    filtered = plot_df[plot_df["n_cases"] >= 100].sort_values(
        by="mean_auc65",
        ascending=False
    ).reset_index(drop=True)
    filtered

    return (filtered,)


@app.cell
def _(date, filtered, os, output_dir):
    filtered["Anesthesiologist"].to_excel(os.path.join(output_dir, f'anesthesiologist_list_{date.today()}.xlsx'), index=False)
    return


@app.cell
def _(filtered, pd, plt, today):
    plt.rcParams['font.family'] = 'Hiragino Sans'
    # Plot
    # Load anesthesiologist-position table
    pos_df = pd.read_excel("data/anesthesiologist_positions_2026-05-17.xlsx")

    # Remove spaces (full-width and half-width) for matching
    pos_df["Anesthesiologist_clean"] = (
        pos_df["Anesthesiologist"]
        .str.replace(" ", "", regex=False)
        .str.replace("　", "", regex=False)
    )

    filtered["Anesthesiologist_clean"] = (
        filtered["Anesthesiologist"]
        .str.replace(" ", "", regex=False)
        .str.replace("　", "", regex=False)
    )

    # Merge position info
    filtered2 = filtered.merge(
        pos_df[["Anesthesiologist_clean", "Position"]],
        on="Anesthesiologist_clean",
        how="left"
    )

    # Rename categories
    # position_map = {
    #     "指導医": "指導医",
    #     "専攻医": "専攻医",
    #     "PAN/PAME": "PAN/PAME",
    #     "International Fellow": "International Fellow"
    # }

    # filtered2["Position"] = (
    #     filtered2["Position"]
    #     .map(position_map)
    #     .fillna("Unknown")
    # )
    filtered2 = (
        filtered2.groupby("Position", as_index=False)["mean_auc65"]
        .mean()
        .sort_values("mean_auc65")
    )
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        filtered2["Position"],      # <-- use Position instead of name
        filtered2["mean_auc65"]
    )

    ax.set_xlabel("Mean MAP AUC below 65 (mmHg·min)")
    ax.set_ylabel("Position")
    ax.set_title("Ranking of Mean MAP AUC < 65 by Position")

    # Show values
    for i, v in enumerate(filtered2["mean_auc65"]):
        ax.text(v + 2, i, f"{v:.1f}", va='center')

    plt.tight_layout()

    plt.savefig(
        f"output/map65/mean_auc65_by_position_{today}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    return


@app.cell
def _(date, main3, os, output_dir):
    main3.to_excel(os.path.join(output_dir, f'main3_{date.today()}.xlsx'), index=False)
    return


@app.cell
def _(main3, smf):
    model_df = main3[
        [
            "MAP_AUC_below65",
            'Group',
            "age",
            "診療科",
            "ASAPS",
            "手術時間（分）",
            "実施麻酔法",
        ]
    ].dropna()

    fit = smf.ols(
        formula='MAP_AUC_below65 ~ C(Group) + age + C(ASAPS) + C(診療科)',
        # formula='MAP_AUC_below65 ~ C(Group) + age + C(診療科) + C(ASAPS) + Q("手術時間（分）") + C(実施麻酔法)',
        data=model_df
    ).fit()

    print(fit.summary())
    return


@app.cell
def _(main3):
    df = main3[main3['Group'].isin(["専攻医", "PAN/PAME"])]
    df['BMI'] = df['Wt_CIS'] / (df['Ht_CIS'] / 100) ** 2
    return (df,)


@app.cell
def _(df):
    df[["MAP_AUC_below65",
            'Group',
            "age",
            'BMI',
            "診療科",
            "ASAPS",
            "手術時間（分）",
            'Anes_type'
        ]]
    return


@app.cell
def _(date, df, os, output_dir):
    df.to_excel(os.path.join(output_dir, f'resi_pan_{date.today()}.xlsx'), index=False)
    return


if __name__ == "__main__":
    app.run()
