"""
Ebata_anesthesia_chart_2026Aug13.py

Draw an anesthesia-record style chart (cf. data/template.jpeg) from the
per-case OR export folders in data/Ebata_2026Aug12/.

Usage
-----
    python Ebata_anesthesia_chart_2026Aug13.py --case 92034
    python Ebata_anesthesia_chart_2026Aug13.py --case 92034 91437 --outdir output
    python Ebata_anesthesia_chart_2026Aug13.py --all          # every case in data_dir
    python Ebata_anesthesia_chart_2026Aug13.py --case 92034 --format png pdf   # save both
    python Ebata_anesthesia_chart_2026Aug13.py --case 92034 --vital-line-scale 1.6   # bolder BP/HR/ETCO2/SpO2
    python Ebata_anesthesia_chart_2026Aug13.py --case 92034 --hide-abg --label-rotation 90

Data-handling notes
--------------------
* バイタルデータ.csv timestamps are 12-digit (%Y%m%d%H%M, no seconds).
  薬剤投与情報.csv / 薬剤投与イベント情報.csv timestamps are 14-digit
  (%Y%m%d%H%M%S). A field made of all 9s ("999999999999" /
  "99999999999999") is this export's sentinel for "no value / still open"
  and is treated as missing.
* The drug panel (gas flows, opioid/pressor infusions, boluses, etc.) is
  driven by 薬剤投与イベント情報.csv, which logs each *actual charted event*
  per drug order (DOSAGE_NO): 開始/流速変更/流量変更/再開 give the real
  pump/flow setpoint at that moment (FLOW column, or parsed out of DETAIL,
  e.g. "0.07 μg/kg/min"), 中断/終了 close the current segment, and
  ワンショット is a single bolus dose. Case.drug_event_timeline() replays
  these events into (start, end, value, unit) rate segments and
  (time, value, unit) boluses -- these are the real numbers a clinician
  set, not derived averages. Only if a case has no event rows at all for a
  matched drug does Case.drug_timeline() fall back to an order-level
  average (total VALUE from 使用薬剤情報.csv / the order's active span from
  薬剤投与情報.csv); this fallback is rare in practice.
* The arterial-line trace is used for the blood-pressure band when an
  A-line was present for the case; otherwise NIBP is used instead (shown
  as a thinner, non-continuous line since it is intermittent).
* This script only reads from data/Ebata_2026Aug12/ and never modifies it.
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=UserWarning)

# "Hiragino Sans" looks slightly crisper on screen, but it ships as a .ttc
# (font collection) that trips a matplotlib PDF-backend bug (UnicodeEncodeError
# in the Type-3/XObject glyph path -- see matplotlib#20835-style reports) as
# soon as any Japanese text needs to go into a PDF/SVG. YuGothic renders
# essentially identically and isn't a .ttc, so it works for every format;
# Arial Unicode MS is the broad-coverage fallback if YuGothic is ever missing.
plt.rcParams["font.family"] = ["YuGothic", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_DEFAULT = os.path.join(HERE, "..", "data", "Ebata_2026Aug12")
OUT_DIR_DEFAULT = os.path.join(HERE, "..", "output")
CP932 = "cp932"

# Rows shown in the lower "drug panel", in display order. Each row's actual
# rate/bolus numbers come from 薬剤投与イベント情報.csv (see
# Case.drug_timeline) -- these specs just say which DRUG_NAME substrings
# belong on which labeled row. `exact` requires an exact name match (used
# for O2/AIR so they don't accidentally match on substrings).
ROW_SPECS = [
    dict(label="O2", patterns=["O2"], exact=True),
    dict(label="AIR", patterns=["AIR"], exact=True),
    dict(label="デスフルラン", patterns=["デスフルラン", "ﾃﾞｽﾌﾙﾗﾝ"]),
    dict(label="セボフルラン", patterns=["セボフルラン", "ｾﾎﾞﾌﾙ"]),
    dict(label="レミフェンタニル", patterns=["レミフェンタニル", "アルチバ"]),
    dict(label="フェンタニル",     patterns=["フェンタニル"], exclude=["レミフェンタニル"]),
    dict(label="レミマゾラム",     patterns=["レミマゾラム"]),
    dict(label="プロポフォール",   patterns=["プロポフォール"]),
    dict(label="ミダゾラム",       patterns=["ドルミカム", "ミダゾラム"]),
    dict(label="ロクロニウム",     patterns=["エスラックス", "ロクロニウム"]),
    dict(label="スガマデクス",     patterns=["スガマデクス"]),
    dict(label="ロピバカイン(硬膜外)", patterns=["アナペイン", "ロピバカイン"]),
    dict(label="フェニレフリン",   patterns=["ネオシネジン"]),
    dict(label="エフェドリン",     patterns=["エフェドリン"]),
    dict(label="ノルアドレナリン", patterns=["ノルアドレナリン", "ノルアド"]),
    dict(label="アドレナリン",     patterns=["アドレナリン"], exclude=["ノルアド"]),
    dict(label="バソプレシン",     patterns=["ピトレシン"]),
    dict(label="デキサメタゾン",   patterns=["デキサート"]),
    dict(label="アセトアミノフェン", patterns=["アセリオ"]),
    dict(label="フルルビプロフェン", patterns=["ロピオン"]),
    dict(label="オンダンセトロン", patterns=["オンダンセトロン"]),
    dict(label="ICG(ジアグノグリーン)", patterns=["ジアグノグリーン"]),
]

# 薬剤投与イベント情報.csv CONTENT values that (re)start vs. stop a rate segment.
RATE_CONTENT = {"開始", "流速変更", "流量変更", "再開"}
PAUSE_CONTENT = {"中断", "終了"}
_NUM_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(.*)$")


def parse_detail(detail, flow=None):
    """Pull a (value, unit) pair out of a 薬剤投与イベント情報 row: prefer the
    numeric FLOW column when present, else parse it from the DETAIL text
    (e.g. "0.25 μg/kg/min", "50 μg")."""
    if pd.notna(flow):
        unit = ""
        if isinstance(detail, str):
            m = _NUM_RE.match(detail.strip())
            if m:
                unit = m.group(2).strip()
        return float(flow), unit
    if not isinstance(detail, str):
        return None, ""
    m = _NUM_RE.match(detail.strip())
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2).strip()


# --------------------------------------------------------------------- IO --

def _read(folder, filename, **kw):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding=CP932, **kw)


def _parse_dt(series, fmt):
    s = series.astype(str).str.strip()
    s = s.mask(s.str.match(r"^9+$"))
    s = s.mask(s.isin(["0", "nan", ""]))
    return pd.to_datetime(s, format=fmt, errors="coerce")


def find_case_folder(data_dir, case_no):
    case_str = str(case_no).strip().zfill(8)
    if not os.path.isdir(data_dir):
        return None
    for d in sorted(os.listdir(data_dir)):
        if d.startswith("OR_") and d.endswith(case_str):
            return os.path.join(data_dir, d)
    return None


def list_all_cases(data_dir):
    cases = []
    for d in sorted(os.listdir(data_dir)):
        if d.startswith("OR_"):
            m = re.search(r"(\d{8})$", d)
            if m:
                cases.append(m.group(1))
    return cases


def _item(df, item_code, col="VALUE"):
    if df.empty or "ITEM_CODE" not in df.columns:
        return None
    row = df[df["ITEM_CODE"] == item_code]
    if row.empty:
        return None
    val = row[col].iloc[0]
    return None if pd.isna(val) else val


def _items(df, item_code, col="VALUE"):
    """All values for an ITEM_CODE (multi-SEQ_NO fields like procedure names)."""
    if df.empty or "ITEM_CODE" not in df.columns:
        return []
    row = df[df["ITEM_CODE"] == item_code]
    return [v for v in row[col].tolist() if pd.notna(v)]


class Case:
    def __init__(self, data_dir, case_no):
        self.case_no = str(case_no).strip().zfill(8)
        self.folder = find_case_folder(data_dir, case_no)
        if self.folder is None:
            raise FileNotFoundError(f"No OR_* folder found for case {case_no} in {data_dir}")
        self._load()

    def _load(self):
        vs = _read(self.folder, "バイタルデータ.csv")  # バイタルデータ
        if not vs.empty:
            vs["dt"] = _parse_dt(vs["STARTED_AT"], "%Y%m%d%H%M")
            vs["NUMERICAL_VALUE"] = pd.to_numeric(vs["NUMERICAL_VALUE"], errors="coerce")
        self.vitals_raw = vs

        basic = _read(self.folder, "手術基本情報.csv")  # 手術基本情報
        self.basic = basic

        events = _read(self.folder, "手術イベント記録.csv")  # 手術イベント記録
        if not events.empty:
            events["dt"] = _parse_dt(events["STARTED_AT"], "%Y%m%d%H%M")
        self.events = events

        self.drug_use = _read(self.folder, "使用薬剤情報.csv")  # 使用薬剤情報
        drug_admin = _read(self.folder, "薬剤投与情報.csv")  # 薬剤投与情報
        if not drug_admin.empty:
            drug_admin["ST"] = _parse_dt(drug_admin["STARTED_AT"], "%Y%m%d%H%M%S")
            drug_admin["EN"] = _parse_dt(drug_admin["ENDED_AT"], "%Y%m%d%H%M%S")
        self.drug_admin = drug_admin

        drug_events = _read(self.folder, "薬剤投与イベント情報.csv")  # 薬剤投与イベント情報
        if not drug_events.empty:
            drug_events["dt"] = _parse_dt(drug_events["DATETIME"], "%Y%m%d%H%M%S")
        self.drug_events = drug_events

        self.patient = _read(self.folder, "患者属性情報.csv")  # 患者属性情報
        self.inout = _read(self.folder, "INOUTバランス.csv")  # INOUTバランス
        self.confirmed = _read(self.folder, "手術確定情報.csv")  # 手術確定情報
        fluid_out = _read(self.folder, "体液OUTイベント情報.csv")  # 体液OUTイベント情報
        if not fluid_out.empty:
            fluid_out["dt"] = _parse_dt(fluid_out["STARTED_AT"], "%Y%m%d%H%M")
        self.fluid_out = fluid_out

        # Key times
        self.anes_start = pd.to_datetime(_item(self.basic, "FANSTDTTM"), format="%Y%m%d%H%M", errors="coerce")
        self.anes_end = pd.to_datetime(_item(self.basic, "LANEDDTTM"), format="%Y%m%d%H%M", errors="coerce")
        self.surg_start = pd.to_datetime(_item(self.basic, "FOPESTDTTM"), format="%Y%m%d%H%M", errors="coerce")
        self.surg_end = pd.to_datetime(_item(self.basic, "LOPEEDDTTM"), format="%Y%m%d%H%M", errors="coerce")

        if pd.isna(self.anes_start) and not vs.empty:
            self.anes_start = vs["dt"].min()
        if pd.isna(self.anes_end) and not vs.empty:
            self.anes_end = vs["dt"].max()

        self.weight = pd.to_numeric(_item(self.patient, "WEIT", col="NUMERICAL_VALUE"), errors="coerce")
        self.age = _item(self.patient, "AGE", col="NUMERICAL_VALUE")
        self.sex = _item(self.patient, "SEX")
        self.dept = _item(self.patient, "DEPARTMENT")

        proc_names = _items(self.confirmed, "OPENM")
        self.procedure = " / ".join(dict.fromkeys(proc_names)) if proc_names else None
        dx_names = _items(self.confirmed, "FOP01024")
        self.diagnosis = " / ".join(dict.fromkeys(dx_names)) if dx_names else None
        self.asaps = _item(self.confirmed, "JSASAPS")

    # ---- vitals -------------------------------------------------------
    def vital_series(self, name):
        vs = self.vitals_raw
        if vs.empty:
            return pd.Series(dtype=float)
        sub = vs[vs["NAME"] == name].dropna(subset=["dt", "NUMERICAL_VALUE"])
        if sub.empty:
            return pd.Series(dtype=float)
        s = sub.groupby("dt")["NUMERICAL_VALUE"].first().sort_index()
        return s

    def first_available_series(self, names):
        for n in names:
            s = self.vital_series(n)
            if not s.empty:
                return s, n
        return pd.Series(dtype=float), None

    def has_art_line(self):
        s = self.vital_series("*ART(MEAN)")
        return not s.empty

    # ---- surgical events ------------------------------------------------
    def event_periods(self, start_label, end_label):
        """Pair up sequential start/end events into (start, end) tuples."""
        ev = self.events
        if ev.empty:
            return []
        starts = sorted(ev.loc[ev["VALUE"] == start_label, "dt"].dropna().tolist())
        ends = sorted(ev.loc[ev["VALUE"] == end_label, "dt"].dropna().tolist())
        periods = []
        ei = 0
        for s in starts:
            e = None
            while ei < len(ends) and ends[ei] < s:
                ei += 1
            if ei < len(ends):
                e = ends[ei]
                ei += 1
            periods.append((s, e if e is not None else self.anes_end))
        return periods

    def event_points(self, label):
        ev = self.events
        if ev.empty:
            return []
        return sorted(ev.loc[ev["VALUE"] == label, "dt"].dropna().tolist())

    # ---- drug orders ------------------------------------------------------
    def _matched_dosage_nos(self, patterns, exclude=None, exact=False):
        du = self.drug_use
        if du.empty:
            return du
        name = du["DRUG_NAME"].fillna("")
        mask = pd.Series(False, index=du.index)
        for p in patterns:
            mask |= (name.str.strip() == p) if exact else name.str.contains(re.escape(p), regex=True)
        if exclude:
            for p in exclude:
                mask &= ~name.str.contains(re.escape(p), regex=True)
        return du.loc[mask]

    def drug_orders(self, patterns, exclude=None, exact=False):
        """One row per DOSAGE_NO: name, total VALUE, unit, first start, last end."""
        da = self.drug_admin
        matched = self._matched_dosage_nos(patterns, exclude, exact)[["DOSAGE_NO", "DRUG_NAME", "VALUE", "UNIT"]] \
            if not self.drug_use.empty else pd.DataFrame(columns=["DOSAGE_NO", "DRUG_NAME", "VALUE", "UNIT"])
        if matched.empty or da.empty:
            return []
        out = []
        for _, row in matched.iterrows():
            seg = da[da["DOSAGE_NO"] == row["DOSAGE_NO"]].dropna(subset=["ST"])
            if seg.empty:
                continue
            start = seg["ST"].min()
            end = seg["EN"].max()
            if pd.isna(end):
                end = self.anes_end if pd.notna(self.anes_end) else seg["ST"].max()
            out.append(dict(
                dosage_no=row["DOSAGE_NO"], drug_name=row["DRUG_NAME"],
                value=row["VALUE"], unit=row["UNIT"], start=start, end=end,
            ))
        out.sort(key=lambda d: d["start"])
        return out

    def drug_event_timeline(self, dosage_no):
        """Reconstruct exact charted rate changes / boluses for one order
        (DOSAGE_NO) from 薬剤投与イベント情報.csv, e.g.:
          開始/流速変更/流量変更/再開 -> a new rate segment starts (real
          pump/flow setpoint, in FLOW or parsed from DETAIL)
          中断/終了                  -> the current rate segment ends
          ワンショット                -> a single bolus dose (from DETAIL)
        Returns dict(rate_segments=[(start,end,value,unit), ...],
                     boluses=[(dt,value,unit), ...]).
        """
        ev = self.drug_events
        empty = dict(rate_segments=[], boluses=[])
        if ev.empty:
            return empty
        sub = ev[ev["DOSAGE_NO"] == dosage_no].dropna(subset=["dt"]).sort_values("dt")
        if sub.empty:
            return empty
        rate_segments, boluses = [], []
        cur_start = cur_val = cur_unit = None
        for _, row in sub.iterrows():
            t, content = row["dt"], row["CONTENT"]
            if content == "ワンショット":
                val, unit = parse_detail(row.get("DETAIL"), row.get("FLOW"))
                if val is not None:
                    boluses.append((t, val, unit))
                continue
            if content in RATE_CONTENT:
                if cur_start is not None and cur_val is not None:
                    rate_segments.append((cur_start, t, cur_val, cur_unit))
                val, unit = parse_detail(row.get("DETAIL"), row.get("FLOW"))
                if val is None:  # e.g. 再開 with no explicit rate -> carry previous
                    val, unit = cur_val, cur_unit
                cur_start, cur_val, cur_unit = t, val, unit
            elif content in PAUSE_CONTENT:
                if cur_start is not None and cur_val is not None:
                    rate_segments.append((cur_start, t, cur_val, cur_unit))
                cur_start = cur_val = cur_unit = None
        if cur_start is not None and cur_val is not None:
            end = self.anes_end if pd.notna(self.anes_end) else cur_start
            rate_segments.append((cur_start, end, cur_val, cur_unit))
        return dict(rate_segments=rate_segments, boluses=boluses)

    def drug_timeline(self, patterns, exclude=None, exact=False):
        """Merge drug_event_timeline() across every order matching `patterns`.
        Falls back to an order-average estimate (see drug_orders) only when
        this case has no 薬剤投与イベント情報 rows for the drug at all."""
        matched = self._matched_dosage_nos(patterns, exclude, exact)
        rate_segments, boluses = [], []
        found_events = False
        for dosage_no in matched.get("DOSAGE_NO", []):
            tl = self.drug_event_timeline(dosage_no)
            if tl["rate_segments"] or tl["boluses"]:
                found_events = True
            rate_segments += tl["rate_segments"]
            boluses += tl["boluses"]
        if not found_events:
            for order in self.drug_orders(patterns, exclude, exact):
                segs = self.drug_order_segments(order["dosage_no"])
                dur_min = (order["end"] - order["start"]) / pd.Timedelta(minutes=1)
                if len(segs) <= 2 and dur_min < 3:
                    boluses.append((order["start"], order["value"], order["unit"]))
                else:
                    rate = order["value"] / max(dur_min, 1.0)
                    rate_segments.append((order["start"], order["end"], rate, f"{order['unit']}/min"))
        rate_segments.sort(key=lambda s: s[0])
        boluses.sort(key=lambda b: b[0])
        return dict(rate_segments=rate_segments, boluses=boluses)

    # ---- IN/OUT summary ---------------------------------------------------
    def inout_summary(self):
        io = self.inout
        get = lambda code: _item(io, code, col="NUMERICAL_VALUE")
        crl = pd.to_numeric(get("CRL_合計"), errors="coerce") or 0
        col = pd.to_numeric(get("COL_合計"), errors="coerce") or 0
        alb = pd.to_numeric(get("ALB_合計"), errors="coerce") or 0
        ffp = pd.to_numeric(get("FFP_合計"), errors="coerce") or 0
        pc = pd.to_numeric(get("PC_合計"), errors="coerce") or 0
        rcc = pd.to_numeric(get("RCC_合計"), errors="coerce") or 0
        bl = pd.to_numeric(get("BL_合計"), errors="coerce") or 0
        uv = pd.to_numeric(get("UV_合計"), errors="coerce") or 0
        total_in = crl + col + alb + ffp + pc + rcc
        total = total_in - bl - uv
        return dict(crystalloid=crl, colloid=col, albumin=alb, ffp=ffp, pc=pc,
                    rbc=rcc, bleeding=bl, urine=uv, total=total)

    def drug_order_segments(self, dosage_no):
        da = self.drug_admin
        if da.empty:
            return []
        seg = da[da["DOSAGE_NO"] == dosage_no].dropna(subset=["ST"]).sort_values("ST")
        return seg["ST"].tolist()

    def abg_points(self):
        """Timepoints where an arterial blood gas was drawn: pH/pCO2/pO2."""
        out = []
        for dt in sorted(self.vital_series("pH").index):
            ph = self.vital_series("pH").get(dt)
            pco2 = self.vital_series("pCO2").get(dt)
            po2 = self.vital_series("pO2").get(dt)
            out.append(dict(dt=dt, pH=ph, pCO2=pco2, pO2=po2))
        return out


# ------------------------------------------------------------------ utils --

def to_hours(dt_like, t0):
    if dt_like is None or (np.isscalar(dt_like) and pd.isna(dt_like)):
        return np.nan
    return (dt_like - t0) / pd.Timedelta(hours=1)


def fmt_num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def fmt_dur(td):
    if td is None or pd.isna(td):
        return "N/A"
    total_min = int(td.total_seconds() // 60)
    h, m = divmod(total_min, 60)
    return f"{h}時間{m}分"


# ------------------------------------------------------------------ plot --

def compute_style(n_rows, total_hr, n_abg=0, n_lanes=0, font_scale=1.0, line_scale=1.0,
                   vital_line_scale=1.0):
    """Derive every size on the chart -- figure dimensions, font sizes, line
    weights -- from what *this* case actually needs, so a 20-minute 3-drug
    case and an 11-hour 20-drug case both come out legible without hand-
    tuning per case.

    * The drug panel gets a fixed physical height per row (`row_h_in`), so
      font size never has to shrink just because a case used more drugs --
      the figure grows taller instead of the text shrinking.
    * Line/marker weight eases off a little as this case's drug-row density
      (rows per hour) goes up, since dense charts read better as finer
      strokes; it firms up for short, sparse cases.
    * `font_scale` / `line_scale` are global multipliers on top of that
      per-case sizing, for personal taste -- also exposed as --font-scale /
      --line-scale on the CLI. `vital_line_scale` is an additional multiplier
      on top of `line_scale`, just for the main-plot vitals traces (HR,
      ETCO2, SpO2 -- the BP band is a fill, not a line) -- --vital-line-scale
      on the CLI.
    """
    n_rows = max(n_rows, 1)

    row_h_in = 0.54          # physical height per drug-panel row (inches)
    main_h_in = 6.4          # main vitals panel
    events_h_in = 0.45 * n_lanes + 0.2   # event-lane panel: shrinks when there's little/nothing to show
    drug_h_in = row_h_in * (n_rows + 1)
    content_h_in = main_h_in + events_h_in + drug_h_in

    fs = {k: v * font_scale for k, v in dict(
        title=19, subtitle=14, footnote=10, axis=14, tick=12,
        abg_label=12, abg_marker=13, intubation=10, event_label=12,
        row_label=13, value=10.5, xtick=10.5, xlabel=13, legend=13, summary=13,
    ).items()}

    def line_h(pt):
        """Approximate rendered line height (inches) for a `pt`-sized line,
        including typical leading."""
        return pt * 1.35 / 72

    # Header/footnote margins are reserved as constant *inches* derived from
    # the actual font sizes above, not a fraction of figure height -- fig_h
    # varies a lot case to case (a short simple case vs. an 11-hour 20-drug
    # one), and a fixed fraction would reserve a different absolute amount
    # of room for the same lines of text, clipping them on short figures.
    pad_top_in = 0.11
    gap_before_plot_in = 0.16
    h1, h2 = line_h(fs["title"]), line_h(fs["subtitle"])
    top_in_base = pad_top_in + h1 + h2 + gap_before_plot_in
    abg_extra_in = 1.55 * font_scale if n_abg else 0.0   # room for the pH/pCO2/pO2 rows above ax_main
    top_in = top_in_base + abg_extra_in

    # Bottom margin, built bottom-up: figure edge -> pad -> footnote line 2
    # -> gap -> footnote line 1 -> gap -> x-axis label -> pad -> ax_drug.
    bottom_pad_in, gap_in, gap2_in, xlabel_pad_in = 0.06, 0.09, 0.03, 0.10
    hx, hf = line_h(fs["xlabel"]), line_h(fs["footnote"])
    footnote2_off = bottom_pad_in
    footnote1_off = footnote2_off + hf + gap2_in
    bottom_in = footnote1_off + hf + gap_in + hx + xlabel_pad_in

    fig_h = content_h_in + top_in + bottom_in
    fig_w = float(np.clip(1.55 * total_hr + 9, 18, 32))
    top_frac = (fig_h - top_in) / fig_h
    bottom_frac = bottom_in / fig_h

    # Rows-per-hour is a proxy for how tightly packed value labels will be;
    # denser charts get thinner lines/markers so clusters don't smear.
    density = n_rows / max(total_hr, 1)
    lw_rate = float(np.interp(density, [1, 6], [1.9, 1.2])) * line_scale
    lw_vital = float(np.interp(total_hr, [1, 14], [2.0, 1.1])) * line_scale * vital_line_scale

    return dict(
        fig_w=fig_w, fig_h=fig_h, top=top_frac, bottom=bottom_frac,
        height_ratios=[main_h_in, events_h_in, drug_h_in], content_h_in=content_h_in,
        lw_rate=lw_rate, lw_vital=lw_vital, line_scale=line_scale, fs=fs,
        # Header/footnote baselines as figure-fraction y (va="top"/"bottom"
        # respectively), each placed by the constant inch offsets above so
        # they never collide with the plot regardless of fig_h.
        header1_y=1 - pad_top_in / fig_h,
        header2_y=1 - (pad_top_in + h1) / fig_h,
        footnote1_y=footnote1_off / fig_h,
        footnote2_y=footnote2_off / fig_h,
    )


def plot_case(case: Case, out_path, dpi=150, font_scale=1.0, line_scale=1.0, formats=None,
              show_abg=True, label_rotation=45, vital_line_scale=1.0):
    """Render one case's chart and save it. `out_path` sets the base name;
    its own extension is used unless `formats` is given, e.g.
    formats=["png", "pdf"] saves both from the same render.

    show_abg: set False to omit the pH/pCO2/pO2 rows above the main plot
      (the extra headroom they need is dropped too, not just left blank).
    label_rotation: angle (degrees) for the drug-panel value labels; 0 is
      horizontal, 45 (default) reduces overlap when values are packed
      close together, 90 is vertical.
    vital_line_scale: extra multiplier (on top of line_scale) for just the
      main-plot vitals traces -- HR, ETCO2, SpO2.
    """
    t0 = case.anes_start
    t1 = case.anes_end if pd.notna(case.anes_end) else t0 + pd.Timedelta(hours=1)
    total_hr = max(to_hours(t1, t0), 0.5)

    # Drug-panel rows, ABG points, and event lanes are pure data (no axes
    # needed yet) -- compute them first so the figure can be sized to fit
    # this case.
    rows = []
    for spec in ROW_SPECS:
        tl = case.drug_timeline(spec["patterns"], spec.get("exclude"), spec.get("exact", False))
        if tl["rate_segments"] or tl["boluses"]:
            rows.append(dict(label=spec["label"], timeline=tl))
    abg = case.abg_points() if show_abg else []
    pneumo_periods = case.event_periods("気腹開始", "気腹終了")
    lanes = [(lbl, per, col) for lbl, per, col in [("気腹", pneumo_periods, "#7fb3d5")] if per]

    style = compute_style(len(rows), total_hr, len(abg), len(lanes), font_scale, line_scale,
                           vital_line_scale)
    fs = style["fs"]

    fig = plt.figure(figsize=(style["fig_w"], style["fig_h"]))
    gs = fig.add_gridspec(
        nrows=3, ncols=2, width_ratios=[5.0, 1.5], height_ratios=style["height_ratios"],
        hspace=0.08, wspace=0.03, left=0.16, right=0.985, top=style["top"], bottom=style["bottom"],
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_ev = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_drug = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[:, 1])
    ax_side.axis("off")

    header1 = (
        f"症例 {case.case_no}   {case.dept or ''}   "
        f"{'' if case.age is None else f'{case.age:.0f}歳'} {case.sex or ''}   "
        f"ASA-PS {case.asaps or ''}"
    )
    header2 = f"{case.diagnosis or ''} — {case.procedure or ''}"
    fig.text(0.02, style["header1_y"], header1, fontsize=fs["title"], fontweight="bold", ha="left", va="top")
    fig.text(0.02, style["header2_y"], header2, fontsize=fs["subtitle"], ha="left", va="top", color="#333333")
    # Two explicit lines rather than one long string: fig.text doesn't wrap
    # on its own, and this needs to stay legible on both the widest (long
    # case) and narrowest (short case, min figure width) charts. Baselines
    # come from compute_style as constant inches from the bottom edge, so
    # they never collide with the drug panel's x-axis label above them.
    fig.text(
        0.02, style["footnote1_y"],
        "※ 投与量・流量・ガス濃度は薬剤投与イベント情報.csv（開始/流速変更/流量変更/中断/再開/終了/ワンショット）"
        "に記録された実際の設定値・単回投与量。",
        fontsize=fs["footnote"], color="#777777", ha="left", va="bottom",
    )
    fig.text(
        0.02, style["footnote2_y"],
        "イベント記録のない薬剤のみ、オーダー単位の合計投与量を実投与時間で割った平均レートで代替表示"
        "（行ラベルには反映されません）。",
        fontsize=fs["footnote"], color="#777777", ha="left", va="bottom",
    )

    # ---------------------------------------------------------- main plot
    ax_main.set_xlim(0, total_hr)
    ax_main.set_ylim(0, 180)
    ax_spo2 = ax_main.twinx()
    ax_spo2.set_ylim(0, 100)

    if case.has_art_line():
        sys_s, dia_s = case.vital_series("*ART(SYS)"), case.vital_series("*ART(DIA)")
        bp_label = "血圧 (Aライン)"
    else:
        sys_s, dia_s = case.vital_series("*NIBP(SYS)"), case.vital_series("*NIBP(DIA)")
        bp_label = "血圧 (NIBP)"
    if not sys_s.empty and not dia_s.empty:
        sys_s = sys_s.where((sys_s >= 30) & (sys_s <= 220))
        dia_s = dia_s.where((dia_s >= 15) & (dia_s <= 150))
        idx = sys_s.index.union(dia_s.index)
        sys_r = sys_s.reindex(idx).interpolate(limit=15).ffill(limit=15).bfill(limit=15)
        dia_r = dia_s.reindex(idx).interpolate(limit=15).ffill(limit=15).bfill(limit=15)
        xh = [to_hours(d, t0) for d in idx]
        ax_main.fill_between(xh, dia_r.values, sys_r.values, color="#d62728", alpha=0.55, linewidth=0, zorder=2)

    hr_s, _ = case.first_available_series(["*HR", "*PR"])
    if not hr_s.empty:
        xh = [to_hours(d, t0) for d in hr_s.index]
        ax_main.plot(xh, hr_s.values, color="#2ca02c", lw=style["lw_vital"], zorder=3)

    etco2_s, _ = case.first_available_series(["EtCO2", "etCO2(mmHg)"])
    if not etco2_s.empty:
        etco2_s = etco2_s[etco2_s > 0]
        xh = [to_hours(d, t0) for d in etco2_s.index]
        ax_main.plot(xh, etco2_s.values, color="black", lw=style["lw_vital"] * 0.9, zorder=3)

    spo2_s = case.vital_series("SpO2")
    if not spo2_s.empty:
        xh = [to_hours(d, t0) for d in spo2_s.index]
        ax_spo2.plot(xh, spo2_s.values, color="#1f77b4", lw=style["lw_vital"] * 1.1, zorder=4)

    ax_main.set_ylabel("mmHg / bpm", fontsize=fs["axis"])
    ax_main.tick_params(axis="y", labelsize=fs["tick"])
    ax_spo2.tick_params(axis="y", labelsize=fs["tick"])
    ax_main.grid(axis="x", alpha=0.15)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ABG annotations above the main axis
    abg_rows = [("pH", "pH", "{:.3f}"), ("pCO2", "pCO2", "{:.1f}"), ("pO2", "pO2", "{:.1f}")]
    y_pos = {"pH": 1.24, "pCO2": 1.15, "pO2": 1.06}
    if abg:
        for label, _, _ in abg_rows:
            ax_main.annotate(label, xy=(0, y_pos[label]), xycoords=("axes fraction", "axes fraction"),
                              ha="right", va="center", fontsize=fs["abg_label"], xytext=(-8, 0),
                              textcoords="offset points", annotation_clip=False)
        for pt in abg:
            xh = to_hours(pt["dt"], t0)
            for label, key, fmt in abg_rows:
                v = pt.get(key)
                if v is not None and pd.notna(v):
                    ax_main.annotate(fmt.format(v), xy=(xh, y_pos[label]), xycoords=("data", "axes fraction"),
                                      ha="center", va="center", fontsize=fs["abg_label"], annotation_clip=False)
            ax_main.annotate("▼", xy=(xh, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                              fontsize=fs["abg_marker"], annotation_clip=False)

    # Point-event markers, sitting just under the vitals baseline like the
    # paper-chart template -- a white disc behind each symbol keeps it
    # legible over the BP/HR/ETCO2 traces near y=0:
    #   ✗ 麻酔開始/終了   ◎ 手術開始/終了   I 挿管   E 抜管   P 体位変換
    surg_valid = pd.notna(case.surg_start) and pd.notna(case.surg_end)
    marker_defs = [
        ("✗", [t0, t1]),
        ("◎", [case.surg_start, case.surg_end] if surg_valid else []),
        ("I", case.event_points("挿管")),
        ("E", case.event_points("抜管")),
        ("P", case.event_points("体位変換")),
    ]
    for sym, times in marker_defs:
        for t in times:
            if pd.isna(t):
                continue
            x = to_hours(t, t0)
            ax_main.annotate(
                sym, xy=(x, -0.035), xycoords=("data", "axes fraction"), ha="center", va="center",
                fontsize=fs["intubation"] * 1.3, annotation_clip=False, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none"),
            )

    # ------------------------------------------------------------ events
    # (lanes was already built above, before figure sizing; 麻酔/手術 are
    # marked with the ✗/◎ symbols above instead of a bar here)
    n_lanes = max(len(lanes), 1)
    ax_ev.set_ylim(0, n_lanes + 1)
    for i, (label, periods, color) in enumerate(lanes):
        y = n_lanes - i
        bars = [
            (to_hours(s, t0), max(to_hours(e, t0) - to_hours(s, t0), 0.01))
            for s, e in periods if pd.notna(s) and pd.notna(e)
        ]
        ax_ev.broken_barh(bars, (y - 0.35, 0.7), color=color, edgecolor="none")
        ax_ev.text(-0.008, y / (n_lanes + 1), label, ha="right", va="center", fontsize=fs["event_label"],
                   transform=ax_ev.transAxes)
    ax_ev.set_yticks([])
    ax_ev.set_xlim(0, total_hr)
    for spine in ("top", "right", "left"):
        ax_ev.spines[spine].set_visible(False)
    plt.setp(ax_ev.get_xticklabels(), visible=False)

    # ------------------------------------------------------------- drugs
    # (rows was already built above, before figure sizing)
    n = max(len(rows), 1)
    ax_drug.set_ylim(0, n + 1)
    ax_drug.set_xlim(0, total_hr)
    VAL_FS = fs["value"]
    for i, row in enumerate(rows):
        y = n - i
        tl = row["timeline"]
        header_unit = tl["rate_segments"][0][3] if tl["rate_segments"] else (
            tl["boluses"][0][2] if tl["boluses"] else "")

        # Merge every labeled x-position (rate changes + boluses) for this
        # row so closely spaced values can be staggered together, rather
        # than each loop staggering independently and colliding anyway.
        label_xs = sorted({to_hours(s, t0) for s, _, _, _ in tl["rate_segments"]} |
                           {to_hours(t_, t0) for t_, _, _ in tl["boluses"]})
        prev_x, tier = None, 0
        tier_of = {}
        for x in label_xs:
            if pd.isna(x):
                continue
            tier = (tier + 1) % 3 if (prev_x is not None and x - prev_x < 0.12) else 0
            tier_of[x] = tier
            prev_x = x
        base_off, step_off, rot = 0.24, 0.16, label_rotation

        for s, e, val, unit in tl["rate_segments"]:
            x0, x1 = to_hours(s, t0), to_hours(e, t0)
            if pd.isna(x0) or pd.isna(x1):
                continue
            ax_drug.hlines(y, x0, x1, color="#333333", lw=style["lw_rate"], zorder=3)
            ax_drug.plot([x0, x1], [y, y], marker="|", color="#333333", ms=8, ls="none", zorder=3)
            yoff = base_off + step_off * tier_of.get(x0, 0)
            ax_drug.text(x0, y + yoff, fmt_num(val), fontsize=VAL_FS, ha="left", va="bottom",
                         rotation=rot, rotation_mode="anchor")

        for t_, val, unit in tl["boluses"]:
            x = to_hours(t_, t0)
            if pd.isna(x):
                continue
            yoff = base_off + step_off * tier_of.get(x, 0)
            ax_drug.plot([x], [y], marker="|", color="#333333", ms=10, mew=1.3 * line_scale, zorder=3)
            txt = fmt_num(val) if (not unit or unit == header_unit) else f"{fmt_num(val)}{unit}"
            ax_drug.text(x, y + yoff, txt, fontsize=VAL_FS, ha="left", va="bottom",
                         rotation=rot, rotation_mode="anchor")

        header = f"{row['label']} ({header_unit})" if header_unit else row["label"]
        ax_drug.text(-0.008, y / (n + 1), header, ha="right", va="center", fontsize=fs["row_label"],
                     transform=ax_drug.transAxes)

    ax_drug.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_drug.spines[spine].set_visible(False)
    xt = np.arange(0, total_hr + 0.01, 0.5)
    ax_drug.set_xticks(xt)
    ax_drug.set_xticklabels([f"{v:g}" for v in xt], fontsize=fs["xtick"])
    ax_drug.set_xlabel("麻酔開始からの経過時間 (h)", fontsize=fs["xlabel"])

    # --------------------------------------------------------------- side
    legend_items = [
        ("SpO2 (右軸)", "line", "#1f77b4"),
        ("ETCO2", "line", "black"),
        ("心拍数", "line", "#2ca02c"),
        (bp_label, "patch", "#d62728"),
        ("気腹", "patch", "#7fb3d5"),
        ("✗ 麻酔開始/終了", "symbol", "#333333"),
        ("◎ 手術開始/終了", "symbol", "#333333"),
        ("I 挿管 / E 抜管", "symbol", "#333333"),
        ("P 体位変換", "symbol", "#333333"),
    ]
    # ax_side spans the full figure height, which itself varies per case
    # (compute_style grows it with the drug panel), so legend/summary line
    # spacing is defined in *inches* here and converted to axes-fraction via
    # content_h_in -- otherwise a dense case's tall figure would spread the
    # legend out with huge gaps.
    line_step = 0.36 / style["content_h_in"]
    swatch_h = 0.16 / style["content_h_in"]
    ly = 0.99
    for label, kind, color in legend_items:
        if kind == "line":
            ax_side.plot([0.03, 0.13], [ly, ly], color=color, lw=2.2 * line_scale,
                         transform=ax_side.transAxes, clip_on=False)
        elif kind == "patch":
            ax_side.add_patch(Rectangle((0.03, ly - swatch_h / 2), 0.10, swatch_h, color=color,
                                         transform=ax_side.transAxes, clip_on=False))
        # "symbol" entries carry their own glyph (✗/◎) as part of the label
        # text, so there's no separate swatch to draw.
        ax_side.text(0.18, ly, label, fontsize=fs["legend"], ha="left", va="center",
                     transform=ax_side.transAxes)
        ly -= line_step

    pneumo_total = sum(
        (e - s for s, e in pneumo_periods if pd.notna(s) and pd.notna(e)), pd.Timedelta(0)
    )
    io = case.inout_summary()
    lines = [
        "【時間】",
        f"麻酔時間: {fmt_dur(t1 - t0)}",
    ]
    if pd.notna(case.surg_start) and pd.notna(case.surg_end):
        lines.append(f"手術時間: {fmt_dur(case.surg_end - case.surg_start)}")
    if pneumo_periods:
        lines.append(f"気腹時間(合計): {fmt_dur(pneumo_total)}")
    lines += ["", "【IN/OUT】", f"晶質液: {io['crystalloid']:.0f} mL"]
    if io["colloid"]:
        lines.append(f"膠質液: {io['colloid']:.0f} mL")
    if io["albumin"]:
        lines.append(f"ALB: {io['albumin']:.0f} mL")
    if io["ffp"]:
        lines.append(f"FFP: {io['ffp']:.0f} mL")
    if io["pc"]:
        lines.append(f"PC: {io['pc']:.0f} mL")
    if io["rbc"]:
        lines.append(f"RBC: {io['rbc']:.0f} mL")
    lines.append(f"出血: {io['bleeding']:.0f} mL")
    lines.append(f"尿量: {io['urine']:.0f} mL")
    lines.append(f"total: {io['total']:+.0f} mL")
    ly -= 0.22 / style["content_h_in"]
    ax_side.text(0.03, ly, "\n".join(lines), fontsize=fs["summary"], ha="left", va="top",
                 transform=ax_side.transAxes, linespacing=1.6)

    base, ext = os.path.splitext(out_path)
    if formats is None:
        formats = [ext.lstrip(".") or "png"]
    saved = []
    for fmt in formats:
        p = f"{base}.{fmt.lstrip('.')}"
        # dpi only affects rasterized elements (PNG/JPEG); a PDF/SVG comes
        # out fully vector regardless, so it stays crisp at any zoom/print
        # size and is the better choice for a printable/archival copy.
        fig.savefig(p, dpi=dpi, facecolor="white")
        saved.append(p)
    plt.close(fig)
    return saved


# ------------------------------------------------------------------- CLI --

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs="+", help="Case number(s), e.g. --case 92034 91437")
    ap.add_argument("--all", action="store_true", help="Generate a chart for every case in data-dir")
    ap.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    ap.add_argument("--outdir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--font-scale", type=float, default=1.0,
                     help="Multiply every font size by this (default 1.0). Per-case sizing "
                          "already adapts to how many drug rows/hours a case has; this is an "
                          "extra multiplier on top, for personal taste.")
    ap.add_argument("--line-scale", type=float, default=1.0,
                     help="Multiply every line/marker weight by this (default 1.0).")
    ap.add_argument("--vital-line-scale", type=float, default=1.0,
                     help="Extra multiplier (on top of --line-scale) for just the main-plot "
                          "vitals traces -- HR, ETCO2, SpO2 (default 1.0). E.g. 1.5 makes only "
                          "those lines 50%% thicker without touching drug-panel lines/markers.")
    ap.add_argument("--format", nargs="+", default=["png"], choices=["png", "pdf", "svg", "jpg"],
                     help="Output format(s) to save, e.g. --format png pdf (default: png). "
                          "PDF/SVG are vector -- sharp at any zoom/print size, good for archiving "
                          "or printing a paper chart.")
    ap.add_argument("--hide-abg", action="store_true",
                     help="Omit the pH/pCO2/pO2 rows above the main plot (and the headroom "
                          "they reserve) for cases where you don't want blood-gas results shown.")
    ap.add_argument("--label-rotation", type=float, default=45,
                     help="Angle in degrees for the drug-panel value labels (default 45). "
                          "0 = horizontal, 90 = vertical; steeper angles pack tighter without "
                          "overlapping when a case has lots of closely spaced dose changes.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.all:
        case_nos = list_all_cases(args.data_dir)
    elif args.case:
        case_nos = args.case
    else:
        ap.error("Provide --case <caseNo> [...] or --all")
        return

    for case_no in case_nos:
        try:
            case = Case(args.data_dir, case_no)
            out_path = os.path.join(args.outdir, f"anesthesia_chart_{case.case_no}.png")
            saved = plot_case(case, out_path, font_scale=args.font_scale, line_scale=args.line_scale,
                               formats=args.format, show_abg=not args.hide_abg,
                               label_rotation=args.label_rotation, vital_line_scale=args.vital_line_scale)
            print(f"[ok] case {case.case_no} -> {', '.join(saved)}")
        except Exception as e:
            print(f"[skip] case {case_no}: {e}")


if __name__ == "__main__":
    main()
