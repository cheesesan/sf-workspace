import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

APP_TITLE = "BDI Dashboard"
DEFAULT_XLSX_PATH = Path("data") / "BDI DATA.xlsx"
DEFAULT_SHEET = "BDI INDEX"


# -------------------------
# Helpers: cleaning & load
# -------------------------
def _clean_col_name(name: str) -> str:
    # Convert line breaks/tabs to spaces, then normalize spacing
    s = str(name).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def detect_header_row(raw: pd.DataFrame, max_scan: int = 25) -> int:
    patterns = [r"\bdate\b", r"\bdates\b", r"\btime\b", r"\bdatetime\b"]
    for i in range(min(max_scan, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        if any(row.str.contains(p).any() for p in patterns):
            return i
    return 1  # fallback


def detect_date_col(columns: list[str]) -> str | None:
    for c in columns:
        lc = str(c).strip().lower()
        if lc in {"date", "dates", "time", "datetime"}:
            return c
    for c in columns:
        lc = str(c).strip().lower()
        if "date" in lc or "time" in lc:
            return c
    return None


def load_excel(file_or_path, sheet_name: str, header_row: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(file_or_path, sheet_name=sheet_name, header=None)
    raw_preview = raw.head(15)

    if header_row is None:
        header_row = detect_header_row(raw)

    headers = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    df = df.dropna(how="all")

    # drop empty-header columns
    df = df.loc[:, [c for c in df.columns if not (pd.isna(c) or str(c).strip() == "")]]

    # clean & deduplicate headers
    cleaned = [_clean_col_name(c) for c in df.columns]
    seen = {}
    new_cols = []
    for c in cleaned:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c} ({seen[c]})")
        else:
            seen[c] = 1
            new_cols.append(c)
    df.columns = new_cols

    # detect date col
    date_col = detect_date_col(df.columns.tolist())
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        if date_col != "DATE":
            df = df.rename(columns={date_col: "DATE"})

    # numeric conversion (except DATE)
    for c in df.columns:
        if c != "DATE":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, raw_preview


# -------------------------
# Analytics helpers
# -------------------------
def ensure_date(df: pd.DataFrame, raw_preview: pd.DataFrame) -> pd.DataFrame:
    if "DATE" in df.columns:
        return df

    st.error("Cannot auto-detect a DATE column. Please select which column is the date.")
    with st.expander("Debug preview (raw top rows)"):
        st.dataframe(raw_preview, use_container_width=True)

    date_pick = st.selectbox("Pick the date column", options=df.columns.tolist())
    df = df.copy()
    df[date_pick] = pd.to_datetime(df[date_pick], errors="coerce")
    df = df.dropna(subset=[date_pick]).sort_values(date_pick)
    df = df.rename(columns={date_pick: "DATE"})
    return df


def add_returns_and_changes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    s = out[col]
    out[f"{col}_dchg"] = s.diff(1)
    out[f"{col}_dret"] = s.pct_change(1)

    # Approx trading-day offsets
    out[f"{col}_mchg"] = s - s.shift(21)
    out[f"{col}_ychg"] = s - s.shift(252)
    out[f"{col}_mpct"] = s / s.shift(21) - 1
    out[f"{col}_ypct"] = s / s.shift(252) - 1
    return out


def rolling_volatility(df: pd.DataFrame, col: str, window: int = 20) -> pd.Series:
    ret = df[col].pct_change(1)
    return ret.rolling(window).std() * np.sqrt(252)  # annualized


def pick_groups(cols: list[str]) -> dict[str, list[str]]:
    upper = {c: c.upper() for c in cols}

    idx_keys = {"BDI", "BCI", "BPI", "BSI", "BHSI"}
    index_cols = [c for c in cols if any(k in upper[c] for k in idx_keys)]

    tc_cols = [
        c for c in cols
        if re.search(r"\b\d*TC\b|\bTCA\b|\bT\/C\b|\bTC\s*AV\b|\b5TC\s*AV\b", upper[c])
    ]
    fleet_cols = [c for c in cols if any(k in upper[c] for k in ["FLEET", "ORDER", "ORDERBOOK", "SCRAP", "DEMOL", "DELIV", "DWT", "SUPPLY"])]

    tc_cols = [c for c in tc_cols if c not in index_cols]
    fleet_cols = [c for c in fleet_cols if c not in index_cols and c not in tc_cols]

    return {"Index": index_cols, "TC Avg": tc_cols, "Fleet": fleet_cols}


def metric_card(latest: pd.Series, prev: pd.Series | None, name: str, fmt: str = "{:,.0f}"):
    val = latest.get(name, np.nan)
    if pd.isna(val):
        st.metric(name, "—")
        return
    if prev is None or pd.isna(prev.get(name, np.nan)):
        st.metric(name, fmt.format(val))
    else:
        delta = val - prev[name]
        st.metric(name, fmt.format(val), f"{delta:,.0f}")


def plot_multi_line(df: pd.DataFrame, date_col: str, series: list[str], title: str):
    long = df[[date_col] + series].melt(id_vars=[date_col], var_name="Metric", value_name="Value")
    long = long.dropna(subset=["Value"])
    fig = px.line(long, x=date_col, y="Value", color="Metric", title=title)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def plot_single(df: pd.DataFrame, y: str, title: str):
    tmp = df[["DATE", y]].dropna()
    fig = px.line(tmp, x="DATE", y=y, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Date-range utilities (FINAL FIX)
# -------------------------
def quick_range_start(quick: str, end_date: pd.Timestamp) -> pd.Timestamp:
    if quick == "All":
        return pd.NaT
    if quick == "MTD":
        return pd.Timestamp(end_date.year, end_date.month, 1)
    if quick == "YTD":
        return pd.Timestamp(end_date.year, 1, 1)
    if quick == "Past Week":
        return end_date - pd.Timedelta(days=7)
    if quick == "Past Month":
        return end_date - pd.DateOffset(months=1)
    if quick == "Past 3 Months":
        return end_date - pd.DateOffset(months=3)
    if quick == "Past 6 Months":
        return end_date - pd.DateOffset(months=6)
    if quick == "Past Year":
        return end_date - pd.DateOffset(years=1)
    if quick == "Past 2 Years":
        return end_date - pd.DateOffset(years=2)
    return pd.NaT


# -------------------------
# App
# -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("BDI Dashboard")
    st.caption("Excel-based dashboard: Index / TC Avg / Fleet + Spread, Rolling Vol, YoY/MoM changes")

    # Sidebar - Data
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload your BDI DATA.xlsx (optional)", type=["xlsx"])
        sheet_name = st.text_input("Sheet name", value=DEFAULT_SHEET)

        auto_header = st.checkbox("Auto-detect header row", value=True)
        header_row_input = st.number_input(
            "Header row (0-based, if not auto)",
            min_value=0, max_value=80, value=1, step=1,
            disabled=auto_header
        )
        debug = st.checkbox("Debug: show raw preview", value=False)

    # Load
    file_src = uploaded if uploaded is not None else DEFAULT_XLSX_PATH
    if uploaded is None and not DEFAULT_XLSX_PATH.exists():
        st.error(f"Cannot find default file: {DEFAULT_XLSX_PATH}. Upload an Excel or put it under data/BDI DATA.xlsx")
        st.stop()

    hdr = None if auto_header else int(header_row_input)
    try:
        df, raw_preview = load_excel(file_src, sheet_name=sheet_name, header_row=hdr)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    if debug:
        with st.expander("Raw preview (top rows)"):
            st.dataframe(raw_preview, use_container_width=True)

    df = ensure_date(df, raw_preview)

    if df.empty:
        st.error("No data rows after parsing. Please check your sheet and header settings.")
        st.stop()

    # Ensure sorted
    df = df.sort_values("DATE").reset_index(drop=True)

    st.success(f"Loaded rows: {len(df):,} | columns: {len(df.columns):,}")
    st.caption(f"Data available through: **{df['DATE'].max().date()}**")

    all_metrics = [c for c in df.columns if c != "DATE"]
    groups = pick_groups(all_metrics)

    # Sidebar - Filters (FINAL: never exceed max data date)
    with st.sidebar:
        st.header("Filters")

        min_date = df["DATE"].min().date()
        max_date = df["DATE"].max().date()

        # ✅ key idea: end date is ALWAYS max_date (last available data day)
        end_default = pd.Timestamp(max_date)

        quick = st.selectbox(
            "Quick range",
            ["Custom", "Past Week", "Past Month", "Past 3 Months", "Past 6 Months", "Past Year", "Past 2 Years", "MTD", "YTD", "All"],
            index=1,
        )

        if quick == "Custom":
            # Custom picker but max selectable date is still max_date, so no error possible
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,  # ✅ critical: cannot pick beyond last data date
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
        else:
            start_ts = quick_range_start(quick, end_default)
            if pd.isna(start_ts) or quick == "All":
                start_date, end_date = min_date, max_date
            else:
                start_date = max(start_ts.date(), min_date)
                end_date = max_date  # ✅ always last data date

            # show the applied range
            st.caption(f"Applied range: **{start_date} → {end_date}** (end = last data date)")

        tab_default = "Index"
        _ = st.selectbox("Default tab", ["Index", "TC Avg", "Fleet"], index=["Index", "TC Avg", "Fleet"].index(tab_default))

        st.divider()
        st.subheader("Index analytics")
        spread_on = st.checkbox("Show BDI vs BPI spread (BDI - BPI)", value=True)
        vol_on = st.checkbox("Show rolling volatility (annualized)", value=True)
        vol_window = st.number_input("Vol window (days)", min_value=5, max_value=120, value=20, step=1, disabled=not vol_on)
        show_yoy_mom = st.checkbox("Show YoY/MoM change", value=True)

    # Apply date filter
    dff = df[(df["DATE"].dt.date >= start_date) & (df["DATE"].dt.date <= end_date)].copy()
    if dff.empty:
        st.warning("No data in selected date range.")
        st.stop()

    # KPI row (top)
    st.subheader("Quick view (latest in range)")
    latest = dff.iloc[-1]
    prev = dff.iloc[-2] if len(dff) >= 2 else None

    kpi_candidates = [c for c in ["BDI", "BPI", "BCI", "BSI", "BHSI"] if c in dff.columns]
    if not kpi_candidates:
        kpi_candidates = all_metrics[:4]

    cols = st.columns(min(4, len(kpi_candidates)))
    for i, name in enumerate(kpi_candidates[: len(cols)]):
        with cols[i]:
            metric_card(latest, prev, name)

    # Tabs
    tabs = st.tabs(["Index", "TC Avg", "Fleet"])

    # -------------------------
    # Tab: Index
    # -------------------------
    with tabs[0]:
        st.markdown("### Index")
        index_cols = groups["Index"] if groups["Index"] else [c for c in all_metrics[:5]]

        left, right = st.columns([1, 1])

        with left:
            selected_index = st.multiselect(
                "Select index series to plot",
                options=[c for c in all_metrics],
                default=[c for c in ["BDI", "BPI", "BCI", "BSI"] if c in all_metrics] or index_cols[:3],
                key="index_series",
            )
            if selected_index:
                plot_multi_line(dff, "DATE", selected_index, "Index series")

        with right:
            if spread_on and ("BDI" in dff.columns) and ("BPI" in dff.columns):
                tmp = dff.copy()
                tmp["BDI_BPI_SPREAD"] = tmp["BDI"] - tmp["BPI"]
                plot_single(tmp, "BDI_BPI_SPREAD", "BDI vs BPI spread (BDI - BPI)")
            else:
                st.info("Spread chart requires both BDI and BPI columns (or turn it off in sidebar).")

        st.markdown("### Analytics (choose a base series)")
        base_series = st.selectbox(
            "Base series",
            options=[c for c in all_metrics],
            index=0 if "BDI" not in all_metrics else all_metrics.index("BDI"),
            key="base_series_index",
        )

        analytics_df = dff[["DATE", base_series]].copy().dropna()
        if analytics_df.empty:
            st.warning("Selected base series has no data in range.")
        else:
            analytics_df = add_returns_and_changes(analytics_df, base_series)

            a1, a2 = st.columns([1, 1])

            with a1:
                if vol_on:
                    analytics_df["ROLL_VOL"] = rolling_volatility(analytics_df, base_series, window=int(vol_window))
                    plot_single(analytics_df, "ROLL_VOL", f"Rolling volatility ({vol_window}d, annualized)")
                else:
                    st.info("Rolling volatility is off (toggle in sidebar).")

            with a2:
                if show_yoy_mom:
                    show_cols = [f"{base_series}_mchg", f"{base_series}_ychg"]
                    exist_cols = [c for c in show_cols if c in analytics_df.columns]
                    if len(exist_cols) == 2:
                        tmp2 = analytics_df[["DATE"] + exist_cols].copy()
                        tmp2 = tmp2.rename(columns={exist_cols[0]: "MoM Change", exist_cols[1]: "YoY Change"})
                        plot_multi_line(tmp2.rename(columns={"MoM Change":"MoM Change", "YoY Change":"YoY Change"}), "DATE", ["MoM Change", "YoY Change"], "MoM / YoY change (absolute)")
                    else:
                        st.info("Not enough history to compute MoM/YoY changes in this range.")
                else:
                    st.info("YoY/MoM is off (toggle in sidebar).")

        st.markdown("### BDI DATA (table)")
        default_table_cols = ["DATE"] + ([c for c in ["BDI", "BPI", "BCI", "BSI"] if c in all_metrics] or all_metrics[:5])
        table_cols = st.multiselect(
            "Table columns",
            options=["DATE"] + all_metrics,
            default=default_table_cols,
            key="index_table_cols",
        )
        table = dff[table_cols].copy()
        table["DATE"] = table["DATE"].dt.date
        st.dataframe(table, use_container_width=True, height=420)
        st.download_button(
            "Download filtered table as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="bdi_dashboard_index_filtered.csv",
            mime="text/csv",
        )

    # -------------------------
    # Tab: TC Avg
    # -------------------------
    with tabs[1]:
        st.markdown("### TC Avg")
        tc_cols = groups["TC Avg"]
        if not tc_cols:
            st.warning("No TC Avg-like columns detected (keywords: TC/4TC/5TC/AVG). You can still select any numeric columns below.")
            tc_cols = all_metrics

        selected_tc = st.multiselect(
            "Select TC Avg series",
            options=all_metrics,
            default=tc_cols[:3],
            key="tc_series",
        )
        if selected_tc:
            plot_multi_line(dff, "DATE", selected_tc, "TC Avg series")

        st.markdown("### TC Avg analytics (YoY/MoM + Vol)")
        base_tc = st.selectbox("Base TC series", options=all_metrics, index=0, key="base_series_tc")
        tc_df = dff[["DATE", base_tc]].copy().dropna()
        if not tc_df.empty:
            tc_df = add_returns_and_changes(tc_df, base_tc)
            if vol_on:
                tc_df["ROLL_VOL"] = rolling_volatility(tc_df, base_tc, window=int(vol_window))
                plot_single(tc_df, "ROLL_VOL", f"{base_tc} rolling vol ({vol_window}d, annualized)")
            if show_yoy_mom:
                pct_cols = [f"{base_tc}_mpct", f"{base_tc}_ypct"]
                if all(c in tc_df.columns for c in pct_cols):
                    tmp = tc_df[["DATE"] + pct_cols].copy()
                    tmp = tmp.rename(columns={pct_cols[0]: "MoM %", pct_cols[1]: "YoY %"})
                    fig = px.line(
                        tmp.melt(id_vars="DATE", var_name="Metric", value_name="Value"),
                        x="DATE",
                        y="Value",
                        color="Metric",
                        title=f"{base_tc} MoM% / YoY% (fraction)",
                    )
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough history for MoM/YoY % in this range.")
        else:
            st.info("Selected TC series has no data in range.")

        st.markdown("### TC DATA (table)")
        default_cols = ["DATE"] + selected_tc[:6]
        table_cols = st.multiselect(
            "Table columns",
            options=["DATE"] + all_metrics,
            default=default_cols if len(default_cols) > 1 else ["DATE"] + all_metrics[:5],
            key="tc_table_cols",
        )
        table = dff[table_cols].copy()
        table["DATE"] = table["DATE"].dt.date
        st.dataframe(table, use_container_width=True, height=420)
        st.download_button(
            "Download filtered table as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="bdi_dashboard_tc_filtered.csv",
            mime="text/csv",
        )

    # -------------------------
    # Tab: Fleet
    # -------------------------
    with tabs[2]:
        st.markdown("### Fleet")
        fleet_cols = groups["Fleet"]

        if not fleet_cols:
            st.warning("No fleet-like columns detected (keywords: FLEET/ORDERBOOK/SCRAP/DELIV/DWT). You can select any numeric columns below.")
            fleet_cols = all_metrics

        selected_fleet = st.multiselect(
            "Select Fleet series",
            options=all_metrics,
            default=fleet_cols[:3],
            key="fleet_series",
        )
        if selected_fleet:
            plot_multi_line(dff, "DATE", selected_fleet, "Fleet series")

        st.markdown("### Fleet DATA (table)")
        table_cols = st.multiselect(
            "Table columns",
            options=["DATE"] + all_metrics,
            default=["DATE"] + selected_fleet[:6] if selected_fleet else ["DATE"] + all_metrics[:5],
            key="fleet_table_cols",
        )
        table = dff[table_cols].copy()
        table["DATE"] = table["DATE"].dt.date
        st.dataframe(table, use_container_width=True, height=420)
        st.download_button(
            "Download filtered table as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="bdi_dashboard_fleet_filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
