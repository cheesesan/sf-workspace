import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import yaml
import bcrypt
from pathlib import Path

import os
import csv
import uuid
from datetime import datetime
from pathlib import Path



APP_TITLE = "BDI Dashboard"
DEFAULT_XLSX_PATH = Path("data") / "BDI DATA.xlsx"
DEFAULT_SHEET = "BDI INDEX"

# =========================
# Auth (username/password)
# =========================
AUTH_FILE = Path("auth.yaml")

def load_auth() -> dict:
    if not AUTH_FILE.exists():
        return {"users": {}}
    with AUTH_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "users" not in data:
        data["users"] = {}
    return data

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def logout():
    st.session_state["authed"] = False
    st.session_state.pop("user", None)

def require_login() -> None:
    # init session
    if "authed" not in st.session_state:
        st.session_state["authed"] = False

    # already logged in
    if st.session_state["authed"]:
        return

    # login UI
    st.set_page_config(page_title="BDI Dashboard", layout="wide")
    st.title("BDI Dashboard")
    st.caption("Please sign in to continue.")

    auth = load_auth()
    users = auth.get("users", {})

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="e.g., rachel")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        u = users.get(username)
        if u and verify_password(password, u.get("password_hash", "")):
            st.session_state["authed"] = True
            st.session_state["user"] = {"username": username, "name": u.get("name", username)}
            st.success(f"Welcome, {st.session_state['user']['name']}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()
# =========================
# Contact / Feedback
# =========================
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)
FEEDBACK_CSV = FEEDBACK_DIR / "feedback.csv"

def save_feedback(
    user: dict | None,
    page: str,
    subject: str,
    message: str,
    uploaded_files: list,
) -> str:
    """
    Save feedback to feedback/feedback.csv and store attachments under feedback/uploads/<id>/
    Return feedback_id.
    """
    feedback_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    upload_dir = FEEDBACK_DIR / "uploads" / feedback_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    attachments = []
    for f in uploaded_files or []:
        # Streamlit UploadedFile: has .name and .getbuffer()
        safe_name = f.name.replace("/", "_").replace("\\", "_")
        out_path = upload_dir / safe_name
        with open(out_path, "wb") as w:
            w.write(f.getbuffer())
        attachments.append(str(out_path))

    row = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "username": (user or {}).get("username", ""),
        "display_name": (user or {}).get("name", ""),
        "page": page,
        "subject": subject.strip(),
        "message": message.strip(),
        "attachments": ";".join(attachments),
    }

    # write header if new
    file_exists = FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return feedback_id

def render_contact_button(current_page: str):
    """
    Show a right-top Contact button that opens a modal dialog.
    """
    # Right-top area
    top_left, top_right = st.columns([0.8, 0.2])
    with top_right:
        if st.button(
            "Contact",
            use_container_width=True,
            key=f"contact_btn_{current_page}"
    ):

            st.session_state["show_contact"] = True

    if "show_contact" not in st.session_state:
        st.session_state["show_contact"] = False

    if st.session_state["show_contact"]:
        with st.modal("Contact / Feedback"):
            st.write("Send feedback to the dashboard owner (you).")

            user = st.session_state.get("user", None)
            if user:
                st.caption(f"Signed in as: **{user.get('name','')}**")

            subject = st.text_input("Subject", placeholder="e.g., Bug report / Feature request / Data issue")
            message = st.text_area("Message", height=160, placeholder="Describe what you saw and what you expected...")

            files = st.file_uploader(
                "Optional: attach screenshots/files",
                accept_multiple_files=True
            )

            col_a, col_b = st.columns(2)
            with col_a:
                submitted = st.button("Submit", type="primary", use_container_width=True)
            with col_b:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["show_contact"] = False
                    st.rerun()

            if submitted:
                if not message.strip():
                    st.error("Please enter a message.")
                else:
                    fid = save_feedback(
                        user=user,
                        page=current_page,
                        subject=subject or "(no subject)",
                        message=message,
                        uploaded_files=files,
                    )
                    st.success(f"Thanks! Feedback received. (ID: {fid})")
                    st.session_state["show_contact"] = False
if st.session_state.get("authenticated", False):
    render_contact_button(current_page=current_page)

# -------------------------
# Vessel groups
# -------------------------
VESSEL_GROUPS = {
    "CAPE": ["C2", "C3", "C5", "C7", "C8-14", "C9-14", "C10-14", "C14", "C16", "C17"],
    "KMX_82": ["P1A-82", "P2A-82", "P3A-82", "P4-82", "P5-82", "P6-82", "P7-Trial", "P8-Trial"],
    "PMX_74": ["P1A-03", "P2A-03", "P3A-03"],
    "SMX_TESS_63": ["S1B-63", "S1C-63", "S2-63", "S3-63", "S4A-63", "S4B-63", "S5-63", "S8-63", "S9-63", "S10-63", "S15-63"],
    "HANDY_38": ["HS1-38", "HS2-38", "HS3-38", "HS4-38", "HS5-38", "HS6-38", "HS7-38"],
}

# UI labels (sidebar/page only show these)
VESSEL_LABELS = {
    "CAPE": "Capesize",
    "KMX_82": "Kamsarmax (82)",
    "PMX_74": "Panamax (74)",
    "SMX_TESS_63": "Supramax (63)",
    "HANDY_38": "Handysize (38)",
}


# -------------------------
# Helpers: cleaning & load
# -------------------------
def _clean_col_name(name: str) -> str:
    s = str(name).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def detect_header_row(raw: pd.DataFrame, max_scan: int = 25) -> int:
    patterns = [r"\bdate\b", r"\bdates\b", r"\btime\b", r"\bdatetime\b"]
    for i in range(min(max_scan, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        if any(row.str.contains(p).any() for p in patterns):
            return i
    return 1


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
    # NOTE: engine="openpyxl" avoids "Excel file format cannot be determined"
    raw = pd.read_excel(file_or_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    raw_preview = raw.head(15)

    if header_row is None:
        header_row = detect_header_row(raw)

    headers = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    df = df.dropna(how="all")

    # drop empty columns
    df = df.loc[:, [c for c in df.columns if not (pd.isna(c) or str(c).strip() == "")]]

    # clean column names + dedupe
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

    # numeric
    for c in df.columns:
        if c != "DATE":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, raw_preview


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


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# -------------------------
# Analytics helpers
# -------------------------
def add_returns_and_changes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    s = out[col]
    out[f"{col}_dchg"] = s.diff(1)
    out[f"{col}_dret"] = s.pct_change(1)
    out[f"{col}_mchg"] = s - s.shift(21)
    out[f"{col}_ychg"] = s - s.shift(252)
    out[f"{col}_mpct"] = s / s.shift(21) - 1
    out[f"{col}_ypct"] = s / s.shift(252) - 1
    return out


def rolling_volatility(df: pd.DataFrame, col: str, window: int = 20) -> pd.Series:
    ret = df[col].pct_change(1)
    return ret.rolling(window).std() * np.sqrt(252)


def plot_multi_line(df: pd.DataFrame, series: list[str], title: str):
    long = df[["DATE"] + series].melt(id_vars=["DATE"], var_name="Metric", value_name="Value")
    long = long.dropna(subset=["Value"])
    fig = px.line(long, x="DATE", y="Value", color="Metric", title=title)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def plot_single(df: pd.DataFrame, y: str, title: str):
    tmp = df[["DATE", y]].dropna()
    fig = px.line(tmp, x="DATE", y=y, title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def quick_range_start(quick: str, end_date: pd.Timestamp, min_date: pd.Timestamp) -> pd.Timestamp:
    if quick == "All":
        return min_date
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
    return min_date

def _year_col(df: pd.DataFrame) -> pd.Series:
    return df["DATE"].dt.year

def compute_yearly_mean_std(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df[["DATE", col]].dropna().copy()
    tmp["YEAR"] = _year_col(tmp)
    out = tmp.groupby("YEAR")[col].agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": "AVG", "std": "STD", "count": "N"})
    return out

def central_band_bounds_by_year(df: pd.DataFrame, col: str, cover: float) -> pd.DataFrame:
    """
    cover=0.50/0.60/0.75 -> central coverage interval:
    low = q((1-cover)/2), high = q(1-(1-cover)/2)
    """
    tmp = df[["DATE", col]].dropna().copy()
    tmp["YEAR"] = tmp["DATE"].dt.year

    a = (1 - cover) / 2.0
    b = 1 - a

    g = tmp.groupby("YEAR")[col]
    out = pd.DataFrame({
        "YEAR": g.apply(lambda s: s.name).index,  # YEAR index
        "LOW": g.quantile(a).values,
        "HIGH": g.quantile(b).values,
    })

    out["COVER"] = int(cover * 100)
    return out


def contiguous_intervals(dates: pd.Series, mask: pd.Series) -> list[dict]:
    """
    Given boolean mask aligned with dates, return contiguous True intervals.
    """
    if len(dates) == 0:
        return []
    m = mask.fillna(False).to_numpy()
    d = pd.to_datetime(dates).to_numpy()

    intervals = []
    start = None
    for i, flag in enumerate(m):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            intervals.append({"start_time": pd.Timestamp(d[start]), "end_time": pd.Timestamp(d[i-1])})
            start = None
    if start is not None:
        intervals.append({"start_time": pd.Timestamp(d[start]), "end_time": pd.Timestamp(d[len(m)-1])})

    # add duration days
    for it in intervals:
        it["days"] = (it["end_time"] - it["start_time"]).days + 1
    return intervals

def intervals_for_value_range(df: pd.DataFrame, col: str, low: float, high: float) -> pd.DataFrame:
    tmp = df[["DATE", col]].dropna().copy()
    m = (tmp[col] > low) & (tmp[col] <= high)
    intervals = contiguous_intervals(tmp["DATE"], m)
    if not intervals:
        return pd.DataFrame(columns=["range", "start_time", "end_time", "days"])
    out = pd.DataFrame(intervals)
    out.insert(0, "range", f"({low:.3f}, {high:.3f}]")
    return out

def bins_and_intervals_for_year(df: pd.DataFrame, col: str, year: int, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a specific year:
    - Build equal-width bins
    - Histogram counts
    - For each bin, find contiguous time intervals
    """
    tmp = df[["DATE", col]].dropna().copy()
    tmp = tmp[tmp["DATE"].dt.year == year]
    if tmp.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["range", "start_time", "end_time", "days"])

    vmin, vmax = float(tmp[col].min()), float(tmp[col].max())
    if np.isclose(vmin, vmax):
        # degenerate
        hist = pd.DataFrame({"bin": [f"[{vmin:.3f}, {vmax:.3f}]"], "count": [len(tmp)]})
        intervals = pd.DataFrame([{
            "range": f"[{vmin:.3f}, {vmax:.3f}]",
            "start_time": tmp["DATE"].min(),
            "end_time": tmp["DATE"].max(),
            "days": (tmp["DATE"].max() - tmp["DATE"].min()).days + 1
        }])
        return hist, intervals

    edges = np.linspace(vmin, vmax, n_bins + 1)
    cats = pd.cut(tmp[col], bins=edges, include_lowest=True, right=True)
    hist = cats.value_counts().sort_index().reset_index()
    hist.columns = ["bin", "count"]
    hist["bin"] = hist["bin"].astype(str)

    all_intervals = []
    # build mask per bin and extract contiguous intervals
    for i in range(n_bins):
        low, high = edges[i], edges[i+1]
        interval_df = intervals_for_value_range(tmp, col, low, high)
        if not interval_df.empty:
            all_intervals.append(interval_df)

    intervals = pd.concat(all_intervals, ignore_index=True) if all_intervals else pd.DataFrame(columns=["range", "start_time", "end_time", "days"])
    intervals = intervals.sort_values(["start_time"]).reset_index(drop=True)
    return hist, intervals

def format_interval_table(df_int: pd.DataFrame) -> pd.DataFrame:
    if df_int.empty:
        return df_int
    out = df_int.copy()
    out["start_time"] = pd.to_datetime(out["start_time"]).dt.date
    out["end_time"] = pd.to_datetime(out["end_time"]).dt.date
    return out

# -------------------------
# Pages
# -------------------------
render_contact_button(current_page="Home")
def render_home(dff: pd.DataFrame | None, all_metrics: list[str] | None):
    st.title("BDI Dashboard")
    st.subheader("Quick view (latest in range)")
    latest_date = pd.to_datetime(latest_row["DATE"]).date()
    st.caption(f"As of: **{latest_date}**")


    if dff is None or dff.empty:
        st.info("📄 Please upload **BDI DATA.xlsx** from the sidebar, then click **Open page**.")
        return

    latest = dff.iloc[-1]
    prev = dff.iloc[-2] if len(dff) >= 2 else None

    kpi_candidates = [c for c in ["BDI", "BPI", "BCI", "BSI", "BHSI"] if c in dff.columns]
    if not kpi_candidates and all_metrics:
        kpi_candidates = all_metrics[:4]

    cols = st.columns(min(4, len(kpi_candidates)))
    for i, name in enumerate(kpi_candidates[: len(cols)]):
        val = latest.get(name, np.nan)
        if pd.isna(val):
            cols[i].metric(name, "—")
        else:
            if prev is None or pd.isna(prev.get(name, np.nan)):
                cols[i].metric(name, f"{val:,.0f}")
            else:
                cols[i].metric(name, f"{val:,.0f}", f"{(val - prev[name]):,.0f}")

def render_index_page(dff: pd.DataFrame, all_metrics: list[str]):
    st.header("Index")

    default_idx = [c for c in ["BDI", "BPI", "BCI", "BSI"] if c in all_metrics]
    selected_index = st.multiselect(
        "Select index series to plot",
        options=all_metrics,
        default=default_idx,
        key="idx_sel",
    )
    if selected_index:
        plot_multi_line(dff, selected_index, "Index series")

    # ---- Analytics controls (moved INSIDE the page) ----
    st.subheader("Analytics (choose a base series)")

    base_series = st.selectbox(
        "Base series",
        options=all_metrics,
        index=all_metrics.index("BDI") if "BDI" in all_metrics else 0,
        key="idx_base",
    )

    show_vol = st.checkbox("Show rolling volatility", value=True, key="idx_vol_on")
    vol_window = st.number_input(
        "Vol window (days)",
        min_value=5,
        max_value=120,
        value=20,
        step=1,
        disabled=not show_vol,
        key="idx_vol_win",
    )
    show_yoy_mom = st.checkbox("Show YoY / MoM change", value=True, key="idx_yoymom_on")

    analytics_df = dff[["DATE", base_series]].dropna()
    analytics_df = add_returns_and_changes(analytics_df, base_series)

    c1, c2 = st.columns(2)
    with c1:
        if show_vol:
            analytics_df["ROLL_VOL"] = rolling_volatility(analytics_df, base_series, window=int(vol_window))
            plot_single(analytics_df, "ROLL_VOL", f"Rolling volatility ({vol_window}d, annualized)")
    with c2:
        if show_yoy_mom:
            mom = f"{base_series}_mchg"
            yoy = f"{base_series}_ychg"
            tmp = analytics_df.rename(columns={mom: "MoM Change", yoy: "YoY Change"})
            plot_multi_line(tmp, ["MoM Change", "YoY Change"], "MoM / YoY change (absolute)")

    st.subheader("BDI DATA (table)")
    table_cols = st.multiselect(
        "Table columns",
        options=["DATE"] + all_metrics,
        default=["DATE"] + default_idx,
        key="idx_tbl",
    )
    tbl = dff[table_cols].copy()
    tbl["DATE"] = tbl["DATE"].dt.date
    st.dataframe(tbl, use_container_width=True, height=420)

def render_tc_page(dff: pd.DataFrame, all_metrics: list[str]):
    st.header("TC Avg")

    tc_candidates = [c for c in all_metrics if ("TC AV" in c.upper() or "5TC AV" in c.upper())]
    selected_tc = st.multiselect(
        "Select TC Avg series",
        options=all_metrics,
        default=tc_candidates,
        key="tc_sel",
    )
    if selected_tc:
        plot_multi_line(dff, selected_tc, "TC Avg series")

    # ---- Analytics controls (moved INSIDE the page) ----
    st.subheader("Analytics (choose a base series)")

    base_tc = st.selectbox(
        "Base TC series",
        options=all_metrics,
        index=all_metrics.index(selected_tc[0]) if selected_tc else 0,
        key="tc_base",
    )

    show_vol = st.checkbox("Show rolling volatility", value=True, key="tc_vol_on")
    vol_window = st.number_input(
        "Vol window (days)",
        min_value=5,
        max_value=120,
        value=20,
        step=1,
        disabled=not show_vol,
        key="tc_vol_win",
    )
    show_yoy_mom = st.checkbox("Show YoY / MoM change", value=True, key="tc_yoymom_on")

    tc_df = dff[["DATE", base_tc]].dropna()
    tc_df = add_returns_and_changes(tc_df, base_tc)

    c1, c2 = st.columns(2)
    with c1:
        if show_vol:
            tc_df["ROLL_VOL"] = rolling_volatility(tc_df, base_tc, window=int(vol_window))
            plot_single(tc_df, "ROLL_VOL", f"Rolling volatility ({vol_window}d, annualized)")
    with c2:
        if show_yoy_mom:
            mom = f"{base_tc}_mchg"
            yoy = f"{base_tc}_ychg"
            tmp = tc_df.rename(columns={mom: "MoM Change", yoy: "YoY Change"})
            plot_multi_line(tmp, ["MoM Change", "YoY Change"], "MoM / YoY change (absolute)")

    st.subheader("TC DATA (table)")
    table_cols = st.multiselect(
        "Table columns",
        options=["DATE"] + all_metrics,
        default=["DATE"] + (selected_tc[:6] if selected_tc else all_metrics[:6]),
        key="tc_tbl",
    )
    tbl = dff[table_cols].copy()
    tbl["DATE"] = tbl["DATE"].dt.date
    st.dataframe(tbl, use_container_width=True, height=420)

vessel_group_key = st.session_state.get("vessel_group_key", "UNKNOWN")
vessel_group_label = VESSEL_LABELS.get(vessel_group_key, vessel_group_key)
def render_vessel_group_page(dff: pd.DataFrame, vessel_group_key: str):
    st.header("Vessel Group")

    # --- Choose vessel INSIDE the page ---
    key_to_label = VESSEL_LABELS
    labels = list(key_to_label.values())
    default_label = key_to_label.get(vessel_group_key, labels[0])
    default_index = labels.index(default_label) if default_label in labels else 0

    vessel_label = st.radio(
        "Choose vessel type",
        options=labels,
        index=default_index,
        key="vg_radio",
    )

    label_to_key = {v: k for k, v in key_to_label.items()}
    vessel_group_key = label_to_key[vessel_label]
    st.session_state.vessel_group_key = vessel_group_key  # remember choice

    group_cols = existing_cols(dff, VESSEL_GROUPS[vessel_group_key])

    st.subheader(vessel_label)

    if not group_cols:
        st.warning(f"No columns found for {vessel_label}.")
        st.write("Expected columns:", VESSEL_GROUPS[vessel_group_key])
        return

    # --- Plot series (as you already have) ---
    selected_routes = st.multiselect(
        "Select series",
        options=group_cols,
        default=group_cols,
        key=f"routes_{vessel_group_key}",
    )
    if selected_routes:
        plot_multi_line(dff, selected_routes, f"{vessel_label} series")

    # =========================================================
    # Analytics (NEW) — match your Excel/fig logic
    # =========================================================
    st.markdown("---")
    st.subheader("Analytics (annual stats + distribution intervals)")

    # pick ONE route for analytics
    route = st.selectbox(
        "Pick one route / leg for analytics",
        options=group_cols,
        index=0,
        key=f"vg_route_pick_{vessel_group_key}",
    )

    base = dff[["DATE", route]].dropna().copy()
    if base.empty:
        st.info("No data for the selected route in the current time range.")
        return

    # ---- summary box (like your example) ----
    start_dt = base["DATE"].min().date()
    end_dt = base["DATE"].max().date()
    total_days = (base["DATE"].max() - base["DATE"].min()).days + 1
    st.caption(f"Route: **{route}** | Start: **{start_dt}** | End: **{end_dt}** | Total time: **{total_days} days**")

    # ---- 1) yearly average & std ----
    yearly = compute_yearly_mean_std(base, route)
    if yearly.empty:
        st.info("Not enough data to compute yearly stats.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        long = yearly.melt(id_vars=["YEAR"], value_vars=["AVG", "STD"], var_name="Metric", value_name="Value")
        fig = px.line(long, x="YEAR", y="Value", color="Metric", title="Annual average & std")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(yearly, use_container_width=True, height=360)

    # ---- 2) central coverage bands: 50% / 60% / 75% ----
    bands = []
    for cover in [0.50, 0.60, 0.75]:
        bands.append(central_band_bounds_by_year(base, route, cover))
    band_df = pd.concat(bands, ignore_index=True)

    # reshape for plotting upper/lower by year & cover
    band_long = band_df.melt(id_vars=["YEAR", "COVER"], value_vars=["LOW", "HIGH"], var_name="Bound", value_name="Value")
    band_long["Series"] = band_long["COVER"].astype(str) + "% " + band_long["Bound"]

    fig_band = px.line(
        band_long,
        x="YEAR",
        y="Value",
        color="Series",
        title="Central coverage bands by year (50% / 60% / 75%)",
    )
    fig_band.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig_band, use_container_width=True)

    # ---- 3) choose a year -> histogram + value-range time intervals ----
    years = sorted(base["DATE"].dt.year.unique().tolist())
    year_pick = st.selectbox("Year", options=years, index=len(years) - 1, key=f"vg_year_{vessel_group_key}_{route}")

    bins = st.slider("Number of bins", min_value=6, max_value=20, value=10, step=1, key=f"vg_bins_{vessel_group_key}_{route}")

    hist, intervals = bins_and_intervals_for_year(base, route, year_pick, n_bins=int(bins))

    c3, c4 = st.columns([2, 1])
    with c3:
        if not hist.empty:
            fig_h = px.bar(hist, x="bin", y="count", title="Distribution (by value ranges)")
            fig_h.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            fig_h.update_xaxes(tickangle=45)
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No data for this year.")

    with c4:
        st.write("Intervals (continuous ranges)")
        it = format_interval_table(intervals)
        if it.empty:
            st.caption("No continuous intervals found.")
        else:
            # show a compact table (top 30)
            st.dataframe(it.head(30), use_container_width=True, height=360)

    # Optional: show continuous intervals for the 50/60/75 bands in the chosen year
    st.markdown("### Continuous intervals inside central bands (selected year)")
    band_year = band_df[band_df["YEAR"] == year_pick].copy()
    if band_year.empty:
        st.caption("No band data for the selected year.")
    else:
        for cover in [50, 60, 75]:
            row = band_year[band_year["COVER"] == cover]
            if row.empty:
                continue
            low, high = float(row["LOW"].iloc[0]), float(row["HIGH"].iloc[0])
            tmp_year = base[base["DATE"].dt.year == year_pick].copy()
            band_intervals = intervals_for_value_range(tmp_year, route, low, high)
            st.write(f"**{cover}% band**: ({low:.3f}, {high:.3f}]")
            st.dataframe(format_interval_table(band_intervals).head(20), use_container_width=True)

    # =========================================================
    # Original table
    # =========================================================
    st.subheader("Data table")
    table_cols = st.multiselect(
        "Table columns",
        options=["DATE"] + group_cols,
        default=["DATE"] + (selected_routes[:8] if selected_routes else group_cols[:8]),
        key=f"routes_tbl_{vessel_group_key}",
    )
    tbl = dff[table_cols].copy()
    tbl["DATE"] = tbl["DATE"].dt.date
    st.dataframe(tbl, use_container_width=True, height=420)


# -------------------------
# Main
# -------------------------
def main():
    require_login()
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Init session state
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "all_metrics" not in st.session_state:
        st.session_state.all_metrics = None
    if "vessel_group_key" not in st.session_state:
        st.session_state.vessel_group_key = "CAPE"

    # -------------------------
    # Sidebar: ONLY Logo / Data / Filters / Default page + Open page
    # -------------------------
    with st.sidebar:
        if st.session_state.get("authed"):
            name = st.session_state.get("user", {}).get("name", "User")
            st.caption(f"Signed in as: **{name}**")
            st.button("Logout", on_click=logout)

    with st.sidebar:
        # Logo
        try:
            st.image("SF_Logo.png", use_container_width=True)
        except Exception:
            # if logo missing or invalid, don't crash the app
            pass
        st.markdown("---")

        # Data (Upload)
        st.header("Data")
        uploaded = st.file_uploader("Upload your BDI DATA.xlsx", type=["xlsx"], key="uploader")

        # Optional advanced parsing controls INSIDE Data (collapsed)
        with st.expander("Advanced (Excel parsing)", expanded=False):
            sheet_name = st.text_input("Sheet name", value=DEFAULT_SHEET, key="sheet_name")
            auto_header = st.checkbox("Auto-detect header row", value=True, key="auto_header")
            header_row_input = st.number_input(
                "Header row (0-based, if not auto)",
                min_value=0,
                max_value=80,
                value=1,
                step=1,
                disabled=auto_header,
                key="header_row_input",
            )

        # Filters (Quick range) - enabled only AFTER data loaded
        st.header("Filters (Quick range)")
        quick = st.selectbox(
            "Time",
            ["Past Week", "Past Month", "Past 3 Months", "Past 6 Months", "Past Year", "Past 2 Years", "MTD", "YTD", "All"],
            index=0,
            disabled=not st.session_state.data_loaded,
            key="quick_range",
        )

        # Default page + Open page
        st.header("Default page")
        page = st.selectbox(
            "Choose page",
            ["Home", "Index", "TC Avg", "Vessel Group"],
            index=0,
            key="page_pick",
        )

        go = st.button("Open page", use_container_width=True, key="open_page_btn")

    # -------------------------
    # Route change only when click Open page
    # Also: Load Excel only when click Open page (and uploaded exists)
    # -------------------------
    if go:
        st.session_state.active_page = page

        # Only load data if uploaded exists
        if uploaded is None:
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.all_metrics = None
        else:
            hdr = None if st.session_state.auto_header else int(st.session_state.header_row_input)
            try:
                df, raw_preview = load_excel(uploaded, sheet_name=st.session_state.sheet_name, header_row=hdr)
                df = ensure_date(df, raw_preview).sort_values("DATE").reset_index(drop=True)

                if df.empty:
                    raise ValueError("No data rows after parsing.")

                st.session_state.df = df
                st.session_state.all_metrics = [c for c in df.columns if c != "DATE"]
                st.session_state.data_loaded = True

            except Exception as e:
                st.session_state.data_loaded = False
                st.session_state.df = None
                st.session_state.all_metrics = None
                st.warning("⚠️ Failed to load data. Please upload a valid .xlsx and check sheet/header settings.")
                st.caption(f"Debug: {type(e).__name__}: {e}")

    # -------------------------
    # Always render HOME shell (only Title + Quick View)
    # If no data loaded, don't crash; just show prompt.
    # -------------------------
    if not st.session_state.data_loaded or st.session_state.df is None:
        render_home(None, None)
        return
    
    df = st.session_state.df
    all_metrics = st.session_state.all_metrics

    # Apply quick range (end always = last data date)
    min_date = df["DATE"].min().date()
    max_date = df["DATE"].max().date()
    end_default = pd.Timestamp(max_date)
    min_ts = pd.Timestamp(min_date)

    start_ts = quick_range_start(quick, end_default, min_ts)
    start_date = max(start_ts.date(), min_date)
    end_date = max_date

    dff = df[(df["DATE"].dt.date >= start_date) & (df["DATE"].dt.date <= end_date)].copy()
    if dff.empty:
        # still keep home clean
        render_home(None, None)
        st.info("No data in selected range. Try a broader quick range.")
        return

    # -------------------------
    # Render active page
    # -------------------------
    active = st.session_state.active_page
    current_page = st.session_state.get("active_page", "Home")  
    render_contact_button(current_page=current_page)


    if active == "Home":
        render_home(dff, all_metrics)

    elif active == "Index":
        render_index_page(dff, all_metrics)

    elif active == "TC Avg":
        render_tc_page(dff, all_metrics)

    elif active == "Vessel Group":
        # default vessel group key stored in session state
        render_vessel_group_page(dff, st.session_state.vessel_group_key)


if __name__ == "__main__":
    main()
