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

from dotenv import load_dotenv
import os
load_dotenv() 
import json
import requests


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


import os, requests

FEEDBACK_WEBHOOK_URL = os.getenv("FEEDBACK_WEBHOOK_URL", "")
FEEDBACK_SECRET = os.getenv("FEEDBACK_SECRET", "")

def post_feedback_to_google(row: dict) -> None:
    r = requests.post(
        FEEDBACK_WEBHOOK_URL,
        params={"key": FEEDBACK_SECRET},
        json=row,
        timeout=15,
    )
    out = r.json()
    if not out.get("ok"):
        raise RuntimeError(out.get("error", "Unknown webhook error"))


def save_feedback(
    user: dict | None,
    page: str,
    subject: str,
    message: str,
    uploaded_files: list,
) -> str:
    """
    Save feedback to Google Sheet via webhook.
    Attachments are ignored (per your requirement).
    Return feedback_id.
    """
    feedback_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

    row = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "username": (user or {}).get("username", ""),
        "display_name": (user or {}).get("name", ""),
        "page": page,
        "subject": (subject or "").strip(),
        "message": (message or "").strip(),
        "attachments": "",  # ✅ 保留字段但不存附件
    }

    # ✅ 写入 Google Sheet
    post_feedback_to_google(row)
    st.success(f"Thanks! Feedback received and logged to Google Sheet.")


    return feedback_id




def render_contact_button(current_page: str):
    """
    Right-top Contact popover (no st.modal; stable on Streamlit)
    """
    # 把按钮放在右上角：用 columns 把它挤到右边
    _, col_right = st.columns([0.82, 0.18])
    with col_right:
        with st.popover("Contact", use_container_width=True):
            st.write("Send feedback to the dashboard owner (Rachel).")

            user = st.session_state.get("user", None)
            if user:
                st.caption(f"Signed in as: **{user.get('name','')}**")

            with st.form(key=f"contact_form_{current_page}"):
                subject = st.text_input(
                    "Subject",
                    placeholder="e.g., Bug report / Feature request / Data issue",
                    key=f"contact_subject_{current_page}",
                )
                message = st.text_area(
                    "Message",
                    height=160,
                    placeholder="Describe what you saw and what you expected...",
                    key=f"contact_msg_{current_page}",
                )
                files = st.file_uploader(
                    "Optional: attach screenshots/files",
                    accept_multiple_files=True,
                    key=f"contact_files_{current_page}",
                )

                submitted = st.form_submit_button("Submit", type="primary")

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
    s = str(name)

    # normalize various dashes to normal hyphen "-"
    s = s.replace("–", "-").replace("—", "-").replace("-", "-").replace("−", "-")

    # normalize whitespace
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
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
def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    合并重复列名：
    - 支持 'P5-82', 'P5-82 (2)', 'P5-82.1' 等
    合并规则：每行从左到右取第一个非空值
    """
    base_to_cols = {}
    for c in df.columns:
        sc = str(c)
        # 去掉 pandas 的 .1/.2 后缀；也去掉 ' (2)' 这种后缀
        base = re.sub(r"(\s\(\d+\)|\.\d+)$", "", sc).strip()
        base_to_cols.setdefault(base, []).append(c)

    out = df.copy()
    for base, cols in base_to_cols.items():
        if len(cols) <= 1:
            continue
        merged = out[cols].bfill(axis=1).iloc[:, 0]
        out[base] = merged
        drop_cols = [c for c in cols if c != base]
        out = out.drop(columns=drop_cols)

    return out
def _nearest_date_options(df: pd.DataFrame) -> list:
    """给 selectbox 用：返回 df 中可选日期（date 类型）"""
    if df is None or df.empty or "DATE" not in df.columns:
        return []
    return sorted(pd.to_datetime(df["DATE"]).dt.date.unique().tolist())

def _get_value_by_date(df: pd.DataFrame, pick_date, col: str):
    """严格按日期取值（同一天多行时取最后一行）"""
    if df is None or df.empty:
        return np.nan
    if "DATE" not in df.columns or col not in df.columns:
        return np.nan
    tmp = df.copy()
    tmp["D"] = pd.to_datetime(tmp["DATE"]).dt.date
    sub = tmp[tmp["D"] == pick_date]
    if sub.empty:
        return np.nan
    return sub.iloc[-1][col]

def fit_poly_models(x: pd.Series, y: pd.Series):
    """
    复刻你Excel里的三种模型：
    - Linear: y = a*x + b
    - Quadratic: y = a*x^2 + b*x + c
    - Cubic: y = a*x^3 + b*x^2 + c*x + d
    返回：coeff dict + 便于显示的函数
    """
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df_xy) < 10:
        return None  # 数据太少就不拟合

    xv = df_xy["x"].astype(float).to_numpy()
    yv = df_xy["y"].astype(float).to_numpy()

    out = {}
    for deg in [1, 2, 3]:
        coef = np.polyfit(xv, yv, deg)
        out[deg] = coef  # numpy poly coef: highest power first
    return out

def predict_with_poly(coef: np.ndarray, x_val: float) -> float:
    p = np.poly1d(coef)
    return float(p(x_val))

def _fmt_2(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return "—"
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "—"


import io
import requests

@st.cache_data(ttl=300)  # 5分钟缓存，避免每次刷新都打Google
def load_google_sheet_as_df(sheet_id: str, tab_name: str) -> pd.DataFrame:
    # Public Google Sheet -> CSV export
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_name}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [_clean_col_name(c) for c in df.columns]

    # detect + parse DATE
    date_col = detect_date_col(df.columns.tolist())
    if date_col is None:
        raise ValueError("No DATE column found in Google Sheet.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    if date_col != "DATE":
        df = df.rename(columns={date_col: "DATE"})

    # numeric conversion
    for c in df.columns:
        if c == "DATE":
            continue

        s = df[c]

        if s.dtype == "object":
            s = s.astype(str)

        # 常见空值符号
            s = s.replace({"": np.nan, "—": np.nan, "–": np.nan, "N/A": np.nan, "na": np.nan})

        # 去 NBSP & 空格
            s = s.str.replace("\u00A0", " ", regex=False).str.strip()

        # 去千分位逗号
            s = s.str.replace(",", "", regex=False)

        # ✅ 关键：去掉所有“非数字/非小数点/非负号”的字符（比如 $、USD、/day）
            s = s.str.replace(r"[^0-9\.\-]", "", regex=True)

        # 清理成空字符串的
            s = s.replace({"": np.nan, "-": np.nan})

        df[c] = pd.to_numeric(s, errors="coerce")

    df = coalesce_duplicate_columns(df)
    return df.reset_index(drop=True)

import io
import urllib.parse
import requests


@st.cache_data(ttl=300)
def load_google_sheet_raw(sheet_id: str, tab_name: str) -> pd.DataFrame:
    # URL encode sheet/tab name (handles spaces like "PMX 5TC")
    tab_q = urllib.parse.quote(tab_name, safe="")

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_q}"

    r = requests.get(url, timeout=30)
    # helpful debug if Google returns HTML error page
    if r.status_code != 200:
        raise RuntimeError(f"Google Sheet HTTP {r.status_code}: {r.text[:200]}")

    # If not CSV, you'll see HTML here (permission/redirect)
    ct = (r.headers.get("content-type") or "").lower()
    if "text/csv" not in ct and "application/vnd.ms-excel" not in ct:
        # still try parse, but show first chars for debugging
        raise RuntimeError(f"Unexpected response type: {ct}. Body head: {r.text[:200]}")

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [_clean_col_name(c) for c in df.columns]

    date_col = detect_date_col(df.columns.tolist())
    if date_col is None:
        raise ValueError(f"No DATE column found in Google Sheet tab: {tab_name}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if date_col != "DATE":
        df = df.rename(columns={date_col: "DATE"})

    for c in df.columns:
        if c == "DATE":
            continue
        s = df[c]
        if s.dtype == "object":
            s = s.astype(str)
            s = s.replace({"": np.nan, "—": np.nan, "–": np.nan, "N/A": np.nan, "na": np.nan})
            s = s.str.replace("\u00A0", " ", regex=False).str.strip()
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
            s = s.replace({"": np.nan, "-": np.nan})
        df[c] = pd.to_numeric(s, errors="coerce")

    df = coalesce_duplicate_columns(df)
    return df.reset_index(drop=True)

FFA_SHEET_ID = "1ma1-ZyBYVhzAUG51yUh0uSdwDl3AbhtxPyJeu8LdjvM"
FFA_TABS = ["PMX 5TC", "PMX 4TC"]



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
    # numeric (robust): remove commas/spaces in strings then to_numeric
    for c in df.columns:
        if c == "DATE":
            continue

        s = df[c]
    
    # 只有 object/string 才需要清洗
        if s.dtype == "object":
            s = (
                s.astype(str)
                .str.replace(",", "", regex=False)     # 去掉千分位逗号
                .str.replace("\u00A0", " ", regex=False)  # NBSP
                .str.strip()
            )
        # 空字符串转成 NaN（否则 to_numeric 会变成 NaN 也行，但这里更干净）
            s = s.replace({"": np.nan, "—": np.nan, "-": np.nan})

        df[c] = pd.to_numeric(s, errors="coerce")
    df = coalesce_duplicate_columns(df)
    
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

def moving_average(df: pd.DataFrame, col: str, window: int = 20) -> pd.Series:
    """
    Simple moving average (SMA)
    """
    return df[col].rolling(window).mean()


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

def render_seasonality_compare(
    df: pd.DataFrame,
    col: str,
    default_years: int = 3,
    title_prefix: str = "Seasonality",
    key_prefix: str = "season",
):
    """
    Seasonality compare:
    - x: Month (Jan..Dec)
    - y: monthly average value
    - line: Year
    Years selectable.
    """
    if df is None or df.empty or col not in df.columns:
        st.info("No data available for seasonality.")
        return

    tmp = df[["DATE", col]].dropna().copy()
    if tmp.empty:
        st.info("No valid data for seasonality.")
        return

    tmp["YEAR"] = tmp["DATE"].dt.year
    tmp["MONTH"] = tmp["DATE"].dt.month

    available_years = sorted(tmp["YEAR"].unique().tolist())
    if not available_years:
        st.info("No years found.")
        return

    # 默认选最近 N 年
    default_sel = available_years[-default_years:] if len(available_years) >= default_years else available_years

    show = st.checkbox(
        "Show seasonality comparison (by month)",
        value=True,
        key=f"{key_prefix}_show_{col}",
    )
    if not show:
        return

    years_sel = st.multiselect(
        "Choose years to compare",
        options=available_years,
        default=default_sel,
        key=f"{key_prefix}_years_{col}",
    )
    if not years_sel:
        st.info("Please select at least one year.")
        return

    tmp = tmp[tmp["YEAR"].isin(years_sel)].copy()

    # monthly mean
    agg = (
        tmp.groupby(["YEAR", "MONTH"], as_index=False)[col]
        .mean()
        .rename(columns={col: "Value"})
    )

    # Month labels + ensure order Jan..Dec
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    agg["Month"] = agg["MONTH"].apply(lambda m: month_names[m-1])

    fig = px.line(
        agg,
        x="Month",
        y="Value",
        color="YEAR",
        markers=True,
        category_orders={"Month": month_names},
        title=f"{title_prefix}: {col} (Monthly Avg by Year)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Year")
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
    if quick == "Past 3 Years":
        return end_date - pd.DateOffset(years=3)
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

import pandas as pd
import streamlit as st

def _fmt_num(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        # 你 routes 有的像 $/tonne, 有的像 index，先统一显示到整数
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def render_markets_snapshot(dff: pd.DataFrame, vessel_groups: dict, vessel_labels: dict):

    """
    在 Home 页面展示类似 Breamar 的 Markets Snapshot:
    - 每个船型一列
    - 列内展示 routes 的最新值 + (可选) 日变化
    """
    if dff is None or dff.empty:
        return

    latest = dff.iloc[-1]
    col_map = { _clean_col_name(c): c for c in dff.columns }
    asof = pd.to_datetime(latest["DATE"]).date() if "DATE" in dff.columns else None
    prev = dff.iloc[-2] if len(dff) >= 2 else None

    st.markdown("#### Routes")

    keys_in_order = ["CAPE", "KMX_82", "PMX_74", "SMX_TESS_63", "HANDY_38"]
    cols = st.columns(len(keys_in_order), gap="small")

    def _fmt_2(v):
        """只用于 Markets snapshot：千分位 + 两位小数；缺失显示 —"""
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            return "—"
        try:
            return f"{float(v):,.2f}"
        except Exception:
            return "—"

    def _fmt_chg_2(v):
        """变化值：带符号 + 两位小数；缺失空"""
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            return ""
        try:
            return f"{float(v):+,.2f}"
        except Exception:
            return ""

    for ui_col, gkey in zip(cols, keys_in_order):
        label = vessel_labels.get(gkey, gkey)
        routes = vessel_groups.get(gkey, [])

        with ui_col:
            st.markdown(f"**{label}**")

            rows = []
            for r in routes:
                real_col = col_map.get(_clean_col_name(r), None)
                v = latest.get(real_col, None) if real_col else None


                dv = None
                if prev is not None:
                    pv = prev.get(real_col, None) if (prev is not None and real_col) else None
                    if pd.notna(v) and pd.notna(pv):
                        try:
                            dv = float(v) - float(pv)
                        except Exception:
                            dv = None

                rows.append(
                    {
                        "Route": r,
                        "Value": _fmt_2(v),          # ✅ 两位小数
                        "Chg": _fmt_chg_2(dv),       # ✅ 两位小数（带符号）
                        "_chg_val": 0 if dv is None else dv,
                    }
                )

            df_show = pd.DataFrame(rows)

            def _style_chg(val):
                # val 现在是字符串 "+1,234.56"，我们需要转回 float 来上色
                try:
                    v = float(str(val).replace(",", ""))
                except Exception:
                    return ""
                if v > 0:
                    return "color: #16a34a; font-weight: 600;"
                if v < 0:
                    return "color: #dc2626; font-weight: 600;"
                return "color: #6b7280;"

            styled = df_show[["Route", "Value", "Chg"]].style.applymap(_style_chg, subset=["Chg"])

            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=min(36 * (len(routes) + 1), 420),
            )

def build_markets_snapshot(latest: pd.Series, prev: pd.Series | None,
                           vessel_groups: dict, vessel_labels: dict) -> dict:
    out = {}
    # ✅ map cleaned -> real column
    col_map = { _clean_col_name(c): c for c in latest.index }

    keys_in_order = ["CAPE", "KMX_82", "PMX_74", "SMX_TESS_63", "HANDY_38"]
    for gkey in keys_in_order:
        label = vessel_labels.get(gkey, gkey)
        routes = vessel_groups.get(gkey, [])
        route_map = {}

        for r in routes:
            real_col = col_map.get(_clean_col_name(r))
            if not real_col:
                continue

            v = latest.get(real_col)
            if v is None or pd.isna(v):
                continue

            dv = None
            if prev is not None:
                pv = prev.get(real_col)
                if pv is not None and pd.notna(pv):
                    try:
                        dv = float(v) - float(pv)
                    except Exception:
                        dv = None

            route_map[r] = {
                "value": round(float(v), 2),
                "chg": None if dv is None else round(float(dv), 2),
            }

        out[label] = route_map



    return out

# -------------------------
# Pages
# -------------------------
def render_home(dff: pd.DataFrame | None, all_metrics: list[str] | None):
    st.title("BDI Dashboard")
    st.subheader("Quick view (latest)")
    
    if dff is None or dff.empty:
        st.info("Please upload an Excel file and click Open page to load data.")
        return

    tmp = dff.dropna(subset=["DATE"]).sort_values("DATE")
    if tmp.empty:
        st.warning("No valid DATE rows in the selected range.")
        return

    latest_row = tmp.iloc[-1]
    latest_date = pd.to_datetime(latest_row["DATE"]).date()
    st.caption(f"As of: **{latest_date}**")


    if dff is None or dff.empty:
        st.info("📄 Please upload **BDI DATA.xlsx** from the sidebar, then click **Open page**.")
        return

    latest = dff.iloc[-1]
    prev = dff.iloc[-2] if len(dff) >= 2 else None
    from ai.gemini import ask_gemini

    st.markdown("---")
    st.subheader("🤖 AI Market Summary")

    q = st.text_area(
        "Ask AI about the current market in this selected time range",
        placeholder="e.g. Summarize today’s market tone in 3 bullet points.",
        key="home_ai_q",
    )

    if st.button("Generate AI Summary", key="home_ai_btn"):
        if dff is None or dff.empty:
            st.warning("No data in current range.")
        else:
            latest = dff.iloc[-1]
            prev = dff.iloc[-2] if len(dff) >= 2 else None
            asof = str(pd.to_datetime(latest["DATE"]).date())

        # 1) KPI snapshot
            kpi_snapshot = {
                "As of": asof,
                "BDI": latest.get("BDI"),
                "BPI": latest.get("BPI"),
                "BCI": latest.get("BCI"),
                "BSI": latest.get("BSI"),
            }

        # 2) Markets snapshot (routes)
            markets_snapshot = build_markets_snapshot(
                latest=latest,
                prev=prev,
                vessel_groups=VESSEL_GROUPS,
                vessel_labels=VESSEL_LABELS,
            )
            # 1.5) TC Average snapshot
            tc_cols = [
                "BCI 5TC AV",
                "BPI 82 TC AV",
                "BPI 74 TC AV",
                "BSI 63 TC AV",  
                "BHSI 38 TC AV",
                "BHSI 28 TC AV",
                ]
            tc_snapshot = {}
            for c in tc_cols:
                if c in dff.columns:
                    v = latest.get(c)
                    pv = prev.get(c) if prev is not None else None
                    tc_snapshot[c] = {
                        "value": None if pd.isna(v) else round(float(v), 2),
                        "chg": None if (prev is None or pd.isna(v) or pd.isna(pv)) else round(float(v - pv), 2),
                    }


        # 3) 合并为一个 snapshot（喂给 Gemini）
            snapshot = {
                "kpi": kpi_snapshot,
                "markets": markets_snapshot,
                "tc_avg": tc_snapshot
            }

            prompt = f"""
    You are a dry bulk shipping market analyst.
    Use simple business language.

    Data snapshot (latest in selected range):
    {snapshot}

    User question:
    {q if q.strip() else "Give a concise market summary in 3 bullet points."}

    Rules:
    - Be concise (3-6 bullet points)
    - No fake numbers (only interpret what is given)
    - Describe: overall tone + which segment is stronger/weaker (Capesize/Kamsarmax/Panamax/Supramax/Handy)
    - Also interpret TC Average (tc_avg): mention which TC segment is rising/failing and notable movers
    - Use 'value' and 'chg' fields only; do not invent numbers
    - If changes (chg) are present, mention notable movers (largest rises/falls)
    - Add 1 risk note (volatility / event risk) but do not invent events
    """

            with st.spinner("Gemini is thinking..."):
                ans = ask_gemini(prompt)

            st.markdown(ans)


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
    render_markets_snapshot(dff, VESSEL_GROUPS, VESSEL_LABELS)
    # -------------------------
    # TC Average (latest) metrics row(s)
    # -------------------------
    st.markdown("##### TC Average (latest)")

    tc_metric_cols = [
        "BCI 5TC AV",
        "BPI 82 TC AV",
        "BPI 74 TC AV",
        "BSI 63 TC AV",
        "BHSI 38 TC AV",
        "BHSI 28 TC AV",
    ]
    tc_metric_cols = [c for c in tc_metric_cols if c in dff.columns]

    if tc_metric_cols:
        # 每行最多放 4 个，和上面 KPI 的视觉一致
        per_row = 4
        for r in range(0, len(tc_metric_cols), per_row):
            row_cols = tc_metric_cols[r : r + per_row]
            ui_cols = st.columns(len(row_cols))

            for j, name in enumerate(row_cols):
                val = latest.get(name, np.nan)

                if pd.isna(val):
                    ui_cols[j].metric(name, "—")
                else:
                    # delta vs previous day (same logic as KPI)
                    if prev is None or pd.isna(prev.get(name, np.nan)):
                        ui_cols[j].metric(name, f"{val:,.0f}")
                    else:
                        ui_cols[j].metric(name, f"{val:,.0f}", f"{(val - prev[name]):,.0f}")
    else:
        st.caption("No TC Average columns found in the uploaded data.")
            

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
        st.markdown("---")
    st.subheader("Seasonality (Month-by-Month)")

    season_base = st.selectbox(
        "Choose a series for seasonality",
        options=all_metrics,
        index=all_metrics.index("BDI") if "BDI" in all_metrics else 0,
        key="idx_season_base",
    )

    render_seasonality_compare(
        dff,
        col=season_base,
        default_years=3,
        title_prefix="Index Seasonality",
        key_prefix="idx_season",
    )


    # ---- Analytics controls (moved INSIDE the page) ----
    st.subheader("Analytics (choose a base series)")

    base_series = st.selectbox(
        "Base series",
        options=all_metrics,
        index=all_metrics.index("BDI") if "BDI" in all_metrics else 0,
        key="idx_base",
    )

    show_ma = st.checkbox("Show moving average", value=True, key="idx_ma_on")

    ma_window = st.number_input(
        "Moving average window (days)",
        min_value=5,
        max_value=120,
        value=20,
        step=1,
        disabled=not show_ma,
        key="idx_ma_win",
    )

    show_yoy_mom = st.checkbox("Show YoY / MoM change", value=True, key="idx_yoymom_on")

    analytics_df = dff[["DATE", base_series]].dropna()
    analytics_df = add_returns_and_changes(analytics_df, base_series)

    c1, c2 = st.columns(2)
    with c1:
        if show_ma:
            analytics_df["MA"] = moving_average(analytics_df, base_series, window=int(ma_window))
            plot_single(analytics_df, "MA", f"Moving Average ({ma_window} days)")
    with c2:
        if show_yoy_mom:
            mom = f"{base_series}_mchg"
            yoy = f"{base_series}_ychg"
            tmp = analytics_df.rename(columns={mom: "MoM Change", yoy: "YoY Change"})
            plot_multi_line(tmp, ["MoM Change", "YoY Change"], "MoM / YoY change")


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
        st.markdown("---")
    st.subheader("Seasonality (Month-by-Month)")

    season_tc = st.selectbox(
        "Choose a TC series for seasonality",
        options=selected_tc if selected_tc else all_metrics,
        index=0,
        key="tc_season_base",
    )

    render_seasonality_compare(
        dff,
        col=season_tc,
        default_years=3,
        title_prefix="TC Avg Seasonality",
        key_prefix="tc_season",
    )
def render_ffa_page(
    index_df_full: pd.DataFrame,   # 用全量历史拟合回归（不要用 dff 截断）
    dff_filtered: pd.DataFrame,    # 用筛选范围做 ratio daily 更直观
    all_metrics: list[str],
):
    st.header("FFA")

    # -------------------------
    # 0) 选择 PMX 5TC / PMX 4TC (Google sheet tab)
    # -------------------------
    c0a, c0b = st.columns([1, 2])
    with c0a:
        ffa_tab = st.selectbox("Choose FFA sheet", options=FFA_TABS, index=0, key="ffa_tab_pick")
    with c0b:
        st.caption(f"Source: Drybulk Forward Closing Number ({ffa_tab})")

    try:
        ffa_df = load_google_sheet_raw(FFA_SHEET_ID, ffa_tab)
    except Exception as e:
        st.error("Failed to load FFA Google Sheet.")
        st.caption(f"{type(e).__name__}: {e}")
        return

    # 合约列（M+/Q+/Y+）
    contract_cols = [c for c in ffa_df.columns if c != "DATE"]
    if not contract_cols:
        st.warning("No contract columns found in the FFA sheet.")
        return

    # 为了后面下拉框
    ffa_dates = _nearest_date_options(ffa_df)

    st.markdown("---")

    # =========================================================
    # 1) CALCULATION: 输入 FFA -> 预测未来 index value
    # =========================================================
    st.subheader("1) Calculation (FFA ➜ Index Forecast)")

    # 选择要预测的“航线/指标”（来自你 index 数据里现有列）
    # 这里给你更灵活：可选任意 metric（含 routes / index / tc）
    target_col = st.selectbox(
        "Choose target index/route to forecast (y)",
        options=all_metrics,
        index=all_metrics.index("P5-82") if "P5-82" in all_metrics else 0,
        key="ffa_calc_target",
    )

    # x-series：按 PMX 5TC/4TC 给默认映射；如果不存在就让用户自己选
    default_x = None
    if "5TC" in ffa_tab:
        default_x = "BPI 82 TC AV" if "BPI 82 TC AV" in index_df_full.columns else None
    else:
        default_x = "BPI 74 TC AV" if "BPI 74 TC AV" in index_df_full.columns else None

    x_candidates = [c for c in all_metrics if "TC AV" in c.upper() or "5TC AV" in c.upper()] or all_metrics

    x_col = st.selectbox(
        "Choose base TC series for regression (x)",
        options=x_candidates,
        index=(x_candidates.index(default_x) if (default_x in x_candidates) else 0),
        key="ffa_calc_xcol",
    )

    # 选择一个日期 + 合约，从 forward curve 直接取 x_val（也可手动覆盖）
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        pick_date = st.selectbox("Pick Date (from FFA sheet)", options=ffa_dates, index=len(ffa_dates)-1, key="ffa_calc_date")
    with c2:
        pick_contract = st.selectbox("Pick Contract", options=contract_cols, index=0, key="ffa_calc_contract")

    auto_x = _get_value_by_date(ffa_df, pick_date, pick_contract)

    with c3:
        x_val = st.number_input(
            "FFA input (x) — you can override",
            value=(float(auto_x) if pd.notna(auto_x) else 0.0),
            step=10.0,
            key="ffa_calc_xval",
        )

    # 现场拟合回归（用全量 index_df_full）
    if x_col not in index_df_full.columns or target_col not in index_df_full.columns:
        st.warning("Selected x or y column not found in index dataset.")
        return

    coefs = fit_poly_models(index_df_full[x_col], index_df_full[target_col])
    if coefs is None:
        st.info("Not enough historical overlap data to fit regression.")
        return

    # 输出三种模型预测
    pred_linear = predict_with_poly(coefs[1], x_val)
    pred_quad   = predict_with_poly(coefs[2], x_val)
    pred_cubic  = predict_with_poly(coefs[3], x_val)

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("FFA input (x)", _fmt_2(x_val))
    o2.metric("Linear forecast", _fmt_2(pred_linear))
    o3.metric("Quadratic forecast", _fmt_2(pred_quad))
    o4.metric("Cubic forecast", _fmt_2(pred_cubic))

    with st.expander("Show fitted coefficients (debug)"):
        st.write("Linear (deg=1):", coefs[1])
        st.write("Quadratic (deg=2):", coefs[2])
        st.write("Cubic (deg=3):", coefs[3])

    st.markdown("---")

    # =========================================================
    # 2) 差值对比：两组 (Date + Contract) -> spread = 2 - 1
    # =========================================================
    st.subheader("2) Spread (two picks comparison)")

    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown("**Leg 1**")
        d1 = st.selectbox("Date (Leg 1)", options=ffa_dates, index=len(ffa_dates)-1, key="ffa_spread_d1")
        k1 = st.selectbox("Contract (Leg 1)", options=contract_cols, index=0, key="ffa_spread_k1")
        v1 = _get_value_by_date(ffa_df, d1, k1)
        st.caption(f"Value 1: **{_fmt_2(v1)}**")

    with sc2:
        st.markdown("**Leg 2**")
        d2 = st.selectbox("Date (Leg 2)", options=ffa_dates, index=len(ffa_dates)-1, key="ffa_spread_d2")
        k2 = st.selectbox("Contract (Leg 2)", options=contract_cols, index=1 if len(contract_cols) > 1 else 0, key="ffa_spread_k2")
        v2 = _get_value_by_date(ffa_df, d2, k2)
        st.caption(f"Value 2: **{_fmt_2(v2)}**")

    spread = np.nan
    if pd.notna(v1) and pd.notna(v2):
        spread = float(v2) - float(v1)

    # 上色显示
    if pd.isna(spread):
        st.metric("Spread (Leg2 - Leg1)", "—")
    else:
        color = "#16a34a" if spread > 0 else ("#dc2626" if spread < 0 else "#6b7280")
        st.markdown(
            f"**Spread (Leg2 - Leg1):** <span style='color:{color}; font-weight:700'>{_fmt_2(spread)}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================
    # 3) RATIO: daily ratio + yearly avg ratio
    # =========================================================
    st.subheader("3) Ratio (FFA / Index)")

    # 选一个合约 + 一个 index route
    rc1, rc2 = st.columns([1, 1])
    with rc1:
        ratio_contract = st.selectbox("Choose FFA contract", options=contract_cols, index=0, key="ffa_ratio_contract")
    with rc2:
        ratio_route = st.selectbox("Choose index/route (denominator)", options=all_metrics, index=all_metrics.index("P5-82") if "P5-82" in all_metrics else 0, key="ffa_ratio_route")

    # daily ratio：用你当前筛选范围 dff_filtered（更符合你的侧边栏 quick range）
    if ratio_route not in dff_filtered.columns:
        st.warning("Selected route not found in current index dataset.")
        return

    # 把 FFA 也按当前 quick range 对齐
    # dff_filtered 的 range 是 [start_date, end_date]，这里直接用 DATE 交集
    ffa_tmp = ffa_df[["DATE", ratio_contract]].dropna().copy()
    idx_tmp = dff_filtered[["DATE", ratio_route]].dropna().copy()

    merged = pd.merge(ffa_tmp, idx_tmp, on="DATE", how="inner")
    merged = merged.rename(columns={ratio_contract: "FFA", ratio_route: "INDEX"})
    merged["RATIO"] = merged["FFA"] / merged["INDEX"]

    if merged.empty:
        st.info("No overlapping dates between FFA sheet and selected index series in current range.")
        return

    # daily chart/table
    fig_ratio = px.line(merged, x="DATE", y="RATIO", title=f"Daily Ratio: {ratio_contract} / {ratio_route}")
    fig_ratio.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_ratio, use_container_width=True)

    show_cols = st.multiselect(
        "Daily table columns",
        options=["DATE", "FFA", "INDEX", "RATIO"],
        default=["DATE", "FFA", "INDEX", "RATIO"],
        key="ffa_ratio_tbl_cols",
    )
    tbl = merged[show_cols].copy()
    tbl["DATE"] = pd.to_datetime(tbl["DATE"]).dt.date
    st.dataframe(tbl.tail(200), use_container_width=True, height=420)

    # yearly avg ratio
    merged["YEAR"] = pd.to_datetime(merged["DATE"]).dt.year
    yearly = (
        merged.groupby("YEAR", as_index=False)
        .agg(AVG_FFA=("FFA", "mean"), AVG_INDEX=("INDEX", "mean"))
    )
    yearly["AVG_RATIO"] = yearly["AVG_FFA"] / yearly["AVG_INDEX"]

    st.markdown("**Yearly averages & ratio**")
    st.dataframe(yearly, use_container_width=True, height=280)

    # ---- Optional analytics (on any series) ----
    st.subheader("Extra Analytics (optional)")

    base_series = st.selectbox(
        "Base series",
        options=all_metrics,
        index=0,
        key="ffa_extra_base",
    )

    show_ma = st.checkbox("Show moving average", value=True, key="ffa_extra_ma_on")
    ma_window = st.number_input(
        "Moving average window (days)",
        min_value=5,
        max_value=120,
        value=20,
        step=1,
        disabled=not show_ma,
        key="ffa_extra_ma_win",
    )

    show_yoy_mom = st.checkbox("Show YoY / MoM change", value=True, key="ffa_extra_yoymom_on")

    tmp_df = dff_filtered[["DATE", base_series]].dropna().copy()
    tmp_df = add_returns_and_changes(tmp_df, base_series)

    c1, c2 = st.columns(2)
    with c1:
        if show_ma:
            tmp_df["MA"] = moving_average(tmp_df, base_series, window=int(ma_window))
            plot_single(tmp_df, "MA", f"Moving Average ({ma_window} days)")
    with c2:
        if show_yoy_mom:
            mom = f"{base_series}_mchg"
            yoy = f"{base_series}_ychg"
            tmp2 = tmp_df.rename(columns={mom: "MoM Change", yoy: "YoY Change"})
            plot_multi_line(tmp2, ["MoM Change", "YoY Change"], "MoM / YoY change")

    st.subheader("Table")
    table_cols = st.multiselect(
        "Table columns",
        options=["DATE"] + all_metrics,
        default=["DATE", base_series],
        key="ffa_tbl",
    )
    tbl = dff_filtered[table_cols].copy()
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
        st.markdown("---")
    st.subheader("Seasonality (Month-by-Month)")

    season_route = st.selectbox(
        "Choose one route for seasonality",
        options=group_cols,
        index=0,
        key=f"vg_season_base_{vessel_group_key}",
    )

    render_seasonality_compare(
        dff,
        col=season_route,
        default_years=3,
        title_prefix=f"{vessel_label} Seasonality",
        key_prefix=f"vg_season_{vessel_group_key}",
    )

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
# Auto-load from Google Sheet on first visit
# -------------------------

    if not st.session_state.get("data_loaded"):
        sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
        tab_name = os.getenv("GOOGLE_SHEET_TAB", "bdi index")

        if sheet_id:
            try:
                df0 = load_google_sheet_as_df(sheet_id, tab_name)
                if not df0.empty:
                    st.session_state.df = df0
                    st.session_state.all_metrics = [c for c in df0.columns if c != "DATE"]
                    st.session_state.data_loaded = True
            except Exception as e:
            # 不要让首页崩掉；只是提示
                st.session_state.data_loaded = False
                st.session_state.df = None
                st.session_state.all_metrics = None
                st.sidebar.warning("⚠️ Auto-load Google Sheet failed.")
                st.sidebar.caption(f"{type(e).__name__}: {e}")

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

        # Filters (Quick range) - enabled only AFTER data loaded
        st.header("Filters (Quick range)")
        quick = st.selectbox(
            "Time",
            ["Past Week", "Past Month", "Past 3 Months", "Past 6 Months", "Past Year", "Past 2 Years", "Past 3 Years",
            "MTD", "YTD", "All", "Custom"],
            index=0,
            disabled=not st.session_state.data_loaded,
            key="quick_range",
        )
        # ✅ NEW: only show when Custom
        custom_start = None
        custom_end = None
        if st.session_state.data_loaded and quick == "Custom":
    # 这里的 min/max 要用“全量 df”日期范围，所以先安全取 session_state.df
            _df0 = st.session_state.df
            _min_d = _df0["DATE"].min().date()
            _max_d = _df0["DATE"].max().date()

            c1, c2 = st.columns(2)
            with c1:
                custom_start = st.date_input(
                    "Start date",
                    value=_min_d,
                    min_value=_min_d,
                    max_value=_max_d,
                    key="custom_start",
                )
            with c2:
                custom_end = st.date_input(
                    "End date",
                    value=_max_d,
                    min_value=_min_d,
                    max_value=_max_d,
                    key="custom_end",
                )

        # Default page + Open page
        st.header("Default page")
        page = st.selectbox(
            "Choose page",
            ["Home", "Index", "TC Avg", "Vessel Group", "FFA"],
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

    # 如果没上传，就不动当前数据（继续用 Google Sheet auto-load 的）


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

    if quick == "Custom":
        # ✅ 用 sidebar 选择的日期
        start_date = custom_start if custom_start else min_date
        end_date = custom_end if custom_end else max_date

        # ✅ 防呆：start > end 就交换，或你也可以直接 st.sidebar.error
        if start_date > end_date:
            start_date, end_date = end_date, start_date
    else:
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
    elif active == "FFA":
    # 用全量 df 拟合 regression；用 dff 做 ratio 的日频对齐（符合 quick range）
        render_ffa_page(
            index_df_full=st.session_state.df,
            dff_filtered=dff,
            all_metrics=all_metrics,
        )



if __name__ == "__main__":
    main()
