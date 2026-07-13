import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# Font setup (English only - no Japanese font needed)
# ============================================================
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# Ticker Configuration
# ============================================================
# Group 1: Main trading instruments
GROUP1 = {
    "Nikkei Futures": {"symbol": "NKD=F",   "has_volume": True,  "risk_label": False},
    "SOX":            {"symbol": "^SOX",     "has_volume": True,  "risk_label": False},
    "Nasdaq Futures": {"symbol": "NQ=F",     "has_volume": True,  "risk_label": False},
    "USD/JPY":        {"symbol": "USDJPY=X", "has_volume": False, "risk_label": False},
}

# Group 2: Risk / macro indicators
GROUP2 = {
    "Nikkei 225":  {"symbol": "^N225",     "has_volume": True,  "risk_label": False},
    "VIX":         {"symbol": "^VIX",      "has_volume": False, "risk_label": True},
    "Nikkei VI":   {"symbol": "^NKVI.OS",  "has_volume": False, "risk_label": True},
    "US 10Y Yield":{"symbol": "^TNX",      "has_volume": False, "risk_label": False},
}

MA_SETTINGS = {
    "5m":  {"ma_short": 9,  "ma_long": 21, "rsi": True,  "prev_close": True},
    "30m": {"ma_short": 20, "ma_long": 50, "rsi": True,  "prev_close": True},
    "1d":  {"ma_short": 20, "ma_long": 60, "rsi": True,  "prev_close": True},
}

PERIOD_SETTINGS = {
    "5m":  "7d",
    "30m": "15d",
    "1d":  "3mo",  # shortened to avoid NKD=F rollover issues
}

# ============================================================
# Data Fetching
# ============================================================
@st.cache_data(ttl=300)
def fetch_data(symbol, interval, end_date=None):
    try:
        if interval == "1d" and end_date:
            end_dt   = pd.Timestamp(end_date) + timedelta(days=1)
            start_dt = end_dt - timedelta(days=365)
            df = yf.download(
                symbol,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True
            )
        else:
            df = yf.download(
                symbol,
                period=PERIOD_SETTINGS[interval],
                interval=interval,
                progress=False,
                auto_adjust=True
            )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in df.columns:
                return None

        if 'Volume' not in df.columns:
            df['Volume'] = 0.0

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        return df if not df.empty else None

    except Exception:
        return None

# ============================================================
# Indicator Calculations
# ============================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def get_prev_close(df):
    try:
        dates = df.index.normalize().unique()
        if len(dates) < 2:
            return None
        prev_day  = sorted(dates)[-2]
        prev_data = df[df.index.normalize() == prev_day]
        return float(prev_data['Close'].iloc[-1]) if not prev_data.empty else None
    except:
        return None

def calc_trend(df, ma_col):
    try:
        if ma_col not in df.columns:
            return "-"
        ma_vals = df[ma_col].dropna()
        if len(ma_vals) < 2:
            return "N/A"
        cur  = float(df['Close'].iloc[-1])
        ma_n = float(ma_vals.iloc[-1])
        ma_p = float(ma_vals.iloc[-2])
        if cur > ma_n and ma_n > ma_p:
            return "Uptrend ↑"
        elif cur < ma_n and ma_n < ma_p:
            return "Downtrend ↓"
        else:
            return "Sideways →"
    except:
        return "-"

def calc_range_position(df, days=20):
    try:
        recent = df['Close'].tail(days)
        hi, lo = float(recent.max()), float(recent.min())
        cur    = float(df['Close'].iloc[-1])
        if hi == lo:
            return 50.0, "Mid"
        pos   = (cur - lo) / (hi - lo) * 100
        label = "High zone" if pos >= 80 else ("Low zone" if pos < 40 else "Mid zone")
        return pos, label
    except:
        return None, "-"

# ============================================================
# Single Chart Drawing
# ============================================================
def draw_chart(fig, outer_gs, row, col, name, info, df, interval, settings):
    has_volume = info["has_volume"]
    is_risk    = info["risk_label"]
    show_rsi   = settings["rsi"]
    show_prev  = settings["prev_close"]
    ma_short   = settings["ma_short"]
    ma_long    = settings["ma_long"]

    # Layout: 3-panel (RSI) or 2-panel
    if show_rsi:
        hr    = [3, 1, 1] if has_volume else [4, 1, 0.01]
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_gs[row, col], hspace=0.06, height_ratios=hr)
        ax_main = fig.add_subplot(inner[0])
        ax_rsi  = fig.add_subplot(inner[1], sharex=ax_main)
        ax_vol  = fig.add_subplot(inner[2], sharex=ax_main)
    else:
        hr    = [4, 1] if has_volume else [4, 0.01]
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row, col], hspace=0.06, height_ratios=hr)
        ax_main = fig.add_subplot(inner[0])
        ax_rsi  = None
        ax_vol  = fig.add_subplot(inner[1], sharex=ax_main)

    # No data
    if df is None or df.empty:
        ax_main.set_facecolor('#1a1a2e')
        ax_main.text(0.5, 0.5, f"{name}\nNo Data / Error",
                     ha='center', va='center', fontsize=16,
                     color='#ff5252', transform=ax_main.transAxes, fontweight='bold')
        for ax in [ax_rsi, ax_vol]:
            if ax: ax.set_visible(False)
        return

    # MA calculation
    df = df.copy()
    df[f'MA{ma_short}'] = df['Close'].rolling(ma_short).mean()
    df[f'MA{ma_long}']  = df['Close'].rolling(ma_long).mean()
    if interval == "1d":
        df['MA5']   = df['Close'].rolling(5).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
    if show_rsi:
        df['RSI'] = calc_rsi(df['Close'])

    # Stats
    current        = float(df['Close'].iloc[-1])
    prev_close_val = get_prev_close(df)
    pct_chg        = ((current - prev_close_val) / prev_close_val * 100) if prev_close_val else 0
    trend          = calc_trend(df, f'MA{ma_long}')
    rng_pos, rng_label = calc_range_position(df)

    risk_str = ""
    if is_risk:
        risk_str = "  [Risk-OFF ▲]" if pct_chg > 0 else "  [Risk-ON ▼]"

    # Price formatting
    if current < 10:
        price_fmt = f"{current:,.3f}"
    elif current > 10000:
        price_fmt = f"{current:,.1f}"
    else:
        price_fmt = f"{current:,.2f}"

    # ============================================================
    # Candle chart
    # ============================================================
    idx    = np.arange(len(df))
    opens  = df['Open'].values
    closes = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    width  = 0.6

    ax_main.set_facecolor('#0d0d1a')
    for i in range(len(df)):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        if any(np.isnan(v) for v in [o, c, h, l]):
            continue
        color = '#ef5350' if c >= o else '#42a5f5'
        ax_main.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        body_h = max(abs(c - o), (h - l) * 0.005)
        rect   = Rectangle((i - width/2, min(o, c)), width, body_h,
                            facecolor=color, edgecolor=color, linewidth=0.3, zorder=2)
        ax_main.add_patch(rect)

    # MA lines
    ma_list = [(f'MA{ma_short}', '#ffeb3b', 1.3),
               (f'MA{ma_long}',  '#ff9800', 1.3)]
    if interval == "1d":
        ma_list += [('MA5', '#00e5ff', 1.1), ('MA200', '#e040fb', 1.1)]

    for col_name, color, lw in ma_list:
        if col_name in df.columns:
            ax_main.plot(idx, df[col_name].values, color=color,
                         linewidth=lw, label=col_name, alpha=0.9, zorder=3)

    # Prev close line
    if show_prev and prev_close_val:
        ax_main.axhline(prev_close_val, color='#00e676', linewidth=1.2,
                        linestyle='--', alpha=0.85,
                        label=f'Prev {price_fmt}', zorder=4)

    # High / Low annotation
    try:
        hi_i = int(np.nanargmax(highs))
        lo_i = int(np.nanargmin(lows))
        ax_main.annotate(f'H: {highs[hi_i]:,.1f}', xy=(hi_i, highs[hi_i]),
                         fontsize=9, color='#ff8a80', ha='center', va='bottom', fontweight='bold')
        ax_main.annotate(f'L: {lows[lo_i]:,.1f}',  xy=(lo_i, lows[lo_i]),
                         fontsize=9, color='#82b1ff', ha='center', va='top',  fontweight='bold')
    except:
        pass

    # ---- Title (large, prominent) ----
    color_pct = '#ff5252' if pct_chg >= 0 else '#40c4ff'
    title_name  = f"{name}"
    title_price = f"{price_fmt}  ({pct_chg:+.2f}%){risk_str}"
    title_info  = (f"Trend: {trend}  |  Range: {rng_label} ({rng_pos:.0f}%)"
                   if rng_pos is not None else f"Trend: {trend}")

    range_str = f"  |  {rng_label} ({rng_pos:.0f}%)" if rng_pos is not None else ""
    ax_main.set_title(
        f"{title_name}   {price_fmt}  ({pct_chg:+.2f}%){risk_str}\n"
        f"{trend}{range_str}",
        fontsize=12, color='#ffffff', pad=5,
        loc='left', fontweight='bold',
        backgroundcolor='#0d0d1a'
    )

    # Axes styling
    ax_main.tick_params(colors='#9e9e9e', labelsize=8)
    ax_main.yaxis.tick_right()
    for sp in ax_main.spines.values():
        sp.set_color('#2a2a4a')
    ax_main.grid(color='#1e1e3a', linewidth=0.5, alpha=0.8)
    ax_main.legend(fontsize=10, loc='upper left', facecolor='#0d0d1a',
                   labelcolor='#e0e0e0', framealpha=0.8, ncol=2)
    ax_main.set_xlim(-1, len(df))
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---- RSI panel ----
    if ax_rsi:
        if show_rsi and 'RSI' in df.columns:
            ax_rsi.set_facecolor('#0d0d1a')
            rv = df['RSI'].values
            ax_rsi.plot(idx, rv, color='#ce93d8', linewidth=1.0)
            ax_rsi.axhline(70, color='#ef9a9a', linewidth=0.7, linestyle='--')
            ax_rsi.axhline(30, color='#90caf9', linewidth=0.7, linestyle='--')
            ax_rsi.fill_between(idx, rv, 70, where=(rv >= 70), alpha=0.25, color='#ef5350')
            ax_rsi.fill_between(idx, rv, 30, where=(rv <= 30), alpha=0.25, color='#42a5f5')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_yticks([30, 70])
            ax_rsi.set_ylabel('RSI', fontsize=8, color='#9e9e9e')
            ax_rsi.tick_params(colors='#9e9e9e', labelsize=7)
            ax_rsi.yaxis.tick_right()
            for sp in ax_rsi.spines.values():
                sp.set_color('#2a2a4a')
            ax_rsi.grid(color='#1e1e3a', linewidth=0.4, alpha=0.6)
        else:
            ax_rsi.set_visible(False)
        plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # X-axis tick helper
    def set_xticks(ax_t):
        n    = len(df)
        step = max(1, n // 6)
        pos  = list(range(0, n, step))
        labs = []
        for p in pos:
            try:
                t = df.index[p]
                labs.append(t.strftime('%m/%d') if interval == "1d"
                             else t.strftime('%m/%d\n%H:%M'))
            except:
                labs.append("")
        ax_t.set_xticks(pos)
        ax_t.set_xticklabels(labs, fontsize=7.5, color='#9e9e9e')

    # ---- Volume panel ----
    if has_volume:
        ax_vol.set_facecolor('#0d0d1a')
        vol   = df['Volume'].values.astype(float)
        vcols = ['#ef5350' if closes[i] >= opens[i] else '#42a5f5'
                 for i in range(len(df))]
        ax_vol.bar(idx, vol, color=vcols, alpha=0.55, width=0.7)
        vol_ma = pd.Series(vol).rolling(20).mean().values
        ax_vol.plot(idx, vol_ma, color='#fff176', linewidth=1.0, label='Vol MA20')
        ax_vol.set_ylabel('Vol', fontsize=8, color='#9e9e9e')
        ax_vol.tick_params(colors='#9e9e9e', labelsize=7)
        ax_vol.yaxis.tick_right()
        for sp in ax_vol.spines.values():
            sp.set_color('#2a2a4a')
        ax_vol.grid(color='#1e1e3a', linewidth=0.3, alpha=0.5)
        set_xticks(ax_vol)
    else:
        ax_vol.set_visible(False)
        target = ax_rsi if (ax_rsi and show_rsi) else ax_main
        if target:
            set_xticks(target)
            plt.setp(target.get_xticklabels(), visible=True)


# ============================================================
# Generate one 2x2 image for a group of 4 tickers
# ============================================================
def generate_image(group_dict, group_title, interval, end_date=None):
    settings = MA_SETTINGS[interval]
    now_str  = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
    label    = {"5m": "5min", "30m": "30min", "1d": "Daily"}[interval]

    fig = plt.figure(figsize=(16, 14), facecolor='#0a0a1a')
    outer_gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.52, wspace=0.14,
        left=0.02, right=0.98, top=0.90, bottom=0.03
    )
    fig.suptitle(
        f"{group_title}  [{label}]  Updated: {now_str} JST",
        fontsize=14, color='#e0e0e0', fontweight='bold', y=0.950
    )

    for i, (name, info) in enumerate(group_dict.items()):
        row = i // 2
        col = i % 2
        df  = fetch_data(
            info["symbol"], interval,
            end_date if interval == "1d" else None
        )
        try:
            draw_chart(fig, outer_gs, row, col, name, info, df, interval, settings)
        except Exception as e:
            ax = fig.add_subplot(outer_gs[row, col])
            ax.set_facecolor('#1a1a2e')
            ax.text(0.5, 0.5, f"{name}\nError:\n{str(e)[:60]}",
                    ha='center', va='center', fontsize=11,
                    color='#ff5252', transform=ax.transAxes)
    return fig


# ============================================================
# Helper: render one tab's two images
# ============================================================
def render_tab(interval, end_date=None):
    label = {"5m": "5min", "30m": "30min", "1d": "Daily"}[interval]

    if st.button(f"📊 Generate {label} Charts", key=f"btn_{interval}"):
        now_str = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y%m%d_%H%M")

        # --- Image 1 ---
        st.subheader("📈 Chart 1: Nikkei Futures / SOX / Nasdaq Futures / USD-JPY")
        with st.spinner("Fetching data for Chart 1..."):
            fig1 = generate_image(GROUP1, "Market Chart 1", interval, end_date)
        st.pyplot(fig1)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=100,
                     bbox_inches='tight', facecolor='#0a0a1a')
        buf1.seek(0)
        st.download_button(
            label="💾 Download Chart 1 PNG",
            data=buf1,
            file_name=f"chart1_{interval}_{now_str}.png",
            mime="image/png",
            key=f"dl1_{interval}"
        )
        plt.close(fig1)

        st.divider()

        # --- Image 2 ---
        st.subheader("📉 Chart 2: Nikkei 225 / VIX / Nikkei VI / US 10Y Yield")
        with st.spinner("Fetching data for Chart 2..."):
            fig2 = generate_image(GROUP2, "Market Chart 2", interval, end_date)
        st.pyplot(fig2)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=100,
                     bbox_inches='tight', facecolor='#0a0a1a')
        buf2.seek(0)
        st.download_button(
            label="💾 Download Chart 2 PNG",
            data=buf2,
            file_name=f"chart2_{interval}_{now_str}.png",
            mime="image/png",
            key=f"dl2_{interval}"
        )
        plt.close(fig2)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Market Chart App", layout="wide", page_icon="📈")

st.markdown("""
<style>
.stApp { background-color: #0a0a1a; color: #e0e0e0; }
.stButton > button {
    background-color: #1565c0; color: white;
    font-size: 15px; padding: 10px 30px;
    border-radius: 8px; border: none; font-weight: bold;
}
.stButton > button:hover { background-color: #1976d2; }
.stTabs [data-baseweb="tab"] { color: #90caf9; font-size: 15px; font-weight: bold; }
h2 { color: #90caf9 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Market Overview Chart App")
st.caption("Data: Yahoo Finance  |  Cache: 5min  |  2 charts per timeframe (4 tickers each)")

tab1, tab2, tab3 = st.tabs(["⚡ 5min", "🕐 30min", "📅 Daily"])

with tab1:
    render_tab("5m")

with tab2:
    render_tab("30m")

with tab3:
    st.subheader("📅 Select End Date (Daily Chart)")
    today    = datetime.today().date()
    min_date = today - timedelta(days=30)
    end_date = st.date_input(
        "End date — chart shows data up to this date",
        value=today, min_value=min_date, max_value=today, key="end_date"
    )
    render_tab("1d", end_date=end_date)
