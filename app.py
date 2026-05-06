import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

try:
    import japanize_matplotlib
except ImportError:
    pass

# ============================================================
# 設定
# ============================================================
TICKERS = {
    "日経先物":   {"symbol": "NKD=F",    "has_volume": True,  "risk_label": False},
    "日経平均":   {"symbol": "^N225",     "has_volume": True,  "risk_label": False},
    "SOX":        {"symbol": "^SOX",      "has_volume": True,  "risk_label": False},
    "Nasdaq先物": {"symbol": "NQ=F",      "has_volume": True,  "risk_label": False},
    "VIX":        {"symbol": "^VIX",      "has_volume": False, "risk_label": True},
    "日経VI":     {"symbol": "^NKVI.OS",  "has_volume": False, "risk_label": True},
    "ドル円":     {"symbol": "USDJPY=X",  "has_volume": False, "risk_label": False},
    "米10年金利": {"symbol": "^TNX",      "has_volume": False, "risk_label": False},
}

TICKER_NAMES = list(TICKERS.keys())

MA_SETTINGS = {
    "5m":  {"ma_short": 9,  "ma_long": 21, "rsi": False, "prev_close": False},
    "30m": {"ma_short": 20, "ma_long": 50, "rsi": True,  "prev_close": True},
    "1d":  {"ma_short": 20, "ma_long": 60, "rsi": True,  "prev_close": True},
}

PERIOD_SETTINGS = {
    "5m":  "5d",
    "30m": "15d",
    "1d":  "6mo",
}

# ============================================================
# データ取得
# ============================================================
@st.cache_data(ttl=300)
def fetch_data(symbol, interval, end_date=None):
    try:
        period = PERIOD_SETTINGS[interval]
        if interval == "1d" and end_date:
            end_dt = pd.Timestamp(end_date) + timedelta(days=1)
            start_dt = end_dt - timedelta(days=365)
            df = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"),
                             end=end_dt.strftime("%Y-%m-%d"),
                             interval=interval, progress=False, auto_adjust=True)
        else:
            df = yf.download(symbol, period=period, interval=interval,
                             progress=False, auto_adjust=True)
        if df.empty:
            return None
        # MultiIndex対応
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(subset=['Close'], inplace=True)
        return df
    except Exception as e:
        return None

# ============================================================
# 指標計算
# ============================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def get_prev_close(df):
    if df is None or df.empty:
        return None
    try:
        df2 = df.copy()
        df2['date'] = df2.index.normalize()
        dates = df2['date'].unique()
        if len(dates) < 2:
            return None
        prev_day = sorted(dates)[-2]
        prev_data = df2[df2['date'] == prev_day]
        return float(prev_data['Close'].iloc[-1])
    except:
        return None

def calc_trend(df, ma_col):
    try:
        if ma_col not in df.columns or df[ma_col].dropna().empty:
            return "データ不足"
        current = float(df['Close'].iloc[-1])
        ma_now = float(df[ma_col].iloc[-1])
        ma_prev = float(df[ma_col].dropna().iloc[-2]) if len(df[ma_col].dropna()) >= 2 else ma_now
        if current > ma_now and ma_now > ma_prev:
            return "上昇継続↑"
        elif current < ma_now and ma_now < ma_prev:
            return "下降継続↓"
        else:
            return "横ばい→"
    except:
        return "-"

def calc_range_position(df, days=20):
    try:
        recent = df['Close'].tail(days)
        high = float(recent.max())
        low = float(recent.min())
        current = float(df['Close'].iloc[-1])
        if high == low:
            return 50.0, "中段"
        pos = (current - low) / (high - low) * 100
        if pos >= 80:
            label = "高値圏"
        elif pos >= 40:
            label = "中段"
        else:
            label = "安値圏"
        return pos, label
    except:
        return None, "-"

# ============================================================
# 1銘柄チャート描画
# ============================================================
def draw_chart(fig, outer_gs, row, col, name, info, df, interval, settings):
    has_volume = info["has_volume"]
    is_risk = info["risk_label"]
    show_rsi = settings["rsi"]
    show_prev = settings["prev_close"]
    ma_short = settings["ma_short"]
    ma_long = settings["ma_long"]

    # サブグリッド（RSIあり：3段、なし：2段）
    if show_rsi:
        heights = [3, 1, 1] if has_volume else [4, 1, 1]
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_gs[row, col],
            hspace=0.08, height_ratios=heights
        )
        ax_main = fig.add_subplot(inner[0])
        ax_rsi  = fig.add_subplot(inner[1], sharex=ax_main)
        ax_vol  = fig.add_subplot(inner[2], sharex=ax_main) if has_volume else fig.add_subplot(inner[2])
    else:
        heights = [4, 1] if has_volume else [5, 0]
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row, col],
            hspace=0.08, height_ratios=[4, 1]
        )
        ax_main = fig.add_subplot(inner[0])
        ax_rsi  = None
        ax_vol  = fig.add_subplot(inner[1], sharex=ax_main) if has_volume else fig.add_subplot(inner[1])

    # ---- データなし処理 ----
    if df is None or df.empty:
        ax_main.text(0.5, 0.5, f"{name}\nデータ取得失敗", ha='center', va='center',
                     fontsize=13, color='red', transform=ax_main.transAxes)
        ax_main.set_facecolor('#1a1a2e')
        for ax in [ax_rsi, ax_vol]:
            if ax:
                ax.set_visible(False)
        return

    # ---- MA計算 ----
    df = df.copy()
    df[f'MA{ma_short}'] = df['Close'].rolling(ma_short).mean()
    df[f'MA{ma_long}']  = df['Close'].rolling(ma_long).mean()

    if interval == "1d":
        df['MA5']   = df['Close'].rolling(5).mean()
        df['MA200'] = df['Close'].rolling(200).mean()

    if show_rsi:
        df['RSI'] = calc_rsi(df['Close'])

    # ---- 基本情報 ----
    current   = float(df['Close'].iloc[-1])
    prev_close_val = get_prev_close(df)
    pct_chg   = ((current - prev_close_val) / prev_close_val * 100) if prev_close_val else 0
    trend_col = f'MA{ma_long if interval != "5m" else ma_long}'
    trend     = calc_trend(df, f'MA{ma_long}')
    rng_pos, rng_label = calc_range_position(df)

    # タイトル組み立て
    price_str = f"{current:,.2f}"
    pct_str   = f"{pct_chg:+.2f}%"
    risk_str  = ""
    if is_risk:
        if pct_chg > 0:
            risk_str = " [リスクオフ]"
        else:
            risk_str = " [リスクオン]"

    title_line1 = f"{name}  {price_str}  ({pct_str}){risk_str}"
    title_line2 = f"トレンド:{trend}  レンジ位置:{rng_label}({rng_pos:.0f}%)" if rng_pos is not None else f"トレンド:{trend}"

    # ---- X軸インデックス ----
    idx = np.arange(len(df))
    xs  = df.index

    # ---- ローソク足描画 ----
    ax_main.set_facecolor('#0d0d1a')
    width = 0.6
    up_color   = '#ff4444'
    down_color = '#4444ff'
    flat_color = '#888888'

    opens  = df['Open'].values
    closes = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values

    for i in range(len(df)):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        if np.isnan(o) or np.isnan(c):
            continue
        color = up_color if c >= o else down_color
        ax_main.plot([i, i], [l, h], color=color, linewidth=0.8)
        rect = Rectangle((i - width/2, min(o, c)), width, abs(c - o),
                          facecolor=color, edgecolor=color, linewidth=0.5)
        ax_main.add_patch(rect)

    # ---- MA線 ----
    ma_colors = {
        f'MA{ma_short}': '#ffff00',
        f'MA{ma_long}':  '#ff8800',
        'MA5':   '#00ffff',
        'MA200': '#ff00ff',
    }
    ma_labels = {
        f'MA{ma_short}': f'MA{ma_short}',
        f'MA{ma_long}':  f'MA{ma_long}',
        'MA5':   'MA5',
        'MA200': 'MA200',
    }
    for col_name, color in ma_colors.items():
        if col_name in df.columns:
            vals = df[col_name].values
            ax_main.plot(idx, vals, color=color, linewidth=1.0,
                         label=ma_labels[col_name], alpha=0.9)

    # ---- 前日終値ライン ----
    if show_prev and prev_close_val:
        ax_main.axhline(prev_close_val, color='#00ff88', linewidth=1.0,
                        linestyle='--', alpha=0.8, label=f'前日終値 {prev_close_val:,.2f}')

    # ---- 高値・安値 ----
    period_high = float(df['High'].max())
    period_low  = float(df['Low'].min())
    hi_idx = int(df['High'].idxmax()) if hasattr(df['High'].idxmax(), '__index__') else df['High'].values.argmax()
    lo_idx = int(df['Low'].idxmin())  if hasattr(df['Low'].idxmin(),  '__index__') else df['Low'].values.argmin()
    try:
        hi_i = df.index.get_loc(df['High'].idxmax())
        lo_i = df.index.get_loc(df['Low'].idxmin())
        ax_main.annotate(f'H:{period_high:,.1f}', xy=(hi_i, period_high),
                         fontsize=7, color='#ff6666', ha='center', va='bottom')
        ax_main.annotate(f'L:{period_low:,.1f}',  xy=(lo_i, period_low),
                         fontsize=7, color='#6699ff', ha='center', va='top')
    except:
        pass

    # ---- タイトル・装飾 ----
    ax_main.set_title(f"{title_line1}\n{title_line2}", fontsize=9,
                      color='white', pad=4, loc='left',
                      fontweight='bold', backgroundcolor='#0d0d1a')
    ax_main.set_facecolor('#0d0d1a')
    ax_main.tick_params(colors='#aaaaaa', labelsize=7)
    ax_main.yaxis.tick_right()
    ax_main.spines[:].set_color('#333355')
    ax_main.grid(color='#222244', linewidth=0.4, alpha=0.6)
    ax_main.legend(fontsize=6, loc='upper left', facecolor='#0d0d1a',
                   labelcolor='white', framealpha=0.7)
    ax_main.set_xlim(-1, len(df))
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---- RSIパネル ----
    if ax_rsi and show_rsi and 'RSI' in df.columns:
        ax_rsi.set_facecolor('#0d0d1a')
        ax_rsi.plot(idx, df['RSI'].values, color='#cc88ff', linewidth=1.0)
        ax_rsi.axhline(70, color='#ff4444', linewidth=0.6, linestyle='--', alpha=0.7)
        ax_rsi.axhline(30, color='#4444ff', linewidth=0.6, linestyle='--', alpha=0.7)
        ax_rsi.fill_between(idx, df['RSI'].values, 70,
                             where=(df['RSI'].values >= 70), alpha=0.2, color='red')
        ax_rsi.fill_between(idx, df['RSI'].values, 30,
                             where=(df['RSI'].values <= 30), alpha=0.2, color='blue')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_yticks([30, 50, 70])
        ax_rsi.set_ylabel('RSI', fontsize=7, color='#aaaaaa')
        ax_rsi.tick_params(colors='#aaaaaa', labelsize=6)
        ax_rsi.yaxis.tick_right()
        ax_rsi.spines[:].set_color('#333355')
        ax_rsi.grid(color='#222244', linewidth=0.4, alpha=0.6)
        plt.setp(ax_rsi.get_xticklabels(), visible=False)
    elif ax_rsi:
        ax_rsi.set_visible(False)

    # ---- 出来高パネル ----
    if has_volume and ax_vol is not None:
        ax_vol.set_facecolor('#0d0d1a')
        vol = df['Volume'].values.astype(float)
        colors_vol = [up_color if closes[i] >= opens[i] else down_color for i in range(len(df))]
        ax_vol.bar(idx, vol, color=colors_vol, alpha=0.6, width=0.7)
        vol_ma = pd.Series(vol).rolling(20).mean().values
        ax_vol.plot(idx, vol_ma, color='#ffff88', linewidth=0.8, label='Vol MA20')
        ax_vol.set_ylabel('Vol', fontsize=7, color='#aaaaaa')
        ax_vol.tick_params(colors='#aaaaaa', labelsize=6)
        ax_vol.yaxis.tick_right()
        ax_vol.spines[:].set_color('#333355')
        ax_vol.grid(color='#222244', linewidth=0.4, alpha=0.4)
        # X軸ラベル（最後の行だけ表示）
        n = len(df)
        step = max(1, n // 6)
        tick_positions = list(range(0, n, step))
        tick_labels = []
        for p in tick_positions:
            try:
                t = df.index[p]
                if interval == "1d":
                    tick_labels.append(t.strftime('%m/%d'))
                else:
                    tick_labels.append(t.strftime('%m/%d\n%H:%M'))
            except:
                tick_labels.append("")
        ax_vol.set_xticks(tick_positions)
        ax_vol.set_xticklabels(tick_labels, fontsize=6, color='#aaaaaa')
    else:
        if ax_vol is not None:
            ax_vol.set_visible(False)
        # X軸ラベルをRSIか本体に
        target_ax = ax_rsi if (ax_rsi and show_rsi) else ax_main
        n = len(df)
        step = max(1, n // 6)
        tick_positions = list(range(0, n, step))
        tick_labels = []
        for p in tick_positions:
            try:
                t = df.index[p]
                if interval == "1d":
                    tick_labels.append(t.strftime('%m/%d'))
                else:
                    tick_labels.append(t.strftime('%m/%d\n%H:%M'))
            except:
                tick_labels.append("")
        if target_ax:
            target_ax.set_xticks(tick_positions)
            target_ax.set_xticklabels(tick_labels, fontsize=6, color='#aaaaaa')
            plt.setp(target_ax.get_xticklabels(), visible=True)


# ============================================================
# 8銘柄まとめて1枚画像生成
# ============================================================
def generate_image(interval, end_date=None):
    settings = MA_SETTINGS[interval]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    fig = plt.figure(figsize=(28, 36), facecolor='#0a0a1a')
    outer_gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.12,
                                 left=0.03, right=0.97, top=0.96, bottom=0.03)

    interval_label = {"5m": "5分足", "30m": "30分足", "1d": "日足"}[interval]
    fig.suptitle(f"マーケット概況  {interval_label}  ({now_str} 更新)",
                 fontsize=16, color='white', fontweight='bold', y=0.985)

    for i, name in enumerate(TICKER_NAMES):
        row = i // 4
        col = i % 4
        info = TICKERS[name]
        symbol = info["symbol"]

        with st.spinner(f"{name} データ取得中..."):
            df = fetch_data(symbol, interval, end_date if interval == "1d" else None)

        try:
            draw_chart(fig, outer_gs, row, col, name, info, df, interval, settings)
        except Exception as e:
            ax = fig.add_subplot(outer_gs[row, col])
            ax.set_facecolor('#1a1a2e')
            ax.text(0.5, 0.5, f"{name}\nエラー: {str(e)[:40]}",
                    ha='center', va='center', fontsize=10, color='red',
                    transform=ax.transAxes)

    return fig


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="マーケット概況チャート", layout="wide",
                   page_icon="📈")

st.markdown("""
<style>
body { background-color: #0a0a1a; color: white; }
.stApp { background-color: #0a0a1a; }
.stButton > button {
    background-color: #2244aa;
    color: white;
    font-size: 16px;
    padding: 10px 30px;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover { background-color: #3355cc; }
</style>
""", unsafe_allow_html=True)

st.title("📈 マーケット概況チャート")

tab1, tab2, tab3 = st.tabs(["⚡ 5分足", "🕐 30分足", "📅 日足"])

def render_tab(interval, end_date=None):
    label = {"5m": "5分足", "30m": "30分足", "1d": "日足"}[interval]
    col1, col2 = st.columns([1, 4])
    with col1:
        btn = st.button(f"📊 {label}チャート生成", key=f"btn_{interval}")
    if btn:
        with st.spinner("チャート生成中..."):
            fig = generate_image(interval, end_date)
        st.pyplot(fig, use_container_width=True)

        # PNG ダウンロード
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#0a0a1a')
        buf.seek(0)
        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="💾 PNG ダウンロード",
            data=buf,
            file_name=f"market_{interval}_{now_str}.png",
            mime="image/png",
            key=f"dl_{interval}"
        )
        plt.close(fig)

with tab1:
    render_tab("5m")

with tab2:
    render_tab("30m")

with tab3:
    st.subheader("基準日選択（日足）")
    today = datetime.today().date()
    min_date = today - timedelta(days=30)
    end_date = st.date_input(
        "基準日（この日を最終日としてチャートを表示）",
        value=today,
        min_value=min_date,
        max_value=today,
        key="end_date"
    )
    render_tab("1d", end_date=end_date)
