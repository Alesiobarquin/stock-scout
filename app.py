import streamlit as st
import pandas as pd
import yfinance as yf
import streamlit.components.v1 as components
import json
import os
import time
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Scout Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Theme & UI Polish
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Font Override */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Hide Streamlit Header & Sidebar Toggle */
    [data-testid="stHeader"] {background: rgba(0,0,0,0); border-bottom: none;}
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}

    /* Card Container */
    .signal-card {
        background-color: #161719;
        border: 1px solid #2C2F36;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .signal-card:hover {
        border-color: #4A4E58;
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    }
    
    /* Header Section */
    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .ticker-symbol {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    
    /* Conviction Badge */
    .conviction-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .conviction-high { background-color: rgba(39, 174, 96, 0.15); color: #2ecc71; border: 1px solid rgba(39, 174, 96, 0.3); }
    .conviction-med { background-color: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid rgba(241, 196, 15, 0.3); }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }
    .metric-item {
        background: #1E2024;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #2C2F36;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8B949E;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 18px;
        font-weight: 600;
        color: #E6E6E6;
    }
    .metric-sub {
        font-size: 12px;
        font-weight: 500;
        margin-left: 6px;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: #2C2F36;
        margin: 20px 0;
    }

    /* Content Box */
    .thesis-box {
        background-color: #1E2024; 
        padding: 20px; 
        border-radius: 8px; 
        border-left: 3px solid #3498db;
    }
    
    /* Standard Streamlit Overrides */
    div[data-testid="stMetric"] { background-color: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
    button[kind="primary"] { border-radius: 6px !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & MAPPING
# -----------------------------------------------------------------------------
JSON_PATH = "data/latest_report.json"
CSV_PATH = "data/performance_log.csv"

def load_latest_json():
    """Loads the active signal report with your specific JSON keys."""
    if not os.path.exists(JSON_PATH):
        return []
    try:
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
            # Return the full list of catalysts
            if "catalysts" in data and isinstance(data["catalysts"], list):
                return data["catalysts"]
            return []
    except Exception as e:
        st.error(f"Error reading JSON: {e}")
        return []

def load_history_csv():
    """Loads and normalizes the CSV Log."""
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    try:
        # Define expected headers from alpha_scout.py
        # ["Date", "Ticker", "Entry_Price", "Conviction", "Market_Cap", "ATR_Value", "Stop_Loss", "Target_Price", "Thesis", "Status"]
        expected_cols = [
            "Date", "Ticker", "Entry_Price", "Conviction", 
            "Market_Cap", "ATR_Value", "Stop_Loss_Target", "Target_Price", 
            "Thesis", "Status"
        ]

        # Load CSV with header=None since we see raw data starting smoothly
        # quotechar='"' helps, but we will also manually strip to be safe.
        df = pd.read_csv(CSV_PATH, header=None, names=expected_cols, quotechar='"', skipinitialspace=True)
        
        # 1. Handle potential header row if file was recreated properly
        if not df.empty and str(df.iloc[0]['Date']).strip() == 'Date':
            df = df.iloc[1:].reset_index(drop=True)

        # 2. Aggressive Cleaning of Tickers (User specifically asked for this)
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.replace('"', '').str.replace("'", "")

        # 3. Clean other string columns just in case
        for col in ['Thesis', 'Status', 'Market_Cap']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace('"', '')

        # Standardize formatting
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Date', ascending=False)

        # Ensure numeric columns are actually numeric
        cols_to_clean = ['Entry_Price', 'Target_Price', 'ATR_Value', 'Stop_Loss_Target']
        for c in cols_to_clean:
            if c in df.columns:
                # Remove '$' if present from price columns before converting
                df[c] = df[c].astype(str).str.replace('$', '').str.replace(',', '')
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df
    except Exception as e:
        # Graceful failure
        # st.info(f"Could not load history: {e}") 
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 3. CACHED DATA WRAPPER
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
# -----------------------------------------------------------------------------
# 3. CACHED DATA WRAPPER
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_data_bundle(last_modified=None, json_modified=None):
    """Bundles data loading and provides a sync timestamp. 
       last_modified arg forces cache invalidation when file changes."""
    json_data = load_latest_json()
    csv_data = load_history_csv()
    sync_time = time.strftime("%H:%M:%S")
    return json_data, csv_data, sync_time

# -----------------------------------------------------------------------------
# 4. LIVE MARKET DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices(tickers):
    """
    Fetches real-time prices for a list of tickers.
    Cached for 5 minutes to prevent rate limiting.
    """
    if not tickers:
        return {}
    
    clean_tickers = list(set([str(t).upper().strip() for t in tickers]))
    price_map = {}
    
    try:
        # 1. Single Ticker (Faster method)
        if len(clean_tickers) == 1:
            t = clean_tickers[0]
            ticker_obj = yf.Ticker(t)
            # fast_info provides the latest cached price from exchange
            price = ticker_obj.fast_info.last_price
            price_map[t] = price
            
        # 2. Batch Tickers (Efficient for History Table)
        else:
            ticker_str = " ".join(clean_tickers)
            data = yf.download(ticker_str, period="1d", group_by='ticker', progress=False, threads=True)
            
            for t in clean_tickers:
                try:
                    price = data[t]['Close'].iloc[-1]
                    price_map[t] = price
                except:
                    price_map[t] = None
                    
    except Exception as e:
        print(f"YFinance Error: {e}")
        
    return price_map

# -----------------------------------------------------------------------------
# 4. CHART WIDGET
# -----------------------------------------------------------------------------
def render_chart(ticker):
    html_code = f"""
    <div class="tradingview-widget-container" style="height:600px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "height": 600,
        "width": "100%",
        "symbol": "{ticker}",
        "interval": "D",
        "timezone": "America/New_York",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """
    components.html(html_code, height=600)

# -----------------------------------------------------------------------------
# 5. MAIN APP
# -----------------------------------------------------------------------------
def main():
    # 1. Auto-Refresh (every 30 mins)
    st_autorefresh(interval=30 * 60 * 1000, key="datarefresh")

    # 2. Sidebar Controls
    with st.sidebar:
        st.header("Terminal Controls")
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
    # 3. Load Data Bundle
    # Pass file modification time to force cache clear if file changed
    try:
        csv_mtime = os.path.getmtime(CSV_PATH)
    except OSError:
        csv_mtime = 0
    
    try:
        json_mtime = os.path.getmtime(JSON_PATH)
    except OSError:
        json_mtime = 0
        
    signals, df, last_sync = get_data_bundle(last_modified=csv_mtime, json_modified=json_mtime)

    # 4. UI Header
    c_title, c_sync = st.columns([3, 1])
    with c_title:
        st.title("Stock Scout Terminal")
    with c_sync:
        st.markdown(f"""
            <div style="text-align: right; color: #666; font-size: 0.8em; margin-top: 25px;">
                Last Sync: {last_sync}
            </div>
        """, unsafe_allow_html=True)
    
    tab_hist, tab_active = st.tabs(["📜 Performance History", "🚀 Active Signals"])

    # --- TAB 2: ACTIVE SIGNALS (FROM JSON) ---
    with tab_active:
        if signals:
            # Fetch live prices for all active signals at once
            active_tickers = [s.get("ticker") for s in signals if s.get("ticker")]
            live_prices = fetch_live_prices(active_tickers)

            for i, signal in enumerate(signals):
                # Map JSON keys specifically based on your provided file
                ticker = signal.get("ticker", "UNKNOWN")
                entry = signal.get("current_price", 0.0) # Using AI estimated price as entry reference
                
                # Use 'calculated_target' if available, otherwise 0
                target = signal.get("calculated_target", 0.0)
                
                # Use 'calculated_stop_loss' if available
                stop_val = signal.get("calculated_stop_loss", 0.0)
                
                # Thesis text
                thesis = signal.get("thesis", "No thesis provided.")
                catalyst_txt = signal.get("catalyst_details", "")
                
                # New Fields
                conviction = signal.get("conviction_score", "N/A")
                cv_score = int(conviction) if isinstance(conviction, (int, float)) else 0
                badge_class = "conviction-high" if cv_score >= 8 else "conviction-med"
                
                # Live Price
                live_price = live_prices.get(ticker, entry)
                
                # Calc Move
                pct_move = 0.0
                if entry > 0 and live_price:
                    pct_move = ((live_price - entry) / entry) * 100
                
                move_class = "positive" if pct_move >= 0 else "negative"
                move_icon = "▲" if pct_move >= 0 else "▼"

                # Navigation Friendly ID
                st.markdown(f"<div id='signal-{ticker}'></div>", unsafe_allow_html=True)

                # --- CUSTOM CARD LAYOUT ---
                # --- CUSTOM CARD LAYOUT ---
                with st.container():
                    st.markdown(f"""
<div class="signal-card">
<div class="signal-header">
<div style="display: flex; align-items: center; gap: 15px;">
<span class="ticker-symbol">${ticker}</span>
<span class="conviction-badge {badge_class}">Conviction {conviction}/10</span>
</div>
</div>
<div class="metrics-grid">
<div class="metric-item">
<div class="metric-label">Live Price</div>
<div class="metric-value">${live_price:.2f}<span class="metric-sub {move_class}">{move_icon} {abs(pct_move):.2f}%</span></div>
</div>
<div class="metric-item">
<div class="metric-label">Entry Price</div>
<div class="metric-value">${entry:.2f}</div>
</div>
<div class="metric-item">
<div class="metric-label">Target</div>
<div class="metric-value" style="color: #2ecc71;">${target:.2f}</div>
</div>
<div class="metric-item">
<div class="metric-label">Stop Loss</div>
<div class="metric-value" style="color: #e74c3c;">${stop_val:.2f}</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

                    # Summary & Actions within Expander
                    with st.expander(f"Thinking & Technicals", expanded=(i==0)):
                        
                        c_actions, c_gap = st.columns([1, 3])
                        with c_actions:
                            st.link_button(
                                f"Trade ${ticker} on Robinhood 🏹", 
                                f"https://robinhood.com/us/en/stocks/{ticker}/",
                                type="primary", 
                                use_container_width=True
                            )
                        
                        st.divider()

                        # Additional keys from JSON for context box
                        risks = signal.get("risk", "N/A")
                        absorption = signal.get("absorption_status", "N/A")

                        # Content - Styled Box
                        st.markdown(f"""
<div class="thesis-box" style="margin-top: 15px;">
<h4 style="margin-top:0; color: #3498db;">📝 Investment Thesis</h4>
<p style="color: #cfd8dc; font-size: 1.05em; line-height: 1.6;">{thesis}</p>
<hr style="border-top: 1px solid #444; margin: 15px 0;">
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div>
<p style="margin-bottom:4px; font-size: 0.85em; color: #90a4ae; text-transform: uppercase;">Catalyst</p>
<p>{catalyst_txt}</p>
</div>
<div>
<p style="margin-bottom:4px; font-size: 0.85em; color: #90a4ae; text-transform: uppercase;">Absorption</p>
<p>{absorption}</p>
</div>
</div>
<p style="margin-top: 15px; font-size: 0.9em; color: #ffab91;"><strong>⚠️ Risks:</strong> {risks}</p>
</div>
""", unsafe_allow_html=True)
                        
                        st.write("### Technical Chart")
                        render_chart(ticker)

                    st.markdown("</div>", unsafe_allow_html=True) # Close card container

        else:
            st.info("No active signals found. Run the backend script.")

    # --- TAB 1: HISTORY (FROM CSV) ---
    with tab_hist:
        # df = load_history_csv() # Now using bundled data
        
        if not df.empty and 'Ticker' in df.columns:
            
            # 1. Fetch Live Prices for table
            tickers = df['Ticker'].unique().tolist()
            with st.spinner("Syncing live prices..."):
                live_prices_map = fetch_live_prices(tickers)
            
            # 2. Add Live Columns
            df['Live Price'] = df['Ticker'].map(live_prices_map)
            
            # --- P/L CALCULATION LOGIC ---
            def calc_bracket_pl(row):
                """Respects BOTH Target and Stop Loss."""
                try:
                    curr = float(row['Live Price'])
                    ent = float(row['Entry_Price'])
                    tgt = float(row['Target_Price']) if pd.notnull(row['Target_Price']) else None
                    stop = float(row['Stop_Loss_Target']) if pd.notnull(row['Stop_Loss_Target']) else None

                    if ent <= 0: return 0.0
                    
                    # 1. Take Profit Hit?
                    if tgt and curr >= tgt:
                        return ((tgt - ent) / ent) * 100
                    
                    # 2. Stop Loss Hit?
                    if stop and curr <= stop:
                        return ((stop - ent) / ent) * 100

                    # 3. Still Open
                    return ((curr - ent) / ent) * 100
                except:
                    return 0.0

            def calc_runners_pl(row):
                """Respects Stop Loss ONLY (Let winners run)."""
                try:
                    curr = float(row['Live Price'])
                    ent = float(row['Entry_Price'])
                    stop = float(row['Stop_Loss_Target']) if pd.notnull(row['Stop_Loss_Target']) else None

                    if ent <= 0: return 0.0
                    
                    # 1. Stop Loss Hit?
                    if stop and curr <= stop:
                        return ((stop - ent) / ent) * 100

                    # 2. Still Open (Ignore Target)
                    return ((curr - ent) / ent) * 100
                except:
                    return 0.0

            df['P/L (Bracket)'] = df.apply(calc_bracket_pl, axis=1)
            df['P/L (Runners)'] = df.apply(calc_runners_pl, axis=1)

            # 3. Select & Order Columns
            display_cols = [
                "Date", "Ticker", "Entry_Price", "Live Price", 
                "Target_Price", "Stop_Loss_Target", "ATR_Value", 
                "P/L (Bracket)", "P/L (Runners)"
            ]
            
            # Filter columns that actually exist
            valid_cols = [c for c in display_cols if c in df.columns]
            view_df = df[valid_cols].copy()

            # --- NEW: Total P/L Summary & Period ---
            total_bracket = view_df["P/L (Bracket)"].sum()
            total_runners = view_df["P/L (Runners)"].sum()
            
            # Calculate Period
            if 'Date' in view_df.columns and not view_df['Date'].empty:
                min_date = view_df['Date'].min()
                max_date = view_df['Date'].max()
                
                # Calculate duration
                from dateutil.relativedelta import relativedelta
                diff = relativedelta(max_date, min_date)
                
                parts = []
                if diff.years > 0:
                    parts.append(f"{diff.years} year{'s' if diff.years != 1 else ''}")
                if diff.months > 0:
                    parts.append(f"{diff.months} month{'s' if diff.months != 1 else ''}")
                if diff.days > 0 or (diff.years == 0 and diff.months == 0):
                    parts.append(f"{diff.days} day{'s' if diff.days != 1 else ''}")
                
                period_str = ", ".join(parts)
            else:
                period_str = "N/A"

            c1, c2, c3 = st.columns(3)
            c1.metric("Total P/L (Bracket)", f"{total_bracket:.2f}%", help="Sum of P/L respecting both Targets and Stops.")
            c2.metric("Total P/L (Runners)", f"{total_runners:.2f}%", help="Sum of P/L respecting ONLY Stops (letting winners run).")
            c3.metric("Time Elapsed", period_str, help=f"Time between first and last alert.\n({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})")
            
            st.divider()
            # ------------------------------

            # 4. Display Table
            selection = st.dataframe(
                view_df,
                width="stretch",
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                column_config={
                    "Date": st.column_config.DatetimeColumn("Alert Date", format="MM-DD HH:mm"),
                    "Entry_Price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "Live Price": st.column_config.NumberColumn("Live", format="$%.2f"),
                    "Target_Price": st.column_config.NumberColumn("Target", format="$%.2f"),
                    "Stop_Loss_Target": st.column_config.TextColumn("Stop Loss"),
                    "ATR_Value": st.column_config.NumberColumn("ATR", format="%.2f"),
                    "P/L (Bracket)": st.column_config.NumberColumn(
                        "P/L (Bracket)", 
                        format="%.2f%%",
                        help="P/L capped at Target and Stop."
                    ),
                    "P/L (Runners)": st.column_config.NumberColumn(
                        "P/L (Runners)", 
                        format="%.2f%%",
                        help="P/L capped at Stop ONLY."
                    )
                }
            )

            # 5. Chart Selection
            sel_ticker = None
            if len(selection.selection.rows) > 0:
                idx = selection.selection.rows[0]
                sel_ticker = view_df.iloc[idx]['Ticker']
            elif not view_df.empty:
                sel_ticker = view_df.iloc[0]['Ticker']
            
            if sel_ticker:
                st.divider()
                st.caption(f"Analyzing History: {sel_ticker}")
                render_chart(sel_ticker)
        else:
            st.warning("Performance log is empty or could not be loaded.")

if __name__ == "__main__":
    main()