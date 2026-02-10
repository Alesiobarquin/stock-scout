import streamlit as st
import pandas as pd
import yfinance as yf
import streamlit.components.v1 as components
import csv
import json
import os
import time
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

load_dotenv()

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
        grid-template-columns: repeat(3, 1fr);
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
        # Force reload from disk to avoid stale reads
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
    """
    Loads the CSV Log with support for both Legacy (10 cols) and New (13 cols) formats.
    """
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    try:
        # 1. Read raw lines to detect format version per row? 
        # Actually, pandas is smart. Let's try reading with new headers.
        # If passed columns > actual data, it fills NaN.
        # If passed columns < actual data, it weirds out.
        
        # New Schema (13 Cols)
        new_cols = [
            "Date", "Ticker", "Entry_Price", "Conviction", 
            "Market_Cap", "ATR_Value", "Stop_Loss_Target", "DeRisk_Target", 
            "Thesis", "Status", "Shares_Count", "Position_Cost", "Risk_R_Unit"
        ]
        
        # Legacy Schema (10 Cols - for reference)
        # ["Date", "Ticker", "Entry_Price", "Conviction", "Market_Cap", "ATR_Value", "Stop_Loss", "Target_Price", "Thesis", "Status"]
        
        # We'll read without header, then map manually based on width.
        df_raw = pd.read_csv(CSV_PATH, header=None, quotechar='"', skipinitialspace=True)
        
        if df_raw.empty:
            return pd.DataFrame()

        # Handle Header Row if present
        if str(df_raw.iloc[0][0]).strip() == 'Date':
            df_raw = df_raw.iloc[1:].reset_index(drop=True)

        processed_rows = []
        
        for _, row in df_raw.iterrows():
            # Convert row to list, removing any NaNs at end if pandas added them
            vals = row.dropna().tolist()
            
            # --- LEGACY ROW (10 Cols) ---
            if len(vals) == 10:
                processed_rows.append({
                    "Date": vals[0], "Ticker": vals[1], "Entry_Price": vals[2], "Conviction": vals[3],
                    "Market_Cap": vals[4], "ATR_Value": vals[5], "Stop_Loss_Target": vals[6], 
                    "DeRisk_Target": vals[7], # Old Target -> New DeRisk (Breakeven) trigger for visual simplicity
                    "Thesis": vals[8], "Status": vals[9],
                    "Shares_Count": 0, "Position_Cost": 0.0, "Risk_R_Unit": 0.0 # Default for legacy
                })
            
            # --- NEW ROW (13 Cols) ---
            elif len(vals) >= 13:
                processed_rows.append({
                    "Date": vals[0], "Ticker": vals[1], "Entry_Price": vals[2], "Conviction": vals[3],
                    "Market_Cap": vals[4], "ATR_Value": vals[5], "Stop_Loss_Target": vals[6], 
                    "DeRisk_Target": vals[7], "Thesis": vals[8], "Status": vals[9],
                    "Shares_Count": vals[10], "Position_Cost": vals[11], "Risk_R_Unit": vals[12]
                })

        df = pd.DataFrame(processed_rows)
        
        if df.empty:
            return pd.DataFrame()

        # Cleaning & Typing
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.replace('"', '').str.replace("'", "")

        for col in ['Thesis', 'Status', 'Market_Cap']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace('"', '')

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Date', ascending=False)

        # Numeric Conversion
        cols_to_clean = ['Entry_Price', 'DeRisk_Target', 'ATR_Value', 'Stop_Loss_Target', 'Shares_Count', 'Position_Cost', 'Risk_R_Unit']
        for c in cols_to_clean:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace('$', '').str.replace(',', '')
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

        return df

    except Exception as e:
        # st.error(f"CSV Read Error: {e}")
        return pd.DataFrame()

def mark_trade_closed(ticker_to_close):
    """
    Updates the CSV to mark a trade as CLOSED.
    Uses robust reading/writing to preserve quotes.
    """
    if not os.path.exists(CSV_PATH):
        return
        
    try:
        # Read all lines as raw strings to minimize parsing errors during write-back? 
        # Safer to use csv module directly for read/write to preserve structure.
        rows = []
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        updated = False
        header = rows[0]
        
        # Find index of Ticker and Status columns
        try:
            # Assuming standard order if header is missing or just assuming indices
            # But let's look for header
            ticker_idx = 1
            status_idx = 9 
            
            if "Ticker" in header:
                ticker_idx = header.index("Ticker")
            if "Status" in header:
                status_idx = header.index("Status")
        except:
            pass # Fallback to indices 1 and 9
            
        for row in rows:
            if len(row) > status_idx:
                # Clean ticker from CSV (remove quotes if manual parsing didn't)
                csv_ticker = row[ticker_idx].replace('"', '').replace("'", "").strip()
                csv_status = row[status_idx].replace('"', '').replace("'", "").strip()
                
                if csv_ticker == ticker_to_close and csv_status == 'OPEN':
                    row[status_idx] = "CLOSED"
                    updated = True
        
        if updated:
            with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerows(rows)
            # st.toast(f"Closed trade for {ticker_to_close}")
            time.sleep(0.5) # Give file system a moment
            st.cache_data.clear()
            st.rerun()
            
    except Exception as e:
        st.error(f"Failed to close trade: {e}")

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
    
    # --- TABS CONFIGURATION ---
    tab_hist, tab_active, tab_info = st.tabs(["📜 Performance History", "🚀 Active Signals", "ℹ️ How it Works"])

    # --- TAB 3: HOW IT WORKS ---
    with tab_info:
        st.markdown("""
         NOTE: This portfolio is currently operting with a simulated budget of $500.

        ### 🦅 The Scout (Technical Scanner)
        The system scans **8,000+ stocks** every morning at 9:00 AM EST looking for:
        - **Volatility Contraction**: Tighter price action indicating potential explosive moves.
        - **Momentum & RS**: Stocks outperforming the S&P 500.
        - **Volume**: Unusual buying pressure.

        ### 🧠 The Analyst (AI Agent)
        Once a chart setup is found, an AI Agent reads:
        - Recent News & PRs
        - Earnings & SEC Filings
        - Social Sentiment
        It assigns a **Conviction Score (0-10)** based on the catalyst strength.

        ### 🛡️ The Execution (Risk Management)
        - **Risk Unit**: Every trade risks exactly **2%** of the portfolio (2R).
        - **Hard Stops**: If a stock hits its stop loss, it is cut immediately. 
        
        #### Status Legend
        - 🟢 **OPEN**: Active trade. Monitor the "Runner Progress".
        - 🔴 **STOPPED_OUT**: Trade hit stop loss. Loss is realized.
        - ⚪ **CLOSED**: Trade was manually closed or hit profit target.

        NOTE: Future Plans: Add user login so individuals can set budgets, define risk tolerance, and further customize their trading experience.
        
        """)

    # --- TAB 2: ACTIVE SIGNALS (FROM JSON) ---
    with tab_active:
        # --- METRIC: CAPITAL AT RISK ---
        # Sum of Position_Cost for all OPEN trades
        total_active_exposure = 0.0
        if not df.empty and 'Position_Cost' in df.columns and 'Status' in df.columns:
            open_pos = df[df['Status'] == 'OPEN']
            if not open_pos.empty:
                total_active_exposure = open_pos['Position_Cost'].sum()
        
        st.metric("Capital at Risk (Total Active Exposure)", f"${total_active_exposure:,.2f}")
        st.divider()

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
<a href="https://robinhood.com/us/en/stocks/{ticker}/" target="_blank" class="ticker-symbol" style="text-decoration:none; color:#FFFFFF;">${ticker}</a>
<span class="conviction-badge {badge_class}">Conviction {conviction}/10</span>
</div>
<div style="text-align: right; color: #8B949E; font-size: 14px;">
<div>{signal.get('market_cap', 'N/A')} Cap</div>
<div>⏱️ {signal.get('hold_time_estimate', 'N/A')}</div>
</div>
</div>
""", unsafe_allow_html=True)

                    # --- ACTIVE TRADE CHECK ---
                    # Check if this ticker is currently OPEN in our CSV
                    is_open_trade = False
                    trade_shares = 0
                    trade_cost = 0.0
                    trade_risk = 0.0
                    
                    if not df.empty:
                        open_trade = df[(df['Ticker'] == ticker) & (df['Status'] == 'OPEN')]
                        if not open_trade.empty:
                            is_open_trade = True
                            row = open_trade.iloc[0]
                            trade_shares = int(row.get('Shares_Count', 0))
                            trade_cost = float(row.get('Position_Cost', 0.0))
                            trade_risk = float(row.get('Risk_R_Unit', 0.0))
                            # Override target/stop from CSV if available for consistency
                            entry = float(row.get('Entry_Price', entry))
                            target = float(row.get('DeRisk_Target', target)) # This is Breakeven trigger now
                            stop_val = float(row.get('Stop_Loss_Target', stop_val))

                    # METRICS GRID
                    st.markdown(f"""
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
<div class="metric-label">Breakeven Trigger</div>
<div class="metric-value" style="color: #f1c40f;">${target:.2f}</div>
</div>
<div class="metric-item">
<div class="metric-label">Stop Loss</div>
<div class="metric-value" style="color: #e74c3c;">${stop_val:.2f}</div>
</div>
<div class="metric-item">
<div class="metric-label">Suggested Order</div>
<div class="metric-value" style="font-size: 16px;">{signal.get('shares_count', 0)} shs <span style="color: #888; font-size: 12px;">(${signal.get('position_cost', 0):,.0f})</span></div>
</div>
<div class="metric-item">
<div class="metric-label">Risk / Share</div>
<div class="metric-value" style="color: #e74c3c; font-size: 16px;">${signal.get('risk_r_unit', 0):.2f}</div>
</div>
</div>
""", unsafe_allow_html=True)

                    # --- RUNNER VISUALIZATION ---
                    # Progress towards DeRisk Target (0% = Entry, 100% = DeRisk, <0% = Losing)
                    if entry > 0 and (target - entry) != 0:
                        progress = (live_price - entry) / (target - entry)
                    else:
                        progress = 0
                        
                    # Clamp for bar display (0.0 to 1.0)
                    bar_fill = max(0.0, min(1.0, progress))
                    bar_color = "#e74c3c" # Red default
                    if progress > 0 and progress < 1: bar_color = "#f39c12" # Orange/Yellow working
                    if progress >= 1: bar_color = "#2ecc71" # Green achieved
                    
                    # Custom HTML Progress Bar
                    st.markdown(f"""
                    <div style="margin-bottom: 20px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px; font-size:12px; color:#888;">
                            <span>Entry</span>
                            <span>RUNNER PROGRESS</span>
                            <span>Breakeven Trigger</span>
                        </div>
                        <div style="background-color: #333; width: 100%; height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background-color: {bar_color}; width: {bar_fill*100}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)



                    # --- RISK MANAGEMENT CARD (If Open) ---
                    if is_open_trade:
                        st.info(f"⚡ **ACTIVE POSITION:** Holding **{trade_shares} shares** (Cost: ${trade_cost:,.0f}) | Risk: **${trade_risk:.2f}/share**")
                    
                    st.markdown("</div>", unsafe_allow_html=True) # End Card HTML section (metrics part)



                    # Summary & Actions (Directly in card now)
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
<div style="margin-top: 15px; text-align: right;">
<a href="{signal.get('recency_proof', '#')}" target="_blank" style="color: #3498db; text-decoration: none; font-size: 0.9em;">🔗 View Proof / Source</a>
</div>
</div>
""", unsafe_allow_html=True)
                        

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
            def calc_strategy_return(row):
                """
                Calculates Strategy Return %. 
                - If STOPPED_OUT: (Stop - Entry) / Entry
                - If OPEN but Live < Stop: (Stop - Entry) / Entry  [Virtual Stop]
                - If OPEN and Safe: (Live - Entry) / Entry
                - If CLOSED: N/A (or 0 if missing exit data)
                """
                try:
                    status = str(row.get('Status', '')).strip().upper()
                    ent = float(row.get('Entry_Price', 0.0))
                    shares = float(row.get('Shares_Count', 0.0))
                    
                    if ent <= 0: return 0.0, 0.0 # Return P/L %, P/L $

                    stop_loss = float(row.get('Stop_Loss_Target', 0.0))
                    
                    # 1. Officially Stopped (Realized)
                    if 'STOPPED_OUT' in status:
                        pct = ((stop_loss - ent) / ent) * 100
                        pl_dollar = (stop_loss - ent) * shares
                        return pct, pl_dollar
                    
                    # 2. Virtual Stop (Open but violated)
                    # Note: We check if Live Price is valid (>0) to avoid false triggers
                    curr = float(row.get('Live Price', 0.0))
                    if 'OPEN' in status and curr > 0 and stop_loss > 0 and curr < stop_loss:
                         pct = ((stop_loss - ent) / ent) * 100
                         pl_dollar = (stop_loss - ent) * shares
                         return pct, pl_dollar
                    
                    # 3. Normal Live Return (Unrealized)
                    if 'OPEN' in status:
                         pct = ((curr - ent) / ent) * 100
                         pl_dollar = (curr - ent) * shares
                         return pct, pl_dollar

                    # 4. Closed (Manual) - Data Missing for Exit Price usually
                    # TODO: accurate exit price needed for closed trades. 
                    return 0.0, 0.0 
                except:
                    return 0.0, 0.0

            # Apply calculation
            pl_results = df.apply(calc_strategy_return, axis=1)
            df['Strategy Return %'] = pl_results.apply(lambda x: x[0])
            df['P/L ($)'] = pl_results.apply(lambda x: x[1])

            # --- UI DATA REFINEMENTS ---
            
            # 1. Status Badges
            def get_status_badge(s):
                s = str(s).strip().upper()
                if s == 'OPEN': return "🟢 OPEN"
                if s == 'STOPPED_OUT': return "🔴 STOPPED_OUT"
                if s == 'CLOSED': return "⚪ CLOSED"
                return s
            
            df['Status'] = df['Status'].apply(get_status_badge)

            # 2. Handle Legacy Zeros (Clean up display)
            cols_to_blank = ['Shares_Count', 'Position_Cost']
            for c in cols_to_blank:
                if c in df.columns:
                     df[c] = df[c].apply(lambda x: None if (isinstance(x, (int, float)) and x == 0) else x)

            # 3. Select & Order Columns
            display_cols = [
                "Date", "Ticker", "Entry_Price", "Live Price", 
                "DeRisk_Target", "Stop_Loss_Target", 
                "Shares_Count", "Position_Cost",
                "Strategy Return %", "P/L ($)", "Status"
            ]
            
            # Filter columns that actually exist
            valid_cols = [c for c in display_cols if c in df.columns]
            view_df = df[valid_cols].copy()

            # --- TRANSFORM TICKER TO LINK ---
            if 'Ticker' in view_df.columns:
                 view_df['Ticker'] = view_df['Ticker'].apply(lambda t: f"https://robinhood.com/us/en/stocks/{t}/")

            # --- FINANCIAL CALCULATIONS ---
            # Total P/L ($) - Sum of all VALID P/L (Realized + Unrealized)
            total_pl_dollars = view_df["P/L ($)"].sum()
            
            # Account Size (for Tooltip only now)
            try:
                account_size = float(os.getenv("ACCOUNT_SIZE", 500.0))
            except:
                account_size = 500.0
                
            # Total Equity = Current Market Value of OPEN Positions
            # Logic: Sum(Entry Cost of Open) + Sum(P/L of Open)
            # OR simply: Sum(Live Price * Shares) for Open
            
            total_holdings_value = 0.0
            
            if 'Status' in view_df.columns and 'Live Price' in view_df.columns and 'Shares_Count' in view_df.columns:
                # Filter for OPEN trades (checking specifically for the Badge string or raw status if we reused df)
                # We are using view_df which has the badge "🟢 OPEN"
                
                # It's safer to recalculate from original df or parse view_df
                # Let's use the raw 'df' which has 'Status' as badges now.
                # Actually, let's just iterate view_df since we have the data.
                
                opens_mask = view_df['Status'].astype(str).str.contains("OPEN")
                open_trades = view_df[opens_mask]
                
                if not open_trades.empty:
                    # Calculate Market Value: Live Price * Shares
                    # Note: Handle NaNs or Nones
                    open_trades['Market_Value'] = open_trades['Live Price'].fillna(0.0) * open_trades['Shares_Count'].fillna(0.0)
                    total_holdings_value = open_trades['Market_Value'].sum()

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

            # --- METRICS ROW ---
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.metric(
                    label="Total Equity",
                    value=f"${total_holdings_value:,.2f}",
                    delta=f"${total_pl_dollars:,.2f}",
                    help=f"Initial Investment of ${account_size:,.0f}"
                )
            
            with c2:
                st.metric(
                    label="Net P/L ($)",
                    value=f"${total_pl_dollars:,.2f}",
                    help="Total Realized + Unrealized P/L"
                )

            with c3:
                # Total Return % = (Total P/L / Initial Investment) * 100
                total_return_pct = (total_pl_dollars / account_size) * 100
                st.metric(
                    label="Total Return",
                    value=f"{total_return_pct:.2f}%",
                    help="Portfolio Growth % (P/L relative to Initial Acct Size)"
                )
                
            with c4:
                st.metric("Time Elapsed", period_str)
            
            st.divider()
            # ------------------------------

            # 4. Display Table
            # Apply styling
            def style_status_rows(row):
                status = str(row.get('Status', '')).strip().upper()
                live = float(row.get('Live Price', 0.0))
                stop = float(row.get('Stop_Loss_Target', 0.0))

                if "STOPPED_OUT" in status:
                    # Very subtle red background
                    return ['background-color: rgba(231, 76, 60, 0.08)'] * len(row)
                
                # Check for Violated Stops in OPEN trades
                if "OPEN" in status and live > 0 and stop > 0 and live < stop:
                    # Very subtle red background for violated stops
                    return ['background-color: rgba(231, 76, 60, 0.08)'] * len(row)

                return [''] * len(row)
            
            def style_return_col(val):
                if not isinstance(val, (int, float)): return ''
                if val > 0: return 'color: #2ecc71;' # Green
                if val < 0: return 'color: #e74c3c;' # Red
                return ''
                
            def style_pl_dollar_col(val):
                if not isinstance(val, (int, float)): return ''
                if val > 0: return 'color: #2ecc71; font-weight: bold;' 
                if val < 0: return 'color: #e74c3c; font-weight: bold;'
                return ''

            # Create styled dataframe (Row style + Column style)
            styled_view = view_df.style.apply(style_status_rows, axis=1)\
                                       .map(style_return_col, subset=['Strategy Return %'])\
                                       .map(style_pl_dollar_col, subset=['P/L ($)'])
            
            selection = st.dataframe(
                styled_view,
                width=1400, # Use width=1400 or just let it stretch
                # use_container_width=True, # Deprecated? User said stretch. 
                # Actually, check logic above. User code had:
                # selection = st.dataframe(..., width="stretch" (which isn't valid int), or use_container_width)
                # Correction: 'use_container_width' is the modern way. 'width' takes int.
                # Let's use use_container_width=True
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                column_config={
                    "Date": st.column_config.DatetimeColumn("Alert Date", format="MM-DD HH:mm"),
                    "Ticker": st.column_config.LinkColumn(
                        "Ticker", 
                        display_text="https://robinhood\\.com/us/en/stocks/(.*?)/"
                    ),
                    "Entry_Price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "Live Price": st.column_config.NumberColumn("Live", format="$%.2f"),
                    "DeRisk_Target": st.column_config.NumberColumn("De-Risk Trigger", format="$%.2f"),
                    "Stop_Loss_Target": st.column_config.NumberColumn("Stop Loss", format="$%.2f"),
                    "Shares_Count": st.column_config.NumberColumn("Shares", format="%.4f"),
                    "Position_Cost": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
                    "Strategy Return %": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                    "P/L ($)": st.column_config.NumberColumn("P/L ($)", format="$%.2f"),
                    "Status": st.column_config.TextColumn("Status"),
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
                # Ensure we have just the ticker symbol (remove Robinhood URL if present)
                if isinstance(sel_ticker, str) and "robinhood.com" in sel_ticker:
                    sel_ticker = sel_ticker.rstrip('/').split('/')[-1]

                st.divider()
                st.caption(f"Analyzing History: {sel_ticker}")
                render_chart(sel_ticker)
        else:
            st.warning("Performance log is empty or could not be loaded.")

if __name__ == "__main__":
    main()