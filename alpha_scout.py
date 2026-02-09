import os
import json
import time
import requests
import re
import csv
import pandas as pd
import yfinance as yf
import subprocess
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
import pytz
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- CONFIGURATION ---
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview") 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "data/latest_report.json"
PERFORMANCE_LOG_FILE = "data/performance_log.csv"

# Risk Management Configuration
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 1000.0))
RISK_PER_TRADE = 0.02       # 2% risk of account equity per trade
MAX_OPEN_POSITIONS = 3      # Max active trades
HARD_STOP_CAP = 0.07        # Max 7% stop loss width

# --- DATA MODELS ---
class Catalyst(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (e.g., AAPL)")
    current_price: float = Field(..., description="AI estimated price (will be overwritten by API).")
    market_cap: str = Field(..., description="Market Cap string.")
    conviction_score: int = Field(..., description="1-10 Score.")
    thesis: str = Field(..., description="1-2 sentence thesis.")
    catalyst_details: str = Field(..., description="Event details.")
    absorption_status: str = Field(..., description="Why market hasn't reacted.")
    earnings_date: str = Field(..., description="Next earnings date.")
    relative_volume: str = Field(..., description="Vol vs 30d avg.")
    stop_loss_trigger: str = Field(..., description="AI suggested stop (text description).")
    sentiment: str = Field(..., description="Bullish/Bearish.")
    prediction_market: str = Field(..., description="Odds.")
    recency_proof: str = Field(..., description="Link/Timestamp.")
    risk: str = Field(..., description="Risk factor.")
    expected_upside: str = Field(..., description="Upside potential.")
    mispricing_evidence: str = Field(..., description="Evidence.")
    x_sentiment: Optional[str] = Field(None, description="X buzz.")
    hold_time_estimate: str = Field(..., description="Hold time (e.g., '2-3 Days' for Earnings, '2-4 Weeks' for Trend).")
    
    # Technical Fields (Populated by yfinance & Risk Engine)
    atr_value: Optional[float] = None
    calculated_stop_loss: Optional[float] = None
    breakeven_trigger: Optional[float] = None    # Replaces calculated_target
    
    # Position Sizing Fields
    shares_count: Optional[int] = None
    position_cost: Optional[float] = None
    risk_r_unit: Optional[float] = None          # Dollar risk per share

class ScoutReport(BaseModel):
    catalysts: List[Catalyst]

# --- HELPER FUNCTIONS ---
def parse_market_cap_to_millions(cap_str: str) -> float:
    clean = cap_str.upper().replace('$', '').replace(',', '').strip()
    try:
        if 'B' in clean: return float(re.search(r"[\d\.]+", clean).group()) * 1000
        elif 'M' in clean: return float(re.search(r"[\d\.]+", clean).group())
        elif 'T' in clean: return float(re.search(r"[\d\.]+", clean).group()) * 1000000
        return 0.0
    except: return 0.0

def parse_upside_percentage(upside_str: str) -> float:
    matches = re.findall(r'(\d+(?:\.\d+)?)%', upside_str)
    if not matches: return 0.0
    values = [float(x) for x in matches]
    return sum(values) / len(values)

def check_portfolio_capacity():
    """Checks if the portfolio is at capacity based on open positions."""
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return True # File doesn't exist, so capacity is fine
    
    try:
        df = pd.read_csv(PERFORMANCE_LOG_FILE)
        if 'Status' not in df.columns:
            return True
            
        open_positions = df[df['Status'] == 'OPEN']
        count = len(open_positions)
        
        if count >= MAX_OPEN_POSITIONS:
            print(f"[!] PORTFOLIO CAPACITY REACHED: {count}/{MAX_OPEN_POSITIONS} Open Positions.")
            print("[!] Pausing scans to prevent over-exposure.")
            return False
        return True
    except Exception as e:
        print(f"[!] Error checking portfolio capacity: {e}")
        return True # Default to allow if check fails (or fail safe? User said 'EXIT'. If read fails, maybe safe to run? Let's assume safe to run if read fails, but print error)

# --- PORTFOLIO HEALTH CHECK ---
def manage_portfolio_health():
    """
    Auto-Gardener: Checks active trades for Stop Loss or Breakeven triggers.
    Runs at the start of main() to ensure portfolio capacity is accurate.
    """
    print("[-] Running Auto-Gardener: Checking Portfolio Health...")
    
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return

    try:
        df = pd.read_csv(PERFORMANCE_LOG_FILE)
        # Ensure necessary columns exist
        required_cols = ['Ticker', 'Status', 'Stop_Loss', 'Breakeven_Trigger', 'Entry_Price', 'Shares_Count']
        if not all(col in df.columns for col in required_cols):
            return

        # Filter OPEN trades
        open_trades = df[df['Status'] == 'OPEN'].copy()
        
        if open_trades.empty:
            print("[-] No open positions to manage.")
            return

        tickers = open_trades['Ticker'].unique().tolist()
        print(f"[*] Monitoring Open Positions: {', '.join(tickers)}")

        # Fetch current prices
        current_prices = {}
        processed_tickers = []
        
        try:
            # Batch fetch for efficiency
            # Using yf.download is efficient for multiple tickers
            if len(tickers) > 0:
                data = yf.download(tickers, period="1d", interval="1m", progress=False)
                
                # Check if data is empty
                if data.empty:
                    print("[!] No price data received.")
                else:
                    # Extract Close prices
                    # Use 'Close' column. If multi-index (multiple tickers), it has a level for Ticker
                    closes = data['Close']
                    
                    if len(tickers) == 1:
                        # Single ticker case: 'closes' is a Series (or DataFrame with 1 col if keep names)
                        # Access safely
                        current_prices[tickers[0]] = closes.iloc[-1]
                        processed_tickers.append(tickers[0])
                    else:
                        # Multiple tickers case
                        for t in tickers:
                            try:
                                if t in closes.columns:
                                    price = closes[t].iloc[-1]
                                    if pd.notna(price):
                                        current_prices[t] = price
                                        processed_tickers.append(t)
                            except:
                                pass
        except Exception as e:
            print(f"[!] Batch fetch failed, trying individual: {e}")
        
        # Fallback / Individual ensure
        for t in tickers:
            if t not in processed_tickers:
                try:
                    p = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
                    current_prices[t] = p
                except:
                    print(f"[!] Could not fetch price for {t}")

        updates_made = False
        
        # Iterate to check logic
        for index, row in open_trades.iterrows():
            ticker = row['Ticker']
            if ticker not in current_prices:
                continue
                
            current_price = float(current_prices[ticker])
            
            try:
                stop_loss = float(row['Stop_Loss'])
                breakeven = float(row['Breakeven_Trigger'])
                entry_price = float(row['Entry_Price'])
                shares = float(row['Shares_Count'])
            except:
                continue
            
            # SCENARIO A: STOP LOSS HIT
            if current_price <= stop_loss:
                # Update Status in the MAIN dataframe (using original index)
                df.at[index, 'Status'] = 'STOPPED_OUT'
                updates_made = True
                
                loss_amount = (entry_price - stop_loss) * shares
                
                print(f"[{ticker}] 🛑 STOP LOSS TRIGGERED at ${current_price:.2f} (Stop: ${stop_loss :.2f})")
                
                alert_msg = (
                    f"🚫 **STOP LOSS TRIGGERED: ${ticker}**\n"
                    f"📉 Price crossed ${stop_loss:.2f}. Marked as CLOSED in DB.\n"
                    f"💸 Est. Loss: -${loss_amount:.2f}\n"
                    f"🔄 **Slot Freed.**"
                )
                send_telegram_alert(alert_msg)

            # SCENARIO B: BREAKEVEN TRIGGER HIT
            elif current_price >= breakeven:
                 # Just Alert (Do not close)
                 print(f"[{ticker}] ⚠️ DE-RISK TRIGGERED at ${current_price:.2f}")
                 
                 alert_msg = (
                    f"⚠️ **DE-RISK ALERT: ${ticker}**\n"
                    f"🚀 Price hit ${current_price:.2f}.\n"
                    f"🛡️ **Move Hard Stop to Breakeven.**"
                 )
                 send_telegram_alert(alert_msg)

        if updates_made:
            df.to_csv(PERFORMANCE_LOG_FILE, index=False, quoting=csv.QUOTE_ALL)
            print("[*] Portfolio Log Updated: Stopped out positions closed.")

    except Exception as e:
        print(f"[!] Error in Auto-Gardener: {e}")


# --- QUANTITATIVE ENGINE ---
def enrich_with_technical_data(catalyst: Catalyst) -> Optional[Catalyst]:
    """
    Fetches live data, calculates Position Size, Dynamic Stops, and Breakeven Trigger.
    """
    print(f"[*] Fetching technical data for {catalyst.ticker}...")
    try:
        ticker = yf.Ticker(catalyst.ticker)
        hist = ticker.history(period="1mo")
        
        if hist.empty or len(hist) < 15:
            print(f"[!] Insufficient data for {catalyst.ticker}. Skipping.")
            return None

        # Real-Time Price
        current_price = hist['Close'].iloc[-1]
        
        # ATR (14) Calculation
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr_value = true_range.rolling(window=14).mean().iloc[-1]

        # --- RISK MANAGEMENT ENGINE ---
        
        # 1. Dynamic Stop Loss Calculation
        atr_stop_price = current_price - (1.5 * atr_value)
        hard_stop_price = current_price * (1 - HARD_STOP_CAP)
        
        # USE WHICHEVER IS TIGHTER (Higher Price)
        final_stop_loss = max(atr_stop_price, hard_stop_price)
        
        # 2. Risk per Share & Metrics
        risk_per_share = current_price - final_stop_loss
        if risk_per_share <= 0:
            print(f"[!] Invalid risk/reward for {catalyst.ticker} (Stop > Price). Skipping.")
            return None
            
        # 3. Position Sizing
        risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE
        shares_to_buy = int(risk_amount / risk_per_share)
        
        # 4. Cost Basis Constraint (Max 20% allocation)
        projected_cost = shares_to_buy * current_price
        max_allocation = ACCOUNT_SIZE * 0.20
        
        if projected_cost > max_allocation:
            # Cap shares to max allocation
            shares_to_buy = int(max_allocation / current_price)
            projected_cost = shares_to_buy * current_price
            
        if shares_to_buy < 1:
            print(f"[!] Position size too small for {catalyst.ticker}. Skipping.")
            return None

        # 5. Breakeven Trigger (The "Runner" Strategy)
        # Move stop to breakeven when price hits Entry + 1.5R
        breakeven_trigger = current_price + (1.5 * risk_per_share)

        # Update Object
        catalyst.current_price = round(current_price, 2)
        catalyst.atr_value = round(atr_value, 2)
        catalyst.calculated_stop_loss = round(final_stop_loss, 2)
        catalyst.breakeven_trigger = round(breakeven_trigger, 2)
        
        # Position Metrics
        catalyst.shares_count = shares_to_buy
        catalyst.position_cost = round(projected_cost, 2)
        catalyst.risk_r_unit = round(risk_per_share, 2)
        
        # Update display strings for Telegram (Optional, or handled in format_telegram_message)
        catalyst.stop_loss_trigger = f"${final_stop_loss:.2f}" 

        return catalyst
    except Exception as e:
        print(f"[!] Error processing {catalyst.ticker}: {e}")
        return None

# --- PERSISTENT LOGGING LOGIC ---
def log_to_performance_csv(catalyst: Catalyst):
    """
    Logs the signal to CSV with updated metrics.
    """
    os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True)
    
    # 1. Duplicate Prevention
    if os.path.exists(PERFORMANCE_LOG_FILE):
        try:
            df = pd.read_csv(PERFORMANCE_LOG_FILE)
            if not df.empty and 'Ticker' in df.columns and 'Status' in df.columns:
                is_active = df[(df['Ticker'] == catalyst.ticker) & (df['Status'] == 'OPEN')]
                if not is_active.empty:
                    print(f"[-] Skipping Log: {catalyst.ticker} is already OPEN in {PERFORMANCE_LOG_FILE}")
                    return
        except Exception as e:
            print(f"[!] Warning: Could not read existing CSV for duplicate check: {e}")

    # 2. Prepare Data
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    row_data = [
        ny_time,
        catalyst.ticker,
        catalyst.current_price,        # Entry Price
        catalyst.conviction_score,     # Conviction
        catalyst.market_cap,           # Market Cap
        catalyst.atr_value,            # ATR
        catalyst.calculated_stop_loss, # Stop Loss (Hard or ATR)
        catalyst.breakeven_trigger,    # Breakeven Trigger (Replaces Target)
        catalyst.thesis,               # Thesis
        'OPEN',                        # Status
        catalyst.shares_count,         # Shares Count
        catalyst.position_cost,        # Position Cost
        catalyst.risk_r_unit           # Risk per Share ($)
    ]
    
    headers = [
        "Date", "Ticker", "Entry_Price", "Conviction", 
        "Market_Cap", "ATR_Value", "Stop_Loss", "Breakeven_Trigger", 
        "Thesis", "Status", "Shares_Count", "Position_Cost", "Risk_R_Unit"
    ]

    # 3. Write to CSV (Append Mode)
    file_exists = os.path.isfile(PERFORMANCE_LOG_FILE)
    try:
        with open(PERFORMANCE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row_data)
        print(f"[*] Successfully logged {catalyst.ticker} to {PERFORMANCE_LOG_FILE}")
    except IOError as e:
        print(f"[!] Failed to write to CSV: {e}")



# --- AGENT SETUP ---
def get_alpha_scout_response():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    three_days_ago = (now - timedelta(days=3)).strftime('%Y-%m-%d')

    system_instruction = """
    Role: You are “Stock Scout,” a Quant-Fundamental Analyst.
    STRATEGY: PEAD, Biotech PDUFA, Insider Aggression.
    FILTERS:
    1. 8% RULE: Discard if stock is already up >8% today.
    2. Liquidity: Market Cap $500M - $10B.
    FORBIDDEN DATES: Ignore results from previous years.
    """

    prompt = f"""
    Current Date: {today_str}
    Find unpriced bullish catalysts between {three_days_ago} and {today_str}.
    Provide 'current_price' (estimate), 'absorption_status', 'market_cap', and 'hold_time_estimate'.
    """

    tools = [types.Tool(google_search=types.GoogleSearch())]
    
    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        try:
            print(f"[-] AI Query Attempt {attempt+1}/{max_retries}...")
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools,
                    response_mime_type="application/json",
                    response_schema=ScoutReport
                )
            )
            return response.parsed, False
        except Exception as e:
            error_str = str(e).lower()
            if attempt < max_retries - 1:
                # Retry on: 503 (Overloaded), 429 (Rate Limit), Disconnected, Timeout
                if any(x in error_str for x in ["503", "overloaded", "disconnected", "timeout", "network"]):
                    wait_time = base_delay * (2 ** attempt) # Exponential backoff
                    print(f"[!] Network/API Error ({error_str[:50]}...). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif "429" in error_str:
                    wait_time = 30
                    print(f"[!] Rate limit (429). Cooling down for {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            # If we're out of retries for the primary model, try the BACKUP
            if attempt == max_retries - 1:
                print(f"[!] Primary model {MODEL_ID} failed after {max_retries} attempts.")
                print(f"[*] Switching to Fallback Model: gemini-3.0-flash...")
                try:
                    response = client.models.generate_content(
                        model="gemini-3.0-flash-preview",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=tools,
                            response_mime_type="application/json",
                            response_schema=ScoutReport
                        )
                    )
                    return response.parsed, True
                except Exception as backup_e:
                    print(f"[!] Critical: Backup model also failed: {backup_e}")
                    raise backup_e
            
            # If it's a non-retryable error (not caught above) or logic fails
            print(f"[!] API call failed: {e}")
            raise e

# --- OUTPUT HANDLERS ---
def save_to_json(report: ScoutReport):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        f.write(report.model_dump_json(indent=2))

def format_telegram_message(catalyst: Catalyst, is_fallback: bool = False) -> str:
    robinhood_link = f"https://robinhood.com/us/en/stocks/{catalyst.ticker}/"
    try:
        ny_time = datetime.now(pytz.timezone("America/New_York")).strftime('%H:%M %Z')
    except:
        ny_time = datetime.now().strftime('%H:%M UTC')

    model_note = " (Backup Model)" if is_fallback else ""

    return (
        f"🚀 *Stock Scout Signal: ${catalyst.ticker}*{model_note}\n"
        f"✅ *Entry:* ${catalyst.current_price:.2f}\n"
        f"� *Suggested Order:* Buy {catalyst.shares_count} Shares (~${catalyst.position_cost:.0f} Total)\n"
        f"🛑 *Stop Loss:* ${catalyst.calculated_stop_loss:.2f} (Risk: ${catalyst.risk_r_unit:.2f}/share)\n"
        f"🏃 *Strategy:* Move Stop to Breakeven at ${catalyst.breakeven_trigger:.2f}. Then let winners run.\n"
        f"━━━━━\n"
        f"*Conviction: {catalyst.conviction_score}/10* | *Mkt Cap:* {catalyst.market_cap}\n"
        f"*Est. Hold Time:* {catalyst.hold_time_estimate}\n"
        f"━━━━━\n"
        f"💡 *Thesis:*\n{catalyst.thesis}\n\n"
        f"⏳ *Absorption Status:*\n{catalyst.absorption_status}\n\n"
        f"📅 *Catalyst:*\n{catalyst.catalyst_details}\n\n"
        f"🔗 *Proof:* {catalyst.recency_proof}\n"
        f"_Generated at {ny_time}_"
    )

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
    try: requests.post(url, json=payload).raise_for_status()
    except Exception as e: print(f"[!] Telegram Error: {e}")

# --- MAIN EXECUTION ---
def main():
    print("[-] Stock Scout initializing (Risk-Managed Runner Mode)...")
    
    # 1. Manage Portfolio Health (Auto-Gardener)
    # Runs FIRST to clear out stopped positions and free up slots.
    manage_portfolio_health()
    
    # 2. Portfolio Capacity Check
    if not check_portfolio_capacity():
        return # Exit script if capacity reached

    try:
        report, is_fallback = get_alpha_scout_response()
        if not report or not report.catalysts:
            print("[!] No catalysts found.")
            return
        
        print(f"[-] Analyzing {len(report.catalysts)} raw candidates...")
        valid_catalysts = []
        
        for c in report.catalysts:
            # 1. Basic Filters
            mcap_millions = parse_market_cap_to_millions(c.market_cap)
            is_liquid = mcap_millions >= 300
            if c.sentiment != "Bullish" or not is_liquid: continue

            # 2. QUANTITATIVE VERIFICATION
            enriched_c = enrich_with_technical_data(c)
            if not enriched_c: continue

            # 3. Final Scoring
            if enriched_c.conviction_score >= 7:
                valid_catalysts.append(enriched_c)

        if not valid_catalysts:
            print("[-] No catalysts met the Hybrid criteria.")
            return
        
        valid_catalysts.sort(key=lambda c: c.conviction_score, reverse=True)
        top_picks = valid_catalysts[:3]
        
        save_to_json(ScoutReport(catalysts=valid_catalysts))
        
        for item in top_picks:
            # LOG FIRST (Persistent & Duplicate Checked)
            log_to_performance_csv(item)
            # THEN ALERT
            msg = format_telegram_message(item, is_fallback=is_fallback)
            send_telegram_alert(msg)
            time.sleep(1)
        
        print(f"[*] Processed {len(valid_catalysts)} valid signals. Sent {len(top_picks)} alerts.")
        


    except Exception as e:
        print(f"[!] Critical Error: {e}")

if __name__ == "__main__":
    main()