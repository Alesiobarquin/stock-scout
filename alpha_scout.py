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
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 500.0))
RISK_PER_TRADE = 0.02       # 2% risk of account equity per trade
MAX_OPEN_POSITIONS = 5      # Max active trades
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
    shares_count: Optional[float] = None
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

def get_available_buying_power() -> tuple[float, float]:
    """
    Calculates Buying Power = Current Equity - Cost of Open Positions.
    Returns (buying_power, current_equity)
    """
    base_equity = calculate_current_equity()
    
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return base_equity, base_equity
        
    try:
        df = pd.read_csv(PERFORMANCE_LOG_FILE)
        open_cost = 0.0
        
        # Check if we have cost column
        if 'Status' in df.columns and 'Position_Cost' in df.columns:
            open_trades = df[df['Status'] == 'OPEN'].copy()
            if not open_trades.empty:
                # Clean currency strings if present
                if open_trades['Position_Cost'].dtype == 'object':
                     open_trades['Position_Cost'] = open_trades['Position_Cost'].astype(str).str.replace('$', '').str.replace(',', '')
                open_cost = pd.to_numeric(open_trades['Position_Cost'], errors='coerce').fillna(0.0).sum()
                
        buying_power = base_equity - open_cost
        return max(0.0, buying_power), base_equity
        
    except Exception as e:
        print(f"[!] Error calculating buying power: {e}")
        return 0.0, base_equity

def calculate_current_equity() -> float:
    """
    Calculates the current buying power / equity based on realized P/L.
    Base Capital + Sum(Realized P/L).
    """
    base_equity = float(os.getenv("ACCOUNT_SIZE", 500.0))
    
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return base_equity
        
    try:
        df = pd.read_csv(PERFORMANCE_LOG_FILE)
        if 'Realized_PL' in df.columns:
            total_realized = df['Realized_PL'].sum()
            current_equity = base_equity + total_realized
            # Ensure we don't trade with negative equity (blown account)
            return max(0.0, current_equity)
        return base_equity
    except Exception as e:
        print(f"[!] Error calculating equity: {e}")
        return base_equity

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
        
        # Ensure necessary columns exist (Schema Evolution)
        required_cols = ['Ticker', 'Status', 'Stop_Loss', 'Breakeven_Trigger', 'Entry_Price', 'Shares_Count', 'Date']
        new_cols = ['Exit_Price', 'Realized_PL']
        
        # Add new columns if missing
        schema_updated = False
        for col in new_cols:
            if col not in df.columns:
                df[col] = 0.0
                schema_updated = True
                
        if schema_updated:
             updates_made = True
                
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
            if len(tickers) > 0:
                data = yf.download(tickers, period="1d", interval="1m", progress=False)
                if data.empty:
                    print("[!] No price data received.")
                else:
                    closes = data['Close']
                    if len(tickers) == 1:
                        # Handle single ticker series
                        if isinstance(closes, pd.DataFrame):
                             current_prices[tickers[0]] = closes.iloc[-1].item()
                        else:
                             current_prices[tickers[0]] = closes.iloc[-1]
                        processed_tickers.append(tickers[0])
                    else:
                        for t in tickers:
                            try:
                                if t in closes.columns:
                                    current_prices[t] = closes[t].iloc[-1]
                                    processed_tickers.append(t)
                            except:
                                pass
        except Exception as e:
            print(f"[!] Batch fetch failed, trying individual: {e}")
        
        # Fallback
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
                entry_date_str = str(row['Date'])
            except:
                continue
            
            # Helper for closing
            def close_position(reason, exit_price):
                pl = (exit_price - entry_price) * shares
                df.at[index, 'Status'] = reason
                df.at[index, 'Exit_Price'] = round(exit_price, 2)
                df.at[index, 'Realized_PL'] = round(pl, 2)
                return pl

            # 1. PRIORITY: STOP LOSS CHECK
            # We check this FIRST so that even if a trade is old (stale), 
            # if it hit the stop, we record it as a Stop Loss exit (honoring the stop price).
            if current_price <= stop_loss:
                pl = close_position('STOPPED_OUT', stop_loss) # Executing at STOP price (slippage ignored)
                updates_made = True
                print(f"[{ticker}] 🛑 STOP LOSS TRIGGERED at ${current_price:.2f} (Stop: ${stop_loss :.2f})")
                alert_msg = (
                    f"🚫 **STOP LOSS TRIGGERED: ${ticker}**\n"
                    f"📉 Price crossed ${stop_loss:.2f}.\n"
                    f"💸 Realized Loss: -${abs(pl):.2f}\n"
                    f"🔄 **Slot Freed.**"
                )
                send_telegram_alert(alert_msg)
                continue

            # 2. TIME STOP CHECKS (Stale Trades)
            # Only check if we are NOT stopped out.
            days_held = 0
            try:
                entry_dt = pd.to_datetime(entry_date_str)
                if entry_dt.tzinfo:
                   entry_dt = entry_dt.tz_localize(None)
                
                now_dt = datetime.now()
                days_held = (now_dt - entry_dt).days
            except Exception as e:
                print(f"[!] Date parse error for {ticker}: {e}")

            if days_held >= 7:
                pl = close_position('CLOSED', current_price) # Manual CLOSE/Time Stop at Market Price
                updates_made = True
                print(f"[{ticker}] ⏳ TIME STOP TRIGGERED (Held {days_held} days). Closing at ${current_price:.2f}")
                alert_msg = (
                    f"🕑 **TIME STOP: ${ticker}**\n"
                    f"📅 Held > 7 Days. Closing to free capital.\n"
                    f"💰 P/L: ${pl:.2f}\n"
                    f"🔄 **Slot Freed.**"
                )
                send_telegram_alert(alert_msg)
                continue 
            
            # 3. BREAKEVEN ALERT
            if current_price >= breakeven:
                 # Just Alert for now
                 print(f"[{ticker}] ⚠️ DE-RISK TRIGGERED at ${current_price:.2f}")
                 alert_msg = (
                    f"⚠️ **DE-RISK ALERT: ${ticker}**\n"
                    f"🚀 Price hit ${current_price:.2f}.\n"
                    f"🛡️ **Move Hard Stop to Breakeven (${entry_price:.2f}).**"
                 )
                 send_telegram_alert(alert_msg)

        # 4. BUDGET ENFORCEMENT (LIQUIDATION ENGINE)
        # Re-check portfolio state after stop checks
        open_active = df[df['Status'] == 'OPEN'].copy()
        
        if not open_active.empty:
            # Calculate Equity
            base_eq = float(os.getenv("ACCOUNT_SIZE", 500.0))
            if 'Realized_PL' in df.columns:
                 # Clean metric
                 if df['Realized_PL'].dtype == 'object':
                     df['Realized_PL'] = pd.to_numeric(df['Realized_PL'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0.0)
                 realized = df['Realized_PL'].sum()
            else:
                 realized = 0.0
            
            curr_equity = base_eq + realized
            
            # Calculate Total Exposure
            if open_active['Position_Cost'].dtype == 'object':
                open_active['Position_Cost'] = pd.to_numeric(open_active['Position_Cost'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0.0)
            
            total_exposure = open_active['Position_Cost'].sum()
            
            # Budget Check
            if total_exposure > curr_equity:
                excess = total_exposure - curr_equity
                print(f"[!] BUDGET EXCEEDED: Exposure ${total_exposure:,.2f} > Equity ${curr_equity:,.2f}. Reducing by ${total_exposure - curr_equity:,.2f}...")
                
                # Rank by Performance (Worst first) to cut losers
                performance_list = []
                for idx, row in open_active.iterrows():
                    tick = row['Ticker']
                    entry = float(row.get('Entry_Price', 0))
                    cost = float(row.get('Position_Cost', 0))
                    
                    # Get current price from our earlier fetch or fallback
                    curr = 0.0
                    if tick in current_prices:
                        curr = float(current_prices[tick])
                    else:
                        # Fallback fetch if missed (shouldn't happen often)
                        try:
                           curr = yf.Ticker(tick).fast_info.last_price
                        except:
                           curr = entry # Assume flat if data fail to avoid panic sell? Or Sell? Assume Sell to be safe.
                    
                    unrealized_pct = (curr - entry) / entry if entry > 0 else -1.0
                    performance_list.append({
                        'index': idx,
                        'ticker': tick,
                        'unrealized_pct': unrealized_pct,
                        'cost': cost,
                        'current_price': curr
                    })
                
                # Sort: Lowest P/L first
                performance_list.sort(key=lambda x: x['unrealized_pct'])
                
                # Liquidation Loop
                for item in performance_list:
                    if total_exposure <= curr_equity:
                        break # Budget restored
                        
                    # Liquidate
                    idx = item['index']
                    tick = item['ticker']
                    cp = item['current_price']
                    
                    # Call close logic manually since helper 'close_position' is local to loop
                    # Re-implementing simplified close here using DataFrame index directly
                    entry_p = float(df.at[idx, 'Entry_Price'])
                    shs = float(df.at[idx, 'Shares_Count'])
                    pl_val = (cp - entry_p) * shs
                    
                    df.at[idx, 'Status'] = 'CLOSED'
                    df.at[idx, 'Exit_Price'] = round(cp, 2)
                    df.at[idx, 'Realized_PL'] = round(pl_val, 2)
                    
                    total_exposure -= item['cost'] # Reduce tracked exposure
                    updates_made = True
                    
                    print(f"[{tick}] ✂️ BUDGET CUT: Closing worst performer ({item['unrealized_pct']*100:.2f}%). Freed ${item['cost']:.2f}")
                    
                    alert_txt = (
                        f"✂️ **BUDGET ENFORCEMENT: ${tick}**\n"
                        f"📉 Portfolio Overextended. Liquidating weak position.\n"
                        f"💰 Realized P/L: ${pl_val:.2f}\n"
                        f"📉 Return: {item['unrealized_pct']*100:.2f}%"
                    )
                    send_telegram_alert(alert_txt)

        if updates_made:
            df.to_csv(PERFORMANCE_LOG_FILE, index=False, quoting=csv.QUOTE_ALL)
            print("[*] Portfolio Log Updated: Positions managed.")

    except Exception as e:
        print(f"[!] Error in Auto-Gardener: {e}")


# --- QUANTITATIVE ENGINE ---
def enrich_with_technical_data(catalyst: Catalyst, total_equity: float, buying_power: float) -> Optional[Catalyst]:
    """
    Fetches live data, calculates Position Size based on Total Equity risk, 
    but Caps max size at Available Buying Power.
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
            
        # 3. Position Sizing (Risk Constraint)
        # Use Total Equity for the 2% Risk Rule (Standard Portfolio Sizing)
        risk_amount = total_equity * RISK_PER_TRADE
        shares_risk_based = risk_amount / risk_per_share
        
        # 4. Buying Power Constraint (Cash Constraint)
        # Cannot buy more than we have cash for
        shares_cash_based = buying_power / current_price
        
        # 5. Max Allocation Constraint (20% of Portfolio Value)
        max_allocation_shares = (total_equity * 0.20) / current_price
        
        # Final Shares = Minimum of all constraints
        shares_to_buy = min(shares_risk_based, shares_cash_based, max_allocation_shares)
        
        if shares_to_buy < 1:
             # Try allowing fractional if > 0.1? Or just skip if too small.
             if shares_to_buy * current_price < 20.0: # Minimum trade size check
                 print(f"[!] Position size too small (${shares_to_buy * current_price:.2f}). Skipping.")
                 return None
        
        # Recalculate cost
        projected_cost = shares_to_buy * current_price
            
        if shares_to_buy < 0.0001:
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
        catalyst.shares_count = round(shares_to_buy, 4)
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
        catalyst.risk_r_unit,          # Risk per Share ($)
        0.0,                           # Exit_Price (New)
        0.0                            # Realized_PL (New)
    ]
    
    headers = [
        "Date", "Ticker", "Entry_Price", "Conviction", 
        "Market_Cap", "ATR_Value", "Stop_Loss", "Breakeven_Trigger", 
        "Thesis", "Status", "Shares_Count", "Position_Cost", "Risk_R_Unit",
        "Exit_Price", "Realized_PL"
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
        f"🏹 *Robinhood:* {robinhood_link}\n"
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
    
    # Calculate Equity & Buying Power
    buying_power, current_equity = get_available_buying_power()
    print(f"[*] Equity: ${current_equity:,.2f} | Buying Power: ${buying_power:,.2f}")
    
    # 2. Portfolio Capacity Check (Capital Based)
    # Define a minimum trade size effective for fees/slippage (e.g., $20)
    MIN_TRADE_CAPITAL = 20.0 
    
    if buying_power < MIN_TRADE_CAPITAL:
        print(f"[!] Insufficient capital to trade (Available: ${buying_power:.2f}). Pausing.")
        return # Exit script if verified no cash


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
            # Pass Buying Power constraint
            enriched_c = enrich_with_technical_data(c, current_equity, buying_power)
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