import os
import json
import time
import requests
import re
import csv
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
import pytz

# --- CONFIGURATION ---
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-002") 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "data/latest_report.json"
PERFORMANCE_LOG_FILE = "data/performance_log.csv"

# --- DATA MODELS (UPGRADED) ---
class Catalyst(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (e.g., AAPL)")
    current_price: float = Field(..., description="The stock price at the time of alert generation.")
    market_cap: str = Field(..., description="Market Cap string (e.g., '$450M', '$2.1B').")
    conviction_score: int = Field(..., description="1-10 Score. 8+ requires hard date and institutional backing.")
    thesis: str = Field(..., description="1-2 sentence thesis focusing on the inefficiency.")
    catalyst_details: str = Field(..., description="Specific event details and timing.")
    absorption_status: str = Field(..., description="Why the news is not fully priced in (e.g., 'Low volume reaction', 'Post-market news').")
    earnings_date: str = Field(..., description="Next earnings date. Mark 'Past' if recently reported, or specific date.")
    relative_volume: str = Field(..., description="Current vol vs 30d avg (e.g., '2.5x 30d avg').")
    stop_loss_trigger: str = Field(..., description="Specific price or event that invalidates the thesis.")
    sentiment: str = Field(..., description="Bullish, Bearish, or Mixed")
    prediction_market: str = Field(..., description="Source, Odds, and 24h change if available.")
    recency_proof: str = Field(..., description="Source Link and Timestamp.")
    risk: str = Field(..., description="Primary invalidation factor.")
    expected_upside: str = Field(..., description="Quantified potential (e.g., '10-20% drift').")
    mispricing_evidence: str = Field(..., description="Why not priced in (e.g., 'Price flat despite news').")
    x_sentiment: Optional[str] = Field(None, description="X buzz summary.")

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

# --- PERFORMANCE LOGGER (NEW) ---
def log_alert_to_csv(catalyst: Catalyst):
    """Appends a catalyst alert to the performance log CSV."""
    os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(PERFORMANCE_LOG_FILE)
    
    row_data = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        catalyst.ticker,
        catalyst.current_price,
        catalyst.conviction_score,
        catalyst.market_cap,
        catalyst.expected_upside,
        catalyst.thesis,
        'OPEN'  # Default status
    ]
    
    with open(PERFORMANCE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ["Timestamp", "Ticker", "Entry_Price", "Conviction", "Market_Cap", "Expected_Upside", "Thesis", "Status"]
            writer.writerow(headers)
        writer.writerow(row_data)
    
    print(f"[*] Logged alert for {catalyst.ticker} to {PERFORMANCE_LOG_FILE}")

# --- AGENT SETUP ---
def get_alpha_scout_response():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    now = datetime.now()
    current_year = now.year
    three_days_ago = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    today_str = now.strftime('%Y-%m-%d')

    system_instruction = f"""
    Role: You are “Alpha Scout,” a Quant-Fundamental Analyst finding "Asymmetric Upside."
    STRATEGY FOCUS: PEAD, Biotech PDUFA Run-ups, Insider Aggression (Form 4), Macro Rotations.
    STRICT FILTERS:
    1. "Priced-In" Check: DISCARD any stock that has already moved >8% on the day of the news.
    2. Source Weighting: Prioritize SEC Edgar (Form 4, 8-K) and FDA Calendars.
    3. Liquidity: Focus on Market Caps between $500M and $10B. Avoid Micro-caps (<$300M).
    4. Risk Management: You must identify a specific 'Stop Loss Trigger'.
    FORBIDDEN DATES: Today is {today_str} ({current_year}). You MUST strictly ignore any search results from {current_year - 1} or earlier.
    """

    prompt = f"""
    Current Date: {today_str}
    Perform a deep sweep for unpriced bullish catalysts published ONLY between {three_days_ago} and {today_str}.
    CRITICAL: For each candidate, you MUST find the current stock price. You must also provide a clear 'absorption_status' explaining why the market has not fully reacted yet (e.g., 'News broke post-market', 'Low relative volume reaction', 'Overshadowed by macro event').
    Compile the JSON report.
    """

    tools = [types.Tool(google_search=types.GoogleSearch())]
    response = client.models.generate_content(
        model=MODEL_ID, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction, tools=tools,
            response_mime_type="application/json", response_schema=ScoutReport
        )
    )
    return response.parsed

# --- OUTPUT HANDLERS ---
def save_to_json(report: ScoutReport):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"[*] Data saved to {DATA_FILE}")

def format_telegram_message(catalyst: Catalyst) -> str:
    """Formats the catalyst data into a highly readable Telegram message."""
    try:
        # Use pytz for accurate timezone conversion
        ny_tz = pytz.timezone("America/New_York")
        now_ny = datetime.now(ny_tz).strftime('%H:%M %Z')
    except pytz.UnknownTimeZoneError:
        now_ny = datetime.now().strftime('%H:%M UTC')

    robinhood_link = f"https://robinhood.com/us/en/stocks/{catalyst.ticker}/"

    return (
        f"🚀 *Alpha Scout Signal: ${catalyst.ticker}*\n"
        f"📈 *Price:* ${catalyst.current_price:.2f} | *Mkt Cap:* {catalyst.market_cap}\n"
        f"🔗 *[Trade on Robinhood]({robinhood_link})*\n"
        f"━━━━━\n"
        f"*Conviction: {catalyst.conviction_score}/10*\n"
        f"*Upside:* {catalyst.expected_upside}\n"
        f"━━━━━\n"
        f"💡 *Thesis:*\n{catalyst.thesis}\n\n"
        f"📊 *Absorption Status:*\n{catalyst.absorption_status}\n\n"
        f"📅 *Catalyst:*\n{catalyst.catalyst_details}\n\n"
        f"🛑 *Stop Loss:*\n{catalyst.stop_loss_trigger}\n"
        f"━━━━━\n"
        f"🔗 *Proof:* {catalyst.recency_proof}\n"
        f"_Generated at {now_ny}_"
    )

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram credentials not found. Skipping message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        print(f"[*] Telegram sent.")
    except Exception as e:
        print(f"[!] Failed to send Telegram: {e}")

# --- MAIN EXECUTION (UPGRADED) ---
def main():
    print("[-] Alpha Scout initializing (Quant-Fundamental Mode)...")
    try:
        report = get_alpha_scout_response()
        if not report or not report.catalysts:
            print("[!] No catalysts found.")
            return
        
        print(f"[-] Analyzing {len(report.catalysts)} raw candidates...")
        filtered_catalysts = []
        for c in report.catalysts:
            upside_val = parse_upside_percentage(c.expected_upside)
            mcap_millions = parse_market_cap_to_millions(c.market_cap)
            is_liquid_enough = mcap_millions >= 300
            is_sweet_spot = 500 <= mcap_millions <= 10000
            
            final_score = c.conviction_score + 0.5 if is_sweet_spot else c.conviction_score
            
            if (c.sentiment == "Bullish" and upside_val >= 8 and is_liquid_enough and final_score >= 7.5):
                c.conviction_score = min(10, int(final_score))
                filtered_catalysts.append(c)

        if not filtered_catalysts:
            print("[-] No catalysts met the strict Quant-Fundamental criteria.")
            save_to_json(report)
            return
        
        filtered_catalysts.sort(key=lambda c: c.conviction_score, reverse=True)
        top_picks = filtered_catalysts[:3]
        
        final_report = ScoutReport(catalysts=filtered_catalysts)
        save_to_json(final_report)
        
        # LOG before ALERTING
        for item in top_picks:
            log_alert_to_csv(item)  # <-- New logging step
            msg = format_telegram_message(item)
            send_telegram_alert(msg)
            time.sleep(1)
        
        print(f"[*] Processed {len(filtered_catalysts)} valid signals. Sent {len(top_picks)} alerts.")

    except Exception as e:
        print(f"[!] Critical Error: {e}")

if __name__ == "__main__":
    main()