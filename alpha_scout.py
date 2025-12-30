import os
import json
import time
import requests
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# Update 'gemini-1.5-pro-002' to 'gemini-3-pro' when available
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-002") 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "data/latest_report.json"

# --- DATA MODELS ---
class Catalyst(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (e.g., AAPL)")
    conviction_score: int = Field(..., description="1-10 Score. 8+ requires hard date.")
    thesis: str = Field(..., description="1-2 sentence thesis.")
    catalyst_details: str = Field(..., description="Specific event details and timing.")
    sentiment: str = Field(..., description="Bullish, Bearish, or Mixed")
    prediction_market: str = Field(..., description="Source, Odds, and 24h change if available.")
    recency_proof: str = Field(..., description="Source Link and Timestamp.")
    risk: str = Field(..., description="Primary invalidation factor.")

class ScoutReport(BaseModel):
    catalysts: List[Catalyst]

# --- AGENT SETUP ---
def get_alpha_scout_response():
    client = genai.Client(api_key=os.getenv("AIzaSyA62C1qiD5fzVXC-LePfQPGv0MEI9qySpE"))

    # System Instructions
    system_instruction = """
    Role: You are “Alpha Scout,” a senior event‑driven analyst.
    
    Constraints:
    1. 72h Recency: ONLY items published in the last 72 hours.
    2. Source Priority: SEC.gov, FDA.gov, Official IR, Tier-1 news (Reuters/Bloomberg).
    3. Prediction Markets: MUST cross-reference all picks with Polymarket or Kalshi odds. 
       If direct odds aren't found, search for proxy markets (e.g., "Fed rate cut odds" for bank stocks).
    4. Logic: Deduplicate news; Normalize names to Tickers.
    5. Conviction: Rate 1–10. Score 8+ requires a "Hard Date" (confirmed earnings, PDUFA, split date).
    
    Task:
    Search for the most significant financial catalysts (FDA approvals, Earnings surprises, M&A, Macro events) 
    from the last 72 hours. Verify sentiment and probability using prediction market data found via search.
    Return a list of valid catalysts.
    """

    # Tool Configuration: Google Search
    tools = [types.Tool(google_search=types.GoogleSearch())]

    # Prompt Construction
    prompt = f"""
    Current Date: {datetime.now().strftime('%Y-%m-%d')}
    
    Perform a deep sweep for financial catalysts. 
    1. Search for trending tickers and breaking news on SEC, FDA, and Reuters in the last 72h.
    2. For each potential candidate, SEARCH for its specific prediction market odds on Polymarket or Kalshi.
    3. Compile the data into the defined JSON schema.
    """

    # Generate Content with Structured Output
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

    return response.parsed

# --- OUTPUT HANDLERS ---
def save_to_json(report: ScoutReport):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"[*] Data saved to {DATA_FILE}")

def format_telegram_message(catalyst: Catalyst) -> str:
    return (
        f"🚀 *{catalyst.ticker}* - *{catalyst.conviction_score}/10*\n"
        f"- *Thesis:* {catalyst.thesis}\n"
        f"- *Catalyst:* {catalyst.catalyst_details}\n"
        f"- *Sentiment:* {catalyst.sentiment}\n"
        f"- *Prediction Market:* {catalyst.prediction_market}\n"
        f"- *Recency Proof:* {catalyst.recency_proof}\n"
        f"- *Risk:* {catalyst.risk}"
    )

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram credentials not found. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        print(f"[*] Telegram sent for ticker.")
    except Exception as e:
        print(f"[!] Failed to send Telegram: {e}")

# --- MAIN EXECUTION ---
def main():
    print("[-] Alpha Scout initializing...")
    try:
        report = get_alpha_scout_response()
        
        if not report or not report.catalysts:
            print("[!] No catalysts found.")
            return

        # Filter for Conviction >= 7
        high_conviction = [c for c in report.catalysts if c.conviction_score >= 7]
        
        if not high_conviction:
            print("[-] No catalysts met the conviction threshold (>=7).")
            # We still save the full report for records, or you can choose to save empty
            save_to_json(report) 
            return

        # Save filtered list to JSON (or full list, depending on preference. 
        # Instructions say "Overwrite with the result", implying the filtered result or full result.
        # We will save the high conviction ones to be safe).
        filtered_report = ScoutReport(catalysts=high_conviction)
        save_to_json(filtered_report)

        # Send Telegram Messages
        for item in high_conviction:
            msg = format_telegram_message(item)
            send_telegram_alert(msg)
            time.sleep(1) # Rate limit safety

    except Exception as e:
        print(f"[!] Critical Error: {e}")
        # Optional: Send error log to Telegram

if __name__ == "__main__":
    main()