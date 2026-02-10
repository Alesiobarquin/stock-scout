
import os
import sys
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import pytz

# Add the project directory to sys.path to import Catalyst
sys.path.append('/Users/alesio/Developer/Projects/stock-scout')

from alpha_scout import Catalyst, format_telegram_message

def test_format():
    test_catalyst = Catalyst(
        ticker="AAPL",
        current_price=150.00,
        market_cap="2.5T",
        conviction_score=8,
        thesis="Apple is doing great.",
        catalyst_details="New iPhone launch.",
        absorption_status="Not yet priced in.",
        earnings_date="2024-05-01",
        relative_volume="1.2x",
        stop_loss_trigger="$140.00",
        sentiment="Bullish",
        prediction_market="70%",
        recency_proof="https://apple.com",
        risk="Supply chain issues.",
        expected_upside="10%",
        mispricing_evidence="Under-valued R&D.",
        hold_time_estimate="2-4 Weeks",
        shares_count=10,
        position_cost=1500.00,
        calculated_stop_loss=140.00,
        breakeven_trigger=160.00,
        risk_r_unit=10.00
    )
    
    msg = format_telegram_message(test_catalyst)
    print("--- FORMATTED MESSAGE ---")
    print(msg)
    print("-------------------------")
    
    if "https://robinhood.com/us/en/stocks/AAPL/" in msg:
        print("SUCCESS: Robinhood link found in message.")
    else:
        print("FAILURE: Robinhood link NOT found in message.")
        sys.exit(1)

if __name__ == "__main__":
    test_format()
