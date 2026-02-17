import pandas as pd
import yfinance as yf
df = pd.read_csv("data/performance_log.csv")
# Find rows with Exit_Price as NaN but Status is CLOSED/STOPPED
broken_mask = (df["Status"].isin(["CLOSED", "STOPPED_OUT"])) & (df["Exit_Price"].isna())
if broken_mask.any():
    print(f"Found {broken_mask.sum()} broken rows. Resetting to OPEN...")
    df.loc[broken_mask, "Status"] = "OPEN"
    df.loc[broken_mask, "Exit_Price"] = 0.0
    df.loc[broken_mask, "Realized_PL"] = 0.0
    df.to_csv("data/performance_log.csv", index=False, quoting=1) # QUOTE_ALL
    print("Repaired CSV.")
else:
    print("No broken rows found.")

