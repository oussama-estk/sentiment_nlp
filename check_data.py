# =================================================================
# check_data.py
# Purpose: Load data, check structure, and visualize class balance.
# =================================================================

import pandas as pd

def check_data():
    print("--- Step 1: Loading Data ---")
    try:
        # Read JSON file
        df = pd.read_json('data.json')
        print("Success: Data loaded.")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 1. Inspect Data
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n" + "="*30 + "\n")

    sentiment_counts = df['sentiment'].value_counts()
    print("--- Sentiment Distribution ---")
    print(sentiment_counts)
    print("\n" + "="*30 + "\n")

if __name__ == "__main__":
    check_data()