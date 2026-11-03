"""
generate_banking_data.py
------------------------
Generates synthetic retail + wholesale banking data for the
databricks_bank_lakehouse_360 demo.

Output: Parquet files under ./landing/<table_name>/
Tables:
    landing_customer
    landing_account
    landing_card
    landing_merchant
    landing_transaction   (partitioned by txn_date)

Run:
    pip install faker pandas pyarrow numpy
    python generate_banking_data.py
"""

import os
import random
import uuid
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from faker import Faker

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

OUT_DIR = "landing"
START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)   # 2 years
N_CONSUMER_CUSTOMERS = 5000
N_WHOLESALE_CUSTOMERS = 300
N_MERCHANTS = 800
TXN_PER_DAY_AVG = 3000          # consumer daily volume average

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Reference data
# -------------------------------------------------------------------
CONSUMER_SEGMENTS = ["Retail", "Mass Affluent", "Private"]
RISK_RATINGS = ["Low", "Medium", "High"]
CURRENCIES = ["MYR", "USD", "SGD"]
ACCOUNT_TYPES_CONSUMER = ["Savings", "Current"]
CARD_TYPES = ["Debit", "Credit"]
CARD_STATUSES = ["Active", "Frozen", "Expired", "Blocked"]
CHANNELS_CONSUMER = ["ATM", "POS", "Online", "Branch"]
AUTH_RESULTS = ["Approved", "Approved", "Approved", "Approved", "Declined"]

INDUSTRIES_WHOLESALE = [
    "Manufacturing", "Trading", "Construction", "Logistics",
    "Technology", "Healthcare", "Energy", "Retail Chain",
]
CHANNELS_WHOLESALE = ["Wire", "RTGS", "Cheque", "Online Banking"]
TXN_TYPES_WHOLESALE = ["Transfer", "Trade Finance", "Payroll", "Supplier Payment"]

# Merchant category codes (MCC) with realistic weights
MCC_POOL = [
    ("5411", "Grocery Stores", 0.18),
    ("5812", "Restaurants", 0.15),
    ("5541", "Fuel Stations", 0.12),
    ("5912", "Pharmacies", 0.08),
    ("5311", "Department Stores", 0.10),
    ("4111", "Transport", 0.07),
    ("5732", "Electronics", 0.06),
    ("7995", "Gambling", 0.02),
    ("6011", "ATM Cash", 0.10),
    ("4899", "Utilities", 0.05),
    ("5999", "Misc Retail", 0.07),
]
MCC_CODES = [m[0] for m in MCC_POOL]
MCC_NAMES = {m[0]: m[1] for m in MCC_POOL}
MCC_WEIGHTS = [m[2] for m in MCC_POOL]
MCC_WEIGHTS = np.array(MCC_WEIGHTS) / sum(MCC_WEIGHTS)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def uid(prefix=""):
    return f"{prefix}{uuid.uuid4().hex[:12].upper()}"

def date_range(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def log_normal_amount(mu=4.5, sigma=1.1, low=1.0, high=15000.0):
    amt = np.random.lognormal(mean=mu, sigma=sigma)
    return round(float(np.clip(amt, low, high)), 2)

def wholesale_amount():
    # large corporate values
    amt = np.random.lognormal(mean=10.5, sigma=1.2)
    return round(float(np.clip(amt, 1000.0, 5_000_000.0)), 2)

# -------------------------------------------------------------------
# 1. Customers
# -------------------------------------------------------------------
def build_customers():
    rows = []
    for _ in range(N_CONSUMER_CUSTOMERS):
        cust_id = uid("C")
        onboard = fake.date_between(start_date="-6y", end_date=START_DATE)
        rows.append({
            "customer_id": cust_id,
            "customer_segment": "Consumer",
            "sub_segment": random.choice(CONSUMER_SEGMENTS),
            "full_name": fake.name(),
            "gender": random.choice(["M", "F"]),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=75),
            "nationality": random.choices(
                ["MY", "SG", "ID", "IN", "CN"], weights=[0.7, 0.1, 0.08, 0.07, 0.05]
            )[0],
            "risk_rating": random.choices(RISK_RATINGS, weights=[0.7, 0.25, 0.05])[0],
            "onboard_date": onboard,
            "home_city": fake.city(),
        })

    for _ in range(N_WHOLESALE_CUSTOMERS):
        cust_id = uid("W")
        onboard = fake.date_between(start_date="-10y", end_date=START_DATE)
        rows.append({
            "customer_id": cust_id,
            "customer_segment": "Wholesale",
            "sub_segment": random.choice(INDUSTRIES_WHOLESALE),
            "full_name": fake.company(),
            "gender": None,
            "date_of_birth": None,
            "nationality": "MY",
            "risk_rating": random.choices(RISK_RATINGS, weights=[0.5, 0.35, 0.15])[0],
            "onboard_date": onboard,
            "home_city": fake.city(),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT_DIR}/landing_customer", index=False)
    print(f"[customer] {len(df)} rows")
    return df

# -------------------------------------------------------------------
# 2. Accounts
# -------------------------------------------------------------------
def build_accounts(customers):
    rows = []
    for _, c in customers.iterrows():
        n_accts = 1 if c.customer_segment == "Wholesale" else random.choices([1, 2, 3], [0.65, 0.25, 0.10])[0]
        for _ in range(n_accts):
            if c.customer_segment == "Wholesale":
                acct_type = random.choice(["Current", "Trade", "Settlement"])
            else:
                acct_type = random.choice(ACCOUNT_TYPES_CONSUMER)
            rows.append({
                "account_id": uid("A"),
                "customer_id": c.customer_id,
                "account_type": acct_type,
                "currency": random.choices(CURRENCIES, weights=[0.85, 0.10, 0.05])[0],
                "open_date": fake.date_between(start_date=c.onboard_date, end_date=START_DATE),
                "status": "Active",
                "branch_code": f"BR{random.randint(1, 60):03d}",
            })
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT_DIR}/landing_account", index=False)
    print(f"[account] {len(df)} rows")
    return df

# -------------------------------------------------------------------
# 3. Cards
# -------------------------------------------------------------------
def build_cards(accounts, customers):
    rows = []
    cust_map = customers.set_index("customer_id")["customer_segment"].to_dict()
    accts_by_cust = accounts.groupby("customer_id")["account_id"].apply(list).to_dict()

    for cust_id, segment in cust_map.items():
        if segment != "Consumer":
            continue
        cust_accts = accts_by_cust.get(cust_id, [])
        if not cust_accts:
            continue
        n_cards = random.choices([1, 2], [0.75, 0.25])[0]
        for _ in range(n_cards):
            acct_id = random.choice(cust_accts)
            acct_open = accounts.loc[accounts.account_id == acct_id, "open_date"].iloc[0]
            issue = fake.date_between(start_date=acct_open, end_date=START_DATE)
            expiry = issue + timedelta(days=365 * random.choice([3, 4, 5]))
            rows.append({
                "card_id": uid("D"),
                "account_id": acct_id,
                "customer_id": cust_id,
                "card_type": random.choice(CARD_TYPES),
                "card_network": random.choice(["VISA", "MasterCard", "AMEX"]),
                "card_status": "Active",
                "issue_date": issue,
                "expiry_date": expiry,
                "credit_limit": round(np.random.choice([5000, 10000, 20000, 50000]), 2),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT_DIR}/landing_card", index=False)
    print(f"[card] {len(df)} rows")
    return df

# -------------------------------------------------------------------
# 4. Merchants
# -------------------------------------------------------------------
def build_merchants():
    rows = []
    for _ in range(N_MERCHANTS):
        mcc = random.choices(MCC_CODES, weights=MCC_WEIGHTS)[0]
        rows.append({
            "merchant_id": uid("M"),
            "mcc": mcc,
            "mcc_description": MCC_NAMES[mcc],
            "merchant_name": fake.company(),
            "country": random.choices(["MY", "SG", "US", "CN"], weights=[0.8, 0.08, 0.07, 0.05])[0],
            "onboard_date": fake.date_between(start_date="-8y", end_date=START_DATE),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT_DIR}/landing_merchant", index=False)
    print(f"[merchant] {len(df)} rows")
    return df

# -------------------------------------------------------------------
# 5. Transactions
# -------------------------------------------------------------------
def consumer_txn_amount(mcc):
    # ATM cash â†’ fixed denominations
    if mcc == "6011":
        return random.choice([50, 100, 200, 300, 500, 800, 1000])
    # Utilities / grocery lower, electronics higher
    if mcc in ("4899", "5411", "5912"):
        return log_normal_amount(mu=4.0, sigma=0.7, low=5, high=1500)
    if mcc in ("5732",):
        return log_normal_amount(mu=6.0, sigma=0.9, low=100, high=8000)
    return log_normal_amount()

def build_transactions(customers, accounts, cards, merchants):
    # maps
    acct_by_cust = accounts.groupby("customer_id")["account_id"].apply(list).to_dict()
    cards_by_cust = cards.groupby("customer_id").apply(
        lambda d: d.to_dict("records"), include_groups=False
    ).to_dict() if not cards.empty else {}
    consumers = customers[customers.customer_segment == "Consumer"]
    wholesalers = customers[customers.customer_segment == "Wholesale"]
    merchants_list = merchants.to_dict("records")

    all_rows = []

    for day in date_range(START_DATE, END_DATE):
        dow = day.weekday()
        # weekend uplift for consumer
        base = TXN_PER_DAY_AVG * (1.25 if dow >= 5 else 1.0)
        n_consumer = int(np.random.poisson(base))

        # ---- consumer txns ----
        sampled = consumers.sample(n=min(n_consumer, len(consumers)), replace=True)
        for _, c in sampled.iterrows():
            cs = cards_by_cust.get(c.customer_id)
            if not cs:
                continue
            card = random.choice(cs)
            if card["card_status"] != "Active":
                continue
            mcc = random.choices(MCC_CODES, weights=MCC_WEIGHTS)[0]
            merchant = random.choice([m for m in merchants_list if m["mcc"] == mcc] or merchants_list)
            channel = "ATM" if mcc == "6011" else random.choice(["POS", "Online"])
            # time-of-day: consumer peaks evening
            hour = int(np.clip(np.random.normal(18, 4), 0, 23))
            ts = datetime.combine(day, datetime.min.time()).replace(hour=hour,
                                                                   minute=random.randint(0, 59),
                                                                   second=random.randint(0, 59))
            all_rows.append({
                "txn_id": uid("T"),
                "customer_id": c.customer_id,
                "account_id": card["account_id"],
                "card_id": card["card_id"],
                "merchant_id": merchant["merchant_id"],
                "segment": "Consumer",
                "channel": channel,
                "txn_type": "Purchase" if channel != "ATM" else "Withdrawal",
                "mcc": mcc,
                "amount": consumer_txn_amount(mcc),
                "currency": "MYR",
                "txn_ts": ts,
                "auth_result": random.choices(AUTH_RESULTS, weights=[0.97, 0.0, 0.0, 0.0, 0.03])[0],
                "counterparty_country": merchant["country"],
            })

        # ---- wholesale txns ---- (lower volume)
        n_wholesale = max(1, int(np.random.poisson(15)))
        sampled_w = wholesalers.sample(n=min(n_wholesale, len(wholesalers)), replace=True)
        for _, w in sampled_w.iterrows():
            accts = acct_by_cust.get(w.customer_id)
            if not accts:
                continue
            # business-hours concentration
            hour = int(np.clip(np.random.normal(11, 2), 8, 17))
            ts = datetime.combine(day, datetime.min.time()).replace(hour=hour,
                                                                   minute=random.randint(0, 59),
                                                                   second=random.randint(0, 59))
            all_rows.append({
                "txn_id": uid("T"),
                "customer_id": w.customer_id,
                "account_id": random.choice(accts),
                "card_id": None,
                "merchant_id": None,
                "segment": "Wholesale",
                "channel": random.choice(CHANNELS_WHOLESALE),
                "txn_type": random.choice(TXN_TYPES_WHOLESALE),
                "mcc": None,
                "amount": wholesale_amount(),
                "currency": random.choices(["MYR", "USD", "SGD"], weights=[0.7, 0.2, 0.1])[0],
                "txn_ts": ts,
                "auth_result": "Approved",
                "counterparty_country": random.choices(["MY", "SG", "US", "CN"], weights=[0.6, 0.15, 0.15, 0.10])[0],
            })

    df = pd.DataFrame(all_rows)
    df["txn_date"] = pd.to_datetime(df["txn_ts"]).dt.date.astype(str)
    df.to_parquet(
        f"{OUT_DIR}/landing_transaction",
        index=False,
        partition_cols=["txn_date"],
        engine="pyarrow",
    )
    print(f"[transaction] {len(df)} rows (partitioned by txn_date)")
    return df

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating banking sample data...")
    customers = build_customers()
    accounts = build_accounts(customers)
    cards = build_cards(accounts, customers)
    merchants = build_merchants()
    txns = build_transactions(customers, accounts, cards, merchants)
    print("Done. Output in ./landing")