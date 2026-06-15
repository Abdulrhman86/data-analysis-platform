"""Generate a deliberately messy sample dataset for smoke-testing / the in-app demo.

Run:  python scripts/generate_sample_data.py
Writes: app/static/sample_sales.csv
"""
import os

import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
n = 220

regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]

df = pd.DataFrame({
    "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.randint(0, 540, n), unit="D"),
    "region": rng.choice(regions, n),                                   # categorical
    "product_category": rng.choice(categories, n, p=[.35, .25, .2, .15, .05]),
    "customer_id": ["CUST-%04d" % i for i in rng.randint(1000, 9999, n)],  # high cardinality
    "units": rng.randint(1, 20, n).astype(float),
    "unit_price": np.round(rng.uniform(5, 200, n), 2),
    "discount": np.round(rng.uniform(0, 0.4, n), 2),
    "satisfaction": rng.randint(1, 6, n).astype(float),                 # 1-5 (regression target)
})
df["revenue"] = np.round(df["units"] * df["unit_price"] * (1 - df["discount"]), 2)

# classification target: churn likelihood rises with low satisfaction + high discount
churn_prob = (0.6 - 0.1 * df["satisfaction"] + 0.5 * df["discount"]).clip(0, 1)
df["churned"] = (rng.uniform(0, 1, n) < churn_prob.values).astype(int)

# --- inject realistic messiness ---
df.loc[rng.choice(n, 18, replace=False), "units"] = np.nan        # missing numeric
df.loc[rng.choice(n, 12, replace=False), "unit_price"] = np.nan   # missing numeric
df.loc[rng.choice(n, 10, replace=False), "region"] = np.nan       # missing categorical
df.loc[rng.choice(n, 5, replace=False), "order_date"] = pd.NaT    # missing date
df.loc[3, "unit_price"] = 9999.0                                  # outlier
df.loc[7, "units"] = 500.0                                        # outlier

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.abspath(os.path.join(here, "..", "app", "static", "sample_sales.csv"))
df.to_csv(out, index=False)
print("wrote", out, df.shape)
print(df.dtypes.to_string())
print("missing per column:")
print(df.isna().sum().to_string())
