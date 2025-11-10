# ==========================================
# Generate Synthetic India Population Dataset
# For: india_population_dashboard_karthikeya
# Author: Karthikeya
# ==========================================

import os
import numpy as np
import pandas as pd

# Ensure folder exists
os.makedirs("data/raw", exist_ok=True)

# 36 States + UTs (2024 classification)
states = {
    "AN": "Andaman & Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CT": "Chhattisgarh",
    "DN": "Dadra & Nagar Haveli and Daman & Diu",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu & Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PY": "Puducherry",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
}

np.random.seed(42)
rows = []

for code, name in states.items():
    # Assign a base population between 0.05 M – 200 M
    base = np.random.uniform(0.05e6, 200e6)
    # yearly growth rate between 0 – 3 %
    growth = np.random.uniform(0.0, 0.03)

    for year in range(2011, 2022):
        # simulate small random noise ±1 %
        noise = np.random.uniform(-0.01, 0.01)
        pop = base * ((1 + growth + noise) ** (year - 2011))
        rows.append([code, name, year, int(pop)])

# Create DataFrame
df = pd.DataFrame(rows, columns=["state_code", "state_name", "year", "population"])

# Save CSV
df.to_csv("data/raw/population_state_year.csv", index=False)
print("✅ Dataset generated at data/raw/population_state_year.csv")
print(df.head())
