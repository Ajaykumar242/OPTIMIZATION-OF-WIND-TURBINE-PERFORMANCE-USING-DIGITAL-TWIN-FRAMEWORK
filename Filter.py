import pandas as pd
import os

# folder where your resampled CSV files are
input_folder = "resampled_outputs"

# folder to save Excel files
output_folder = "excel_2022"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# date range
start_date = "2022-01-01"
end_date = "2022-12-31"

# -----------------------------
# HOURLY DATA
# -----------------------------
hourly = pd.read_csv(
    os.path.join(input_folder, "data_2021_hourly.csv"),
    parse_dates=["Datetime"]
)

hourly_2022 = hourly[
    (hourly["Datetime"] >= start_date) &
    (hourly["Datetime"] <= end_date)
]

hourly_2022.to_excel(
    os.path.join(output_folder, "data_2022_hourly.xlsx"),
    index=False
)

print("Hourly 2022 saved")

# -----------------------------
# DAILY DATA
# -----------------------------
daily = pd.read_csv(
    os.path.join(input_folder, "data_2021_daily.csv"),
    parse_dates=["Datetime"]
)

daily_2022 = daily[
    (daily["Datetime"] >= start_date) &
    (daily["Datetime"] <= end_date)
]

daily_2022.to_excel(
    os.path.join(output_folder, "data_2022_daily.xlsx"),
    index=False
)

print("Daily 2022 saved")

# -----------------------------
# MONTHLY DATA
# -----------------------------
monthly = pd.read_csv(
    os.path.join(input_folder, "data_2021_monthly.csv"),
    parse_dates=["Datetime"]
)

monthly_2022 = monthly[
    (monthly["Datetime"] >= start_date) &
    (monthly["Datetime"] <= end_date)
]

monthly_2022.to_excel(
    os.path.join(output_folder, "data_2022_monthly.xlsx"),
    index=False
)

print("Monthly 2022 saved")
