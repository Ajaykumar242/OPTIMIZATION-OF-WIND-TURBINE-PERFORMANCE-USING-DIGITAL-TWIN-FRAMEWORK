import pandas as pd
import os

file_name = "data_2021.csv"
output_folder = "resampled_outputs"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

time_col = "Datetime"

cols = [
    "RotorSpeed",
    "GeneratorSpeed",
    "GeneratorTemperature",
    "WindSpeed",
    "PowerOutput",
    "SpeiseSpannung",
    "StatusAnlage",
    "MaxWindHeute",
    "offsetWindDirection",
    "PitchDeg"
]

chunk_size = 1000000

hourly_list = []

reader = pd.read_csv(file_name, chunksize=chunk_size)

for i, chunk in enumerate(reader):
    print("chunk:", i + 1)

    # keep only needed columns (saves RAM)
    chunk = chunk[[time_col] + cols]

    # convert time
    chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")

    # remove bad timestamps
    chunk = chunk.dropna(subset=[time_col])

    # numeric conversion (StatusAnlage might already be integer, but this is safe)
    for c in cols:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

    # set index for resampling
    chunk = chunk.set_index(time_col).sort_index()

    # 1 Hz -> hourly mean
    hourly_chunk = chunk.resample("1h").mean()

    hourly_list.append(hourly_chunk)

# combine all hourly chunks
hourly = pd.concat(hourly_list)

# if same hour appears in multiple chunks, average again
hourly = hourly.groupby(hourly.index).mean()
hourly = hourly.sort_index()

# save hourly
hourly.to_csv(os.path.join(output_folder, "data_2021_hourly.csv"))
print("saved hourly")

# hourly -> daily/monthly
daily = hourly.resample("1D").mean()
monthly = hourly.resample("MS").mean()

daily.to_csv(os.path.join(output_folder, "data_2021_daily.csv"))
monthly.to_csv(os.path.join(output_folder, "data_2021_monthly.csv"))

print("saved daily and monthly")

