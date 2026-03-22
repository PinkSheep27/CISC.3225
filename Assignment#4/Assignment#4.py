import numpy as np

pm25 = np.array([
    [12, 18, 25, 31, 29, 15],
    [20, 22, 35, 42, 38, 28],
    [8,  12, 18, 21, 19, 14],
    [15, 25, 30, 34, 40, 22]
])

#Part I
shape = pm25.shape
num_stations = shape[0]
num_days = shape[1]

print(f"Shape: {shape}")
print(f"Number of stations: {num_stations}")
print(f"Number of days: {num_days}")

#Part II
station_averages = pm25.mean(axis=1, keepdims=True)
diff_from_avg = pm25 - station_averages

#Part III
station_max = pm25.max(axis=1, keepdims=True)
normalized_pm25 = pm25 / station_max

#Part IV
standard = 35
diff_from_standard = pm25 - standard

#Part V
is_unhealthy = pm25 > 35

#Part VI
offset = np.array([1.5, -0.5, 0.8, 1.9])
corrected_pm25 = pm25 + offset.reshape(-1, 1)

#Part VII
print("Station Averages:\n", station_averages)
print("\nDifference from Average:\n", diff_from_avg)
print("\nNormalized:\n", normalized_pm25)
print("\nDifference from EPA Standard:\n", diff_from_standard)
print("\nBoolean Array:\n", is_unhealthy)
print("\nCorrected Measurements:\n", corrected_pm25)