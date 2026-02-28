import numpy as np

np.random.seed(42)
aqi_data = np.random.normal(75,20,28).astype(int)

print(f"Part I: Data Generation")
print(f"Full AQI Array for all 28 neighborhoods: {aqi_data}\n")


mean_aqi = np.mean(aqi_data)
max_aqi = np.max(aqi_data)
min_aqi = np.min(aqi_data)
count_high = np.sum(aqi_data >= 100)
count_low = np.sum(aqi_data < 50)

print(f"Part II: Basic Summary")
print(f"Mean AQI: {mean_aqi:.2f}")
print(f"Maximum AQI: {max_aqi}")
print(f"Minimum AQI: {min_aqi}")
print(f"Number of neighborhoods with AQI >= 100: {count_high}")
print(f"Number of neighborhoods with AQI < 50: {count_low}\n")


high_risk_mask = aqi_data >= 100
unhealthy_count = np.sum(aqi_data >= 101)

print(f"Part III: High-Risk Areas")
print(f"Boolean mask for AQI >= 100: {high_risk_mask}")
print(f"AQI values using the mask (>= 100): {aqi_data[high_risk_mask]}")
print(f"Neighborhoods in the 'Unhealthy' (101+) category: {unhealthy_count}\n")


intervention_indices = [2, 7, 14, 21]
aqi_data[intervention_indices] -= 15
new_mean_aqi = np.mean(aqi_data)

print(f"Part IV: Targeted Intervention")
print(f"Updated AQI Array after filtration systems: {aqi_data}")
print(f"Recomputed Mean AQI: {new_mean_aqi:.2f}\n")

#Did the intervention significantly change the city-wide average?
#No, it did not significantly change the city-wide average.

#Why might local improvements not strongly affect global averages?
#The mean is influenced by all values in the dataset. 
#If the 4 targeted neighborhoods had AQI values that were not extremely high to begin with, then reducing them by 15 might not have a large impact on the overall average.


sorted_aqi = np.sort(aqi_data)
lowest_5 = sorted_aqi[:5]
highest_5 = sorted_aqi[-5:][::-1]

print(f"Part V: Ranking Severity")
print(f"The 5 lowest AQI readings (ascending): {lowest_5}")
print(f"The 5 highest AQI readings (descending): {highest_5}\n")

#Is ranking neighborhoods by AQI useful?
#Yes, it can be useful for prioritizing neighborhoods that need immediate attention.

#Is it always fair to focus only on the worst 5?
#No, it may not be fair to focus only on the worst 5 because neglecting the rest of the neighborhoods can lead to more problems.

#What information is lost when we sort?
#Once sorted, we no longer know which neighborhood corresponds to which AQI value unless we specifically use np.argsort() to keep track of the original indices.


aqi_data[0] = 250
outlier_mean = np.mean(aqi_data)

print(f"Part VI: Outlier Scenario")
print(f"Recomputed Mean AQI with extreme outlier (250): {outlier_mean:.2f}")
#How sensitive is the mean to extreme values?
#The mean is very sensitive to extreme values.
#An outlier can significantly skew the average, making it unrepresentative of the typical conditions in the city.

#Should a single outlier influence city-wide policy?
#No, a single outlier should not influence city-wide policy as it may not reflect the overall air quality conditions in the city.