import numpy as np

np.random.seed(42)

# Part 1: Generate Data & Neighborhoods

aqi_values = np.random.normal(75,20,28).astype(int)

neighborhoods = np.array([f"N{i:02d}" for i in range(28)])

print("Initial Data Arrays")
print(f"Neighborhood Identifier Array:\n{neighborhoods}\n")
print(f"Raw AQI Values Array:\n{aqi_values}\n")

print("Neighborhood AQI Pairings")

for n, aqi in zip(neighborhoods, aqi_values):
    print(f"{n} : AQI = {aqi}")
print()

# Part 2: Sorting AQI Values Only

sorted_aqi = np.sort(aqi_values)

print("Basic Sorted AQI Values")
print(f"The 5 lowest AQI values (ascending): {sorted_aqi[:5]}")
print(f"The 5 highest AQI values (descending): {sorted_aqi[-5:][::-1]}\n")

# Part 3: Ranking Using argsort()

sort_indices = np.argsort(aqi_values)

print("Top 5 Cleanest Neighborhoods")
cleanest_indices = sort_indices[:5]
for idx in cleanest_indices:
    print(f"{neighborhoods[idx]} : AQI = {aqi_values[idx]}")
print()

print("Top 5 Most Polluted Neighborhoods")
polluted_indices = sort_indices[-5:][::-1] 
for idx in polluted_indices:
    print(f"{neighborhoods[idx]} : AQI = {aqi_values[idx]}")
print()

# Part 4: Targeted Intervention

intervention_indices = [2, 7, 14, 21]

aqi_values[intervention_indices] -= 15

print("Post-Intervention Analysis")
print(f"Updated AQI Array after installing filtration systems: \n{aqi_values}")
print(f"Recomputed City-Wide Mean AQI: {np.mean(aqi_values):.2f}\n")

# Part 5: Recomputing Rankings Post-Intervention

# We must recompute argsort because the underlying AQI data has changed
new_sort_indices = np.argsort(aqi_values)
new_worst_indices = new_sort_indices[-5:][::-1]

print("Top 5 Worst Neighborhoods (Post-Intervention)")
for idx in new_worst_indices:
    print(f"{neighborhoods[idx]} : AQI = {aqi_values[idx]}")