import numpy as np
import pandas as pd

pages = ["A", "B", "C", "D"]
M = np.array([
    [0.0, 0.5, 0.0, 0.0],
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0, 1.0],
    [0.0, 0.0, 0.5, 0.0],
])
rank = np.array([0.25, 0.25, 0.25, 0.25])

# 1. Display the transition matrix as a Pandas DataFrame
print("1. Transition Matrix as DataFrame:")
df_M = pd.DataFrame(M, index=pages, columns=pages)
print(df_M)
print("-" * 40)

# 2. Check that each column of the matrix adds up to 1.
column_sums = M.sum(axis=0)
print("2. Column sums (should all be 1.0):")
print(column_sums)
print("-" * 40)

# 3. Update the rank vector once using matrix multiplication.
rank_updated_once = M @ rank
print("3. Rank vector after 1 update:")
print(rank_updated_once)
print("-" * 40)

# 4. Reset the rank vector and use a loop to update it 200 times.
rank = np.array([0.25, 0.25, 0.25, 0.25])
for _ in range(200):
    rank = M @ rank

print("4. Rank vector after 200 iterations:")
print(rank)
print("-" * 40)

# 5. Convert the final rank vector into a Pandas Series using page names.
rank_series = pd.Series(rank, index=pages)
print("5. Final rank vector as a Pandas Series:")
print(rank_series)
print("-" * 40)

# 6. Sort and print the final ranks from highest to lowest.
sorted_ranks = rank_series.sort_values(ascending=False)
print("6. Final Ranks (Highest to Lowest):")
print(sorted_ranks)
print("-" * 40)

# 7. Add damping and run the damped version for 20 steps.
damped_rank = np.array([0.25, 0.25, 0.25, 0.25])

for _ in range(20):
    damped_rank = 0.85 * (M @ damped_rank) + 0.15 * (np.ones(len(pages)) / len(pages))

damped_series = pd.Series(damped_rank, index=pages)
sorted_damped_ranks = damped_series.sort_values(ascending=False)

print("7. Damped Version Ranks after 20 steps (Highest to Lowest):")
print(sorted_damped_ranks)