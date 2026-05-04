import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("--- Original Model Results ---")

# --- TASK 1 ---
accuracy = accuracy_score(y_test, y_pred)
print(f"1. Accuracy: {accuracy:.4f}")

# --- TASK 2 ---
print(f"\n2. First 10 true values: {y_test[:10]}")
print(f"   First 10 predictions: {y_pred[:10]}")

# --- TASK 3 ---
probs = model.predict_proba(X_test_scaled)
print(f"\n3. Probabilities for the first test example: {probs[0]}")

# --- TASK 4 ---
incorrect_indices = np.where(y_pred != y_test)[0]

print("\n4. ", end="")
if len(incorrect_indices) > 0:
    first_incorrect_idx = incorrect_indices[0]
    predicted_val = y_pred[first_incorrect_idx]
    true_val = y_test[first_incorrect_idx]
    print(f"First incorrect prediction is at index {first_incorrect_idx}: Predicted Class {predicted_val}, True Class {true_val}")
else:
    print("No incorrect predictions found in this test set.")

# --- TASK 5 ---
confidences = np.max(probs, axis=1) 

correct_mask = y_pred == y_test
incorrect_mask = y_pred != y_test

avg_conf_correct = np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0.0
avg_conf_incorrect = np.mean(confidences[incorrect_mask]) if np.any(incorrect_mask) else 0.0

print(f"\n5. Average confidence for correct predictions: {avg_conf_correct:.4f}")
print(f"   Average confidence for incorrect predictions: {avg_conf_incorrect:.4f}")

# --- TASK 6 ---
model_small_iter = LogisticRegression(max_iter=50)
model_small_iter.fit(X_train_scaled, y_train)
y_pred_small = model_small_iter.predict(X_test_scaled)

accuracy_small = accuracy_score(y_test, y_pred_small)
print(f"\n6. New accuracy with smaller max_iter (50): {accuracy_small:.4f}")

# --- TASK 7 ---
best_accuracy = 0
best_iter = 0

for i in range(20, 150):
    temp_model = LogisticRegression(max_iter=i)
    temp_model.fit(X_train_scaled, y_train)
    temp_pred = temp_model.predict(X_test_scaled)
    temp_acc = accuracy_score(y_test, temp_pred)
    
    if temp_acc > best_accuracy:
        best_accuracy = temp_acc
        best_iter = i

print(f"\n7. Peak accuracy of {best_accuracy:.4f} achieved at max_iter = {best_iter}")

# --- TASK 8 ---
cm = confusion_matrix(y_test, y_pred)
print("\n8. Confusion Matrix (Original Model):\n", cm)