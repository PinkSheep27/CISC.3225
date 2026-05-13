import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_wine()

X = data.data
y = data.target

df = pd.DataFrame(data.data, columns=data.feature_names)

print("Available columns:")
print(data.feature_names)
print(df.head())
print("-" * 30)

# 1. Split the data into training and testing sets using train_test_split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Create and train a DecisionTreeClassifier.
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

# 3. Use the trained model to make predictions on the test data.
y_pred_full = full_tree.predict(X_test)

# 4. Compute and print the accuracy of the decision tree.
acc_full = accuracy_score(y_test, y_pred_full)
print(f"Full Decision Tree Accuracy: {acc_full:.4f}")

# 5. Compute and print the confusion matrix.
cm_full = confusion_matrix(y_test, y_pred_full)
print("Confusion Matrix (Full Tree):")
print(cm_full)
print("-" * 30)

# 7. Train a second decision tree using max_depth=2.
pruned_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
pruned_tree.fit(X_train, y_train)
y_pred_pruned = pruned_tree.predict(X_test)

# 8. Compare the accuracy of the full decision tree with the accuracy of the tree using max_depth=2.
acc_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"Pruned Tree (max_depth=2) Accuracy: {acc_pruned:.4f}")

if acc_full > acc_pruned:
    print("The full decision tree performed better on the test data.")
elif acc_pruned > acc_full:
    print("The pruned tree (max_depth=2) performed better on the test data.")
else:
    print("Both models achieved the same accuracy on the test data.")

# 6. Visualize the decision trees.

plt.figure(figsize=(16, 10))
plot_tree(
    full_tree, 
    feature_names=data.feature_names, 
    class_names=data.target_names, 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization (Full Tree)")
plt.show()

# Visualize the pruned tree to see the difference
plt.figure(figsize=(10, 6))
plot_tree(
    pruned_tree, 
    feature_names=data.feature_names, 
    class_names=data.target_names, 
    filled=True, 
    rounded=True,
    fontsize=12
)
plt.title("Decision Tree Visualization (max_depth=2)")
plt.show()