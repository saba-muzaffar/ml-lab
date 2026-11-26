import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import (accuracy_score, precision_score, 
recall_score, f1_score, 
confusion_matrix, classification_report) 
import matplotlib.pyplot as plt 
import seaborn as sns

# 1. Load data from CSV file 
df = pd.read_csv("play_tennis.csv") 
print("Dataset:") 
print(df.head()) 
print("\nDataset shape:", df.shape) 

# 2. Separate features (X) and target (y) 
X = df.drop('Play', axis=1) 
y = df['Play'] 
X = pd.get_dummies(X)

# 3. Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y )

# 4. Train the Decision Tree model 
model = DecisionTreeClassifier( 
criterion='gini', 
max_depth=3, 
random_state=42) 
model.fit(X_train, y_train)

# 5. Make predictions 
y_pred = model.predict(X_test)

# 6. Evaluate model performance 
acc = accuracy_score(y_test, y_pred) 
prec = precision_score(y_test, y_pred, average='weighted') 
rec = recall_score(y_test, y_pred, average='weighted') 
f1 = f1_score(y_test, y_pred, average='weighted') 
cm = confusion_matrix(y_test, y_pred) 
print("\nModel Performance:") 
print(f"Accuracy: {acc:.4f}") 
print(f"Precision: {prec:.4f}") 
print(f"Recall: {rec:.4f}") 
print(f"F1-score: {f1:.4f}")

# 7. Visualizations 
plt.figure(figsize=(12, 5))

# Decision Tree plot 
plt.subplot(1, 2, 1) 
plot_tree( 
model, 
feature_names=X.columns, 
class_names=["No", "Yes"], 
filled=True, 
rounded=True, 
fontsize=10) 
plt.title("Decision Tree (Play Tennis Dataset)")

# Confusion Matrix 
plt.subplot(1, 2, 2) 
sns.heatmap( 
cm, 
annot=True, 
fmt='d', 
cmap='Blues', 
xticklabels=['No', 'Yes'], 
yticklabels=['No', 'Yes'] ) 
plt.xlabel('Predicted Label') 
plt.ylabel('Actual Label') 
plt.title('Confusion Matrix') 
plt.tight_layout() 
plt.show()

# Classification report 
print("\nClassification Report:") 
print(classification_report(y_test, y_pred))