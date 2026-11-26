import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load data 
df = pd.read_csv("diabetes.csv") 
df.columns =[col.strip() for col in df.columns]

print(df.head())
print(df.columns)

# 2. Separate features and target 
X = df.drop('Outcome', axis=1) 
y = df['Outcome'] 

# 3. Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size=0.2, random_state=42)

# 4. Train the Logistic Regression model 
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train) 

# 5. Make predictions 
y_pred = model.predict(X_test) 
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Evaluate model performance 
acc = accuracy_score(y_test, y_pred) 
prec = precision_score(y_test, y_pred) 
rec = recall_score(y_test, y_pred) 
f1 = f1_score(y_test, y_pred) 
cm = confusion_matrix(y_test, y_pred) 
print("\nModel Performance:") 
print(f"Accuracy: {acc:.4f}") 
print(f"Precision: {prec:.4f}") 
print(f"Recall: {rec:.4f}") 
print(f"F1-score: {f1:.4f}")

# 7. Plot confusion matrix 
plt.figure(figsize=(5,4)) 
sns.heatmap(cm, annot=True, fmt='d', 
cmap='Blues', xticklabels=['No', 'Yes'], 
yticklabels=['No', 'Yes']) 
plt.xlabel('Predicted Label') 
plt.ylabel('Actual Label') 
plt.title('Confusion Matrix') 
plt.grid(True)
plt.show()