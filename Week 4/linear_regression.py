import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# 1. Load data from CSV file
df = pd.read_csv("study_hours.csv")
df.columns = df.columns.str.strip()

# 2. Seperate features(X) and target (Y)
X = df[['Study_Hours']]
Y = df[['Exam_Score']]

# 3. Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)

# 4.Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train,Y_train)

# 5. Make predictions
Y_pred = model.predict(X_test)

# 6. Evaluate model performance
mae = mean_absolute_error(Y_test,Y_pred)
mse = mean_squared_error(Y_test,Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test,Y_pred)

# 7. Plot actual vs predicted values
plt.figure(figsize = (6,4))
plt.scatter(X_test, Y_test, color = 'yellow', label = 'Actual data')
plt.plot(X_test,Y_pred, color = 'pink', linewidth = 2, label = 'Regression Line')
plt.xlabel('Study_Hours')
plt.ylabel('Exam_Score')
plt.title('Linear Regression: Actual vs Predicted values')
plt.legend()
plt.grid(True)
plt.show()

# 8. Display Performance
print("Model Performance")
print(f"Mean Absolute Error(MAE):{mae:.2f}")
print(f"Mean Squared Error(MSE):{mse:.2f}")
print(f"Root Mean Squared Error(RMSE):{rmse:.2f}")
print(f"R Square Score:{r2:.2f}")