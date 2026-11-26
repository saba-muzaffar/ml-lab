import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load the IRIS dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
feature_names = iris.feature_names 
target_names = iris.target_names 

# 2. Create 2D scatter plots for different feature pairs 
plt.figure(figsize=(12, 4)) 
plt.subplot(1, 3, 1) 
for i, name in enumerate(target_names):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolors='k') 
plt.xlabel('Sepal Length (cm)') 
plt.ylabel('Sepal Width (cm)') 
plt.title('Sepal Length vs Sepal Width') 
plt.legend(target_names) 
plt.grid(True) 


plt.subplot(1, 3, 2) 
for i, name in enumerate(target_names):
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap='viridis', s=50, edgecolors='k') 
plt.xlabel('Petal Length (cm)') 
plt.ylabel('Petal Width (cm)') 
plt.title('Petal Length vs Petal Width') 
plt.legend(target_names) 
plt.grid(True)


plt.subplot(1, 3, 3)
for i, name in enumerate(target_names):
     plt.scatter(X[:, 0], X[:, 2], c=y, cmap='viridis', s=50, edgecolors='k') 
plt.xlabel('Sepal Length (cm)') 
plt.ylabel('Petal Length (cm)') 
plt.title('Sepal Length vs Petal Length') 
plt.legend(target_names) 
plt.grid(True) 
plt.tight_layout() 
plt.show() 