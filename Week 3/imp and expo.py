#Import data from CSV file

import pandas as pd
df = pd.read_csv("StudentsPerformance.csv")
print("Original data")
print(df)

#Export data to different formats
df.to_csv("Students_copy.csv", index = False)
df.to_excel("Students.xlsx", index = False)
print("\nData exported sucessfully")