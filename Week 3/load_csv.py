import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("students.csv")
df.columns = df.columns.str.strip()

print(df.columns)
print(df.head())

#BAR CHART
subjects = ['Math','Science','English']
avg_marks = df[subjects].mean()

plt.figure(figsize = (6, 4))
sns.barplot(x = avg_marks.index, y = avg_marks.values, palette = "Blues")
plt.title("Average Marks in each Subject")
plt.xlabel("Subjects")
plt.ylabel("Average Marks")
plt.show()

#HISTOGRAM: DISTURBUTION OF MATH MARKS
plt.figure(figsize = (6, 4))
sns.histplot(df['Math'], bins = 5, kde = True, color ='pink')
plt.title("Disturbution pf Math Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

#SCATTER PLOT: MATH VS SCIENCE MARKS(WITH GRNDER)
plt.figure(figsize = (6,4))
sns.scatterplot(x = 'Math', y = 'Science', hue = 'Gender', data = df, s = 100)
plt.title("Math vs Science Marks")
plt.xlabel("Math Marks")
plt.ylabel("Science Marks")
plt.legend(title = 'Gender')
plt.show()