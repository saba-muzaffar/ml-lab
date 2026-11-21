import pandas as pd
data ={'Name': ['Saba','Iqra','Usmat'], 'Age' : [22,23,24], 'City':['Bandipora','Pahalgam','Kupwara']}
df = pd.DataFrame(data)
print(df)
var=[{'Name':'Saba','Age': 22, 'City':'Bandipora'},{'Name': 'Iqra','Age': 23,'City':'Pahalgam'},{'Name':'Usmat','Age':24,'City':'Kupwara'}]
df= pd.DataFrame(var)
print(df)
df.head()
df.tail()
df.info()
df.describe()

df.shape
df.columns
df.dtypes