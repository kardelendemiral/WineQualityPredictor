import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('winequalityN.csv')

#print(df.shape)
#print(df.tail())
#print(df.info())
#print(df.isnull().sum())

df.dropna(inplace=True)
#print(df.isnull().sum())

df.replace(to_replace="white", value=0, inplace=True)
df.replace(to_replace="red", value=1, inplace=True)


"""plt.hist(df['quality'])
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()"""

X = df.iloc[:, :-1] #all columns except the last one
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=100)

"""print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)"""

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
print(Y_pred)
print(Y_pred.round())

print(np.sqrt(mean_squared_error(Y_test, Y_pred)))
print(np.sqrt(mean_squared_error(Y_test, Y_pred.round())))