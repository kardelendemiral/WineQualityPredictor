import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statistics

def testN_estimatorsparameter(X, Y):

    errors = {}
    for i in range(40):
        nEst = i + 1
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)  # train set contains 4847 items, test set contains 1616 items
        rf = RandomForestRegressor(n_estimators= nEst)
        rf.fit(X_train, Y_train)
        Y_pred = rf.predict(X_test)
        errors[nEst] = np.sqrt(mean_squared_error(Y_test, Y_pred))

    print(errors)

def testOOBscoreParameter(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)  # train set contains 4847 items, test set contains 1616 items
    rf = RandomForestRegressor(oob_score=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    print("Error when oob_score=True: ", np.sqrt(mean_squared_error(Y_test, Y_pred)))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)  # train set contains 4847 items, test set contains 1616 items
    rf = RandomForestRegressor(oob_score=False)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    print("Error when oob_score=False ", np.sqrt(mean_squared_error(Y_test, Y_pred)))


def testBootstrapParameter(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)  # train set contains 4847 items, test set contains 1616 items
    rf = RandomForestRegressor(bootstrap=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    print("Error when bootstrap samples are used: ", np.sqrt(mean_squared_error(Y_test, Y_pred)))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)  # train set contains 4847 items, test set contains 1616 items
    rf = RandomForestRegressor(warm_start=False)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    print("Error when bootstrap samples are not used ", np.sqrt(mean_squared_error(Y_test, Y_pred)))


def testWarmStartParameter(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)  # train set contains 4847 items, test set contains 1616 items
    rf = RandomForestRegressor(warm_start=False)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    print("Error without warm start: ", np.sqrt(mean_squared_error(Y_test, Y_pred)))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)  # train set contains 4847 items, test set contains 1616 items
    rf = RandomForestRegressor(warm_start=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    print("Error with warm start: ", np.sqrt(mean_squared_error(Y_test, Y_pred)))


def plotFeatureImportances(X, Y):
    plt.clf()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    importances = rf.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)

    plt.bar(feature_names, forest_importances, color="maroon", width=0.5)
    plt.xlabel("Features")
    plt.gcf().autofmt_xdate()
    plt.ylabel("Importances")
    plt.title("Feature Importances")
    plt.gcf().subplots_adjust(bottom=0.25)

    plt.show()



df = pd.read_csv('winequalityN.csv')

#print(df.shape)
#print(df.tail())
#print(df.info())
#print(df.isnull().sum())

df.dropna(inplace=True) #drop null values
#print(df.isnull().sum())

df.replace(to_replace="white", value=0, inplace=True)
df.replace(to_replace="red", value=1, inplace=True)


"""plt.hist(df['quality'])
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()"""

X = df.iloc[:, :-1] #all columns except the last one
Y = df.iloc[:, -1]

columns = list(df.columns)
feature_names = columns[: -1]

# try different parameter values:


random_states = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]  # for 10 different splits
errors = []

for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=random_states[i]) # train set contains 4847 items, test set contains 1616 items
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    errors.append(np.sqrt(mean_squared_error(Y_test, Y_pred)))


print(errors)
avg_error = statistics.mean(errors)
std_dev = statistics.pstdev(errors)
print(avg_error, std_dev)

#testN_estimatorsparameter(X, Y)
#testOOBscoreParameter(X,Y)
#testBootstrapParameter(X, Y)
#testWarmStartParameter(X, Y)
#plotFeatureImportances(X, Y)

