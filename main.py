import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statistics

def splitDataAndSave():

    df = pd.read_csv('winequalityN.csv')

    df.dropna(inplace=True)  # drop null values
    df.replace(to_replace="white", value=0, inplace=True)
    df.replace(to_replace="red", value=1, inplace=True)
    X = df.iloc[:, :-1]  # all columns except the last one
    Y = df.iloc[:, -1]

    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, train_size=0.8, random_state=100)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size=0.5, random_state=100)


    X_train.to_pickle('X_train.pkl')
    Y_train.to_pickle('Y_train.pkl')
    X_valid.to_pickle('X_valid.pkl')
    Y_valid.to_pickle('Y_valid.pkl')
    X_test.to_pickle('X_test.pkl')
    Y_test.to_pickle('Y_test.pkl')

def getError(X_train, X_test, Y_train, Y_test):

    rf = RandomForestRegressor(n_estimators = 98, oob_score=False, bootstrap=True, warm_start=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    return np.sqrt(mean_squared_error(Y_test, Y_pred))

def testN_estimatorsparameter(X_train, X_valid, Y_train, Y_valid): #number of trees in the forest

    errors = {}
    for i in range(100):
        nEst = i + 10
        rf = RandomForestRegressor(n_estimators=nEst)
        rf.fit(X_train, Y_train)
        Y_pred = rf.predict(X_valid)
        errors[nEst] = np.sqrt(mean_squared_error(Y_valid, Y_pred))

    print(errors)

def testOOBscoreParameter(X_train, X_valid, Y_train, Y_valid):

    rf = RandomForestRegressor(oob_score=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_valid)

    print("Error when oob_score=True: ", np.sqrt(mean_squared_error(Y_valid, Y_pred)))

    rf = RandomForestRegressor(oob_score=False)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_valid)

    print("Error when oob_score=False ", np.sqrt(mean_squared_error(Y_valid, Y_pred)))


def testBootstrapParameter(X_train, X_valid, Y_train, Y_valid):

    rf = RandomForestRegressor(bootstrap=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_valid)

    print("Error when bootstrap samples are used: ", np.sqrt(mean_squared_error(Y_valid, Y_pred)))

    rf = RandomForestRegressor(warm_start=False)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_valid)

    print("Error when bootstrap samples are not used ", np.sqrt(mean_squared_error(Y_valid, Y_pred)))


def testWarmStartParameter(X_train, X_valid, Y_train, Y_valid):
    rf = RandomForestRegressor(warm_start=False)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_valid)

    print("Error without warm start: ", np.sqrt(mean_squared_error(Y_valid, Y_pred)))

    rf = RandomForestRegressor(warm_start=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_valid)

    print("Error with warm start: ", np.sqrt(mean_squared_error(Y_valid, Y_pred)))


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


random_states = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]  # for 10 different splits
errors = []

"""for i in range(10):
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, train_size=0.8, random_state=random_states[i])
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size=0.5, random_state=random_states[i])
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    errors.append(np.sqrt(mean_squared_error(Y_test, Y_pred)))


print(errors)
avg_error = statistics.mean(errors)
std_dev = statistics.pstdev(errors)
print(avg_error, std_dev)"""

#splitDataAndSave()

X_train = pd.read_pickle('X_train.pkl')
Y_train = pd.read_pickle('Y_train.pkl')
X_valid = pd.read_pickle('X_valid.pkl')
Y_valid = pd.read_pickle('Y_valid.pkl')
X_test = pd.read_pickle('X_test.pkl')
Y_test = pd.read_pickle('Y_test.pkl')


#testN_estimatorsparameter(X_train, X_valid, Y_train,Y_valid)
#testOOBscoreParameter(X_train, X_valid, Y_train,Y_valid)
#testBootstrapParameter(X_train, X_valid, Y_train,Y_valid)
#testWarmStartParameter(X_train, X_valid, Y_train,Y_valid)

print("Training set error:")
print(getError(X_train, X_train, Y_train, Y_train))

print("Validation set error:")
print(getError(X_train, X_valid, Y_train, Y_valid))

print("Test set error:")
print(getError(X_train, X_test, Y_train, Y_test))

