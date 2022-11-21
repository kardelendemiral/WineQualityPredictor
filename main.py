import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statistics
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

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

    rf = RandomForestRegressor(n_estimators=600, oob_score=False, bootstrap=True, warm_start=True)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    return mean_squared_error(Y_test, Y_pred)

def test(X_train, X_valid, Y_train, Y_valid):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 14, num=7)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    oob_score = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'oob_score': oob_score}
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, Y_train)

    print("Best Parameters: ", rf_random.best_params_)

def testWhetherToRound(X_train, X_valid, Y_train, Y_valid):
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_valid)

    print("Error without rounding:", np.sqrt(mean_squared_error(Y_valid, Y_pred)))
    Y_pred_rounded = [round(item) for item in Y_pred]
    print("Error with rounding:", np.sqrt(mean_squared_error(Y_valid, Y_pred_rounded)))

def testN_estimatorsparameter(X_train, X_valid, Y_train, Y_valid): #number of trees in the forest

    errors = {}
    for i in range(10):
        nEst = (i+1)*100
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

def testMinSamplesSplitParameter(X_train, X_valid, Y_train, Y_valid):
    errors = {}

    for i in range(10):
        min_samples_split = i + 2
        rf = RandomForestRegressor(min_samples_split=min_samples_split)
        rf.fit(X_train, Y_train)
        Y_pred = rf.predict(X_valid)
        errors[min_samples_split] = np.sqrt(mean_squared_error(Y_valid, Y_pred))

    print(errors)

def testMinSamplesLeafParameter(X_train, X_valid, Y_train, Y_valid):
    errors = {}

    for i in range(10):
        min_samples_leaf = i + 1
        rf = RandomForestRegressor(min_samples_leaf=min_samples_leaf)
        rf.fit(X_train, Y_train)
        Y_pred = rf.predict(X_valid)
        errors[min_samples_leaf] = np.sqrt(mean_squared_error(Y_valid, Y_pred))

    print(errors)


def plotFeatureImportances(X_train, Y_train):
    plt.clf()
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=random_states[i])
    rf = RandomForestRegressor(n_estimators=600, oob_score=False, bootstrap=True, warm_start=True)
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
#testWhetherToRound(X_train, X_valid, Y_train,Y_valid)
#testMinSamplesSplitParameter(X_train, X_valid, Y_train,Y_valid)
#testMinSamplesLeafParameter(X_train, X_valid, Y_train,Y_valid)

print("Training set error:", getError(X_train, X_train, Y_train, Y_train))

print("Validation set error:", getError(X_train, X_valid, Y_train, Y_valid))

print("Test set error:", getError(X_train, X_test, Y_train, Y_test))

#plotFeatureImportances(X_train, Y_train)

#test(X_train, X_valid, Y_train,Y_valid)


