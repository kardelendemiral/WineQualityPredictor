from numpy import mean
from numpy import std
from sklearn import model_selection
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import statistics

def getError(X_train, X_test, Y_train, Y_test): #returns the error of a particular split

    r = XGBRegressor(min_child_weight = 1, max_depth = 15, learning_rate = 0.05)
    r.fit(X_train, Y_train)
    Y_pred = r.predict(X_test)

    return mean_squared_error(Y_test, Y_pred)

def setErrors(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    T_err = getError(X_train, X_train, Y_train, Y_train)
    V_err = getError(X_train, X_valid, Y_train, Y_valid)
    Tst_err = getError(X_train, X_test, Y_train, Y_test)

    print("Training set error:", T_err)

    print("Validation set error:", V_err)

    print("Test set error:", Tst_err)

    plt.clf()
    plt.bar(["Training Set", "Validation Set", "Test Set"], [T_err, V_err, Tst_err], color="purple", width=0.5)
    plt.xlabel("Sets")
    plt.ylabel("Errors")
    plt.show()


def tenSplits(X, Y):

    random_states = [90, 100, 700, 66, 50, 409, 30, 20, 110, 5] # for 10 different splits
    errors = []

    for i in range(10):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=random_states[i])
        errors.append(getError(X_train, X_test, Y_train, Y_test))

    plt.clf()
    plt.plot(errors)
    plt.show()
    print(errors)
    avg_error = statistics.mean(errors)
    std_dev = statistics.pstdev(errors)
    print(avg_error, std_dev)

def test(X_train, Y_train):
    hyperparameter_grid = {
        'n_estimators': [400, 800, 1000],
        'max_depth': [6, 9, 15],
        'learning_rate': [0.05, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 10, 100]
    }

    r = XGBRegressor()

    random_cv = RandomizedSearchCV(estimator=r,
                                   param_distributions=hyperparameter_grid,
                                   cv=5, n_iter=50,
                                   scoring='neg_mean_absolute_error', n_jobs=4,
                                   verbose=5,
                                   return_train_score=True,
                                   random_state=42)


    result = random_cv.fit(X_train, Y_train)
    print(result.best_params_)

df = pd.read_csv('winequalityN.csv')
df.dropna(inplace=True) #drop null values

df.replace(to_replace="white", value=0, inplace=True)
df.replace(to_replace="red", value=1, inplace=True)

X = df.iloc[:, :-1] #all columns except the last one
Y = df.iloc[:, -1]

X_train, X_rem, Y_train, Y_rem = model_selection.train_test_split(X, Y, train_size=0.8, random_state=100)
X_valid, X_test, Y_valid, Y_test = model_selection.train_test_split(X_rem, Y_rem, test_size=0.5, random_state=100)

"""model = XGBRegressor(min_child_weight = 1, max_depth = 15, learning_rate = 0.05)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_valid)

print(Y_pred)
print(Y_valid)

print(mean_squared_error(Y_pred, Y_valid))"""

setErrors(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
tenSplits(X, Y)

#test2(X_train, Y_train)