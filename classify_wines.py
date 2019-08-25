#!/usr/bin/env python3

# import all of the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run():

    # read in the white wine data
    white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

    # read in the red wine data
    red   = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

    # label the data
    red['type'] = 1
    white['type'] = 0

    # and then concatenate the two datasets
    # NOTE: must set ignore_index=True when 
    #       concatenating to avoid duplicating index
    wines = red.append(white, ignore_index=True)

    # specify the data
    X = wines.iloc[:,0:11]
    print(X)

    # specify the target labels and flatten the array
    y = np.ravel(wines.type)
    print(y)

    # split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

    # Define the scaler
    scaler = StandardScaler().fit(X_train)

    # Scale the train set
    X_train = scaler.transform(X_train)

    # Scale the test set
    X_test = scaler.transform(X_test)


if __name__ == '__main__':
    run()
