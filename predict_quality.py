#!/usr/bin/env python3

# import all of the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.models import load_model


def load_data():

    # read in the red wine data
    red   = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
    # read in the white wine data
    white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

    return red, white

def preproccess(red, white):

    # label the data
    red['type'] = 1
    white['type'] = 0

    # and then concatenate the two datasets
    # NOTE: must set ignore_index=True when 
    #       concatenating to avoid duplicating index
    wines = red.append(white, ignore_index=True)

    # specify the data
    X = wines.iloc[:,0:11]

    # specify the target labels and flatten the array
    y = np.ravel(wines.type)

    return X, y

def build_model(X_train, y_train):

    # now build the model
    model = Sequential()
    model.add(Dense(12, activation='relu',input_shape=(11,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    # print the model in human readable diagram
    plot_model(model, to_file='model.png')

    return model

def train_model(model, X_train, y_train):

    # now we compile and fit
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

    # save the model for future use
    model.save('model.h5')

    return model

def predict(model, X_test, y_test):

    # get the prediction for the test data
    y_pred = model.predict_classes(X_test)
    # evaluate the scores 
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score)

    # Metrics
    print(confusion_matrix(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(f1_score(y_test,y_pred))

def run():

    # get the data from the CSV files and load it into pandas dataframes
    red, white = load_data()

    # label, concatenate, and scale the data
    X, y = preproccess(red, white)

    # split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

    # Define the scaler
    scaler = StandardScaler().fit(X_train)

    # Scale the train set
    X_train = scaler.transform(X_train)
    # Scale the test set
    X_test = scaler.transform(X_test)

    # now build and train the network
    model = build_model(X_train,y_train)
    trained_model = train_model(model, X_train,y_train)
  
    # now test the model
    predict(trained_model, X_test, y_test)


if __name__ == '__main__':
    run()
