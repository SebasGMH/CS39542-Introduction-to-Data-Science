"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
Classifying Digits
"""
from cgi import test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import pandas as pd
import numpy as np

def select_data(data, target, labels = [0,1]):
    # num_dict = dict(zip(target,data))
    # res = dict((k, num_dict[k]) for k in labels if k in num_dict)
    # return list(res.values()), list(res.keys())
    data_ = []
    target_ = []
    index_ = 0
    for r in target:
        if r in labels:
            data_.append(data[index_])
            target_.append(r)
        index_ = index_ + 1
    return tuple(data_),tuple(target_)

def split_data(data, target, test_size = 0.25, random_state = 21):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state,stratify=target)
    return X_train, X_test, y_train, y_test

def fit_model(x_train, y_train, model_type='logreg'):
    #Possible values are 'logreg', 'svm', 'nbayes', and 'rforest'
    if model_type == 'logreg':
        clf = LogisticRegression(penalty='l2',solver = 'saga',max_iter=5000).fit(x_train, y_train)
        picklestring = pickle.dumps(clf)
        return picklestring
    if model_type == 'svm':
        clf = SVC(kernel='rbf').fit(x_train,y_train)
        picklestring = pickle.dumps(clf)
        return picklestring
    if model_type == 'nbayes':
        clf = GaussianNB().fit(x_train,y_train)
        picklestring = pickle.dumps(clf)
        return picklestring
    if model_type == 'rforest':
        clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(x_train,y_train)
        picklestring = pickle.dumps(clf)
        return picklestring

def predict_model(mod_pkl, xes):
    clf = pickle.loads(mod_pkl)
    y_estimate = clf.predict(xes)
    return y_estimate

def score_model(mod_pkl,xes,yes):
    clf = pickle.loads(mod_pkl)
    y_estimate = clf.predict(xes)
    return confusion_matrix(yes, y_estimate)

def compare_models(data, target, test_size = 0.25, random_state = 21, models = ['logreg','nbayes','svm','rforest']):
    X_train, X_test, y_train, y_test= split_data(data, target, test_size, random_state)
    model_=''
    acc_score = 0
    for m in models:
        pckl_str = fit_model(X_train,y_train,m)
        yes = predict_model(pckl_str,X_test)
        if accuracy_score(y_test,yes) > acc_score:
            acc_score = accuracy_score(y_test,yes)
            model_ = m
    return model_, acc_score

# from sklearn import datasets
# digits = datasets.load_digits()

# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
# print(f'The labels for the first 5 entries: {digits.target[:5]}')
# print(data[0:5])