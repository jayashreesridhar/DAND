# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:19:13 2017

@author: JAYASHREE
"""

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from time import time
from New_feature import computeFraction
from tester import test_classifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','bonus','long_term_incentive','exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
### Task 3: Create new feature(s)
submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    #print
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
#print data_dict['CAUSEY RICHARD A']


## updated_features_list
#features_list = ['poi','salary','bonus','long_term_incentive','exercised_stock_options','fraction_from_poi','fraction_to_poi']
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi']

#deferral_payments,bonus,deferred_income,expenses'long_term_incentive', 'restricted_stock','from_this_person_to_poi''fraction_to_poi'
#`features_list =['poi','deferral_payments','bonus','deferred_income','expenses','long_term_incentive', 'restricted_stock']


### Store to my_dataset for easy export below.
my_dataset = data_dict
#No of items in the dictionary
print "Number of employees detail captured",len(data_dict)
print "Number of features in the dataset",len(data_dict['CAUSEY RICHARD A'])

#To find number of POI'sin the dataset
poi_count=0
for key in data_dict.keys():
    if data_dict[key]["poi"]==1 :
        poi_count+=1
print "Number of POI's in the dataset",poi_count

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
## Select features based on k scores
from sklearn.feature_selection import SelectKBest
k= 'all'
k_best = SelectKBest(k=k)
k_best=k_best.fit(features, labels)
features_k=k_best.transform(features)
scores = k_best.scores_ # extract scores attribute
pairs = zip(features_list[1:], scores) # zip with features_list
pairs= sorted(pairs, key=lambda x: x[1], reverse= True) # sort tuples in descending order
#print pairs
## kbest
features_list=['poi','exercised_stock_options','total_stock_value','bonus','salary','fraction_to_poi']
## Features rescaling
scaler=MinMaxScaler()

rescaled_features = scaler.fit_transform(features_k)
#print rescaled_features

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(rescaled_features, labels, test_size=0.3, random_state=42)
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4,3e4,7e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01,0.05,0.075, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
#clf=SVC(kernel='rbf',C=50000.0,gamma=0.01)
clf = clf.fit(features_train,labels_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_


###############################################################################
# Quantitative evaluation of the model quality on the test set

print "Predicting the people names on the testing set"
t0 = time()
y_pred = clf.predict(features_test)
print "done in %0.3fs" % (time() - t0)

print classification_report(labels_test, y_pred)

#from sklearn.cross_validation import StratifiedShuffleSplit
#cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
#for train_idx, test_idx in cv: 
#        features_train = []
#        features_test  = []
#        labels_train   = []
#        labels_test    = []
#        for ii in train_idx:
#            features_train.append( features[ii] )
#            labels_train.append( labels[ii] )
#        for jj in test_idx:
#            features_test.append( features[jj] )
#            labels_test.append( labels[jj] )
#        
#print "Fitting the classifier to the training set"
#t0 = time()
#param_grid = {
#           'C': [0.1,0.25,0.5,0.75,1.25,1.5],
#            'gamma': [0.00015, 0.0205, 0.008, 0.005, 0.01, 0.1],
#            }
##  # for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid,cv=cv)
#        #clf=SVC(kernel='rbf',C=100000.0,gamma=0.00001)
#clf = clf.fit(features, labels)
#print "done in %0.3fs" % (time() - t0)
#print "Best estimator found by grid search:"
#        #print clf.best_estimator_
#print("The best parameters are %s with a score of %0.2f"
#      % (clf.best_params_, clf.best_score_))
#  
#  
#  ###############################################################################
#  # Quantitative evaluation of the model quality on the test set
#  
#  print "Predicting the people names on the testing set"
#t0 = time()
#y_pred = clf.predict(features_test)
#        print "done in %0.3fs" % (time() - t0)
#        accuracy = accuracy_score(y_pred,labels_test)
#        print accuracy

dump_classifier_and_data(clf, my_dataset, features_list)