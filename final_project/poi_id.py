#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from time import time
from New_feature import computeFraction,combine_feature
from tester import test_classifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
#from xgboost.sklearn import XGBClassifier

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
    #data_point.keys()

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
    salary=data_point["salary"]
    bonus=data_point["bonus"]
    combine_salary_bonus=combine_feature(salary,bonus)
    data_point['combine_salary_bonus']=combine_salary_bonus
    exercised_stock_options=data_point['exercised_stock_options']
    total_stock_value=data_point['total_stock_value']
    data_point['combine_stock_value']=combine_feature(exercised_stock_options,total_stock_value)
    


## updated_features_list

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi','combine_salary_bonus','combine_stock_value']



    
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

# Correlation Matrix Plot
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

df = pd.DataFrame.from_dict(my_dataset, orient='index')
# drop non-numeric features, replace NaN with zero
df = df.drop('email_address', axis=1)
# First replace string `NaN` with numpy nan
df.replace(to_replace='NaN', value=np.nan, inplace=True)
#count number of nan's in columns
print df.isnull().sum()
# then fill in nan
df = df.fillna(0)
#print df.head()
# Compute the correlation matrix
corr = df.corr()


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corr, vmax=.8, square=True)
## Select features based on k scores
from sklearn.feature_selection import SelectKBest
k= 'all'
k_best = SelectKBest(k=k)
k_best=k_best.fit(features, labels)
features_k=k_best.transform(features)
scores = k_best.scores_ # extract scores attribute
pairs = zip(features_list[1:], scores) # zip with features_list
pairs= sorted(pairs, key=lambda x: x[1], reverse= True) # sort tuples in descending order
print pairs

## kbest
features_list=['poi','exercised_stock_options','combine_salary_bonus','fraction_to_poi']

## Features rescaling
scaler=MinMaxScaler()

rescaled_features = scaler.fit_transform(features_k)
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(rescaled_features, labels, test_size=0.3, random_state=42)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#t0=time()
#clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#    ### use the trained classifier to predict labels for the test features
#t1=time()
#pred = clf.predict(features_test)
#print "predicting  time:", round(time()-t1, 3), "s"
#accuracy = accuracy_score(pred,labels_test)
#print accuracy
#print classification_report(labels_test,pred)
### Decision Tree Classifier
#
from sklearn.tree import DecisionTreeClassifier
#clf=DecisionTreeClassifier(max_features='log2',min_samples_split=6)
#clf=clf.fit(features_train_k,labels_train)
#pred=clf.predict(features_test_k)
#print "accuracy",accuracy_score(pred,labels_test)
#print clf.feature_importances_
#print classification_report(labels_test,pred)
#SVM Rbf kernel

#==============================================================================
#==============================================================================

## KNN Classifier
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=10,weights='distance')
#clf.fit(features_train, labels_train)
#pred=clf.predict(features_test)
#accuracy = accuracy_score(pred,labels_test)
#print accuracy
#print classification_report(labels_test,pred)

##ADABoostClassifier
param_grid = {
        'n_estimators':[10,50,100,150,200],
         'learning_rate':[0.1,0.5,0.75,1.0],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_features='log2',min_samples_split=3,min_samples_leaf=2),algorithm='SAMME'), param_grid)
#clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_features='log2',min_samples_split=4,min_samples_leaf=2),n_estimators=150,learning_rate=0.4,algorithm='SAMME')
t0=time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
    ### use the trained classifier to predict labels for the test features
t1=time()
print "Best estimator found by grid search:"
print clf.best_estimator_
pred = clf.predict(features_test)
print "predicting  time:", round(time()-t1, 3), "s"
accuracy = accuracy_score(pred,labels_test)
print accuracy
print classification_report(labels_test,pred)



#==============================================================================


#==============================================================================
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)