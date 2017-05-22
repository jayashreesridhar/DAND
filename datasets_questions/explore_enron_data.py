#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#To find number  of data poits in the enron dataset
# add this line
enron_data.pop("TOTAL", 0)
print len(enron_data)
#To find number of features for each person
print enron_data['THE TRAVEL AGENCY IN THE PARK']
print len(enron_data['CAUSEY RICHARD A'])
#To find number of POI'sin the dataset
count=0
for key in enron_data.keys():
    if enron_data[key]["poi"]==1 :
        count+=1
print count
    
#stock value
print enron_data['PRENTICE JAMES']['total_stock_value']
##Number of messages
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
##Value of stock options
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print enron_data['SKILLING JEFFREY K']['total_payments']
#print enron_data.keys()
print enron_data['FASTOW ANDREW S']['total_payments']
print enron_data['LAY KENNETH L']['total_payments']

salary_count=0
e_mail_count=0
payments=0
for key in enron_data.keys():
    if enron_data[key]["salary"] != 'NaN' :
        salary_count+=1
    if enron_data[key]['email_address'] != 'NaN':
        e_mail_count+=1
    if enron_data[key]['total_payments'] =='NaN':
        payments+=1
print salary_count,e_mail_count,payments

poicount=0
for key in enron_data.keys():
    if enron_data[key]["poi"]==1 and enron_data[key]['total_payments'] =='NaN':
        poicount+=1
print poicount
stock_value=[]
for key in enron_data.keys():
    if enron_data[key]['exercised_stock_options'] != 'NaN':
        stock_value.append(enron_data[key]['exercised_stock_options'])
stock_value.sort()
print "min:",stock_value[0]
print "max:",stock_value[len(stock_value)-1]

## PCA IN SKLEARN

# Get the list of features:
for employee, features in enron_data.items():
    all_features_list = features.keys()
    break

# remove the 'email_address' feature
all_features_list.remove('email_address')

# 'poi' has to be the first item in the list for 'targetFeatureSplit' to work:
# remove 'poi' then add it back at the start of the list:
all_features_list.remove('poi')
all_features_list.insert(0,'poi')

print all_features_list

# format the features (replaces NaN, creates numpy arrays)
#data = featureFormat(enron_data, all_features_list, sort_keys = True)

# change the features used:
finance_features = ['poi', 'bonus', 'long_term_incentive']

#### use the correct features in this line:
# format the features (replaces NaN, creates numpy arrays)
data = featureFormat(enron_data, finance_features, sort_keys = True)
# split off the first variable, 'poi', call it 'labels', all other data are 'features'
labels, features = targetFeatureSplit(data)

def doPCA():
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2)
    pca.fit(features)
    return pca
    
pca=doPCA()
print pca.explained_variance_ratio_
first_pc=pca.components_[0]
second_pc=pca.components_[1]

transformed_data=pca.transform(features)

for ii,jj in zip(transformed_data,features):
    plt.scatter(first_pc[0]*ii[0],first_pc[1]*ii[0],color="r")
    plt.scatter(second_pc[0]*ii[1],second_pc[1]*ii[1],color="c")
    plt.scatter(jj[0],jj[1],color="b")
    
plt.xlabel("bonus")
plt.ylabel("long-term incentive")
plt.show()