#!/usr/bin/python
from decimal import *
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# I chose salary as a feature since I believe those with higher salary likely had more power within the company
# and therefore more to lose if the company failed and more to win if the company succeeded. 

# I chose bonus and total_stock_value as a feature since I believe that bonuses and stocks awarded 
# in this company were based upon the discression of many actively creating the fraud in Enron and would mostly 
# shell out alot of money to those who were helping them.

# I chose to use 'from_this_person_to_poi' email count since I believe those who sent more emails to POI likely
# had more to do with the fraud.  

features_list = ['poi',
'fraction_bonus_salary', 'fraction_from_poi', 'fraction_to_poi', 
'fraction_total_stock_value_salary', 'from_poi_to_this_person'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

### Task 2: Remove outliers
data_dict.pop( "TOTAL", 0 )
data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 )
data_dict.pop( "LOCKHART EUGENE E", 0 )


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

### Task 3: Create new feature(s)

# I created 4 fractions to potentially scale features already given in the dataset. 

# I first created the fractions of fraction_from_poi and fraction_to_poi. I then wanted to see how extravagent 
# someone bonus and total_stock_value really was so I created the fractions fraction_bonus_salary and
# fraction_total_stock_value_salary

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    if "NaN" == poi_messages or "NaN" == all_messages:
        fraction = 0
    else:
        fraction = float(Decimal(poi_messages)/Decimal(all_messages))

    return fraction
NaN_Percent = {}
for name in data_dict:
    data_point = data_dict[name]
    for data_result in data_point:
        if data_point[data_result] == "NaN":
            if data_result in NaN_Percent.keys():
                NaN_Percent[data_result] += 1
            else:
                NaN_Percent[data_result] = 1
    salary = data_point["salary"]
    bonus = data_point["bonus"]
    total_stock_value = data_point['total_stock_value']
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    fraction_bonus_salary = computeFraction( salary, bonus )
    # if fraction_bonus_salary >= 1:
    #     print 'fraction_bonus_salary'
    #     print fraction_bonus_salary
    fraction_total_stock_value_salary = computeFraction( salary, total_stock_value )
    # if fraction_total_stock_value_salary >= 1:
    #     print 'fraction_total_stock_value_salary'
    #     print fraction_total_stock_value_salary
    data_point["fraction_from_poi"] = fraction_from_poi
    data_point["fraction_bonus_salary"] = fraction_bonus_salary
    data_point["fraction_total_stock_value_salary"] = fraction_total_stock_value_salary

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi


### Store to my_dataset for easy export below.
my_dataset = data_dict

# Review dataset
for data_name in NaN_Percent:
    NaN_Percent[data_name] = (NaN_Percent[data_name]/146.0)*100.0

import pandas as pd

NaN_Percent = pd.DataFrame(NaN_Percent.items(), columns=['Name', 'Percent Missing'])
NaN_Percent_sorted = NaN_Percent.sort("Percent Missing", ascending=True)
print NaN_Percent_sorted.to_string(index=False)


my_dataset_pandas = pd.DataFrame.from_dict(my_dataset,orient='index')
my_dataset_pandas.to_csv('my_dataset_pandas.csv')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


### Task 4: Employ and tune a classifier

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score

# # Gaussian Niave Bayes


# estimators = [('reduce_dim', PCA()), ("scale", MinMaxScaler()), ('svm', SVC())]
# pipe = Pipeline(steps = estimators)
# param_dict = [{'svm__kernel' : ['rbf', 'poly', 'sigmoid'], 
#                     'svm__C' : [.1, 1., 1000000.],
#                     'svm__tol' : [1.0e-6, 1.0e6], 
#                     'svm__gamma' :[.1, 3., 13.],
#                     'reduce_dim__n_components' : [2,3,4]}]

# gs = GridSearchCV(pipe, param_dict, scoring='f1')
# print "fitting"
# gs.fit(features_train, labels_train)
# clf = gs.best_estimator_
# print clf
# print "done"



estimators = [('reduce_dim', PCA()) ,('NB', GaussianNB())]
pipe = Pipeline(steps = estimators)
param_dict = [{'reduce_dim__n_components' : [2,3,4]}]
gs = GridSearchCV(pipe, 
                  param_dict, scoring='f1')
print "fitting"
gs.fit(features_train, labels_train)
clf = gs.best_estimator_
print "done"


#     Accuracy: 0.67327   Precision: 0.29730  Recall: 0.58450 F1: 0.39413 F2: 0.48986
#     Total predictions: 11000    True positives: 1169    False positives: 2763   
#     False negatives:  831   True negatives: 6237

# [Finished in 1.4s]

# ____________________________________________________________________________________________________________

# Support Vector Machine plus grid_search:

# estimators = [("scale", MinMaxScaler()), ('svm', SVC())]
# pipe = Pipeline(steps = estimators)
# param_dict = [
#                 {'svm__kernel' : ['rbf', 'poly', 'sigmoid'], 
#                     'svm__C' : [.1, 1., 100., 1000., 100000., 1000000.],
#                     'svm__tol' : [1.0e-6, 1.0e6], 
#                     'svm__gamma' :[.1, 3., 5., 10., 13., 20.]}]
# gs = GridSearchCV(pipe, 
#                   param_dict, scoring='f1')
# print "fitting"
# gs.fit(features_train, labels_train)
# clf = gs.best_estimator_
# print "done"


# RESULTS:

# Input Features :['poi', 'fraction_bonus_salary', 'fraction_from_poi', 
# 'fraction_to_poi', 'fraction_total_stock_value_salary' ]  
# Pipeline(steps=[('scale', MinMaxScaler(copy=True, feature_range=(0, 1))), 
# ('svm', SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=13.0, kernel='poly',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=1e-06, verbose=False))])
#     Accuracy: 0.81073   Precision: 0.47121  Recall: 0.33550 F1: 0.39194 F2: 0.35601
#     Total predictions: 11000    True positives:  671    False positives:  753   
#       False negatives: 1329   True negatives: 8247

# Input Features: ['poi', 'from_poi_to_this_person', 'from_this_person_to_poi','salary', 'bonus','total_stock_value' ]  
# # Pipeline(steps=[('scale', MinMaxScaler(copy=True, feature_range=(0, 1))), 
# ('svm', SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=20.0, kernel='poly',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=1e-06, verbose=False))])
#     Accuracy: 0.83629   Precision: 0.41512  Recall: 0.35700 F1: 0.38387 F2: 0.36728
#     Total predictions: 14000    True positives:  714    
# False positives: 1006   False negatives: 1286   True negatives: 10994

# [Finished in 5742.9s]

# ____________________________________________________________________________________________________________

# Decision Tree

# parameters = [{'criterion':['gini', 'entropy'], 'min_samples_split':[2, 3, 4, 5, 6, 10, 20, 30, 40], 
# 'splitter':['best', 'random']}]
# dtc = DecisionTreeClassifier()
# gs = GridSearchCV(dtc, parameters, scoring = 'f1')
# print "fitting"
# gs.fit(features_train, labels_train)
# clf = gs.best_estimator_
# print "done"

# RESULTS:
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=5, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='random')
#     Accuracy: 0.78155   Precision: 0.36413  Recall: 0.27000 F1: 0.31008 F2: 0.28472
#     Total predictions: 11000    True positives:  540    False positives:  943   
#       False negatives: 1460   True negatives: 8057

# [Finished in 1.1s]

# # Works better than grid search to create best precision and recall
# clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 2, splitter = 'random')

# RESULTS:
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='random')
#     Accuracy: 0.76936   Precision: 0.36922  Recall: 0.37900 F1: 0.37404 F2: 0.37700
#     Total predictions: 11000    True positives:  758    False positives: 1295   
#       False negatives: 1242   True negatives: 7705

# [Finished in 1.2s]


# ____________________________________________________________________________________________________________

# # ADABOOST

# dt_stump = DecisionTreeClassifier(criterion = 'entropy', min_samples_split= 2, splitter = 'random')

# clf = AdaBoostClassifier(base_estimator=dt_stump,
#     algorithm="SAMME.R")

# parameters = {'learning_rate':(1.5, 1., .5), 'n_estimators':[10, 20, 40, 60, 80, 100, 300, 600, 800]}
# gs = GridSearchCV(clf, parameters, scoring='f1')
# print "fitting"
# gs.fit(features_train, labels_train)
# clf = gs.best_estimator_
# print "done"

# RESULTS:
# AdaBoostClassifier(algorithm='SAMME.R',
#           base_estimator=DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='random'),
#           learning_rate=1.5, n_estimators=300, random_state=None)
#     Accuracy: 0.77709   Precision: 0.38290  Recall: 0.36950 F1: 0.37608 F2: 0.37210
#     Total predictions: 11000    True positives:  739    False positives: 1191   
#       False negatives: 1261   True negatives: 7809

# [Finished in 3.5s]

# ____________________________________________________________________________________________________________

# PCA plus SVC with Gridsearch

# estimators = [('reduce_dim', PCA()), ("scale", MinMaxScaler()), ('svm', SVC())]
# pipe = Pipeline(steps = estimators)
# param_dict = [{'svm__kernel' : ['rbf', 'poly', 'sigmoid'], 
#                     'svm__C' : [.1, 1., 1000000.],
#                     'svm__tol' : [1.0e-6, 1.0e6], 
#                     'svm__gamma' :[.1, 3., 13.],
#                     'reduce_dim__n_components' : [2,3,4]}]

# gs = GridSearchCV(pipe, param_dict, scoring='f1')
# print "fitting"
# gs.fit(features_train, labels_train)
# clf = gs.best_estimator_
# print clf
# print "done"


# RESULTS:
# Pipeline(steps=[('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=4, random_state=None,
#   svd_solver='auto', tol=0.0, whiten=False)), ('scale', MinMaxScaler(copy=True, feature_range=(0, 1))), 
#   ('svm', SVC(C=1000000.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=1e-06, verbose=False))])
#     Accuracy: 0.76164   Precision: 0.22815  Recall: 0.13050 F1: 0.16603 F2: 0.14272
#     Total predictions: 11000    True positives:  261    False positives:  883   
#       False negatives: 1739   True negatives: 8117

# [Finished in 380.7s]
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


### Task 5: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)