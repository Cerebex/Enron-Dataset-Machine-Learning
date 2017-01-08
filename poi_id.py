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

# Feature List With Newly Created Features
features_list = ['poi',
'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'total_stock_value',
'fraction_bonus_salary', 'fraction_from_poi', 'fraction_to_poi', 
'fraction_total_stock_value_salary', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi' ] 

# Feature List Without Newly Created Features
# features_list = ['poi',
# 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
# 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
# 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'total_stock_value', 'from_poi_to_this_person',
#  'from_messages', 'from_this_person_to_poi' ] 

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
NaN_POI_Percent = {}
for name in data_dict:
    data_point = data_dict[name]
    for data_result in data_point:
        if data_result not in NaN_POI_Percent.keys():
            NaN_POI_Percent[data_result] = 0
        if data_result not in NaN_Percent.keys():
            NaN_Percent[data_result] = 0
    for data_result in data_point:
        if data_point[data_result] == "NaN":
                NaN_Percent[data_result] += 1
    if data_point['poi'] == 1:
        for data_result in data_point:
            if data_point[data_result] == "NaN":
                    NaN_POI_Percent[data_result] += 1
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

# # Review dataset
# for data_name in NaN_Percent:
#     NaN_Percent[data_name] = (NaN_Percent[data_name]/143.0)*100.0
# for data_name in NaN_POI_Percent:
#     NaN_POI_Percent[data_name] = (NaN_POI_Percent[data_name]/143.0)*100.0    

# import pandas as pd

# NaN_Percent = pd.DataFrame(NaN_Percent.items(), columns=['Name', 'Percent Missing'])
# NaN_Percent_sorted = NaN_Percent.sort("Percent Missing", ascending=True)
# print NaN_Percent_sorted.to_string(index=False)

# NaN_POI_Percent = pd.DataFrame(NaN_POI_Percent.items(), columns=['Name', 'Percent Missing'])
# NaN_POI_Percent_sorted = NaN_POI_Percent.sort("Percent Missing", ascending=True)
# print NaN_POI_Percent_sorted.to_string(index=False)

# NaN_POI_Percent_sorted.to_csv('NaN_POI_Percent_sorted.csv')
# NaN_Percent_sorted.to_csv('NaN_Percent_sorted.csv')


# my_dataset_pandas = pd.DataFrame.from_dict(my_dataset,orient='index')
# my_dataset_pandas.to_csv('my_dataset_pandas.csv')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


### Task 4: Employ and tune a classifier

# from sklearn.model_selection import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

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
import numpy as np

estimators = [('reduce_dim', SelectKBest()), ('NB', GaussianNB())]
pipe = Pipeline(steps = estimators)
param_dict = [{'reduce_dim__k' : [6]}]
clf = GridSearchCV(pipe, 
                  param_dict, scoring='f1')


# ____________________________________________________________________________________________________________


### Task 5: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)