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

features_list = ['poi','poi','fraction_to_poi', 
                'fraction_from_poi', 'fraction_bonus_salary',
                'fraction_total_stock_value_salary'    ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers
data_dict.pop( "TOTAL", 0 )
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

for name in data_dict:

    data_point = data_dict[name]
    salary = data_point["salary"]
    bonus = data_point["bonus"]
    total_stock_value = data_point['total_stock_value']
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    fraction_bonus_salary = computeFraction( salary, bonus )
    fraction_total_stock_value_salary = computeFraction( salary, total_stock_value )
    data_point["fraction_from_poi"] = fraction_from_poi
    data_point["fraction_bonus_salary"] = fraction_bonus_salary
    data_point["fraction_total_stock_value_salary"] = fraction_total_stock_value_salary

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.



# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()






# print "start"
# import time
# tic = time.clock()
# from sklearn.svm import SVC
# clf = SVC(kernel="linear")
# print "middle"
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# linear, poly, rbf, sigmoid, precomputed or a callable
# Change SVC(kernel, C, gamma)
# Try rbf nernel with C (say, 10.0, 100., 1000., and 10000.)



# Works
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split=2)
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
# Increase DecisionTreeClassifier(min_samples_split=2 or 50 or 100 or more) or other parameters
# Also try criterion = 'entropy'







# # Principal Component Analysis
# def doPCA():
# 	from sklearn.decomposition import PCA
# 	pca = PCA(n_components = 2)
# 	pca.fit(data)
# 	return pca

# pca = doPCA()
# # Iagen values to tell how much variance each components has
# print pca.explained_variance_ratio_
# first_pc = pca.components_[0]
# second_pc = pca.components_[1]

# transformed_data = pca.transform(data)
# for ii, jj in zip(transformed_data, data):
# 	plt.scatter( first_pc[0]*ii[0], first_pc[1]*ii[0], color="r")
# 	plt.scatter( second_pc[0]*ii[1], second_pc[1]*ii[1], color="c")
# 	plt.scatter( jj[0], jj[1], color="b")

# plt.xlabel("bonus")
# plt.ylabel("long-term incentive")
# plt.show()

# n_components = 4
# # change n_components [10, 15, 25, 50, 100, 250]
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# estimators = [('reduce_dim', PCA(n_components=n_components)), ('clf', SVC())]
# pipe = Pipeline(estimators)
# pipe.fit(features_train, labels_train)


# from sklearn import svm, grid_search
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svr = svm.SVC()
# clf = grid_search.GridSearchCV(svr, parameters)
# clf.best_params_








### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# To train faster:
# features_train = features_train[:len(features_train)] 
# labels_train = labels_train[:len(labels_train)] 



# ADABOOST
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# n_estimators = 50
# # A learning rate of 1. may not be optimal for both SAMME and SAMME.R
# learning_rate = 1.

# dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
# dt_stump.fit(features_train, labels_train)

# clf = AdaBoostClassifier(base_estimator=dt_stump,
#     learning_rate=learning_rate,
#     n_estimators=n_estimators,
#     algorithm="SAMME.R")

# Access data enron_data["LASTNAME FIRSTNAME"]["feature_name"], enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]
# print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

n_components = 2
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

estimators = [('reduce_dim', PCA(n_components=n_components)), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe.fit(features_train, labels_train)


from sklearn import svm, grid_search
parameters = {'kernel':('linear', 'rbf'), 'C':[.1, 1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)


print "fitting"
clf.fit(features_train, labels_train)
print "done done"
# pred = clf.predict(features_test)
# print "fitting2"
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# accuracy = accuracy_score(labels_test, pred)
# prec = precision_score(labels_test, pred)
# recall = recall_score(labels_test, pred)
# print "accuracy:", accuracy
# print "precision:", prec
# print "recall:", recall



# toc = time.clock()
# print toc - tic




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)