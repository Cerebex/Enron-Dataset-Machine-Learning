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

features_list = ['poi','fraction_to_poi', 
                'fraction_from_poi', 'fraction_bonus_salary',
                'fraction_total_stock_value_salary'  ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Count the number of persons of interest compared to total persons in dataset
poi_count = 0 
total_persons_count = 0
for person in data_dict:
    total_persons_count += 1
    for feature in data_dict[person]:
        if feature == "poi" and data_dict[person][feature] == True:
            poi_count += 1

print poi_count
print total_persons_count

### Task 2: Remove outliers
print data_dict["TOTAL"]
data_dict.pop( "TOTAL", 0 )


### Task 3: Create new feature(s)
# I created 4 fractions to potentially scale features already given in the dataset. 


# I first created the fractions of fraction_from_poi and fraction_to_poi. I then wanted to see how extravagent 
# someone bonus and total_stock_value really was so I created the fractions fraction_bonus_salary and
# fraction_total_stock_value_salary. 


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

### Task 4: Pick Classifiers

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split=2)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using testing script.

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)

### Task 6: Dump your classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)