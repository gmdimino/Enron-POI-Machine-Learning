#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

# Imports and settings for pandas, stats, plotting, pipelines.
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from matplotlib import pyplot as plt
import seaborn as sns

import math
import statsmodels.api as sm

from sklearn.pipeline import Pipeline

def get_and_process_data():

  ### Load the dictionary containing the dataset
  with open("final_project_dataset.pkl", "r") as data_file:
      data_dict = pickle.load(data_file)
  print "opened file"

  # Data cleaning and feature engineering
  # See ipython notebook for details

  # Convert dict to dataframe
  df = pd.DataFrame.from_dict(data_dict)

  # Transpose dataframe
  df = df.transpose()

  # Write and read to standardize
  df.to_csv('enron_data.csv')
  df = pd.read_csv('enron_data.csv')

  # Fill in N/A values with 0 and drop email address column
  df = df.fillna(0)
  df = df.drop("email_address", 1)

  ### Task 2: Remove outliers
  # Drop extreme outlier (TOTAL)
  df = df.drop(130)

  # Drop anomalous entries 
  # LOCKHART EUGENE E
  df = df.drop(84)
  # THE TRAVEL AGENCY IN THE PARK
  df = df.drop(127)

  ### Task 3: Create new feature(s)
  # Add new normalized messages to/from POI and drop the old ones.
  # Note that these features are problematic, since they rely on knowing who
  # the POI are.
  df["normalized_from_poi"] = df["from_poi_to_this_person"]/df["to_messages"]
  df["normalized_to_poi"] = df["from_this_person_to_poi"]/df["from_messages"]
  df = df.drop("from_poi_to_this_person", 1)
  df = df.drop("from_this_person_to_poi",1)
  df = df.fillna(0)

  my_dataset = df.transpose().to_dict()
  return my_dataset

def main():

  my_dataset = get_and_process_data() 

  # Naive Bayes
  # Removed problematic features
  best_featureset_nb = ['poi',
     'salary',
     'bonus',
     'exercised_stock_options',
     'deferred_income',
     'long_term_incentive',
     # 'normalized_from_poi',
     # 'normalized_to_poi'
    ]

  from sklearn.naive_bayes import GaussianNB
  clf = GaussianNB()
  print "About to test Naive Bayes"
  test_classifier(clf, my_dataset, best_featureset_nb)
  dump_classifier_and_data(clf, my_dataset, best_featureset_nb)


  # Decision Tree
  best_featureset_dt = ['poi',
         'bonus', 
         'shared_receipt_with_poi', 
         'normalized_to_poi',
  ]

  from sklearn.tree import DecisionTreeClassifier
  clf = DecisionTreeClassifier(min_samples_split=2)
  test_classifier(clf, my_dataset, best_featureset_dt)
  # dump_classifier_and_data(clf, my_dataset, best_featureset_dt)

  print sorted(zip(best_featureset_dt[1:], clf.feature_importances_), 
         key = lambda l: l[1], 
         reverse = True)

  # Random Forest
  best_featureset_rf =  ['poi',
     'bonus', 
     'exercised_stock_options',
     'shared_receipt_with_poi',
     'normalized_to_poi'
  ]

  from sklearn.ensemble import RandomForestClassifier
  clf = RandomForestClassifier(n_estimators=100, min_samples_split=3)
  test_classifier(clf, my_dataset, best_featureset_rf)

  print sorted(zip(best_featureset_rf[1:], clf.feature_importances_), 
         key = lambda l: l[1], 
         reverse = True)

  # dump_classifier_and_data(clf, my_dataset, best_featureset_rf)

  # Logistic Regression (w/ Robust Scaling)
  from sklearn.linear_model import LogisticRegression
  from sklearn.preprocessing import MinMaxScaler, RobustScaler
  clf = Pipeline(steps=[("scaler", RobustScaler()),
			("clf", LogisticRegression(C=100, class_weight='balanced'))])

  test_classifier(clf, my_dataset, best_featureset_dt)
  # dump_classifier_and_data(clf, my_dataset, best_featureset_dt)

if __name__ == '__main__':
  main()

