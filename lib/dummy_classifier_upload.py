#!/usr/bin/python3

print('Loading modules...')

import os, sys, getopt, datetime
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold,cross_validate, cross_val_predict, train_test_split, 
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, make_scorer, balanced_accuracy_score, confusion_matrix, classification_report
from pickle import load, dump
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from get_transformer_feature_names import *

#set the working directory 
os.chdir('/project/lemay_diet_guthealth/echin/stoolcon/')

#set seed
np.random.seed(0)

def main():
	if len(sys.argv) < 3 :
		print("Not enough arguments specified\n Usage: lasso.py <x features path> <y target path> <savepath>")
		sys.exit (1)
	else:
	# print command line arguments
		for arg in sys.argv[0:]:
			print(arg)
	#Load X features data
		X_path = sys.argv[1]
		print('Loading the X features at {}'.format(X_path))
		X_in = pd.read_csv(X_path, index_col = 0, header= 0)
	#	X_in = X_in.drop(['age_cat', 'bmi_cat', 'bin_number'],axis=1) only need this if using un-truncated feature set
		X_in = X_in.sort_index(axis = 0)	
	
	#get the model name from the input x features
		base = os.path.basename(X_path)
		mod_name = os.path.splitext(base)[0]
		#print("\n\n\n:",mod_name)	
	
	#numerical_features = data.columns[1:-1]
	#Load Y target data- this should be just one of the dxa outputs 
		Y_path= sys.argv[2]
		print('Loading Y target at {}'.format(Y_path))
		y_in = pd.read_csv(Y_path, index_col = 0, header = 0)
		y_in = y_in.sort_index(axis=0)
		y_target = y_in[:-1].columns

	#Load the numeric and categorical feature names 
		num_feat = pd.read_csv("data/input/numeric_features.csv", delimiter=',', header=0)
		cat_feat = pd.read_csv("data/input/categorical_features.csv", delimiter=',', header=0)

	#Define the numeric and categorical features 
		numerical_features = [col for col in X_in.columns if col in num_feat.values]
		categorical_features= [col for col in X_in.columns if col in cat_feat.values]

	#numeric transformer
		numeric_transformer = Pipeline(steps = [
			#('yeo', PowerTransformer(method="yeo-johnson", standardize=True)
			('scaler', StandardScaler())])
		
		#set up categorical transformer
		X_cat = X_in[categorical_features]
		enc = OneHotEncoder(handle_unknown="error", sparse=False)
		enc.fit(X_cat)
		enc.transform(X_cat)
		cat_levels=enc.categories_

		categorical_transformer = Pipeline(steps = [
			('onehot', OneHotEncoder(handle_unknown='error',sparse=False, categories=cat_levels))
		])

	#Set up ColumnTransformer
		prep = ColumnTransformer(
			transformers=[
				('num', numeric_transformer, numerical_features),
				('cat', categorical_transformer, categorical_features)
			]
		)
	
	#make the model
		dummy_model = DummyClassifier(strategy = "most_frequent")
		
	#setting up pipeline
		pipeline= Pipeline(steps = [(
			'preprocessor', prep),
			('dummy', dummy_model)]) 

		NUM_TRIALS = 10
		xval_acc = np.zeros(NUM_TRIALS)
		xval_bacc = np.zeros(NUM_TRIALS)
		test_acc = np.zeros(NUM_TRIALS)
		test_bacc = np.zeros(NUM_TRIALS)
	
		for i in range(NUM_TRIALS):
			X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.25,
				random_state=i,shuffle=True, stratify = y_in)
			cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
			scoring = {'accuracy','balanced_accuracy'} #f1_micro = accuracy = recall_micro = precision_micro
			scores = cross_validate(pipeline, X_train, y_train, scoring = scoring, cv = cv)
			xval_acc[i] = np.mean(scores['test_accuracy'])
			xval_bacc[i] = np.mean(scores['test_balanced_accuracy'])
			pipeline.fit(X_train, y_train)
			y_pred = pipeline.predict(X_test)
			test_acc[i] = accuracy_score(y_test, y_pred)
			test_bacc[i] = balanced_accuracy_score(y_test, y_pred)		


	print("Cross-Val accuracy:", np.mean(xval_acc).round(3))
	print("Cross-Val Balanced Accuracy:", np.mean(xval_bacc).round(3))
	print("Held-out Test Accuracy:", np.mean(test_acc).round(3))
	print("Held-out Test Balanced Accuracy:", np.mean(test_bacc).round(3))
	
	#cross validate predict so we can get the classification report and confusion matrix
	cv_pred = cross_val_predict(pipeline, X_in, y_in, cv = cv)
	print(confusion_matrix(y_in, cv_pred))
	print(classification_report(y_in, cv_pred))
	
	#save the results
	savepath = sys.argv[3]
	
	results = pd.DataFrame()
	results['xval_acc'] = xval_acc
	results['xval_bacc'] = xval_bacc
	results['test_acc'] = test_acc
	results['test_bacc'] = test_bacc
	results_name = 'dummy_results_for_{}'.format(y_target[0])+'.csv'
	results_path = savepath + results_name
	results.to_csv(results_path, index=True)
	
	#save the confusion matrix
	cmat = pd.DataFrame(confusion_matrix(y_in, cv_pred))
	cmat_name = 'dummy_confusion_matrix_for_{}'.format(y_target[0])+'.csv'
	cmat_path = savepath + cmat_name
	cmat.to_csv(cmat_path, index=True)
	
	#save the classification report
	crep = pd.DataFrame(classification_report(y_in, cv_pred, digits = 3, output_dict = True)).T
	crep_name = 'dummy_classification_report_for_{}'.format(y_target[0])+'.csv'
	crep_path = savepath + crep_name
	crep.to_csv(crep_path, index = True)

	#save the model 
	mod_name = 'dummy_{}'.format(y_target[0])+'.pkl'
	filename = savepath + mod_name
	dump(pipeline, open(filename, 'wb'))
	
	print("\nResults saved to {}".format(savepath))
	print("\nModel saved to {}".format(filename))
 
if __name__ == "__main__":
	main()

