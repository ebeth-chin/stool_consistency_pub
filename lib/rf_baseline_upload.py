#!/usr/bin/python3

print('Loading modules...')

import os, sys, getopt, datetime
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from pickle import load, dump
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from get_transformer_feature_names import *

#set the working directory 
os.chdir('/project/lemay_diet_guthealth/echin/stoolcon/')

#set seed
np.random.seed(0)


# BASLINE RF, no hyperparameters are tuned# 


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
	
	#numeric transformer - scaling only
		ss_transformer = Pipeline(steps = [('ss', StandardScaler())])
		
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
				('ss', ss_transformer, numerical_features),
				('cat', categorical_transformer, categorical_features)
			]
		)

	#get feature names
		prep.fit(X_in) #fit the preprocessing step
		feature_names = get_transformer_feature_names(prep)
	
	#make the model
		model = RandomForestClassifier(random_state = 0,
					oob_score = True,
					n_jobs = -1)
		
	#setting up pipeline
		pipeline= Pipeline(steps = [(
			'preprocessor', prep),
			('rf', model)]) 


		cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

		scoring = {'accuracy','balanced_accuracy'} 
		refit = 'balanced_accuracy'
		
		train_accuracy = []
		test_accuracy = []
		train_balanced = []
		test_balanced = []
		oob_accuracy = []
		oob_balanced = []
						  
		for train_index,test_index in cv.split(X_in,y_in):
			pipeline.fit(X_in.iloc[train_index],(y_in.iloc[train_index]).values.ravel())
			train_pred = pipeline.predict(X_in.iloc[train_index])
			test_pred = pipeline.predict(X_in.iloc[test_index])
			#accuracy
			train_accuracy.append(accuracy_score(y_in.iloc[train_index], train_pred))
			test_accuracy.append(accuracy_score(y_in.iloc[test_index], test_pred))
			#balanced accuracy
			train_balanced.append(balanced_accuracy_score(y_in.iloc[train_index], train_pred))
			test_balanced.append(balanced_accuracy_score(y_in.iloc[test_index], test_pred))	   
			#oob accuracy 
			oob_accuracy.append(pipeline.named_steps['rf'].oob_score_)
			#oob predictions
			oob_preds = (np.argmax(pipeline.named_steps['rf'].oob_decision_function_, axis = 1))
			oob_balanced.append(balanced_accuracy_score(y_in.iloc[train_index], oob_preds))
		
	xval_results = pd.DataFrame({'train_accuracy':train_accuracy, 'train_balanced_accuracy':train_balanced,
				'test_accuracy':test_accuracy, 'test_balanced_accuracy':test_balanced,
				'oob_accuracy':oob_accuracy, 'oob_balanced_accuracy':oob_balanced})
		
	print("Mean Validation Vanilla Accuracy Score:", np.mean(xval_results['test_accuracy']))
	print("Mean Validation Balanced Accuracy Score:", np.mean(xval_results['test_balanced_accuracy']))
	print("OOB Accuracy Score:", np.mean(xval_results['oob_accuracy']))
	print("OOB Balanced Accuracy:", np.mean(xval_results['oob_balanced_accuracy']))
		

	xval = cross_validate(pipeline, X_in, y_in.values.ravel(), cv= cv, return_train_score = True, return_estimator=True, n_jobs = -1, scoring = scoring)
	print('sanity check: validation accuracy:', np.mean(xval['test_accuracy']))
	print('sanity check: validation balanced acc:', np.mean(xval['test_balanced_accuracy']))

	#get the oob scores for the 'final' model 
	pipeline.fit(X_in, y_in)
	oob_pipe = []
	oob_pipe.append(pipeline.named_steps['rf'].oob_score_)
	pipe_preds = (np.argmax(pipeline.named_steps['rf'].oob_decision_function_, axis = 1))	
	oob_pipe_bacc = []
	oob_pipe_bacc.append(balanced_accuracy_score(y_in, pipe_preds))
	
	oob_pipedf= pd.DataFrame(oob_pipe)
	oob_pipedf = oob_pipedf.rename(columns = {0:'oob_accuracy'})
	oob_pipe_baccdf = pd.DataFrame(oob_pipe_bacc)
	oob_pipe_baccdf = oob_pipe_baccdf.rename(columns = {0:'oob_bacc'})

	#save the results
        
	savepath = sys.argv[3]

	oob_scores = pd.concat([oob_pipedf, oob_pipe_baccdf], axis = 1)
	scores_name = "oob_accuracy_for_{}".format(y_target[0])+"_{}".format(mod_name)+".csv"
	scores_path = savepath + scores_name 
	oob_scores.to_csv(scores_path, index = True)

	results_name = "cv_results_for_{}".format(y_target[0])+"_{}".format(mod_name)+".csv"
	results_path = savepath + results_name
	xval_results.to_csv(results_path, index = True)
	
	#save the model 
	mod_name = 'rf_{}'.format(y_target[0])+"_{}".format(mod_name)+'.pkl'
	filename = savepath + mod_name
	dump(pipeline, open(filename, 'wb'))
	
	print("\nResults saved to {}".format(savepath))
 
if __name__ == "__main__":
	main()
