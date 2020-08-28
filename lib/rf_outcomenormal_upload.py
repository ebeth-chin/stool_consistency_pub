#!/usr/bin/python3

print('Loading modules...')

import os, sys, getopt, datetime
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, balanced_accuracy_score,confusion_matrix, classification_report
from pickle import load, dump
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from get_transformer_feature_names import *

#set the working directory 
os.chdir('/project/lemay_diet_guthealth/echin/stoolcon/')

#set seed
np.random.seed(0)

#OUTCOME NORMAL  w/ STANDARD SCALER ONLY# 

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

	#numeric transformer- ss only
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
	
	#make the model
		model = RandomForestClassifier(random_state = 0,
					oob_score = True,
		#			warm_start = True,
					n_jobs = -1)
		
	#setting up pipeline
		pipeline= Pipeline(steps = [(
			'preprocessor', prep),
			('rf', model)])	

		#set up the parameter grid
		param_grid = {
				'rf__max_features':np.arange(0.1,1,0.05), 
				'rf__max_samples':np.arange(0.1, 1, 0.05), 
				'rf__n_estimators':np.arange(100,1000, 50) 
				} 
		
		cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

		scoring = {'accuracy','balanced_accuracy'} 
		refit = 'balanced_accuracy'
		
		grid_search = GridSearchCV(estimator = pipeline, param_grid = param_grid, n_jobs = -1, cv= cv,
						  scoring  = scoring, refit = refit, return_train_score = True)
						  
		#hyperparameter tune and refit the model to the entire dataset 
		grid_search.fit(X_in, y_in.values.ravel())
		
		print("Best params:", grid_search.best_params_)
		print("Best Validation Balanced Accuracy Score:", grid_search.best_score_)
		print("Best Validation Vanilla Accuracy Score:", grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_])
		print("OOB Score (vanilla accuracy):", grid_search.best_estimator_.named_steps['rf'].oob_score_)
		
		
	#save the results
	
	savepath = sys.argv[3]
	
	#get the predictions
	#these are the probabilities:
	preds = pd.DataFrame(grid_search.best_estimator_.named_steps['rf'].oob_decision_function_) 
	#convert the highest probability to a 1: 
	preds_binary = preds.eq(preds.where(preds != 0).max(1), axis=0).astype(int) 
	preds_binary = preds_binary.rename(columns={0: "not", 1: "normal"})
	#collapse into predicted classes: 
	conditions = [
		(preds_binary['normal'] == 1),
		(preds_binary['not']==1)]
	choices = ['normal', 'not']
	preds_binary['predicted_class'] = np.select(conditions, choices, default='not')
	preds_binary['pred_binary_class'] = np.argmax(grid_search.best_estimator_.named_steps['rf'].oob_decision_function_, axis=1)	
	#merge all the prediction outputs into a single df for saving
	preds_merged = preds.merge(preds_binary, left_index=True, right_index=True)
	preds_merged = preds_merged.rename(columns={0: "not_prob", 1: "normal_prob"})
	preds_merged['subjectid'] = y_in.index
	preds_merged = preds_merged.set_index(['subjectid'])
	preds_name = 'oob_predictions_for_{}'.format(y_target[0])+"_{}".format(mod_name)+'.csv'
	preds_path = savepath + preds_name
	preds_merged.to_csv(preds_path, index = True)


	print('oob balanced accuracy score:', balanced_accuracy_score(y_in, preds_binary['pred_binary_class']))
	
	oob_score = []
	oob_bacc = []
	oob_score.append(grid_search.best_estimator_.named_steps['rf'].oob_score_)
	oob_bacc.append(balanced_accuracy_score(y_in, preds_binary['pred_binary_class']))
	bacc_df = pd.DataFrame(oob_bacc)
	bacc_df = bacc_df.rename(columns =	{0:'oob_balanced_acc'})
	oob_df = pd.DataFrame(oob_score)
	oob_df = oob_df.rename(columns={0:'oob_accuracy'})
	oob_accs = pd.concat([oob_df, bacc_df], axis = 1)
	oob_name = 'oob_accuracy_for_{}'.format(y_target[0])+'_{}'.format(mod_name)+'.csv'
	oob_path = savepath + oob_name
	oob_accs.to_csv(oob_path, index = True)
	
	#save the confusion matrix
	pred_class = preds_binary[["pred_binary_class"]]
	print("Confusion Matrix:\n\n", confusion_matrix(y_in, pred_class))
	cmat = pd.DataFrame(confusion_matrix(y_in, pred_class))
	cmat_name = 'confusion_matrix_for_{}'.format(y_target[0])+"_{}".format(mod_name)+'.csv'
	cmat_path = savepath + cmat_name
	cmat.to_csv(cmat_path, index=True)
	
	#save the classification report
	print("Classification Report:\n\n", classification_report(y_in, pred_class, digits = 3))
	crep = pd.DataFrame(classification_report(y_in, pred_class, digits = 3, output_dict = True)).T
	crep_name = 'classification_report_for_{}'.format(y_target[0])+"_{}".format(mod_name)+'.csv'
	crep_path = savepath + crep_name
	crep.to_csv(crep_path, index = True)
	
	#save the CV results
	results = pd.DataFrame(grid_search.cv_results_)
	name = 'cv_results_for_{}'.format(y_target[0])+'.csv'
	path = savepath + name
	results.to_csv(path, index=True)
	
	# save the cv predictions
	train_pred = cross_val_predict(grid_search.best_estimator_, X_in, y_in.values.ravel(), cv = cv)
	train_actual = y_in
	predictions = pd.DataFrame(index=X_in.index)
	predictions['train_pred'] = train_pred
	predictions['train_actual'] = y_in
	pred_name = 'cv_predictions_for_{}'.format(y_target[0])+"_{}".format(mod_name)+'.csv'
	pred_path = savepath +pred_name
	predictions.to_csv(pred_path,index=True)
	
	print("\nResults saved to {}".format(savepath))
	print("\nModel saved to {}".format(filename))
 
if __name__ == "__main__":
	main()
