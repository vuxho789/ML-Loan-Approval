# -*- coding: utf-8 -*-
"""
This program was written Vu Ho
Created on Mon Apr 18 22:49:39 2022

[Program Summary]
"""

import numpy as np
import pandas as pd
import io
import requests
import os
from pathlib import Path
import sys
import csv
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import f_regression, f_classif
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from scipy import stats
from datetime import datetime
from imblearn.over_sampling import SMOTE
import warnings


warnings.filterwarnings('ignore')


def handling_missing_values(traindata, testdata):
    missing_train_df = traindata.isna().sum().reset_index().rename(columns={"index": "Column", 0: 'Count'})
    missing_train_df['Percent'] = (missing_train_df[['Count']]/len(traindata.index)*100)
    
    col_missing_over_60 = missing_train_df.loc[missing_train_df['Percent'] > 60].sort_values(ascending=False, by='Percent')
    
    # Drop columns which have missing values more than 60% 
    modified_traindata = traindata.drop(col_missing_over_60['Column'], 1)
    modified_testdata = testdata.drop(col_missing_over_60['Column'], 1)
    
    col_missing_under_60 = missing_train_df.loc[(missing_train_df['Percent'] < 60) & (missing_train_df['Percent'] != 0)].sort_values(ascending=False, by='Percent')
    
    # Get a DataFrame of columns which have missing values less than 60%
    missing_under_60_df = traindata[col_missing_under_60['Column'].tolist()]
    
    # Get a DataFrame of numerical columns which have missing values less than 60%
    numerical_missing_df = missing_under_60_df.select_dtypes('number')
    
    # Get a DataFrame of categorical columns which have missing values less than 60%
    categorical_missing_df = missing_under_60_df.select_dtypes('object')
    
    # Replace missing numerical values with median
    for column in numerical_missing_df.columns.tolist():
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        modified_traindata[column] = imputer.fit_transform(traindata[[column]])
        modified_testdata[column] = imputer.transform(testdata[[column]])
        
    # Replace missing categorical values with most frequent values
    for column in categorical_missing_df.columns.tolist():
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        modified_traindata[column] = imputer.fit_transform(traindata[[column]])
        modified_testdata[column] = imputer.transform(testdata[[column]])
    
    return modified_traindata, modified_testdata


def handling_outliers(data):
    # Replace 365243 value with 0 in DAYS_EMPLOYED column
    if 365243 in data['DAYS_EMPLOYED'].values:
        data['DAYS_EMPLOYED'].replace({365243: 0}, inplace = True)

    # Replace XNA value with F (most frequent) in CODE_GENDER column
    if 'XNA' in data['CODE_GENDER'].values:
        data['CODE_GENDER'].replace({'XNA': 'F'}, inplace = True)
        
    return data
       
 
def removing_colinear(data):
    # List of colinear columns to be removed
    colinear = ['AMT_GOODS_PRICE', 'FLOORSMAX_MEDI', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'ELEVATORS_MEDI']
    data.drop(colinear, 1, inplace=True)
    
    return data


def label_encoding(traindata, testdata):
    categorical_features_train = traindata.select_dtypes('object')
    bin_cat_train_df = categorical_features_train.loc[:, categorical_features_train.nunique() == 2]
    bin_cat_train_list = bin_cat_train_df.columns.tolist()
    
    l_encoder = LabelEncoder()
    for column in bin_cat_train_list:
        traindata[column] = l_encoder.fit_transform(traindata[column])
        testdata[column] = l_encoder.transform(testdata[column])

    return traindata, testdata


def OH_encoding(traindata, testdata):
    categorical_features_train = traindata.select_dtypes('object')
    multi_cat_train_list = categorical_features_train.columns.tolist()
    
    categorical_features_test = testdata.select_dtypes('object')
    multi_cat_test_list = categorical_features_test.columns.tolist()
    
    # Inititate an instance of OneHotEncoder class
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    # Transform feature columns with OneHotEncoder (Output is an array)
    oh_encoded_train_array = oh_encoder.fit_transform(traindata[multi_cat_train_list])
    
    oh_encoded_test_array = oh_encoder.transform(testdata[multi_cat_test_list])
    
    # Convert the encoded array to df
    oh_encoded_train_df = pd.DataFrame(oh_encoded_train_array, index=traindata.index)
    oh_encoded_train_df.columns = oh_encoder.get_feature_names(multi_cat_train_list)
    
    oh_encoded_test_df = pd.DataFrame(oh_encoded_test_array, index=testdata.index)
    oh_encoded_test_df.columns = oh_encoder.get_feature_names(multi_cat_train_list)
    
    #Extract only the columns that didnt need to be encoded from original df
    other_train_cols_df = traindata.drop(columns=multi_cat_train_list)
    
    other_test_cols_df = testdata.drop(columns=multi_cat_test_list)
    
    #Concatenate the two dataframes : 
    oh_traindata = pd.concat([other_train_cols_df, oh_encoded_train_df], axis=1)
    oh_testdata = pd.concat([other_test_cols_df, oh_encoded_test_df], axis=1)
    
    return oh_traindata, oh_testdata


def scaling_features(traindata, testdata):
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(traindata)
    X_test = std_scaler.transform(testdata)
    
    X_train = pd.DataFrame(X_train, columns = traindata.columns)
    X_test = pd.DataFrame(X_test,columns = testdata.columns)
    
    return X_train, X_test


def feature_Pearson(X_train, Y_train, X_val, num_features):  
    # configure to select a subset of features
    fs = SelectKBest(score_func=f_regression, k=num_features)
    # Fit the model with training data
    fs.fit(X_train, Y_train)
    
    # Transform training input data
    X_train_fs = fs.transform(X_train)
    
    # Transform validation input data
    X_val_fs = fs.transform(X_val)
    
    return X_train_fs, X_val_fs, fs



def get_features_part1(X_train, Y_train):
    # Split the training set into train set and validation set
    X_train_p1, X_val_p1, Y_train_p1, Y_val_p1 = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
    
    # Perform feature selection on the training and validation set
    # Select top 80 features
    X_train_pearson, X_val_pearson, pearson = feature_Pearson(X_train_p1, Y_train_p1, X_val_p1, 80)
    
    # Get a list of top 80 features
    filter_p1 = pearson.get_support()
    features_list_p1 = np.array(X_train_p1.columns.tolist())
    selected_features_p1 = features_list_p1[filter_p1].tolist()
    
    return selected_features_p1


def training_part1(traindata, testdata):
    X_train_part1 = traindata.drop(['AMT_INCOME_TOTAL', 'SK_ID_CURR'], 1)
    Y_train_part1 = traindata['AMT_INCOME_TOTAL']

    X_test_part1 = testdata.drop(['AMT_INCOME_TOTAL', 'SK_ID_CURR'], 1)
    Y_test_part1 = testdata['AMT_INCOME_TOTAL']
    
    # Get a list of selected features
    selected_features = get_features_part1(X_train_part1, Y_train_part1)
    
    # Apply selected features to the training set and test set
    X_train_part1_selected = X_train_part1[selected_features]
    X_test_part1_selected = X_test_part1[selected_features]
    
    # Select model Ridge Regression
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    
    # Fit the model with training data
    ridge_reg.fit(X_train_part1_selected, Y_train_part1)
    
    # Evaluate the model
    Y_predicted = ridge_reg.predict(X_test_part1_selected)
    
    # Evaluate the predictions
    MSE = mean_squared_error(Y_test_part1, Y_predicted)
    r, p_value = stats.pearsonr(Y_test_part1, Y_predicted)
    
    print('Part 1 - Linear Regression metrics:')
    print(f'MSE: {round(MSE, 2)}')
    print(f'Pearson\'s correlation: {round(r, 2)}')
    print()
    
    # Print summary of MSE and correlation values to csv file
    summary_df = pd.DataFrame(columns=['zid', 'MSE', 'correlation'],\
                              data=[['z5335667', round(MSE, 2), round(r, 2)]])
    summary_df.to_csv('z5335667.PART1.summary.csv', index=False)
    
    
    list_predicted_income = [int(round(x, 0)) for x in Y_predicted]
    
    # Print output of client ID and their predicted income to csv file
    output_df = pd.DataFrame(columns=['SK_ID_CURR', 'predicted_income'])
    output_df['SK_ID_CURR'] = testdata['SK_ID_CURR']
    output_df['predicted_income'] = pd.DataFrame(list_predicted_income)
    output_df.to_csv('z5335667.PART1.output.csv', index=False)


def feature_ANOVA(X_train, Y_train, X_val, num_features):
    # Configure to select a subset of features
    fs = SelectKBest(score_func=f_classif, k=num_features)
    
    # Fit the model with training data
    fs.fit(X_train, Y_train)
    
    # Transform training input data
    X_train_fs = fs.transform(X_train)
    
    # Transform validation input data
    X_val_fs = fs.transform(X_val)
    
    return X_train_fs, X_val_fs, fs


def get_features_part2(X_train, Y_train):
    # Split the training set into train set and validation set
    X_train_p2, X_val_p2, Y_train_p2, Y_val_p2 = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
    
    # Perform feature selection on the training and validation set
    # Select top 75 features
    X_train_anova, X_val_anova, anova = feature_ANOVA(X_train_p2, Y_train_p2, X_val_p2, 75)
    
    # Get a list of top 80 features
    filter_p2 = anova.get_support()
    features_list_p2 = np.array(X_train_p2.columns.tolist())
    selected_features_p2 = features_list_p2[filter_p2].tolist()
    
    return selected_features_p2


def SMOTE_oversampling(X_train, Y_train):
    oversampling = SMOTE(sampling_strategy=0.5)
    X_over, Y_over = oversampling.fit_resample(X_train, Y_train)
    X_over = pd.DataFrame(X_over, columns=X_train.columns)
    Y_over = pd.DataFrame(Y_over)
    return X_over, Y_over


def training_part2(traindata, testdata):
    X_train_part2 = traindata.drop(['TARGET', 'SK_ID_CURR'], 1)
    Y_train_part2 = traindata['TARGET']

    X_test_part2 = testdata.drop(['TARGET', 'SK_ID_CURR'], 1)
    Y_test_part2 = testdata['TARGET']
    
    # Scale the input features
    X_train_part2_scaled, X_test_part2_scaled = scaling_features(X_train_part2, X_test_part2)
    
    # Get a list of selected features
    selected_features = get_features_part2(X_train_part2_scaled, Y_train_part2)
    
    # Apply selected features to the training set and test set
    X_train_part2_selected = X_train_part2_scaled[selected_features]
    X_test_part1_selected = X_test_part2_scaled[selected_features]
    
    # Apply SMOTE oversampling to the training set
    X_train_part2_over, Y_train_part2_over = SMOTE_oversampling(X_train_part2_selected, Y_train_part2)
    
    # Select model Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier()
    
    # Fit the model with training data
    gb_clf.fit(X_train_part2_over, Y_train_part2_over)
    # gb_clf.fit(X_train_part2_selected, Y_train_part2)
    
    # Evaluate the model
    Y_predicted = gb_clf.predict(X_test_part1_selected)
    
    accuracy = accuracy_score(Y_test_part2, Y_predicted)
    avg_precision = precision_score(Y_test_part2, Y_predicted, average='macro')
    avg_recall = recall_score(Y_test_part2, Y_predicted, average='macro')
    avg_f1score = f1_score(Y_test_part2, Y_predicted, average='macro')
    
    print('Part 2 - Classification metrics:')
    print(f'Average precision: {round(avg_precision, 2)}')
    print(f'Average recall: {round(avg_recall, 2)}')
    print(f'F1-Score: {round(avg_f1score,2)}')
    print(f'Accuracy: {round(accuracy, 2)}')
    print()
    
    # Print summary of MSE and correlation values to csv file
    summary_df = pd.DataFrame(columns=['zid', 'average_precision', 'average_recall', 'accuracy'],\
                              data=[['z5335667', round(avg_precision, 2), round(avg_recall, 2), round(accuracy, 2)]])
    summary_df.to_csv('z5335667.PART2.summary.csv', index=False)
    
    list_predicted_target = [int(round(x, 0)) for x in Y_predicted]
    
    # Print output of client ID and their predicted income to csv file
    output_df = pd.DataFrame(columns=['SK_ID_CURR', 'predicted_target'])
    output_df['SK_ID_CURR'] = testdata['SK_ID_CURR']
    output_df['predicted_target'] = pd.DataFrame(list_predicted_target)
    output_df.to_csv('z5335667.PART2.output.csv', index=False)


# ----- MAIN PROGRAM -----
if __name__ == '__main__':
    start_time = datetime.now()
    print(f'Start time: {start_time}')
    
    # Read input file names from the command line
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    
    # Check if the data files exist in the directory
    if not os.path.exists(training_file) or not os.path.exists(testing_file):
        print('No data files in the directory, giving up...')
        sys.exit()
    
    # Use Pandas read_csv() module to read input file into DataFrame
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(testing_file)
    
    # Handle missing data
    processed_train_data, processed_test_data = handling_missing_values(train_data, test_data)

    # Handle outliers
    processed_train_data = handling_outliers(processed_train_data)
    processed_test_data = handling_outliers(processed_test_data)
    
    # Remove colinear features
    processed_train_data = removing_colinear(processed_train_data)   
    processed_test_data = removing_colinear(processed_test_data)
    
    # Apply label encoding
    label_encoded_train, label_encoded_test = label_encoding(processed_train_data, processed_test_data)
    
    # Apply one hot encoding
    oh_encoded_train, oh_encoded_test = OH_encoding(label_encoded_train, label_encoded_test)
    
    # Train model and prepare output files for part 1
    training_part1(oh_encoded_train, oh_encoded_test)
    
    # Train model and prepare output files for part 2
    training_part2(oh_encoded_train, oh_encoded_test)
    
    end_time = datetime.now()
    print(f'End time: {end_time}')
    print('Duration: {}'.format(end_time - start_time))