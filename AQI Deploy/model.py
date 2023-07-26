# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:16:39 2023

@author: KIIT
"""

# EDA Libraries
import pandas as pd
import numpy as np

# Data Transformation Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Data Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Regression Model
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('C:\\Users\\KIIT\\OneDrive\\Desktop\\city_day.csv')

data = data.dropna(axis = 0, subset = ['AQI'])

X_train = data.drop(['AQI', 'AQI_Bucket'], axis = 1)
y_train_reg = data['AQI']
y_train_cl = data['AQI_Bucket']

numerical_features = ['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
nominal_features = ['City']
drop_columns = ['Date', 'Xylene']

def FillMissingValues(X):
    for feat in numerical_features:
        X[feat] = X[feat].fillna(X.groupby('City')[feat].transform('mean'))
    return X

X_train = FillMissingValues(X_train)

numerical_pipeline = Pipeline([('simple imputer', SimpleImputer()), ("std scaler", StandardScaler())])
nominal_pipeline = Pipeline([ ("one hot encoding", OneHotEncoder() ) ])

pipeline = ColumnTransformer([
    ("numerical pipeline", numerical_pipeline, numerical_features),
    ("nominal pipeline", nominal_pipeline, nominal_features),
    ("drop columns", "drop", drop_columns)

])

output_columns = ['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
       'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
       'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
       'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai',
       'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
       'Visakhapatnam']

#transforming the train data through our pipeline

X_train_tr = pipeline.fit_transform(X_train)
X_train_tr = pd.DataFrame(X_train_tr, columns=output_columns)

model = RandomForestRegressor(max_depth = 25, max_features=15, min_samples_split=10, n_estimators=200, random_state = 42)
model.fit(X_train_tr, y_train_reg)

test_list = [['Delhi',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Ahmedabad',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Aizwal',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76],  
                       ['Amritsar',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Bengaluru',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76],
                       ['Amravati',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Bhopal',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Brajrajnagar',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Chandigarh',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Chennai',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Coimbatore',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Ernakulam',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Gurugram',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Guwahati',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Hyderabad',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Jaipur',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Jorapokhar',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Kochi',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Kolkata',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Lucknow',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Mumbai',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Patna',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Shillong',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Talcher',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Thiruvananthapuram',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76], 
                       ['Vishakhapatnam',	'2015-01-10',	221.020000,	361.74,	24.79,	46.39,	55.19,	134.060000,	9.70,	5.91,	34.12,	4.87,	9.44,	4.76] ]


test_sample = ['Lucknow',	'2015-01-10',	221.020000,	361.74,	24.79,	45.39,	55.19,	138.060000,	9.70,	5.91,	34.12,	3.87,	9.44,	4.76]

test_list.append(test_sample)

X_test = pd.DataFrame(test_list, columns = ['City', 'Date', 'PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'])

X_test_tr = pd.DataFrame(pipeline.fit_transform(X_test), columns=output_columns)

X_test_tr = X_test_tr.tail(1)

print(model.predict(X_test_tr))

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

