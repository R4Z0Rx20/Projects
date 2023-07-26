# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:53:12 2023

@author: KIIT
"""

# EDA Libraries
import pandas as pd
import numpy as np

#Flask
from flask import Flask, request, render_template

import pickle

# Data Transformation Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Data Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numerical_features = ['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
nominal_features = ['City']
drop_columns = ['Date', 'Xylene']


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

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    test_sample = [x for x in request.form.values()] 
    for x in test_sample[2:]:
        x = float(x)
    test_list.append(test_sample)
    X_test = pd.DataFrame(test_list, columns = ['City', 'Date', 'PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'])
    X_test_tr = pd.DataFrame(pipeline.fit_transform(X_test), columns=output_columns)
    X_test_tr = X_test_tr.tail(1)
    prediction = model.predict(X_test_tr)

    aqi = round(prediction[0], 2)
    
    if aqi<=50:
        output = 'Good'
    elif aqi<=100:
        output = 'Satisfactory'
    elif aqi<=200:
        output = 'Moderate'
    elif aqi<=300:
        output = 'Poor'
    elif aqi<=400:
        output = 'Very Poor'
    else: 
        output = 'Severe'

    return render_template('index.html', prediction_text=f'Air Quality Index is {aqi}. The Air Quality is {output}')

if __name__ == "__main__":
    app.run()
