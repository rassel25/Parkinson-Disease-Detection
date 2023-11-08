# Midterm project ML Zoomcamp 2023: Prediction of Parkinson's disease

## Description of problem and data

In this project, I examined the problem of whether someone has Parkinson's disease.

According to this problem, I found a dataset on Kaggle.
The link for the dataset: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set.
You can find the dataset in my GitHub repository also: https://github.com/rassel25/zoomcamp/blob/main/parkinsons.data.

The dataset contains the following columns:
- name - ASCII subject name and recording number
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
- MDVP:Flo(Hz) - Minimum vocal fundamental frequency
- MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP - Several measures of variation in fundamental frequency
- MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
- NHR, HNR - Two measures of the ratio of noise to tonal components in the voice
- status - The health status of the subject (one) - Parkinson's, (zero) - healthy
- RPDE, D2 - Two nonlinear dynamical complexity measures
- DFA - Signal fractal scaling exponent
- spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation

The target variable is Status which contains values: 0(healthy) and 1(parkinson disease)

## Description of project

In the project, I first cleaned the data by doing EXPLORATORY DATA ANALYSIS AND DATA PREPROCESSING.

Then I used FEATURE SELECTION techniques to get the top 5 best features. In the end, I used all the features to train our model as this gives the best performance.

After that, I used PIPELINE to do FEATURE SCALING & OVERSAMPLING.

Lastly, I used GRIDSEARCH CV for MODEL SELECTION, HYPERPARAMETERS TUNING.

## Description of files in Github Repository


- Data: 

- Jupyter Notebook: notebook.ipynb with
  - Data preparation and data clearning
  - EDA, feature importance analysis
  - Model selection process and parameter tuning

- Script: train.py - in here the final model is build

- model bin: The final model and Dict Vecotrizer are saved by pickle to model.boin 

- predict.py. contains
  - Loading the model
  - Serving it via a web service via flask

- Pipenv and Pipenv.lock: to build a evironment via Pipenv

- Dockerfile: for running the service

- test_predict.py: to test the dockerfilde

## Description on how to use the model

Build the docker image: docker build -t obesity:latest .

Run the docker image: docker run -it -p 9696:9696 obesity:latest 

Test the docker impage: python test_predict.py

If you want to test another individual than given in test_predict use the following structure: 

individual = {'gender': 'male' or 'female',

                'family_history_with_overweight': 'yes' or 'no',
                
                'consumption_high_caloric_food': 'yes' or 'no',
                
                'consumption_between_meals': 'sometimes', 'frequently', 'always'or 'no',
                
                'smoke': 'yes' or 'no',
                
                'calories_consumption_monitoring': 'yes' or 'no',
                
                'consumption_alcohol': 'sometimes', 'frequently' or 'no',
                
                'transportation_used': 'automotive', 'public_transportation' or 'walk_or_bike',
                
                'age': a number of the age, must not be integer (note that the research only covered ages to 50, prediciton for ages above 50 might be incorrect),
                
                'consumption_vegetables': number of how often a day vegetabels are consumped in average,
                
                'number_meals': number of meals a day in average ,
                
                'consumption_water': number of liters of water to drink on a day in average,
               
               'physical_activity': frequence of pysical activity on a day in average,
               
               'time_techn_devices': time spend with technical devices on a day in average}

