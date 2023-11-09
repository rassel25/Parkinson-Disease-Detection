# Midterm project ML Zoomcamp 2023: Parkinson Disease Detection

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

I have used 5 models to train the model: Support Vector Classifier, kNN Classifier, Gaussian Naive Bayes, Decision Tree Classifier, Random Forest Classifier.

The best model with its Hyperparameters: SVC(C=10, kernel='poly')

The ROC_AUC score, precision score, recall score, and f1-score for the training set are: 0.92, 0.96, 0.92, and 0.94.

The ROC_AUC score, precision score, recall score, and f1-score for the testing set are: 0.87, 0.87, 0.87, and 0.87.

## Description of files in Github Repository

- Data: parkinsons.data

- Jupyter Notebook: notebook.ipynb with
  - Data preparation and data cleaning
  - EDA, feature importance analysis
  - Model selection process and parameter tuning

- Script: train.py - in here the final model is build

- model.bin: The final model with its pipeline and parameters are saved by pickle to model.boin 

- predict.py. contains
  - Loading the model
  - Serving it via flask web service

- Files with dependencies: Pipfile and Pipfile.lock to build an environment via Pipenv

- Dockerfile: for running the service

- test_predict.py: to test the docker file

- Deployment: Used AWS Beanstalk to deploy the dockerized file

## Description of how to use the model

# Docker

- isoloate the environment from the host machine
- You can find docker image here https://hub.docker.com/_/python
- I have chosen python:3.10 to match my python version
- Build the docker image:  docker build -t parkinson-disease . 
- I have chosen python:3.10 to match my python version
- Run the docker image: docker run -it --rm -p 9696:9696 parkinson-disease   
- Test the docker image: python predict_test.py

# Deploy to AWS Elastic Beanstalk

- create an aws account
- install eb cli as dev dependency pipenv install awsebcli --dev
- go to virtual environment pipenv shell
- initial the eb init -p "Docker running on 64bit Amazon Linux 2" parkinson-disease
- ls -a to check whether there is .elasticbeanstalk folder
- ls .elasticbeanstalk/ to check the doc inside the folder config.yml
- run locally to test eb local run --port 9696
- in another terminal run python predict_test.py to test
- implement in the cloud: create a cloud environment -> eb create parkinson-disease-env
- copy the service link to predict_test.py, update our url

https://github.com/rassel25/Parkinson-Disease-Detection/assets/36706178/a2af4c50-b13d-4c1f-9b90-1b960a576de3

I have terminated this service to avoid generating extra fees.

