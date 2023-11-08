# Midterm project ML Zoomcamp 2022: Prediction of obesity level

## Description of problem and data

In this project I examined the problem, if you can predict the obestiy level of an individual given its eating habits and pyhsical condition.

According to this problem I found a dataset in the UCI Machine Learning Repository. 
The link is: https://archive-beta.ics.uci.edu/ml/datasets/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition#Abstract.
You can find the datasat in my github repository also: https://github.com/katrinlaura73/MLZoomcamp/blob/main/MidtermProject/ObesityDataSet_raw_and_data_sinthetic.csv.

In a study the researchers asked the following attributes:
- Gender
- Age
- Height
- Weight
- Family history of Obesity
- Frequent consumption of high caloric food (FAVC)
- Frequency of consumption of vegetables(FCVC)
- Number of main meals (NCP)
- Consumption of food between meals (CAEC)
- Smoking Habits
- Consumption of water daily (CH20)
- Consumption of alcohol (CALC)
- Calories consumption monitoring (SCC)
- Physical activity frequency (FAF)
- Time using technology devices (TUE)
- Transportation used (MTRANS)

They calculated the obesity level from weight and height. The measure is called BMI and is calculated by BMI =  weight / height^2. 
The calculation resulted in the following obesity levels:
- Underweight Less than 18.5
- Normal 18.5 to 24.9
- Overweight (I and II) 25.0 to 29.9
- Obesity I 30.0 to 34.9
- Obesity II 35.0 to 39.9
- Obesity III Higher than 40.

## Description of project

In the project I first cleaned the data.

Then I tested different classification methods: 
- Multiclass LogisticRegression
- Random Forest Decision Tree
- Gradient Boost

I searched the best parameters for the models and compared them. For comparison I used the metric AUC.

The best model turned out to be Random Forest with the parameters n_estimators=140, max_depth=15 and min_samples_leaf=1.

## Description of files in Github Repository


- Data: ObesityDataSet_raw_and_data_sinthetic.csv

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

