import requests

host = 'parkinson-disease-env.eba-7ddpgguy.eu-west-1.elasticbeanstalk.com'
url = f'http://{host}/predict'

individual = {
 "MDVP:Fo(Hz)": 222.236,
 "MDVP:Fhi(Hz)": 231.345,
 "MDVP:Flo(Hz)": 205.495,
 "MDVP:Jitter(%)": 0.00266,
 "MDVP:Jitter(Abs)": 1e-05,
 "MDVP:RAP": 0.00152,
 "MDVP:PPQ": 0.00144,
 "Jitter:DDP": 0.00457,
 "MDVP:Shimmer": 0.01643,
 "MDVP:Shimmer(dB)": 0.145,
 "Shimmer:APQ3": 0.00867,
 "Shimmer:APQ5": 0.01108,
 "MDVP:APQ": 0.012,
 "Shimmer:DDA": 0.02602,
 "NHR": 0.0034,
 "HNR": 25.856,
 "RPDE": 0.364867,
 "DFA": 0.694399,
 "spread1": -6.966321,
 "spread2": 0.095882,
 "D2": 2.278687,
 "PPE": 0.103224
}

response = requests.post(url, json=individual).json()

print(response)


''''def disease(x):
 if x == 0:
  return 'The person does not have parkinson disease'
 else:
  return 'The person has parkinson disease'''

