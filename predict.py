import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('Parkinson_Detection')


@app.route('/predict', methods=['POST'])
def predict():
    individual = request.get_json()
    parkinson_detection = model.predict(individual)

    if parkinson_detection[0] == False:
        result = 'The person does not have parkinson disease'
    else:
        result = 'The person has parkinson disease'

    output = {
        'Parkinson_Detection': str(result)
    }

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
