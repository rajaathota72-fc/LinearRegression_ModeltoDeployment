import pickle
from flask import Flask, render_template,request
import numpy as np
app=Flask(__name__)
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        to_predict_list = [np.array(int_features)]
        result = ValuePredictor(to_predict_list)
        prediction = round(result,2)
        return render_template('result.html',prediction = prediction)
if __name__ == "__main__":
    app.run(debug=True)