from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)


file = open("./gradient_boosting_regressor_model.pkl", 'rb')
model = pickle.load(file)

data = pd.read_csv('./clean_data.csv')
data.head()

@app.route('/')
def index():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('index.html', sex= sex, smoker= smoker, region= region)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

    return str(prediction[0])           

if __name__=="__main__":
    app.run(debug=True)
