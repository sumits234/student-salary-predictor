from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    def to_float(value):
        try:
            return float(value)
        except:
            return 0.0

    features = {
        'CGPA': to_float(request.form.get('CGPA')),
        'Internships': to_float(request.form.get('Internships')),
        'Projects': to_float(request.form.get('Projects')),
        'Workshops/Certifications': to_float(request.form.get('Workshops')),
        'AptitudeTestScore': to_float(request.form.get('AptitudeTestScore')),
        'SoftSkillsRating': to_float(request.form.get('SoftSkills')),
        'ExtracurricularActivities': to_float(request.form.get('Extracurricular')),
        'PlacementTraining': to_float(request.form.get('PlacementTraining')),
    }

    input_df = pd.DataFrame([features])

    # Use 'model' instead of 'pipeline'
    predicted_salary = model.predict(input_df)[0]

    return render_template('result.html', salary=round(predicted_salary, 2))


if __name__ == '__main__':
    app.run(debug=True)
