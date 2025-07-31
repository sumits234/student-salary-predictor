from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load dataset for gap analysis
data = pd.read_csv('placementdata.csv')
feature_columns = [
    'CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
    'AptitudeTestScore', 'SoftSkillsRating',
    'ExtracurricularActivities', 'PlacementTraining'
]

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

    student = {
        'CGPA': to_float(request.form.get('CGPA')),
        'Internships': to_float(request.form.get('Internships')),
        'Projects': to_float(request.form.get('Projects')),
        'Workshops/Certifications': to_float(request.form.get('Workshops')),
        'AptitudeTestScore': to_float(request.form.get('AptitudeTestScore')),
        'SoftSkillsRating': to_float(request.form.get('SoftSkills')),
        'ExtracurricularActivities': to_float(request.form.get('Extracurricular')),
        'PlacementTraining': to_float(request.form.get('PlacementTraining')),
    }

    input_df = pd.DataFrame([student])
    predicted_salary = float(model.predict(input_df)[0])

    # Salary range ±20%
    lower = round(predicted_salary * 0.8, 2)
    upper = round(predicted_salary * 1.2, 2)

    # Domain suggestions
    def recommend_domains(salary):
        if salary < 4.0:
            return ['BPO', 'Technical Support', 'Data Entry', 'Sales Trainee']
        elif salary < 6.0:
            return ['QA Testing', 'Manual Testing', 'Support Engineer']
        elif salary < 8.0:
            return ['Full-Stack Web Dev', 'Mobile App Dev', 'Backend Developer']
        elif salary < 10.0:
            return ['Data Analyst', 'Software Engineer', 'DevOps', 'System Admin']
        elif salary < 12.5:
            return ['AI/ML Engineer', 'Cloud Engineer', 'Business Analyst']
        else:
            return ['AI/ML Specialist', 'Data Scientist', 'Solution Architect', 'Product Manager']

    recommended_domains = recommend_domains(predicted_salary)

    # Gap analysis
    suggestions = []
    if predicted_salary < 4.0:
        diffs = {}
        for col in feature_columns:
            avg = data[col].mean()
            diffs[col] = avg - student[col]
        top_gaps = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:3]
        for feat, gap in top_gaps:
            suggestions.append(f"📌 Improve {feat} (you are {gap:.1f} below average)")

    return render_template('result.html',
                           salary_center=round(predicted_salary, 2),
                           salary_range=f"{lower} – {upper} LPA",
                           is_low=predicted_salary < 4.0,
                           suggestions=suggestions,
                           recommended_domains=recommended_domains)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
