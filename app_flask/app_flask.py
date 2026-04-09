from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

base_path = 'data/processed/' if os.path.exists('data/processed/') else '../data/processed/'
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
model  = joblib.load(os.path.join(base_path, 'best_model.pkl'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income     = float(request.form['income'])
        loan_amt   = float(request.form['loan_amt'])
        total_debt = float(request.form['total_debt'])
        fico       = int(request.form['fico'])
        years      = int(request.form['years'])
        lines      = int(request.form['lines'])

        dti = total_debt / income if income > 0 else 0
        lti = loan_amt / income if income > 0 else 0

        feats = ['credit_lines_outstanding', 'loan_amt_outstanding',
                 'total_debt_outstanding', 'income', 'years_employed',
                 'fico_score', 'debt_to_income', 'loan_to_income']

        user_df = pd.DataFrame([[lines, loan_amt, total_debt,
                                  income, years, fico, dti, lti]],
                                columns=feats)

        pred  = model.predict(scaler.transform(user_df))[0]
        proba = model.predict_proba(scaler.transform(user_df))[0][1]

        result = "RISQUE ELEVE" if pred == 1 else "RISQUE FAIBLE"
        color  = "#DC2626" if pred == 1 else "#16A34A"

        return render_template('index.html',
                               prediction=result,
                               probability=f"{proba:.2%}",
                               dti=f"{dti:.2%}",
                               lti=f"{lti:.2%}",
                               color=color)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)


