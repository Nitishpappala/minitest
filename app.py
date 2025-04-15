from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import pandas as pd

app = Flask(__name__, static_folder='assets')

# Load model and scaler
model = xgb.Booster()
model.load_model('xgboost_gps_spoofing_model.json')
scaler = joblib.load('scaler.pkl')

def engineer_features(input_data):
    df = pd.DataFrame([input_data])
    df['CP_diff'] = 0
    df['CP_consistency'] = 0
    df['signal_quality'] = df['CN0']
    df['PC_PIP_ratio'] = df['PC'] / df['PIP']
    df['EC_LC_ratio'] = df['EC'] / df['LC']
    df['CP_rate'] = 0
    df['PD_rate'] = 0
    df['quality_score'] = ((df['CN0'] >= 45.0) & (np.abs(df['PC'] - df['PIP']) < 1000.0)).astype(int)
    return df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_features = {
                'PRN': float(request.form['PRN']),
                'DO': float(request.form['DO']),
                'PD': float(request.form['PD']),
                'CP': float(request.form['CP']),
                'EC': float(request.form['EC']),
                'LC': float(request.form['LC']),
                'PC': float(request.form['PC']),
                'PIP': float(request.form['PIP']),
                'PQP': float(request.form['PQP']),
                'TCD': float(request.form['TCD']),
                'CN0': float(request.form['CN0'])
            }
            df_processed = engineer_features(input_features)

            features = ['PRN', 'DO', 'PD', 'CP', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'TCD', 'CN0',
                        'CP_consistency', 'signal_quality', 'PC_PIP_ratio', 'EC_LC_ratio',
                        'CP_rate', 'PD_rate', 'quality_score']

            features_scaled = scaler.transform(df_processed[features])
            dtest = xgb.DMatrix(features_scaled)

            pred_probs = model.predict(dtest)
            prediction = np.argmax(pred_probs[0])
            labels = {
                0: "Authentic GPS Signal (Unspoofed)",
                1: "Spoofed GPS Signal (Type 1)",
                2: "Spoofed GPS Signal (Type 2)",
                3: "Spoofed GPS Signal (Type 3)"
            }
            result = labels.get(prediction, f"Unknown Prediction (Raw Output: {prediction})")
            confidence = float(pred_probs[0][prediction]) * 100

            return render_template('index.html', prediction_text=f'{result}', confidence=f'Confidence: {confidence:.2f}%')

        except ValueError:
            return render_template('index.html', prediction_text="Invalid input! Please enter numerical values.")
        except Exception as e:
            return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
