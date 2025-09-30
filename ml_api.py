from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

model = None
X_train = None
y_train = None

def preprocess(df):
    if 'region' in df.columns:
        df = df.drop('region', axis=1)
    df['sex_enc'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker_enc'] = df['smoker'].map({'no': 0, 'yes': 1})
    return df

@app.route('/train', methods=['POST'])
def train():
    global model, X_train, y_train
    file = request.files['file']
    df = pd.read_csv(file)
    df = preprocess(df)

    X_train = df[['age', 'sex_enc', 'bmi', 'children', 'smoker_enc']]
    y_train = df['charges']

    model = LinearRegression()
    model.fit(X_train, y_train)

    return jsonify({'status': 'ok'})

@app.route('/test', methods=['POST'])
def test():
    global model
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400

    file = request.files['file']
    df = pd.read_csv(file)
    df = preprocess(df)

    X_test = df[['age', 'sex_enc', 'bmi', 'children', 'smoker_enc']]
    y_test = df['charges']

    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return jsonify({
        'r2_score': r2,
        'mean_squared_error': mse,
        'mean_absolute_error': mae
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400

    req_data = request.get_json()
    features = req_data['features']  # [age, sex_enc, bmi, children, smoker_enc]

    prediction = model.predict([features])
    return jsonify({'predicted_charge': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
