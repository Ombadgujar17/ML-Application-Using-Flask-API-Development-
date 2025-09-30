# Health Insurance Charges Prediction API

This project is a Flask-based REST API for training, testing, and predicting health insurance charges using a linear regression model.

## Features
- **Train**: Upload a CSV file to train the model.
- **Test**: Upload a CSV file to evaluate the model (R², MSE, MAE).
- **Predict**: Send a JSON payload to get a charge prediction for a single input.

## Endpoints

### `/train` (POST)
- Upload a CSV file with columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.
- Trains the linear regression model.
- **How to use in Postman:**
  - Set method to POST and URL to `http://localhost:5000/train`
  - In the Body tab, select `form-data`
  - Add a key named `file` of type `File` and upload your CSV file

### `/test` (POST)
- Upload a CSV file with the same columns as above.
- Returns model metrics: R² score, mean squared error, mean absolute error.
- **How to use in Postman:**
  - Set method to POST and URL to `http://localhost:5000/test`
  - In the Body tab, select `form-data`
  - Add a key named `file` of type `File` and upload your CSV file

### `/predict` (POST)
- Send a JSON payload with `features`: `[age, sex_enc, bmi, children, smoker_enc]`
  - `sex_enc`: 0 for female, 1 for male
  - `smoker_enc`: 0 for no, 1 for yes
- Returns predicted charge.
- **How to use in Postman:**
  - Set method to POST and URL to `http://localhost:5000/predict`
  - In the Body tab, select `raw` and choose `JSON` format
  - Example JSON body:
    ```json
    {
      "features": [34, 1, 25, 2, 0]
    }
    ```
  - Expected Output in JSON:
    ```json
    {
    "predicted_charge": 5589.815344897121
    }
    ```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API:
   ```bash
   python ml_api.py
   ```

## Notes
- The model must be trained before using `/test` or `/predict`.
- Example data file: `Health_insurance.csv` (included).
