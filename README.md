# Indian Urban House Price Prediction

A machine-learning project that estimates residential property prices in Indian cities using property details, location, size, amenities, and other features.

The project includes:
- **Model training:** EDA, preprocessing, feature selection, model comparison, and Decision Tree hyperparameter tuning.
- **Backend:** FastAPI prediction endpoint.
- **Frontend:** Streamlit interface for entering property details and viewing the estimated price.
- **Dataset:** `indian_urban_house_price_practice_100k.csv` (100,000 property records).

## Project structure

```text
NEW/
├── train.py
├── fastapi_app.py
├── streamlit_app.py
├── indian_urban_house_price_practice_100k.csv
├── pickle.pkl
├── requirements.txt
└── README.md
```

## Tech stack

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- FastAPI and Uvicorn
- Streamlit
- Requests

## Setup

Use Python 3.10 or later.

1. Clone the repository and move into the project folder:

   ```bash
   git clone <your-repository-url>
   cd NEW
   ```

2. (Recommended) Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv venv
   venv\\Scripts\\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the application

The FastAPI backend and Streamlit frontend run in separate terminals. Keep both running.

### 1. Start the FastAPI backend

From the project folder:

```bash
uvicorn fastapi_app:app --reload
```

The API will be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### 2. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, move into the project folder, and run:

```bash
streamlit run streamlit_app.py
```

Streamlit will print the local URL (usually http://localhost:8501). Open it in your browser, enter the property details, and select **Predict House Price**.

## Train the model

Make sure the CSV dataset is in the project folder, then run:

```bash
python train.py
```

The training script performs exploratory data analysis, transforms selected skewed features and the target, builds preprocessing pipelines, compares regression models, and tunes a Decision Tree Regressor using 5-fold K-Fold cross-validation and `RandomizedSearchCV`.

**Model filename note:** `fastapi_app.py` currently loads `pickle.pkl`. The training script saves its tuned Decision Tree pipeline as `decision_tree_house_price.pkl`. If you want the API to use the newly trained model, either change the `joblib.load("pickle.pkl")` line in `fastapi_app.py` to `joblib.load("decision_tree_house_price.pkl")`, or rename/copy the newly generated model to `pickle.pkl`. Restart FastAPI after changing the model.

## API endpoint

`POST /predict`

The endpoint accepts the property fields used by the Streamlit form and returns:
- `Property_ID`
- `predicted_price_in_lakhs`
- `predicted_price_in_INR`

The API expects the saved model to accept the input features after `Property_ID` is removed. The price prediction is converted back from the log-transformed target using `expm1`.

## Notes

- Run commands from the directory containing the Python files and model file.
- Keep `pickle.pkl` in the same directory as `fastapi_app.py`, unless you update the model path.
- The Streamlit app calls the API at `http://127.0.0.1:8000/predict`. If you change the backend address or port, update the URL in `streamlit_app.py`.
- The estimated price is a model prediction based on the supplied dataset and features; it is not a formal property valuation.
- Only load pickle/joblib model files from trusted sources.

## Future improvements

- Add input validation and clearer API error responses.
- Add automated tests for the API and prediction pipeline.
- Deploy the FastAPI backend and Streamlit frontend.
