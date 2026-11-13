
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


# ================= LOAD DATASET =================

df = pd.read_csv('indian_urban_house_price_practice_100k.csv')


# ================= EDA =================

print("--------- Top 5 Observations -------")
print(df.head())

print("--------- Bottom 5 Observations -------")
print(df.tail())

print("------------ Detailed Info of Dataset --------------")
df.info()

print("-------- Statistical Information --------")
print(df.describe())

print("------- Column Names --------")
print(df.columns)

print("-------- Check Null Values ---------")
print(df.isnull().sum())

print("------------- Check Duplicates ----------")
print(df.duplicated().sum())

print("---------- Check Elements in Each Attribute ----------")
for i in df.columns.to_list():
    print(i)
    print("-" * 50)
    print(df[i].unique())

print("----------- Drop Property ID ----------")
df.drop('Property_ID', axis=1, inplace=True)

print("Completed EDA")


# ================= SEPARATE X AND Y =================

print("------------- Separate Input and Output --------------")

X = df.drop('Price_INR_Lakhs', axis=1)
y = df['Price_INR_Lakhs']


# ================= NUMERICAL AND CATEGORICAL COLUMNS =================

num_cols = X.select_dtypes(include='number').columns.to_list()
print("Numerical Columns:", num_cols)

cat_cols = X.select_dtypes(include='object').columns.to_list()
print("Categorical Columns:", cat_cols)


# ================= OUTLIER DETECTION =================

print("--------------- Outlier Detection ------------------")

for col in num_cols:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((X[col] < lower) | (X[col] > upper)).sum()

    print(f'{col}: {outliers} outliers')


# ================= SKEWNESS HANDLING =================

oc = [
    'Distance_to_Metro_km',
    'Distance_to_City_Center_km',
    'Hospital_Distance_km',
    'School_Distance_km',
    'Shopping_Mall_Distance_km',
    'Nearby_Metro',
    'Price_Per_SqFt_INR'
]

print("### Before Transformation")

for i in oc:
    sns.histplot(data=X, x=i, kde=True)
    plt.show()


print("### Applying Log Transformation")

X[oc] = np.log1p(X[oc])

print(X[oc].skew())


print("### After Transformation")

for i in oc:
    sns.histplot(data=X, x=i, kde=True)
    plt.show()


print("-------- Outlier Distribution After Handling --------")

for col in oc:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((X[col] < lower) | (X[col] > upper)).sum()

    print(f'{col}: {outliers} outliers')


# ================= TARGET TRANSFORMATION =================

print("--------- Target Skewness ---------")
print(y.skew())

print("Applying Log Transformation to Target")

y = np.log1p(y)


# ================= RELATIONSHIP BETWEEN X AND Y =================

print("Checking Relationship Between Input and Target")

for i in X.columns:

    plt.figure(figsize=(6, 4))
    plt.scatter(X[i], y)
    plt.xlabel(i)
    plt.ylabel('Log Price')
    plt.title(f'{i} vs Price')
    plt.show()


# ================= PREPROCESSING =================

print("---------------- Preprocessing Steps ----------------")


# Numerical Pipeline

p_num = Pipeline([
    ('si', SimpleImputer(strategy='median')),
    ('ss', RobustScaler())
])


# Ordinal and Nominal Columns

ord_cols = ['Locality_Type', 'Furnishing_Status']

one_hot_cols = [
    'City',
    'Property_Type',
    'Water_Supply',
    'Power_Backup',
    'Facing',
    'Ownership_Type',
    'Construction_Status',
    'Transaction_Type'
]


# Ordinal Pipeline

p_ord = Pipeline([
    ('si', SimpleImputer(strategy='most_frequent')),
    ('oe', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])


# Nominal Pipeline

p_nom = Pipeline([
    ('si', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(
        handle_unknown='ignore'
    ))
])


# Column Transformer

preprocessor = ColumnTransformer([
    ('numeric', p_num, num_cols),
    ('nominal', p_nom, one_hot_cols),
    ('ordinal', p_ord, ord_cols)
])

print("Preprocessor Pipeline:")
print(preprocessor)


# ================= TRAIN TEST SPLIT =================

xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ================= APPLY MODELS =================

results = {}

models = {
    'Linear Regression': LinearRegression(),
    'KNN':KNeighborsRegressor(),
    'Lasso': Lasso(alpha=0.05),
    'Ridge': Ridge(alpha=0.05),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}


for name, model in models.items():

    pipe = Pipeline([
        ('preprocessor', preprocessor),

        ('feature_selection', SelectKBest(
            score_func=mutual_info_regression,
            k=15
        )),

        ('regressor', model)
    ])

    pipe.fit(xtrain, ytrain)

    ypred = pipe.predict(xtest)

    # Inverse Transformation

    ypred_actual = np.expm1(ypred)
    ytest_actual = np.expm1(ytest)

    # Evaluation

    r2 = r2_score(ytest_actual, ypred_actual)

    rmse = root_mean_squared_error(
        ytest_actual,
        ypred_actual
    )

    results[name] = [r2, rmse]

    print("\n", name)
    print("R2 Score:", r2)
    print("RMSE:", rmse)
    print("-------------------")


# ================= FINAL MODEL COMPARISON =================

results_df = pd.DataFrame(
    results,
    index=['R2 Score', 'RMSE']
).T

print("\n========== MODEL COMPARISON ==========")

print(
    results_df.sort_values(
        by='R2 Score',
        ascending=False
    )
)

print("\nCompleted Base Model Training")


##Hyperparamter Tuning

from sklearn.model_selection import KFold, RandomizedSearchCV


# Decision Tree Pipeline

pipe = Pipeline([
    ('preprocessor', preprocessor),

    ('feature_selection', SelectKBest(
        score_func=mutual_info_regression,
        k=15
    )),

    ('regressor', DecisionTreeRegressor(random_state=42))
])


# Hyperparameters

params = {
    'regressor__max_depth': [None, 10, 20, 30, 40],
    'regressor__min_samples_split': [2, 5, 10, 15],
    'regressor__min_samples_leaf': [1, 2, 4, 6],
    'regressor__max_features': [None, 'sqrt', 0.7]
}


# KFold Cross Validation

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Randomized Search CV

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=params,
    n_iter=10,
    scoring='r2',
    cv=kfold,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# Training

random_search.fit(xtrain, ytrain)


# Best Parameters

print("Best Parameters:")
print(random_search.best_params_)

print("Best CV RMSE (Log Scale):")
print(-random_search.best_score_)


# Best Model

best_model = random_search.best_estimator_


# Prediction

ypred = best_model.predict(xtest)


# Inverse Log Transformation

ypred_actual = np.expm1(ypred)
ytest_actual = np.expm1(ytest)


# Evaluation

r2 = r2_score(ytest_actual, ypred_actual)

rmse = root_mean_squared_error(
    ytest_actual,
    ypred_actual
)

print("\nDecision Tree Results")
print("R2 Score:", r2)
print("RMSE:", rmse)


joblib.dump(best_model, 'decision_tree_house_price.pkl')

print("\nDecision Tree Model Saved Successfully!")

### ENSEMLE ALGORIHTMS

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)


results = {}

models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42)
}

for name, model in models.items():

    pipe = Pipeline([
        ('preprocessor', preprocessor),

        ('feature_selection', SelectKBest(
            score_func=mutual_info_regression,
            k=15
        )),

        ('regressor', model)
    ])

    pipe.fit(xtrain, ytrain)

    ypred = pipe.predict(xtest)

    ypred_actual = np.expm1(ypred)
    ytest_actual = np.expm1(ytest)

    r2 = r2_score(ytest_actual, ypred_actual)

    rmse = root_mean_squared_error(
        ytest_actual, ypred_actual
    )

    results[name] = [r2, rmse]

    print("\n", name)
    print("R2 Score:", r2)
    print("RMSE:", rmse)
    print("-------------------")

# Model Comparison

results_df = pd.DataFrame(
    results,
    index=['R2 Score', 'RMSE']
).T

print("\nEnsemble Model Comparison")

print(results_df.sort_values(
    by='R2 Score',
    ascending=False
))



