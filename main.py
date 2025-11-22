import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Loading datasets
base_path = "./dataset/"

channels = pd.read_csv(base_path + "channels.csv")
deliveries = pd.read_csv(base_path + "deliveries.csv")
drivers = pd.read_csv(base_path + "drivers.csv")
hubs = pd.read_csv(base_path + "hubs.csv")
orders = pd.read_csv(base_path + "orders.csv")
payments = pd.read_csv(base_path + "payments.csv")
stores = pd.read_csv(base_path + "stores.csv")


# Merge

df = deliveries.merge(orders, on="order_id", how="left")
df = df.merge(channels, on="channel_id", how="left")
df = df.merge(drivers, on="driver_id", how="left")
df = df.merge(hubs, on="hub_id", how="left")
df = df.merge(stores, on="store_id", how="left")
df = df.merge(payments, on="payment_id", how="left")

# Initial exploration
print("Formato final do dataset:", df.shape)
print("Colunas:", df.columns.tolist())
print(df.head())

# Target
df['target'] = df['status'].map({
    'delivered': 1,
    'cancelled': 0
})

df = df.dropna(subset=['target'])

# Handling missing values

df = df.fillna({
    col: df[col].median() for col in df.select_dtypes(include='number').columns
})

df = df.fillna({
    col: "missing" for col in df.select_dtypes(include='object').columns
})

#X/Y
X = df.drop(columns=['target'])
y = df['target']

# Drop irrelevant columns
cols_to_drop = [
    'order_id', 'channel_id', 'driver_id', 'hub_id',
    'store_id', 'payment_id', 'status'
]

X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipelines

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
# Create pipeline
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', SVC(probability=True))
])

# Train-test split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define parameter grid for SVM
param_grid = [
    {
        'clf__kernel': ['linear'],
        'clf__C': [0.1, 1, 10],
        'clf__class_weight': [None, 'balanced']
    },
    {
        'clf__kernel': ['rbf'],
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 0.01, 0.1],
        'clf__class_weight': [None, 'balanced']
    }
]


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_valid)
print(classification_report(y_valid, y_pred))
print(confusion_matrix(y_valid, y_pred))
f1 = f1_score(y_valid, y_pred)
print(f"F1 Score: {f1}")
print(f"Best Hyperparameters: {grid.best_params_}")

# Save the best model
import joblib
joblib.dump(best_model, "modelo_svm.pkl")
