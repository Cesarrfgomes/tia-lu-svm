import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import joblib

print("Loading merged data...")
df = pd.read_csv('dataset/merged.csv')

print("Filtering target classes...")
df = df[df['delivery_status'].isin(['DELIVERED', 'CANCELLED'])]
df['target'] = df['delivery_status'].map({'DELIVERED': 0, 'CANCELLED': 1})

drop_cols = ['delivery_id', 'delivery_order_id_x', 'delivery_order_id_y', 'order_id',
             'driver_id', 'payment_order_id', 'delivery_status', 'order_status',
             'hub_name', 'store_name', 'channel_name', 'order_moment_created', 'order_moment_accepted', 'order_moment_ready', 'order_moment_collected', 'order_moment_in_expedition',
             'order_moment_delivering', 'order_moment_delivered', 'order_moment_finished']

df = df.drop(columns=drop_cols, errors='ignore')

X = df.drop('target', axis=1)
y = df['target']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)
feature_names = (numeric_features +
                 list(preprocessor.named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features)))


X_df = pd.DataFrame(X_processed, columns=feature_names)

print("Feature Selection...")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_df, y)
selector = SelectFromModel(rf, prefit=True, threshold='mean')
X_selected = selector.transform(X_df)
selected_features = X_df.columns[selector.get_support()]

processed_data = pd.DataFrame(X_selected, columns=selected_features)
processed_data['target'] = y.values
processed_data.to_csv('dataset/processed_data.csv', index=False)
