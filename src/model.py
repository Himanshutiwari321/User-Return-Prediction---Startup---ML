import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.transforms import IQRCapper

# Load Data
df = pd.read_csv('data/gold/feature_data.csv')
X = df.drop(columns=['Return'])
y = df['Return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = ['Page Views', 'Session Duration', 'Time on Page', 'Previous Visits']
categorical_cols = ['Traffic Source']

# Sequential numeric preprocessing
numeric_pipe = Pipeline(steps=[
    ('cap', IQRCapper()),
    ('log', FunctionTransformer(np.log1p, validate=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipe, numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train
pipe.fit(X_train, y_train)

# # Evaluate with custom threshold
# threshold = 0.53
# proba = pipe.predict_proba(X_test)[:, 1]
# y_pred = (proba >= threshold).astype(int)

# print('Accuracy:', accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Save artifacts
artifacts = {
    'model': pipe,
    # 'threshold': threshold,
    'features': list(X.columns)
}

with open('return_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print('Saved: return_model.pkl')





