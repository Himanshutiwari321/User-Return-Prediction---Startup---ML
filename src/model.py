import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Custom Transformer
class IQRCapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns = X.columns
        self.q1_ = X.quantile(0.25)
        self.q3_ = X.quantile(0.75)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - 1.5 * iqr
        self.upper_ = self.q3_ + 1.5 * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.columns:
            X[col] = X[col].clip(self.lower_[col], self.upper_[col])
        return X

# Load Data
df = pd.read_csv('../data/gold/feature_data.csv')
X = df.drop(columns=['Return'])
y = df['Return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = [0, 1, 3, 4]
categorical_cols = [2]

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

# Evaluate with custom threshold
threshold = 0.53
proba = pipe.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save artifacts
artifacts = {
    'model': pipe,
    'threshold': threshold,
    'features': list(X.columns)
}

with open('model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print('Saved: model.pkl')





