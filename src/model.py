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
        self.columns = X.columns  # column names save karo
        
        self.Q1 = X.quantile(0.25)
        self.Q3 = X.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        
        self.lower_bound = self.Q1 - 1.5 * self.IQR
        self.upper_bound = self.Q3 + 1.5 * self.IQR
        
        return self
    
    def transform(self, X):
        X_capped = X.copy()
        
        for col in self.columns:
            X_capped[col] = X_capped[col].clip(
                lower=self.lower_bound[col],
                upper=self.upper_bound[col]
            )
        
        return X_capped

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

with open('return_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print('Saved: return_model.pkl')





