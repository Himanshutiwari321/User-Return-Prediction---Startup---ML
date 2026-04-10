import seaborn as sns
import numpy as  np 
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier


#load data
df = pd.read_csv("../data/gold/feature_data.csv")

#split the data 
X = df.drop("Return", axis=1)
y = df["Return"]

#train/test/split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,random_state=42)

#handle outlier data using capping technique
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
    
#column type indentification 
numaric_cols = [0,1,3,4]  
categorical_cols = [2]

# Preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('iqr', IQRCapper(), numaric_cols),
        ('log_transformer', FunctionTransformer(np.log1p),numaric_cols),
        ('ohe_Traffic_Source', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols )
    ]
)

#Final Pipeline (Preprocess + Model)
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])