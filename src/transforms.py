from sklearn.base import BaseEstimator, TransformerMixin


# Custom Transformer
class IQRCapper(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.columns = X.columns  
        
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
