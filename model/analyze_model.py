import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
from feature_config import FEATURE_COLS, N_FEATURES, MODEL_DIR, FEATURES_LABELLED_CSV

def main():
    print("Loading data...")
    df = pd.read_csv(FEATURES_LABELLED_CSV)
    
    import joblib
    scalers = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    
    X_raw = df[FEATURE_COLS].values
    y = df['muis_score'].values
    
    X = np.zeros_like(X_raw)
    for i, col in enumerate(FEATURE_COLS):
        strategy, scaler = scalers[col]
        values = X_raw[:, i].reshape(-1, 1)
        if strategy == 'log_robust':
            values = np.log1p(values)
            X[:, i] = scaler.transform(values).ravel()
        elif strategy == 'robust':
            X[:, i] = scaler.transform(values).ravel()
        else:
            X[:, i] = values.ravel()
            
    print("Loading XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, 'xgb_model.json'))
    
    print("Computing predictions...")
    y_pred = model.predict(X)
    
    # 1. Performance Metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print("Computing VIF...")
    # 2. VIF (Multicollinearity)
    # Add a constant for VIF calculation
    X_df = pd.DataFrame(X, columns=FEATURE_COLS)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    # calculating VIF for each feature
    # Handle perfect collinearity or division by zero by catching exceptions
    vifs = []
    for i in range(len(X_df.columns)):
        try:
            vif = variance_inflation_factor(X_df.values, i)
        except Exception:
            vif = float('inf')
        vifs.append(vif)
    vif_data["VIF"] = vifs
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    print("Computing SHAP values...")
    # 3. SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)
    
    # Mean absolute SHAP values for global feature importance
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False)
    
    # Residuals
    residuals = y - y_pred
    
    # Save results
    results = {
        "metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2)
        },
        "vif": vif_data.to_dict(orient="records"),
        "shap_importance": shap_importance.to_dict(orient="records"),
        "residuals": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "max": float(np.max(residuals)),
            "min": float(np.min(residuals))
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'model_analysis.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Analysis complete. Saved to model_analysis.json")

if __name__ == "__main__":
    main()
