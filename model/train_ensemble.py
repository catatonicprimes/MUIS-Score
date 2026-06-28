"""
MUIS Project — Ensemble Training and Tuning Script
==================================================
Trains a weighted ensemble of PyTorch ANN and XGBoost Regressor.
Hyperparameters are tuned using Optuna, and validation MAE is minimized.

Author : Teamwork Agent
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)
from model_def import MUISNN

# ================================================================
# EARLY STOPPING HELPER
# ================================================================
class EarlyStopping:
    def __init__(self, patience=20, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

# ================================================================
# DATA LOADING
# ================================================================
def load_data():
    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    X_val   = np.load(os.path.join(MODEL_DIR, 'X_val.npy'))
    X_test  = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(MODEL_DIR, 'y_train_score.npy'))
    y_val   = np.load(os.path.join(MODEL_DIR, 'y_val_score.npy'))
    y_test  = np.load(os.path.join(MODEL_DIR, 'y_test_score.npy'))
    return X_train, X_val, X_test, y_train, y_val, y_test

# ================================================================
# PYTORCH ANN TUNING & TRAINING
# ================================================================
def run_ann_tuning(X_train, y_train, X_val, y_val, n_trials, device):
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f'n_units_l{i}', 32, 128))
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'leaky_relu', 'elu'])
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        
        model = MUISNN(
            input_dim=X_train.shape[1],
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=20)
        
        n_samples = X_train_tensor.size(0)
        for epoch in range(150):
            model.train()
            perm = torch.randperm(n_samples, device=device)
            X_tr_shuffled = X_train_tensor[perm]
            y_tr_shuffled = y_train_tensor[perm]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_tr_shuffled[i:i+batch_size]
                y_batch = y_tr_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_mae = torch.mean(torch.abs(val_preds - y_val_tensor)).item()
                
            early_stopping(val_mae, model)
            if early_stopping.early_stop:
                break
                
        return early_stopping.best_loss

    print("Tuning PyTorch ANN hyperparameters...", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Best ANN CV MAE: {study.best_value:.4f}", flush=True)
    return study.best_params

def train_final_ann(best_params, X_train, y_train, X_val, y_val, device):
    n_layers = best_params['n_layers']
    hidden_dims = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
    dropout_rate = best_params['dropout_rate']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    batch_size = best_params['batch_size']
    activation_fn = best_params['activation_fn']
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    model = MUISNN(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation_fn=activation_fn
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=30)
    
    n_samples = X_train_tensor.size(0)
    for epoch in range(250):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        X_tr_shuffled = X_train_tensor[perm]
        y_tr_shuffled = y_train_tensor[perm]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_tr_shuffled[i:i+batch_size]
            y_batch = y_tr_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_mae = torch.mean(torch.abs(val_preds - y_val_tensor)).item()
            
        early_stopping(val_mae, model)
        if early_stopping.early_stop:
            break
            
    model.load_state_dict(early_stopping.best_state)
    return model

# ================================================================
# XGBOOST TUNING & TRAINING
# ================================================================
def run_xgb_tuning(X_train, y_train, X_val, y_val, n_trials):
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 3, 8)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 5.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True)
        
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='reg:squarederror',
            eval_metric='mae',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        return val_mae

    print("Tuning XGBoost hyperparameters...", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Best XGBoost CV MAE: {study.best_value:.4f}", flush=True)
    return study.best_params

def train_final_xgb(best_params_xgb, X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        max_depth=best_params_xgb['max_depth'],
        n_estimators=best_params_xgb['n_estimators'],
        learning_rate=best_params_xgb['learning_rate'],
        subsample=best_params_xgb['subsample'],
        colsample_bytree=best_params_xgb['colsample_bytree'],
        reg_alpha=best_params_xgb['reg_alpha'],
        reg_lambda=best_params_xgb['reg_lambda'],
        objective='reg:squarederror',
        eval_metric='mae',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

# ================================================================
# MAIN TRAINING PIPELINE
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Ensemble training and tuning")
    parser.add_argument('--trials', type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    # Load arrays
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Optuna tuning
    best_params_ann = run_ann_tuning(X_train, y_train, X_val, y_val, args.trials, device)
    best_params_xgb = run_xgb_tuning(X_train, y_train, X_val, y_val, args.trials)
    
    # Train final best models
    print("\nTraining final ANN model with best parameters...", flush=True)
    best_ann_model = train_final_ann(best_params_ann, X_train, y_train, X_val, y_val, device)
    
    print("Training final XGBoost model with best parameters...", flush=True)
    best_xgb_model = train_final_xgb(best_params_xgb, X_train, y_train, X_val, y_val)
    
    # Save model weights
    ann_path = os.path.join(MODEL_DIR, 'ann_model.pth')
    torch.save(best_ann_model.state_dict(), ann_path)
    print(f"Saved ANN model weights to: {ann_path}", flush=True)
    
    xgb_path = os.path.join(MODEL_DIR, 'xgb_model.json')
    best_xgb_model.save_model(xgb_path)
    print(f"Saved XGBoost model to: {xgb_path}", flush=True)
    
    # Predict on validation set
    best_ann_model.eval()
    with torch.no_grad():
        val_preds_ann = best_ann_model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
        test_preds_ann = best_ann_model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
        
    val_preds_xgb = best_xgb_model.predict(X_val)
    test_preds_xgb = best_xgb_model.predict(X_test)
    
    # Find optimal weight w_ann in [0.0, 1.0] minimizing MAE on validation set
    best_w = 0.5
    best_val_mae = float('inf')
    for w in np.linspace(0.0, 1.0, 101):
        val_preds_ensemble = w * val_preds_ann + (1 - w) * val_preds_xgb
        val_mae = mean_absolute_error(y_val, val_preds_ensemble)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_w = w
            
    best_w = round(float(best_w), 2)
    print(f"\nOptimal ensemble weight w_ann: {best_w:.2f} (Val MAE: {best_val_mae:.4f})", flush=True)
    
    # Predict on held-out test set
    y_pred_test = best_w * test_preds_ann + (1 - best_w) * test_preds_xgb
    
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_r2 = float(r2_score(y_test, y_pred_test))
    
    print("\nHeld-out Test Set Performance:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R2:   {test_r2:.4f}")
    
    if test_mae < 1.79:
        print(f"\n[SUCCESS] Test MAE {test_mae:.4f} improves on the 1.79 baseline!", flush=True)
    else:
        print(f"\n[WARNING] Test MAE {test_mae:.4f} did not improve on the 1.79 baseline.", flush=True)
        
    # Save metrics and hyperparameters to model/training_history.json
    history = {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "ensemble_weight_ann": best_w,
        "best_params": best_params_ann,
        "best_params_xgb": best_params_xgb
    }
    
    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}", flush=True)
    
    # Plot predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual Score')
    plt.ylabel('Ensemble Predicted Score')
    plt.title(f'Ensemble Predicted vs Actual MUIS Score (MAE = {test_mae:.4f})')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(MODEL_DIR, 'ensemble_predicted_vs_actual.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved predicted vs actual plot to: {plot_path}", flush=True)

if __name__ == '__main__':
    main()
