"""
MUIS Project — PyTorch ANN Regressor with Optuna Tuning (Optimized)
==================================================================
An Artificial Neural Network model for predicting the MUIS score (0–10).
Includes 5-fold cross-validation and hyperparameter tuning via Optuna.

Author : Model Developer
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Import N_FEATURES and other configs if needed
sys.path.insert(0, os.path.join(os.path.dirname(MODEL_DIR), 'data'))
try:
    from feature_config import FEATURE_COLS, N_FEATURES
except ImportError:
    FEATURE_COLS = None
    N_FEATURES = 27

# ================================================================
# 1. DATA LOADING
# ================================================================
def load_training_data():
    print('=' * 60, flush=True)
    print('LOADING TRAINING DATA', flush=True)
    print('=' * 60, flush=True)

    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    X_val   = np.load(os.path.join(MODEL_DIR, 'X_val.npy'))
    X_test  = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))

    y_train = np.load(os.path.join(MODEL_DIR, 'y_train_score.npy'))
    y_val   = np.load(os.path.join(MODEL_DIR, 'y_val_score.npy'))
    y_test  = np.load(os.path.join(MODEL_DIR, 'y_test_score.npy'))

    print(f'  X_train: {X_train.shape}', flush=True)
    print(f'  X_val:   {X_val.shape}', flush=True)
    print(f'  X_test:  {X_test.shape}', flush=True)
    print(f'  y_train: [{y_train.min():.2f}, {y_train.max():.2f}] mean={y_train.mean():.2f}', flush=True)

    return X_train, X_val, X_test, y_train, y_val, y_test

# ================================================================
# 2. ANN DEFINITION
# ================================================================
from model_def import MUISNN

# ================================================================
# 3. EARLY STOPPING HELPER
# ================================================================
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
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
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}", flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

# ================================================================
# 5. CROSS-VALIDATION (OPTIMIZED TENSOR OPERATIONS)
# ================================================================
def run_cross_validation(X_cv, y_cv, hidden_dims, dropout_rate, lr, weight_decay, batch_size, activation_fn, epochs=150, patience=15, verbose=False):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cv_maes = []
    
    X_cv_tensor = torch.FloatTensor(X_cv).to(device)
    y_cv_tensor = torch.FloatTensor(y_cv).to(device)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_cv)):
        X_tr, y_tr = X_cv_tensor[train_idx], y_cv_tensor[train_idx]
        X_va, y_va = X_cv_tensor[val_idx], y_cv_tensor[val_idx]
        
        model = MUISNN(
            input_dim=X_cv.shape[1], 
            hidden_dims=hidden_dims, 
            dropout_rate=dropout_rate, 
            activation_fn=activation_fn
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        early_stopping = EarlyStopping(patience=patience)
        n_samples = X_tr.size(0)
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_samples, device=device)
            X_tr_shuffled = X_tr[perm]
            y_tr_shuffled = y_tr[perm]
            
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
                val_preds = model(X_va)
                val_mae = torch.mean(torch.abs(val_preds - y_va)).item()
                
            early_stopping(val_mae, model)
            if early_stopping.early_stop:
                break
                
        model.load_state_dict(early_stopping.best_state)
        model.eval()
        with torch.no_grad():
            best_val_preds = model(X_va)
            best_val_mae = torch.mean(torch.abs(best_val_preds - y_va)).item()
            
        cv_maes.append(best_val_mae)
        if verbose:
            print(f'    Fold {fold_idx + 1}/5: MAE = {best_val_mae:.4f}', flush=True)
        
    return cv_maes

# ================================================================
# 6. FINAL MODEL TRAINING (OPTIMIZED TENSOR OPERATIONS)
# ================================================================
def train_final_model(X_train, y_train, X_val, y_val, best_params, epochs=250, patience=20):
    n_layers = best_params['n_layers']
    hidden_dims = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
    dropout_rate = best_params['dropout_rate']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    batch_size = best_params['batch_size']
    activation_fn = best_params['activation_fn']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_tr_tensor = torch.FloatTensor(X_train).to(device)
    y_tr_tensor = torch.FloatTensor(y_train).to(device)
    X_va_tensor = torch.FloatTensor(X_val).to(device)
    y_va_tensor = torch.FloatTensor(y_val).to(device)
    
    model = MUISNN(
        input_dim=X_train.shape[1], 
        hidden_dims=hidden_dims, 
        dropout_rate=dropout_rate, 
        activation_fn=activation_fn
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses, val_losses = [], []
    n_samples = X_tr_tensor.size(0)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        X_tr_shuffled = X_tr_tensor[perm]
        y_tr_shuffled = y_tr_tensor[perm]
        
        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            X_batch = X_tr_shuffled[i:i+batch_size]
            y_batch = y_tr_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            
        train_loss = epoch_loss / n_samples
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_tensor)
            val_loss = criterion(val_preds, y_va_tensor).item()
            val_mae = torch.mean(torch.abs(val_preds - y_va_tensor)).item()
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        early_stopping(val_mae, model)
        if early_stopping.early_stop:
            print(f'  [INFO] Early stopping triggered at epoch {epoch}', flush=True)
            break
            
    model.load_state_dict(early_stopping.best_state)
    return model, train_losses, val_losses

# ================================================================
# 7. MAIN PIPELINE
# ================================================================
def main():
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data()
    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)
    
    print('\n' + '=' * 60, flush=True)
    print('RUNNING HYPERPARAMETER TUNING (OPTUNA)', flush=True)
    print('=' * 60, flush=True)
    
    trial_counter = [0]
    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f'n_units_l{i}', 16, 128, step=16))
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'leaky_relu', 'elu'])
        
        cv_maes = run_cross_validation(X_cv, y_cv, hidden_dims, dropout_rate, lr, weight_decay, batch_size, activation_fn)
        mean_mae = np.mean(cv_maes)
        trial_counter[0] += 1
        print(f"  Trial {trial_counter[0]:02d}/30: CV MAE = {mean_mae:.4f} | n_layers={n_layers}, lr={lr:.5f}, wd={weight_decay:.5f}, batch={batch_size}", flush=True)
        return mean_mae
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    print("\nOptuna Optimization Complete:", flush=True)
    print(f"  Best trial CV MAE: {study.best_value:.4f}", flush=True)
    print("  Best hyperparameters:", flush=True)
    for k, v in study.best_params.items():
        print(f"    {k}: {v}", flush=True)
        
    # Get best params
    best_params = study.best_params
    n_layers = best_params['n_layers']
    best_hidden_dims = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
    
    # Train final model on X_train using X_val for early stopping
    print('\n' + '=' * 60, flush=True)
    print('TRAINING FINAL MODEL WITH BEST PARAMETERS', flush=True)
    print('=' * 60, flush=True)
    model, train_losses, val_losses = train_final_model(X_train, y_train, X_val, y_val, best_params)
    
    # Save model weights
    weights_path = os.path.join(MODEL_DIR, 'ann_model.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"  [OK] PyTorch weights saved to: {weights_path}", flush=True)
    
    # Evaluate on Test Set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
        
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2 = float(r2_score(y_test, y_pred))
    
    # Run final 5-fold CV using best params to obtain cv_mae_mean and cv_mae_std
    print('\n' + '=' * 60, flush=True)
    print('RUNNING FINAL 5-FOLD CV ON COMBINED SET', flush=True)
    print('=' * 60, flush=True)
    final_cv_maes = run_cross_validation(
        X_cv, y_cv, 
        best_hidden_dims, 
        best_params['dropout_rate'], 
        best_params['lr'], 
        best_params['weight_decay'], 
        best_params['batch_size'], 
        best_params['activation_fn'],
        verbose=True
    )
    cv_mae_mean = float(np.mean(final_cv_maes))
    cv_mae_std = float(np.std(final_cv_maes))
    print(f"\n  Final 5-Fold CV MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f}", flush=True)
    
    # Save metrics in training_history.json
    history = {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
        "best_params": best_params
    }
    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  [OK] Training history saved to: {history_path}", flush=True)
    
    # Generate Plots
    print('\n' + '=' * 60, flush=True)
    print('GENERATING FIGURES', flush=True)
    print('=' * 60, flush=True)
    
    # 1. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, c='#2196F3', edgecolors='none')
    lim_lo = min(y_test.min(), y_pred.min()) - 0.5
    lim_hi = max(y_test.max(), y_pred.max()) + 0.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', linewidth=1.5, label='Perfect prediction (y = x)')
    ax.set_xlabel('Actual MUIS Score')
    ax.set_ylabel('Predicted MUIS Score')
    ax.set_title('PyTorch ANN: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    pred_vs_act_path = os.path.join(MODEL_DIR, 'ann_predicted_vs_actual.png')
    plt.savefig(pred_vs_act_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved predicted vs actual plot: {pred_vs_act_path}", flush=True)
    
    # 2. Loss Curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Training Loss (MSE)', color='#4CAF50')
    ax.plot(val_losses, label='Validation Loss (MSE)', color='#FF9800')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('PyTorch ANN: Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(MODEL_DIR, 'ann_loss_curves.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved loss curves plot: {loss_path}", flush=True)
    
    print('\n' + '=' * 60, flush=True)
    print('ANN TRAINING COMPLETE', flush=True)
    print('=' * 60, flush=True)
    print(f"  Test MAE:  {test_mae:.4f}", flush=True)
    print(f"  Test RMSE: {test_rmse:.4f}", flush=True)
    print(f"  Test R2:   {test_r2:.4f}", flush=True)
    print(f"  CV MAE:    {cv_mae_mean:.4f} ± {cv_mae_std:.4f}", flush=True)

if __name__ == '__main__':
    main()
