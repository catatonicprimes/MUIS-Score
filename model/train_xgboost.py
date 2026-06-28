"""
MUIS Project — XGBoost Baseline Regressor
==========================================
A gradient-boosted tree model for predicting the MUIS score (0–10).

WHY INCLUDE A TREE-BASED BASELINE?
-----------------------------------
1. For tabular data with < 10,000 samples, gradient-boosted trees
   (XGBoost, LightGBM) typically OUTPERFORM neural networks.  This is
   well-documented in Grinsztajn et al. (2022), "Why do tree-based
   models still outperform deep learning on tabular data?"

2. Trees don't need feature normalisation — they split on raw values.
   This means we can skip the log_robust / robust / passthrough logic
   and still get competitive results.  (We DO use the normalised data
   here for a fair comparison with the ANN, but trees would be fine
   with raw data too.)

3. XGBoost provides built-in feature importance, which is invaluable
   for urban planning interpretation.  Knowing that F02_land_use_entropy
   is the #1 predictor validates the Jacobs (1961) diversity thesis.

4. If XGBoost significantly beats the ANN, it tells us that the
   relationships in the data are mostly additive/interaction-based
   (which trees capture naturally) rather than deeply compositional
   (which deep networks capture).

COMPARISON METHODOLOGY:
-----------------------
Both models are trained on the same normalised X_train / X_val / X_test
splits produced by prepare_training_data.py.  Both are evaluated with
the same metrics (MAE, RMSE, R²) on the same held-out test set.

Author : Swaastak
"""

# ----------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------
import numpy as np
import os
import sys
import json

# Non-interactive matplotlib backend (must be before pyplot import)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# -- XGBoost with graceful fallback ----------------------------
try:
    import xgboost as xgb
except ImportError:
    print('+======================================================+')
    print('|  XGBoost not installed.                             |')
    print('|  Run:  pip install xgboost                          |')
    print('+======================================================+')
    sys.exit(1)

# -- Path setup ------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Import FEATURE_COLS from feature_config (one directory up, then data/)
sys.path.insert(0, os.path.join(os.path.dirname(MODEL_DIR), 'data'))
from feature_config import FEATURE_COLS, N_FEATURES


# ================================================================
# 1. DATA LOADING
# ================================================================
def load_training_data():
    """
    Load the .npy arrays produced by prepare_training_data.py.

    We load the SAME normalised data used by the ANN so the comparison
    is fair.  In production, XGBoost could use raw features (trees are
    invariant to monotonic transformations), but for apples-to-apples
    evaluation we use identical inputs.
    """
    print('=' * 60)
    print('LOADING TRAINING DATA')
    print('=' * 60)

    X_train = np.load(os.path.join(MODEL_DIR, 'X_train.npy'))
    X_val   = np.load(os.path.join(MODEL_DIR, 'X_val.npy'))
    X_test  = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))

    y_train = np.load(os.path.join(MODEL_DIR, 'y_train_score.npy'))
    y_val   = np.load(os.path.join(MODEL_DIR, 'y_val_score.npy'))
    y_test  = np.load(os.path.join(MODEL_DIR, 'y_test_score.npy'))

    # Class labels for per-class evaluation
    y_test_class = np.load(os.path.join(MODEL_DIR, 'y_test_class.npy'),
                           allow_pickle=True)

    print(f'  X_train: {X_train.shape}')
    print(f'  X_val:   {X_val.shape}')
    print(f'  X_test:  {X_test.shape}')
    print(f'  y_train: [{y_train.min():.2f}, {y_train.max():.2f}]  '
          f'mean={y_train.mean():.2f}')

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_class


# ================================================================
# 2. MODEL TRAINING
# ================================================================
def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an XGBRegressor with carefully chosen hyperparameters.

    Hyperparameter Rationale:
    -------------------------
    * n_estimators=500:
      Maximum number of boosting rounds.  Early stopping will cut this
      short.  500 is generous enough for convergence on datasets up to
      ~10k samples.

    * max_depth=6:
      Maximum tree depth.  Deeper trees can model more complex interactions
      but overfit faster.  Depth 6 allows up to 6-way feature interactions,
      which is sufficient for 28 features.  (Default is 6; we keep it.)

    * learning_rate=0.05:
      Shrinkage factor per tree.  Lower values (0.01–0.1) require more
      trees but generalise better.  0.05 is a good balance between
      training speed and generalisation.

    * subsample=0.8:
      Row sampling per tree.  Using 80% of rows adds stochasticity,
      which acts as regularisation (similar to Dropout in ANNs).

    * colsample_bytree=0.8:
      Column (feature) sampling per tree.  Prevents the model from
      relying too heavily on any single feature.  Analogous to L2
      regularisation in neural networks.

    * reg_alpha=0.1 (L1):
      L1 regularisation on leaf weights.  Promotes sparsity — some
      leaves get weight 0, effectively pruning them.

    * reg_lambda=1.0 (L2):
      L2 regularisation on leaf weights.  Shrinks large weights toward
      zero, preventing any single tree from overfitting.

    * early_stopping_rounds=30:
      Stops if validation metric hasn't improved for 30 rounds.
      This is the XGBoost equivalent of Keras EarlyStopping.

    Returns
    -------
    model : xgb.XGBRegressor
        Trained XGBoost model.
    """
    print('\n' + '=' * 60)
    print('TRAINING XGBOOST REGRESSOR')
    print('=' * 60)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,           # L1 regularisation
        reg_lambda=5.0,          # L2 regularisation
        objective='reg:squarederror',
        eval_metric='mae',
        random_state=42,
        n_jobs=-1,               # Use all CPU cores
        verbosity=1,
        early_stopping_rounds=30,
    )

    print('  Hyperparameters:')
    print(f'    n_estimators:      300')
    print(f'    max_depth:         4')
    print(f'    learning_rate:     0.05')
    print(f'    subsample:         0.8')
    print(f'    colsample_bytree:  0.8')
    print(f'    reg_alpha (L1):    0.5')
    print(f'    reg_lambda (L2):   5.0')
    print()

    # -- Training with validation-based early stopping -------------
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    best_iteration = getattr(model, 'best_iteration', model.n_estimators)
    print(f'\n  Best iteration: {best_iteration}')
    best_score = getattr(model, 'best_score', 0.0)
    print(f'  Best val MAE:   {best_score:.4f}')

    return model


# ================================================================
# 3. EVALUATION
# ================================================================
def evaluate_model(model, X_test, y_test, y_test_class):
    """
    Evaluate the XGBoost model on the held-out test set using the same
    metrics as the ANN (MAE, RMSE, R²) for direct comparison.
    """
    print('\n' + '=' * 60)
    print('TEST SET EVALUATION')
    print('=' * 60)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f'  MAE:  {mae:.4f}  (avg error in MUIS points)')
    print(f'  RMSE: {rmse:.4f}  (penalises large errors)')
    print(f'  R²:   {r2:.4f}  (variance explained)')

    # -- Per-class MAE ---------------------------------------------
    print('\n  Per-class MAE:')
    for cls in ['High', 'Medium', 'Low']:
        mask = y_test_class == cls
        if mask.sum() > 0:
            cls_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            print(f'    {cls:7s}: MAE = {cls_mae:.4f}  (n = {mask.sum()})')
        else:
            print(f'    {cls:7s}: no samples')

    return y_pred, mae, rmse, r2


# ================================================================
# 4. CROSS-VALIDATION
# ================================================================
def run_cross_validation(X_train, y_train, X_val, y_val):
    """
    5-fold cross-validation on combined train+val data.

    Mirrors the ANN's CV procedure for a fair comparison.
    XGBoost is much faster to train, so this completes quickly.
    """
    print('\n' + '=' * 60)
    print('5-FOLD CROSS-VALIDATION')
    print('=' * 60)

    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)
    print(f'  CV dataset: {X_cv.shape[0]} samples, {X_cv.shape[1]} features')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_maes = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_cv)):
        model_cv = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5.0,
            objective='reg:squarederror',
            eval_metric='mae',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=30
        )

        model_cv.fit(
            X_cv[train_idx], y_cv[train_idx],
            eval_set=[(X_cv[val_idx], y_cv[val_idx])],
            verbose=False,
        )

        pred = model_cv.predict(X_cv[val_idx])
        fold_mae = mean_absolute_error(y_cv[val_idx], pred)
        cv_maes.append(fold_mae)
        print(f'  Fold {fold_idx + 1}/5: MAE = {fold_mae:.4f}')

    mean_mae = np.mean(cv_maes)
    std_mae  = np.std(cv_maes)
    print(f'\n  5-Fold CV MAE: {mean_mae:.4f} ± {std_mae:.4f}')

    return cv_maes


# ================================================================
# 5. FEATURE IMPORTANCE
# ================================================================
def plot_feature_importance(model):
    """
    Plot the top 15 most important features as a horizontal bar chart.

    WHY feature importance matters for MUIS:
    -----------------------------------------
    * If F02_land_use_entropy is #1, it validates Jane Jacobs' thesis
      that diversity of uses is the primary driver of urban vitality.
    * If F01_poi_density ranks high, it confirms that activity intensity
      (Christopher Alexander's Pattern 30: Activity Nodes) matters.
    * If F12_gamma_index is important, it supports the connectivity
      argument from space syntax / network urbanism.
    * Unexpected importances (e.g., F27_food_stand_density ranking high)
      might reveal data artefacts or genuinely interesting local patterns.

    We use 'gain' importance (total gain contributed by splits on each
    feature) rather than 'weight' (number of splits) because gain is
    more indicative of actual predictive power.
    """
    print('\n' + '=' * 60)
    print('FEATURE IMPORTANCE')
    print('=' * 60)

    # Extract importance scores aligned with FEATURE_COLS
    importances = model.feature_importances_

    # Build a sorted list of (feature_name, importance)
    feat_imp = sorted(zip(FEATURE_COLS, importances),
                      key=lambda x: x[1], reverse=True)

    # Print all importances
    print(f'  {"Feature":<30s}  {"Importance":>10s}')
    print(f'  {"-" * 30}  {"-" * 10}')
    for fname, imp in feat_imp:
        bar = '#' * int(imp / max(importances) * 30)
        print(f'  {fname:<30s}  {imp:10.4f}  {bar}')

    # -- Plot top 15 -----------------------------------------------
    top_n = min(15, len(feat_imp))
    top_feats = feat_imp[:top_n]
    names = [f[0] for f in reversed(top_feats)]   # Reverse for horizontal bar
    values = [f[1] for f in reversed(top_feats)]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(names)), values, color='#2196F3', edgecolor='none')

    # Add value labels on bars
    for bar_item, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar_item.get_y() + bar_item.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(f'XGBoost: Top {top_n} Feature Importances')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, 'xgb_feature_importance.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n  [OK] Saved: {path}')

    # -- Save as JSON ----------------------------------------------
    imp_dict = {fname: float(imp) for fname, imp in feat_imp}
    json_path = os.path.join(MODEL_DIR, 'feature_importance.json')
    with open(json_path, 'w') as f:
        json.dump(imp_dict, f, indent=2)
    print(f'  [OK] Saved: {json_path}')

    return imp_dict


# ================================================================
# 6. PREDICTED VS ACTUAL PLOT
# ================================================================
def plot_predicted_vs_actual(y_test, y_pred):
    """
    Scatter plot: predicted vs actual MUIS score (same as ANN version
    for visual comparison).
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_test, y_pred, alpha=0.5, s=20, c='#FF9800', edgecolors='none')

    lim_lo = min(y_test.min(), y_pred.min()) - 0.5
    lim_hi = max(y_test.max(), y_pred.max()) + 0.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            'k--', linewidth=1.5, label='Perfect prediction (y = x)')

    ax.set_xlabel('Actual MUIS Score')
    ax.set_ylabel('Predicted MUIS Score')
    ax.set_title('XGBoost: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, 'xgb_predicted_vs_actual.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Saved: {path}')


# ================================================================
# 7. SAVE MODEL AND RESULTS
# ================================================================
def save_model_and_results(model, mae, rmse, r2, cv_maes, feat_imp):
    """
    Save the XGBoost model as JSON and all metrics for reproducibility.

    WHY JSON format for the model?
    ------------------------------
    XGBoost's JSON format is:
    * Human-readable (you can inspect tree structures)
    * Language-agnostic (can be loaded in C++, Java, etc.)
    * Version-stable (pickle files can break across XGBoost versions)
    """
    print('\n' + '=' * 60)
    print('SAVING MODEL AND RESULTS')
    print('=' * 60)

    # -- Save model ------------------------------------------------
    model_path = os.path.join(MODEL_DIR, 'xgb_model.json')
    model.save_model(model_path)
    print(f'  [OK] Model saved: {model_path}')

    # -- Save metrics ----------------------------------------------
    results = {
        'test_mae':  float(mae),
        'test_rmse': float(rmse),
        'test_r2':   float(r2),
        'cv_maes':     [float(x) for x in cv_maes],
        'cv_mae_mean': float(np.mean(cv_maes)),
        'cv_mae_std':  float(np.std(cv_maes)),
        'best_iteration': int(getattr(model, 'best_iteration', getattr(model, 'n_estimators', 500))),
        'best_score':     float(getattr(model, 'best_score', 0.0)),
        'n_features':     N_FEATURES,
        'feature_importance': feat_imp,
    }

    results_path = os.path.join(MODEL_DIR, 'xgb_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  [OK] Results saved: {results_path}')


# ================================================================
# 8. MODEL COMPARISON
# ================================================================
def print_comparison_summary(xgb_mae, xgb_rmse, xgb_r2,
                              xgb_cv_mean, xgb_cv_std):
    """
    Load ANN results (if available) and print a side-by-side comparison.

    This helps decide which model to deploy:
    * If XGBoost wins by a large margin -> the relationships are mostly
      additive and the ANN's depth is wasted.
    * If ANN wins -> there are deep compositional patterns in the data
      that trees can't capture.
    * If they're similar -> use XGBoost in production (faster, more
      interpretable) and report both in the paper.
    """
    print('\n' + '=' * 60)
    print('MODEL COMPARISON: ANN vs XGBoost')
    print('=' * 60)

    ann_results_path = os.path.join(MODEL_DIR, 'training_history.json')

    if os.path.exists(ann_results_path):
        with open(ann_results_path, 'r') as f:
            ann_results = json.load(f)

        ann_mae  = ann_results.get('test_mae',  float('nan'))
        ann_rmse = ann_results.get('test_rmse', float('nan'))
        ann_r2   = ann_results.get('test_r2',   float('nan'))
        ann_cv   = ann_results.get('cv_mae_mean', float('nan'))
        ann_cv_s = ann_results.get('cv_mae_std',  float('nan'))

        print(f'  {"Metric":<20s}  {"ANN":>12s}  {"XGBoost":>12s}  {"Winner":>10s}')
        print(f'  {"-" * 20}  {"-" * 12}  {"-" * 12}  {"-" * 10}')

        # MAE (lower is better)
        mae_winner = 'XGBoost' if xgb_mae < ann_mae else 'ANN'
        print(f'  {"Test MAE":<20s}  {ann_mae:12.4f}  {xgb_mae:12.4f}  {mae_winner:>10s}')

        # RMSE (lower is better)
        rmse_winner = 'XGBoost' if xgb_rmse < ann_rmse else 'ANN'
        print(f'  {"Test RMSE":<20s}  {ann_rmse:12.4f}  {xgb_rmse:12.4f}  {rmse_winner:>10s}')

        # R² (higher is better)
        r2_winner = 'XGBoost' if xgb_r2 > ann_r2 else 'ANN'
        print(f'  {"Test R²":<20s}  {ann_r2:12.4f}  {xgb_r2:12.4f}  {r2_winner:>10s}')

        # CV MAE (lower is better)
        cv_winner = 'XGBoost' if xgb_cv_mean < ann_cv else 'ANN'
        print(f'  {"CV MAE":<20s}  {ann_cv:7.4f}±{ann_cv_s:.4f}'
              f'  {xgb_cv_mean:7.4f}±{xgb_cv_std:.4f}  {cv_winner:>10s}')

        # -- Recommendation ----------------------------------------
        xgb_wins = sum([
            xgb_mae < ann_mae,
            xgb_rmse < ann_rmse,
            xgb_r2 > ann_r2,
            xgb_cv_mean < ann_cv,
        ])

        print()
        if xgb_wins >= 3:
            print('  [STATS] RECOMMENDATION: XGBoost outperforms the ANN on most metrics.')
            print('     This is expected for tabular data with < 10k samples.')
            print('     Consider using XGBoost for production and reporting both')
            print('     models in the paper for academic completeness.')
        elif xgb_wins <= 1:
            print('  [STATS] RECOMMENDATION: The ANN outperforms XGBoost on most metrics.')
            print('     This suggests the data has non-trivial compositional patterns')
            print('     that benefit from deep feature learning.')
        else:
            print('  [STATS] RECOMMENDATION: Both models perform similarly.')
            print('     Use XGBoost for production (faster, more interpretable).')
            print('     Report both in the paper.')

    else:
        print('  [WARN] ANN results not found (training_history.json missing).')
        print('    Run train_ann.py first, then re-run this script for comparison.')
        print()
        print(f'  XGBoost standalone results:')
        print(f'    Test MAE:  {xgb_mae:.4f}')
        print(f'    Test RMSE: {xgb_rmse:.4f}')
        print(f'    Test R²:   {xgb_r2:.4f}')
        print(f'    CV MAE:    {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}')


# ================================================================
# 9. MAIN
# ================================================================
def main():
    """
    Full XGBoost training pipeline:
      1. Load data
      2. Train XGBRegressor with early stopping
      3. Evaluate on held-out test set
      4. 5-fold cross-validation
      5. Feature importance analysis
      6. Save model, results, and figures
      7. Print comparison with ANN (if available)
    """
    # -- Load ------------------------------------------------------
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_test_class) = load_training_data()

    # -- Train -----------------------------------------------------
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # -- Evaluate --------------------------------------------------
    y_pred, mae, rmse, r2 = evaluate_model(
        model, X_test, y_test, y_test_class
    )

    # -- Cross-Validation ------------------------------------------
    cv_maes = run_cross_validation(X_train, y_train, X_val, y_val)

    # -- Feature Importance ----------------------------------------
    feat_imp = plot_feature_importance(model)

    # -- Predicted vs Actual ---------------------------------------
    print('\n' + '=' * 60)
    print('GENERATING FIGURES')
    print('=' * 60)
    plot_predicted_vs_actual(y_test, y_pred)

    # -- Save ------------------------------------------------------
    save_model_and_results(model, mae, rmse, r2, cv_maes, feat_imp)

    # -- Comparison ------------------------------------------------
    print_comparison_summary(mae, rmse, r2,
                              np.mean(cv_maes), np.std(cv_maes))

    # -- Final Summary ---------------------------------------------
    print('\n' + '=' * 60)
    print('XGBOOST TRAINING COMPLETE')
    print('=' * 60)
    print(f'  Test MAE:    {mae:.4f} MUIS points')
    print(f'  Test RMSE:   {rmse:.4f}')
    print(f'  Test R²:     {r2:.4f}')
    print(f'  CV MAE:      {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}')
    print(f'  Trees used:  {getattr(model, "best_iteration", model.n_estimators)}')
    print(f'  Saved to:    {MODEL_DIR}/')


if __name__ == '__main__':
    main()
