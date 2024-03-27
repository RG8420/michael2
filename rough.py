# Meta Model Improved Code

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from bayes_opt import BayesianOptimization
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define base models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)
lgb_model = lgb.LGBMClassifier(random_state=42)

# Define meta-model
meta_model = RandomForestClassifier(random_state=42)

# Define parameter grids for Bayesian optimization
param_grid_rf = {
    'n_estimators': (10, 1000),
    'max_depth': (2, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'max_features': (1, len(X[0]))
}

param_grid_xgb = {
    'n_estimators': (10, 1000),
    'max_depth': (2, 50),
    'learning_rate': (0.01, 1),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

param_grid_lgb = {
    'n_estimators': (10, 1000),
    'max_depth': (2, 50),
    'learning_rate': (0.01, 1),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}


# Define objective function for Bayesian optimization
def rf_objective(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    rf_model.set_params(n_estimators=int(n_estimators), max_depth=int(max_depth),
                        min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf),
                        max_features=int(max_features))
    return np.mean(cross_val_predict(rf_model, X, y, cv=5))


def xgb_objective(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
    xgb_model.set_params(n_estimators=int(n_estimators), max_depth=int(max_depth),
                         learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree)
    return np.mean(cross_val_predict(xgb_model, X, y, cv=5))


def lgb_objective(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
    lgb_model.set_params(n_estimators=int(n_estimators), max_depth=int(max_depth),
                         learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree)
    return np.mean(cross_val_predict(lgb_model, X, y, cv=5))


# Perform Bayesian optimization for each model
rf_optimizer = BayesianOptimization(rf_objective, param_grid_rf, random_state=42)
xgb_optimizer = BayesianOptimization(xgb_objective, param_grid_xgb, random_state=42)
lgb_optimizer = BayesianOptimization(lgb_objective, param_grid_lgb, random_state=42)

rf_optimizer.maximize(init_points=5, n_iter=20)
xgb_optimizer.maximize(init_points=5, n_iter=20)
lgb_optimizer.maximize(init_points=5, n_iter=20)

# Get best hyperparameters
best_params_rf = rf_optimizer.max['params']
best_params_xgb = xgb_optimizer.max['params']
best_params_lgb = lgb_optimizer.max['params']

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store predictions from base models
rf_preds, xgb_preds, lgb_preds = [], [], []

# Perform cross-validation to generate predictions from base models
for train_idx, val_idx in kf.split(X):
    # Split data for cross-validation
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Initialize base models with best hyperparameters
    rf_model.set_params(**best_params_rf)
    xgb_model.set_params(**best_params_xgb)
    lgb_model.set_params(**best_params_lgb)

    # Fit base models
    rf_model.fit(X_train_fold, y_train_fold)
    xgb_model.fit(X_train_fold, y_train_fold)
    lgb_model.fit(X_train_fold, y_train_fold)

    # Make predictions
    rf_fold_preds = rf_model.predict(X_val_fold)
    xgb_fold_preds = xgb_model.predict(X_val_fold)
    lgb_fold_preds = lgb_model.predict(X_val_fold)

    # Store predictions
    rf_preds.extend(rf_fold_preds)
    xgb_preds.extend(xgb_fold_preds)
    lgb_preds.extend(lgb_fold_preds)

# Stack the predictions vertically
stacked_X_train = np.column_stack((rf_preds, xgb_preds, lgb_preds))

# Fit the meta-model on stacked predictions
meta_model.fit(stacked_X_train, y)

# # Generate predictions from base models on the test data
# rf_test_preds = rf_model.predict(X_test)
# xgb_test_preds = xgb_model.predict(X_test)
# lgb_test_preds = lgb_model.predict(X_test)
#
# # Stack the test predictions vertically
# stacked_X_test = np.column_stack((rf_test_preds, xgb_test_preds, lgb_test_preds))
#
# # Make predictions using the meta-model
# meta_test_preds = meta_model.predict(stacked_X_test)
#
# # Calculate accuracy
# accuracy = accuracy_score(y_test, meta_test_preds)
# print("Accuracy:", accuracy)

