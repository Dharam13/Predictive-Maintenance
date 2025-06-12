import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🚀 Improved XGBoost TTTF Prediction Pipeline (Better Short-Term Accuracy)")
print("=" * 90)

# Step 1: Load and Sample Data Strategically
print("\n📊 Step 1: Loading and Stratified Sampling (200K)")
print("-" * 50)

df = pd.read_csv('TTF_Dataset_Weeks.csv')
print(f"✅ Full dataset loaded: {df.shape}")


# Create stratification bins for balanced sampling (without failure_within_48h)
def create_strata(df):
    """Create stratification categories for balanced sampling"""
    strata = pd.DataFrame()

    # Machine model stratification
    if 'model' in df.columns:
        strata['model'] = df['model']
    else:
        strata['model'] = 'default'

    # TTTF range stratification (short/medium/long term)
    target_cols = ['ttf_comp1_weeks', 'ttf_comp2_weeks', 'ttf_comp3_weeks', 'ttf_comp4_weeks']
    available_target_cols = [col for col in target_cols if col in df.columns]

    if available_target_cols:
        avg_tttf = df[available_target_cols].mean(axis=1)
        strata['tttf_range'] = pd.cut(avg_tttf, bins=4, labels=['very_short', 'short', 'medium', 'long'])
    else:
        strata['tttf_range'] = 'default'

    # Age stratification (more granular)
    if 'age' in df.columns:
        strata['age_group'] = pd.cut(df['age'], bins=4, labels=['new', 'young', 'mature', 'old'])
    else:
        strata['age_group'] = 'default'

    # Error count stratification (if available)
    if 'error_count' in df.columns:
        strata['error_level'] = pd.cut(df['error_count'], bins=3, labels=['low', 'medium', 'high'])
    else:
        strata['error_level'] = 'default'

    # Combine all strata
    strata['combined'] = (strata['model'].astype(str) + '_' +
                          strata['tttf_range'].astype(str) + '_' +
                          strata['age_group'].astype(str) + '_' +
                          strata['error_level'].astype(str))

    return strata['combined']


# Create stratification labels
strata_labels = create_strata(df)
print(f"📊 Created {strata_labels.nunique()} stratification groups")

# Stratified sampling to get 200k representative samples
sample_size = min(200000, len(df))
sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
sample_idx, _ = next(sss.split(df, strata_labels))

df_sample = df.iloc[sample_idx].copy()
print(f"✅ Stratified sample created: {df_sample.shape}")
print(f"📈 Sample represents {df_sample.shape[0] / df.shape[0] * 100:.1f}% of original data")

# Step 2: Define Features and Targets (Updated - Removing failure_within_48h and machineID)
print("\n🎯 Step 2: Feature and Target Definition (Updated)")
print("-" * 50)

# Define target features
target_features = ['ttf_comp1_weeks', 'ttf_comp2_weeks', 'ttf_comp3_weeks', 'ttf_comp4_weeks']
available_targets = [col for col in target_features if col in df_sample.columns]

# Feature groups (UPDATED - Removed failure_within_48h and machineID)
sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
operational_features = ['model', 'age', 'error_count']  # Removed machineID and failure_within_48h
maintenance_features = ['days_since_comp1_maint', 'days_since_comp2_maint',
                        'days_since_comp3_maint', 'days_since_comp4_maint']

# Get available features
available_features = []
for feature_group in [sensor_features, operational_features, maintenance_features]:
    available_features.extend([f for f in feature_group if f in df_sample.columns])

print(f"🎯 Available target features: {available_targets}")
print(f"📋 Available input features ({len(available_features)}): {available_features}")
print(f"🚫 Excluded features: machineID, failure_within_48h")

# Step 3: Train-Test Split BEFORE preprocessing
print("\n📊 Step 3: Train-Test Split (80-20)")
print("-" * 35)

# Prepare initial features and targets
X_initial = df_sample[available_features].copy()
y_initial = df_sample[available_targets].copy()

# Stratify based on model type for better distribution
stratify_col = df_sample['model'] if 'model' in df_sample.columns else None

X_train, X_test, y_train, y_test = train_test_split(
    X_initial, y_initial,
    test_size=0.2,
    random_state=42,
    stratify=stratify_col
)

print(f"📈 Train set: {X_train.shape[0]} samples ({X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100:.1f}%)")
print(f"📉 Test set: {X_test.shape[0]} samples ({X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100:.1f}%)")

# Step 4: Enhanced Data Preprocessing
print("\n🔄 Step 4: Enhanced Data Preprocessing")
print("-" * 40)

# Separate numerical and categorical features
categorical_features = ['model']
numerical_features = [f for f in available_features if f not in categorical_features]

print(f"📊 Categorical features: {categorical_features}")
print(f"🔢 Numerical features ({len(numerical_features)}): {numerical_features}")

# Initialize preprocessors
label_encoders = {}
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Process training data
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Handle categorical variables
print("Encoding categorical variables...")
for cat_feat in categorical_features:
    if cat_feat in available_features:
        # Fit label encoder on training data
        le = LabelEncoder()

        # Handle missing values in training data
        train_values = X_train_processed[cat_feat].fillna('unknown').astype(str)
        le.fit(train_values)
        X_train_processed[cat_feat] = le.transform(train_values)

        # Transform test data (handle unseen categories)
        test_values = X_test_processed[cat_feat].fillna('unknown').astype(str)
        # Handle unseen categories by mapping them to 'unknown'
        test_values = test_values.apply(lambda x: x if x in le.classes_ else 'unknown')
        X_test_processed[cat_feat] = le.transform(test_values)

        # Store encoder for later use
        label_encoders[cat_feat] = le
        print(f"✅ Encoded {cat_feat} ({len(le.classes_)} unique values)")

# Handle missing values for numerical features
if numerical_features:
    print("Handling missing values for numerical features...")
    # Fit imputer on training data
    X_train_processed[numerical_features] = imputer.fit_transform(X_train_processed[numerical_features])
    # Transform test data
    X_test_processed[numerical_features] = imputer.transform(X_test_processed[numerical_features])
    print("✅ Missing values handled")

    # Display missing value statistics
    train_missing = X_train[numerical_features].isnull().sum()
    if train_missing.sum() > 0:
        print("Missing values per feature (before imputation):")
        for feat, count in train_missing[train_missing > 0].items():
            print(f"  {feat}: {count} ({count / len(X_train) * 100:.1f}%)")

# Convert TTF to floor values AFTER split
print("\nConverting TTF to floor values...")
y_train_floor = np.floor(y_train)
y_test_floor = np.floor(y_test)

print("Original vs Floor TTF statistics:")
for i, col in enumerate(available_targets):
    orig_mean = y_train.iloc[:, i].mean()
    floor_mean = y_train_floor.iloc[:, i].mean()
    orig_std = y_train.iloc[:, i].std()
    floor_std = y_train_floor.iloc[:, i].std()
    print(f"{col}: Original={orig_mean:.2f}±{orig_std:.2f}, Floor={floor_mean:.2f}±{floor_std:.2f}")

# Feature scaling
print("\nScaling features...")
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)
print("✅ Features scaled using StandardScaler")

# Display feature statistics after scaling
print("Feature scaling verification (first 5 features):")
for i, feat in enumerate(available_features[:5]):
    train_mean = X_train_scaled[:, i].mean()
    train_std = X_train_scaled[:, i].std()
    print(f"  {feat}: mean={train_mean:.3f}, std={train_std:.3f}")

# Step 5: Enhanced XGBoost Model Training (BETTER FOR SHORT-TERM PREDICTIONS)
print("\n🚀 Step 5: Enhanced XGBoost Model Training (Optimized for Short-Term)")
print("-" * 65)

# XGBoost parameters optimized for better short-term (< 10 weeks) predictions
xgb_params = {
    'n_estimators': 500,  # More trees for better learning
    'learning_rate': 0.05,  # Lower learning rate for precision
    'max_depth': 8,  # Deeper trees to capture complex patterns
    'min_child_weight': 2,  # Reduced for better sensitivity to small changes
    'subsample': 0.9,  # High subsample for stability
    'colsample_bytree': 0.8,  # Feature sampling
    'colsample_bylevel': 0.8,  # Additional regularization
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'gamma': 0.1,  # Minimum split loss (helps with overfitting)
    'random_state': 42,
    'n_jobs': -1,  # Use all cores
    'tree_method': 'hist',  # Faster training
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

print("Enhanced XGBoost Parameters (Optimized for Short-Term Accuracy):")
for param, value in xgb_params.items():
    print(f"  {param}: {value}")

# Create and train XGBoost model with MultiOutput wrapper
print("\nTraining Enhanced XGBoost MultiOutput Regressor...")
xgb_model = MultiOutputRegressor(XGBRegressor(**xgb_params))
xgb_model.fit(X_train_scaled, y_train_floor)
print("✅ XGBoost model training completed!")

# Step 6: Comprehensive Model Evaluation
print("\n📊 Step 6: Comprehensive Model Evaluation")
print("-" * 45)

# Make predictions on both sets
y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

print("🔮 Predictions completed!")
print(f"📊 Train predictions shape: {y_train_pred.shape}")
print(f"📊 Test predictions shape: {y_test_pred.shape}")

# Calculate comprehensive metrics for each target
results = []
for i, target in enumerate(available_targets):
    # Training metrics
    train_mse = mean_squared_error(y_train_floor.iloc[:, i], y_train_pred[:, i])
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_floor.iloc[:, i], y_train_pred[:, i])
    train_r2 = r2_score(y_train_floor.iloc[:, i], y_train_pred[:, i])

    # Test metrics
    test_mse = mean_squared_error(y_test_floor.iloc[:, i], y_test_pred[:, i])
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_floor.iloc[:, i], y_test_pred[:, i])
    test_r2 = r2_score(y_test_floor.iloc[:, i], y_test_pred[:, i])

    # Calculate percentage errors
    train_mape = np.mean(np.abs((y_train_floor.iloc[:, i] - y_train_pred[:, i]) /
                                np.maximum(y_train_floor.iloc[:, i], 1))) * 100
    test_mape = np.mean(np.abs((y_test_floor.iloc[:, i] - y_test_pred[:, i]) /
                               np.maximum(y_test_floor.iloc[:, i], 1))) * 100

    # Special metrics for short-term predictions (< 10 weeks)
    short_term_mask = y_test_floor.iloc[:, i] < 10
    if short_term_mask.sum() > 0:
        short_term_mae = mean_absolute_error(
            y_test_floor.iloc[:, i][short_term_mask],
            y_test_pred[:, i][short_term_mask]
        )
        short_term_mape = np.mean(np.abs((y_test_floor.iloc[:, i][short_term_mask] - y_test_pred[:, i][short_term_mask]) /
                                       np.maximum(y_test_floor.iloc[:, i][short_term_mask], 1))) * 100
    else:
        short_term_mae = 0
        short_term_mape = 0

    results.append({
        'Target': target,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Train_MAPE': train_mape,
        'Test_MAPE': test_mape,
        'Short_Term_MAE': short_term_mae,
        'Short_Term_MAPE': short_term_mape,
        'Short_Term_Count': short_term_mask.sum(),
        'Overfitting': train_r2 - test_r2
    })

# Display comprehensive results
results_df = pd.DataFrame(results)
print("\n🎯 Comprehensive XGBoost Model Performance Summary:")
print(results_df.round(4))

# Overall performance metrics
print(f"\n📊 Overall Performance Metrics:")
print(f"Average Test RMSE: {results_df['Test_RMSE'].mean():.4f}")
print(f"Average Test MAE: {results_df['Test_MAE'].mean():.4f}")
print(f"Average Test R²: {results_df['Test_R2'].mean():.4f}")
print(f"Average Test MAPE: {results_df['Test_MAPE'].mean():.2f}%")
print(f"Average Short-Term (<10 weeks) MAE: {results_df['Short_Term_MAE'].mean():.4f}")
print(f"Average Short-Term (<10 weeks) MAPE: {results_df['Short_Term_MAPE'].mean():.2f}%")
print(f"Average Overfitting: {results_df['Overfitting'].mean():.4f}")

# Model efficiency metrics
print(f"\n⚡ Model Efficiency:")
print(f"Features used: {len(available_features)} (reduced from original)")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Model complexity: {xgb_params['n_estimators']} estimators × {len(available_targets)} targets")
print(f"Algorithm: XGBoost (Better for short-term predictions)")

# Step 7: Feature Importance Analysis
print("\n📈 Step 7: Feature Importance Analysis")
print("-" * 42)

# Get feature importance for each target
feature_importance_data = []
for i, target in enumerate(available_targets):
    estimator = xgb_model.estimators_[i]
    importances = estimator.feature_importances_

    for j, feature in enumerate(available_features):
        feature_importance_data.append({
            'Target': target,
            'Feature': feature,
            'Importance': importances[j]
        })

# Create feature importance DataFrame
importance_df = pd.DataFrame(feature_importance_data)

# Display top features for each target
print("\n🔝 Top 5 Most Important Features per Target:")
for target in available_targets:
    target_importance = importance_df[importance_df['Target'] == target].sort_values('Importance', ascending=False)
    print(f"\n{target}:")
    for _, row in target_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Step 8: Save Enhanced XGBoost Model and Preprocessors
print("\n💾 Step 8: Saving Enhanced XGBoost Model and Preprocessors")
print("-" * 57)

# Create comprehensive model package
model_package = {
    'model': xgb_model,
    'scaler': scaler,
    'imputer': imputer,
    'label_encoders': label_encoders,
    'feature_names': available_features,
    'target_names': available_targets,
    'model_params': xgb_params,
    'performance_metrics': results_df.to_dict('records'),
    'feature_importance': importance_df.to_dict('records'),
    'excluded_features': ['machineID', 'failure_within_48h'],
    'model_version': '3.0_xgboost_enhanced',
    'model_type': 'XGBoost'
}

# Save using joblib (recommended for sklearn models)
joblib.dump(model_package, 'tttf_xgb_model_enhanced.pkl')
print("✅ Enhanced XGBoost model package saved as 'tttf_xgb_model_enhanced.pkl'")

# Also save as pickle for compatibility
with open('tttf_xgb_model_enhanced_backup.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print("✅ Backup XGBoost model saved as 'tttf_xgb_model_enhanced_backup.pkl'")

# Save test data for validation
test_data_package = {
    'X_test_original': X_test,
    'X_test_processed': X_test_processed,
    'X_test_scaled': X_test_scaled,
    'y_test_original': y_test,
    'y_test_floor': y_test_floor,
    'y_test_predictions': y_test_pred,
    'feature_names': available_features,
    'target_names': available_targets
}

joblib.dump(test_data_package, 'test_data_xgb_enhanced.pkl')
print("✅ Enhanced XGBoost test data saved as 'test_data_xgb_enhanced.pkl'")

# Step 9: Sample Predictions Display with Short-Term Focus
print("\n🔮 Step 9: Sample Test Predictions (Focus on Short-Term)")
print("-" * 55)

# Show prediction examples with better formatting, focusing on short-term predictions
short_term_indices = []
for i in range(len(y_test_floor)):
    if any(y_test_floor.iloc[i] < 10):
        short_term_indices.append(i)

if len(short_term_indices) > 0:
    sample_indices = np.random.choice(short_term_indices, min(10, len(short_term_indices)), replace=False)
    print("\nSample Short-Term Predictions (< 10 weeks):")
else:
    sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    print("\nSample General Predictions:")

print("Index | Component           | Actual | Predicted | Error  | Error% | Short-Term")
print("-" * 80)

for idx in sample_indices[:8]:  # Show first 8 samples
    for i, target in enumerate(available_targets):
        actual = y_test_floor.iloc[idx, i]
        predicted = y_test_pred[idx, i]
        error = abs(actual - predicted)
        error_pct = (error / max(actual, 1)) * 100
        is_short_term = "Yes" if actual < 10 else "No"
        print(f"{idx:5d} | {target:18s} | {actual:6.0f} | {predicted:9.1f} | {error:6.1f} | {error_pct:5.1f}% | {is_short_term:9s}")
    print("-" * 80)

# Final summary with focus on short-term improvement
print(f"\n✅ Enhanced XGBoost Training Pipeline Completed Successfully!")
print(f"📊 Model trained on {X_train.shape[0]:,} samples (80%)")
print(f"🧪 Tested on {X_test.shape[0]:,} samples (20%)")
print(f"🎯 Predicting floor values for {len(available_targets)} components")
print(f"⚡ Using {len(available_features)} efficient features")
print(f"🚫 Excluded: machineID, failure_within_48h")
print(f"📈 Average Test R²: {results_df['Test_R2'].mean():.4f}")
print(f"📉 Average Test MAPE: {results_df['Test_MAPE'].mean():.2f}%")
print(f"🎯 Average Short-Term (<10 weeks) MAE: {results_df['Short_Term_MAE'].mean():.4f}")
print(f"🎯 Average Short-Term (<10 weeks) MAPE: {results_df['Short_Term_MAPE'].mean():.2f}%")
print(f"🔧 Algorithm: XGBoost (Superior for short-term accuracy)")
print(f"💾 Enhanced XGBoost model and preprocessors saved successfully")
print("=" * 90)