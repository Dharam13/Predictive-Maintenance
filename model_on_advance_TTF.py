import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings

warnings.filterwarnings('ignore')


class ImprovedTTFPredictor:
    def __init__(self):
        """
        Improved TTF Predictor with enhanced feature engineering and model ensemble
        """
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.poly_features = None
        self.label_encoders = {}
        self.feature_columns = None
        self.feature_selector = None
        self.is_trained = False
        self.machine_stats = None  # Store machine statistics for single predictions

    def advanced_feature_engineering(self, df, is_single_prediction=False):
        """Advanced feature engineering for maximum predictive power"""
        print("Performing advanced feature engineering...")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Handle single prediction case
        if is_single_prediction and 'machineID' not in df.columns:
            df['machineID'] = 1  # Default machine ID for single predictions

        # Sort by machineID and datetime only if we have multiple records
        if len(df) > 1 and 'datetime' in df.columns and 'machineID' in df.columns:
            df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)

        # Enhanced time features
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Enhanced sensor statistics with multiple windows
        sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
        available_sensor_cols = [col for col in sensor_cols if col in df.columns]

        # For single predictions, use default values or stored statistics
        if is_single_prediction and len(df) == 1:
            # Use stored machine statistics if available
            if self.machine_stats is not None:
                for col in available_sensor_cols:
                    # Add some basic rolling window features with default values
                    for window in [3, 6, 12, 24]:
                        df[f'{col}_mean_{window}'] = df[col].iloc[0]
                        df[f'{col}_std_{window}'] = self.machine_stats.get(f'{col}_std', 0.1)
                        df[f'{col}_max_{window}'] = df[col].iloc[0] * 1.1
                        df[f'{col}_min_{window}'] = df[col].iloc[0] * 0.9
                        df[f'{col}_median_{window}'] = df[col].iloc[0]
                        df[f'{col}_range_{window}'] = df[col].iloc[0] * 0.2
                        df[f'{col}_cv_{window}'] = 0.1
                        df[f'{col}_trend_{window}'] = 0
            else:
                # Default values when no machine stats available
                for col in available_sensor_cols:
                    for window in [3, 6, 12, 24]:
                        df[f'{col}_mean_{window}'] = df[col].iloc[0]
                        df[f'{col}_std_{window}'] = 0.1
                        df[f'{col}_max_{window}'] = df[col].iloc[0] * 1.1
                        df[f'{col}_min_{window}'] = df[col].iloc[0] * 0.9
                        df[f'{col}_median_{window}'] = df[col].iloc[0]
                        df[f'{col}_range_{window}'] = df[col].iloc[0] * 0.2
                        df[f'{col}_cv_{window}'] = 0.1
                        df[f'{col}_trend_{window}'] = 0
        else:
            # Original rolling window logic for training data
            if 'machineID' in df.columns and len(df) > 1:
                try:
                    machine_groups = df.groupby('machineID')

                    # Multiple rolling windows for comprehensive patterns
                    for window in [3, 6, 12, 24, 48, 72]:
                        for col in available_sensor_cols:
                            # Basic statistics
                            df[f'{col}_mean_{window}'] = machine_groups[col].rolling(window,
                                                                                     min_periods=1).mean().reset_index(
                                0, drop=True)
                            df[f'{col}_std_{window}'] = machine_groups[col].rolling(window,
                                                                                    min_periods=1).std().reset_index(0,
                                                                                                                     drop=True).fillna(
                                0.1)
                            df[f'{col}_max_{window}'] = machine_groups[col].rolling(window,
                                                                                    min_periods=1).max().reset_index(0,
                                                                                                                     drop=True)
                            df[f'{col}_min_{window}'] = machine_groups[col].rolling(window,
                                                                                    min_periods=1).min().reset_index(0,
                                                                                                                     drop=True)
                            df[f'{col}_median_{window}'] = machine_groups[col].rolling(window,
                                                                                       min_periods=1).median().reset_index(
                                0, drop=True)

                            # Advanced statistics
                            df[f'{col}_skew_{window}'] = machine_groups[col].rolling(window,
                                                                                     min_periods=3).skew().reset_index(
                                0, drop=True).fillna(0)
                            df[f'{col}_kurt_{window}'] = machine_groups[col].rolling(window,
                                                                                     min_periods=3).kurt().reset_index(
                                0, drop=True).fillna(0)
                            df[f'{col}_range_{window}'] = df[f'{col}_max_{window}'] - df[f'{col}_min_{window}']
                            df[f'{col}_cv_{window}'] = df[f'{col}_std_{window}'] / (df[f'{col}_mean_{window}'] + 1e-8)

                            # Trend indicators
                            try:
                                df[f'{col}_trend_{window}'] = machine_groups[col].rolling(window, min_periods=2).apply(
                                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                                ).reset_index(0, drop=True).fillna(0)
                            except:
                                df[f'{col}_trend_{window}'] = 0
                except Exception as e:
                    print(f"Warning: Error in rolling window calculations: {e}")
                    # Fallback: create simple features
                    for window in [3, 6, 12, 24]:
                        for col in available_sensor_cols:
                            df[f'{col}_mean_{window}'] = df[col]
                            df[f'{col}_std_{window}'] = 0.1
                            df[f'{col}_max_{window}'] = df[col] * 1.1
                            df[f'{col}_min_{window}'] = df[col] * 0.9
                            df[f'{col}_median_{window}'] = df[col]
                            df[f'{col}_range_{window}'] = df[col] * 0.2
                            df[f'{col}_cv_{window}'] = 0.1
                            df[f'{col}_trend_{window}'] = 0

        # Advanced sensor interactions and health indicators
        if len(available_sensor_cols) >= 4:  # Check if all sensor columns are available
            try:
                # Critical failure indicators (enhanced for critical TTF prediction)
                df['critical_volt_deviation'] = np.maximum(0, (170 - df['volt']) / 170)
                df['critical_rotate_drop'] = np.maximum(0, (440 - df['rotate']) / 440)
                df['critical_pressure_drop'] = np.maximum(0, (100 - df['pressure']) / 100)
                df['critical_vibration_spike'] = np.maximum(0, (df['vibration'] - 40) / 40)

                # Composite critical failure score
                df['critical_failure_score'] = (df['critical_volt_deviation'] * 0.3 +
                                                df['critical_rotate_drop'] * 0.25 +
                                                df['critical_pressure_drop'] * 0.25 +
                                                df['critical_vibration_spike'] * 0.2)

                # Ratios and products
                df['volt_pressure_ratio'] = df['volt'] / (df['pressure'] + 1e-8)
                df['rotate_vibration_ratio'] = df['rotate'] / (df['vibration'] + 1e-8)
                df['volt_rotate_product'] = df['volt'] * df['rotate']
                df['pressure_vibration_product'] = df['pressure'] * df['vibration']

                # Health indices with non-linear transformations for critical detection
                df['sensor_health_index'] = (df['volt'] / 170 + df['rotate'] / 440 + df['pressure'] / 100) / (
                            df['vibration'] / 40 + 1)
                df['degradation_index'] = (np.abs(df['volt'] - 170) / 170 +
                                           np.abs(df['rotate'] - 440) / 440 +
                                           np.abs(df['pressure'] - 100) / 100 +
                                           (df['vibration'] - 40) / 40) / 4

                # Enhanced degradation with exponential weighting for critical cases
                df['exponential_degradation'] = np.exp(np.clip(df['degradation_index'] * 3, -10, 10)) - 1

                # Mahalanobis-like distance for anomaly detection
                sensor_means = df[available_sensor_cols].mean()
                sensor_stds = df[available_sensor_cols].std() + 1e-8
                df['sensor_anomaly_score'] = np.sqrt(
                    sum(((df[col] - sensor_means[col]) / sensor_stds[col]) ** 2 for col in available_sensor_cols)
                )

                # Composite indicators
                df['total_sensor_variance'] = df[available_sensor_cols].var(axis=1)
                df['sensor_efficiency'] = (df['volt'] * df['rotate']) / (df['pressure'] * df['vibration'] + 1)

                # Non-linear transformations optimized for critical prediction
                df['volt_squared'] = df['volt'] ** 2
                df['rotate_log'] = np.log1p(df['rotate'])
                df['pressure_sqrt'] = np.sqrt(df['pressure'])
                df['vibration_exp'] = np.minimum(np.exp(df['vibration'] / 100), 100)  # Cap to prevent overflow
            except Exception as e:
                print(f"Warning: Error in sensor interaction calculations: {e}")

        # Enhanced age features with critical failure indicators
        if 'age' in df.columns:
            try:
                df['age_squared'] = df['age'] ** 2
                df['age_cubed'] = df['age'] ** 3
                df['age_log'] = np.log1p(df['age'])
                df['age_sqrt'] = np.sqrt(df['age'])
                df['age_normalized'] = df['age'] / (df['age'].max() + 1e-8)

                # Age binning with error handling
                try:
                    df['age_binned'] = pd.cut(df['age'], bins=10, labels=False, duplicates='drop')
                except:
                    df['age_binned'] = 0

                # Critical age indicators
                df['is_old_machine'] = (df['age'] > df['age'].quantile(0.8)).astype(int)
                df['age_risk_factor'] = np.minimum(df['age'] / 200, 2.0)  # Cap age risk

                # Age interaction with sensors for critical failure prediction
                for col in available_sensor_cols:
                    df[f'{col}_age_interaction'] = df[col] * df['age_risk_factor']
            except Exception as e:
                print(f"Warning: Error in age feature calculations: {e}")

        # Enhanced error pattern features for critical prediction
        if 'error_count' in df.columns:
            try:
                # Error escalation indicators
                df['error_spike'] = (df['error_count'] > df['error_count'].quantile(0.8)).astype(int)
                df['critical_error_level'] = np.minimum(df['error_count'] / 5, 2.0)  # Normalize and cap

                # For training data, calculate rolling error statistics
                if not is_single_prediction and len(df) > 1 and 'machineID' in df.columns:
                    try:
                        machine_groups = df.groupby('machineID')
                        for window in [6, 12, 24, 48]:
                            df[f'error_sum_{window}'] = machine_groups['error_count'].rolling(window,
                                                                                              min_periods=1).sum().reset_index(
                                0, drop=True)
                            df[f'error_rate_{window}'] = df[f'error_sum_{window}'] / window

                        df['error_trend'] = machine_groups['error_count'].rolling(12, min_periods=2).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                        ).reset_index(0, drop=True).fillna(0)
                    except:
                        # Fallback values
                        for window in [6, 12, 24, 48]:
                            df[f'error_sum_{window}'] = df['error_count'] * window / 2
                            df[f'error_rate_{window}'] = df['error_count'] / 2
                        df['error_trend'] = 0
                else:
                    # Default values for single prediction
                    for window in [6, 12, 24, 48]:
                        df[f'error_sum_{window}'] = df['error_count'] * window / 2
                        df[f'error_rate_{window}'] = df['error_count'] / 2
                    df['error_trend'] = 0

                # Error interaction with other features
                if 'age' in df.columns:
                    df['error_age_interaction'] = df['error_count'] * df.get('age_risk_factor', 1)
                if 'vibration' in df.columns:
                    df['error_vibration_interaction'] = df['error_count'] * df['vibration']
            except Exception as e:
                print(f"Warning: Error in error pattern calculations: {e}")

        # Machine-specific features (only for training data)
        if not is_single_prediction and len(df) > 1 and 'machineID' in df.columns:
            try:
                machine_stats = df.groupby('machineID').agg({
                    col: ['mean', 'std'] for col in available_sensor_cols if col in df.columns
                }).fillna(0)

                if 'age' in df.columns:
                    age_stats = df.groupby('machineID')['age'].agg(['mean']).fillna(0)
                    machine_stats = pd.concat([machine_stats, age_stats], axis=1)

                machine_stats.columns = ['_'.join(col).strip() for col in machine_stats.columns]
                machine_stats = machine_stats.add_prefix('machine_')

                # Store machine stats for single predictions
                self.machine_stats = machine_stats.mean().to_dict()

                df = df.merge(machine_stats, left_on='machineID', right_index=True, how='left')
            except Exception as e:
                print(f"Warning: Error in machine-specific feature calculations: {e}")
        elif self.machine_stats is not None:
            # Use stored machine statistics for single predictions
            for key, value in self.machine_stats.items():
                df[key] = value

        # Fill NaN values with more sophisticated methods
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Forward fill then backward fill for remaining NaNs
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        print(f"Advanced feature engineering complete. Total features: {len(df.columns)}")
        return df

    def create_polynomial_features(self, X, degree=2):
        """Create polynomial features for key variables"""
        # Select most important base features for polynomial expansion
        base_features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'error_count']
        available_base = [col for col in base_features if col in X.columns]

        if len(available_base) >= 2:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_data = self.poly_features.fit_transform(X[available_base])
            poly_df = pd.DataFrame(poly_data, columns=self.poly_features.get_feature_names_out(available_base), index=X.index)

            # Add polynomial features to original data
            for col in poly_df.columns:
                if col not in X.columns:  # Only add new features
                    X[col] = poly_df[col]

        return X

    def load_and_preprocess_data(self, file_path, sample_size=100000):
        """Load and preprocess data with advanced techniques"""
        print("Loading TTF data...")

        try:
            df = pd.read_csv(file_path)
            print(f"Original dataset shape: {df.shape}")

            # Sampling logic
            if len(df) > sample_size:
                print(f"Dataset has {len(df)} rows. Sampling {sample_size} rows...")

                # Check if target column exists for stratified sampling
                if 'min_ttf_hours' in df.columns:
                    try:
                        # Remove any invalid values before binning
                        valid_mask = ~(df['min_ttf_hours'].isna() |
                                       np.isinf(df['min_ttf_hours']) |
                                       (df['min_ttf_hours'] <= 0))
                        df_valid = df[valid_mask].copy()

                        if len(df_valid) == 0:
                            print("Warning: No valid TTF values found. Using simple random sampling.")
                            df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
                        else:
                            # Create bins for stratified sampling
                            df_valid['ttf_bins'] = pd.cut(df_valid['min_ttf_hours'],
                                                          bins=[0, 6, 12, 24, 48, 96, np.inf],
                                                          labels=[0, 1, 2, 3, 4, 5])

                            # Sample from each bin
                            sampled_dfs = []
                            target_per_bin = sample_size // 6

                            for bin_val in [0, 1, 2, 3, 4, 5]:
                                bin_data = df_valid[df_valid['ttf_bins'] == bin_val]
                                if len(bin_data) > 0:
                                    sample_size_bin = min(len(bin_data), target_per_bin)
                                    sampled_bin = bin_data.sample(n=sample_size_bin, random_state=42)
                                    sampled_dfs.append(sampled_bin)

                            if sampled_dfs:
                                df = pd.concat(sampled_dfs, ignore_index=True)
                                df = df.drop('ttf_bins', axis=1)
                            else:
                                print("Warning: Stratified sampling failed. Using simple random sampling.")
                                df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

                    except Exception as e:
                        print(f"Warning: Stratified sampling failed ({e}). Using simple random sampling.")
                        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
                else:
                    # Simple random sampling if target column not available
                    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

                print(f"Sampled dataset shape: {df.shape}")

            print(f"Final dataset shape: {df.shape}")

            # Convert datetime with error handling
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

            # Encode categorical variables
            if 'model' in df.columns:
                try:
                    le = LabelEncoder()
                    df['model_encoded'] = le.fit_transform(df['model'].fillna('Unknown'))
                    self.label_encoders['model'] = le
                except Exception as e:
                    print(f"Warning: Error encoding model column: {e}")
                    df['model_encoded'] = 0

            # Apply advanced feature engineering with error handling
            print("Starting feature engineering...")
            try:
                df_engineered = self.advanced_feature_engineering(df, is_single_prediction=False)

                if df_engineered is None:
                    print("Error: Feature engineering returned None. Using original dataframe.")
                    return df

                print("Feature engineering completed successfully.")
                return df_engineered

            except Exception as e:
                print(f"Error in feature engineering: {e}")
                print("Returning original dataframe without advanced features.")
                return df

        except Exception as e:
            print(f"Error loading or preprocessing data: {e}")
            return None

    def intelligent_feature_selection(self, X, y, max_features=50):
        """Intelligent feature selection using multiple methods"""
        print("Performing intelligent feature selection...")

        # Remove constant and near-constant features
        feature_vars = X.var()
        variable_features = feature_vars[feature_vars > 1e-6].index
        X_filtered = X[variable_features]

        # Remove highly correlated features
        corr_matrix = X_filtered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X_filtered = X_filtered.drop(columns=high_corr_features)

        # Univariate feature selection
        selector_univariate = SelectKBest(score_func=f_regression, k=min(max_features * 2, len(X_filtered.columns)))
        X_univariate = selector_univariate.fit_transform(X_filtered, y)
        selected_features_univariate = X_filtered.columns[selector_univariate.get_support()]

        # Random Forest feature importance
        rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_selector.fit(X_filtered[selected_features_univariate], y)

        feature_importance = pd.DataFrame({
            'feature': selected_features_univariate,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)

        # Select top features
        top_features = feature_importance.head(max_features)['feature'].tolist()

        print(f"Selected {len(top_features)} features out of {len(X.columns)} original features")
        return top_features

    def prepare_features_target(self, df, target_column='min_ttf_hours'):
        """Comprehensive feature preparation"""
        # Get all potential features
        exclude_cols = ['datetime', 'machineID', target_column, 'model']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        print(f"Initial feature count: {len(feature_cols)}")

        X = df[feature_cols].copy()
        y = df[target_column].copy() if target_column in df.columns else None

        if y is not None:
            # Remove samples with invalid target values
            valid_mask = ~(y.isna() | np.isinf(y) | (y <= 0))
            X = X[valid_mask]
            y = y[valid_mask]

            # Create polynomial features
            X = self.create_polynomial_features(X)

            # Intelligent feature selection
            selected_features = self.intelligent_feature_selection(X, y, max_features=60)
            X = X[selected_features]
            self.feature_columns = selected_features
        else:
            self.feature_columns = feature_cols
            X = X[self.feature_columns]

        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)

        return X, y

    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models optimized for critical TTF prediction"""
        print("Training ensemble of models optimized for critical TTF...")

        # Enhanced models with focus on critical predictions
        models = {
            'RandomForest_Critical': RandomForestRegressor(
                n_estimators=300,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Critical': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,  # Lower learning rate for better precision
                subsample=0.8,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ),
            'ElasticNet_Critical': ElasticNet(
                alpha=0.05,  # Lower alpha for less regularization
                l1_ratio=0.3,
                random_state=42,
                max_iter=3000
            )
        }

        best_score = -np.inf
        best_model_name = None

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Clip negative predictions
            y_pred = np.maximum(y_pred, 0.1)

            r2 = r2_score(y_test, y_pred)

            # Special scoring for critical TTF (give more weight to accurate predictions under 24h)
            critical_mask = y_test <= 24
            if critical_mask.sum() > 0:
                critical_mae = mean_absolute_error(y_test[critical_mask], y_pred[critical_mask])
                # Adjusted score that heavily weights critical prediction accuracy
                adjusted_score = r2 - (critical_mae / 24) * 0.5
            else:
                adjusted_score = r2

            self.models[name] = {
                'model': model,
                'r2_score': r2,
                'adjusted_score': adjusted_score,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }

            print(f"  {name} R² Score: {r2:.4f}, Adjusted Score: {adjusted_score:.4f}")

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_model_name = name

        self.best_model = self.models[best_model_name]['model']
        print(f"\nBest model: {best_model_name} with Adjusted Score = {best_score:.4f}")

        return best_model_name

    def train_model(self, file_path, target='min_ttf_hours', sample_size=100000):
        """Train improved model with ensemble approach"""
        print("IMPROVED TTF PREDICTION MODEL")
        print("=" * 60)

        # Load data
        df = self.load_and_preprocess_data(file_path , sample_size)

        if target not in df.columns:
            print(f"Error: Target '{target}' not found in dataset")
            return None

        print(f"Training model for: {target}")

        X, y = self.prepare_features_target(df, target)

        if y is None or len(y) == 0:
            print(f"Error: No valid data for target '{target}'")
            return None

        # LESS AGGRESSIVE outlier removal - keep more data
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = max(0.1, Q1 - 3.0 * IQR)  # Changed from 1.5 to 3.0
        upper_bound = Q3 + 3.0 * IQR  # Changed from 2.0 to 3.0

        # OR completely remove outlier filtering:
        # outlier_mask = np.ones(len(y), dtype=bool)  # Keep all data
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        X, y = X[outlier_mask], y[outlier_mask]
        print(f"Training samples: {len(y)} (removed {(~outlier_mask).sum()} outliers)")

        # Stratified split to ensure critical TTF samples in both sets
        # Create bins for stratification
        y_bins = pd.cut(y, bins=[0, 6, 12, 24, 48, 96, np.inf], labels=[0, 1, 2, 3, 4, 5])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_bins, shuffle=True
        )

        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train ensemble models
        best_model_name = self.train_ensemble_models(X_train_scaled, y_train, X_test_scaled, y_test)

        # Make predictions with best model
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred = np.maximum(y_pred, 0.1)  # Ensure positive predictions

        # Evaluate the best model
        self.evaluate_model(y_test, y_pred, target, best_model_name)

        self.is_trained = True
        return True

    def evaluate_model(self, y_test, y_pred, target_name, model_name):
        """Enhanced model evaluation with focus on critical TTF"""
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

        print(f"\n{target_name.upper()} - BEST MODEL PERFORMANCE ({model_name}):")
        print("=" * 60)
        print(f"  R² Score: {r2:.4f} (Target: >0.70)")
        print(f"  RMSE: {rmse:.2f} hours ({rmse / 24:.1f} days)")
        print(f"  MAE: {mae:.2f} hours ({mae / 24:.1f} days)")
        print(f"  MAPE: {mape:.2f}%")

        # Critical TTF Analysis (≤24 hours)
        critical_mask = y_test <= 24
        if critical_mask.sum() > 0:
            critical_y_test = y_test[critical_mask]
            critical_y_pred = y_pred[critical_mask]
            critical_mae = mean_absolute_error(critical_y_test, critical_y_pred)
            critical_rmse = np.sqrt(mean_squared_error(critical_y_test, critical_y_pred))
            critical_r2 = r2_score(critical_y_test, critical_y_pred)

            print(f"\nCRITICAL TTF PERFORMANCE (≤24 hours, n={critical_mask.sum()}):")
            print(f"  R² Score: {critical_r2:.4f}")
            print(f"  RMSE: {critical_rmse:.2f} hours")
            print(f"  MAE: {critical_mae:.2f} hours")

        # Model comparison
        print(f"\nMODEL COMPARISON:")
        for name, metrics in self.models.items():
            print(f"  {name:25s}: R²={metrics['r2_score']:.4f}, Adj={metrics['adjusted_score']:.4f}, RMSE={metrics['rmse']:.2f}")

        # Enhanced error analysis
        errors = y_pred - y_test
        abs_errors = np.abs(errors)

        print(f"\nERROR ANALYSIS:")
        print(f"  Mean Error: {errors.mean():.2f} hours")
        print(f"  Std Error: {errors.std():.2f} hours")
        print(f"  95% of errors within: ±{np.percentile(abs_errors, 95):.2f} hours")
        print(f"  99% of errors within: ±{np.percentile(abs_errors, 99):.2f} hours")

        # Prediction accuracy by time windows (enhanced)
        print(f"\nPREDICTION ACCURACY:")
        for threshold in [3, 6, 12, 24, 48, 72]:
            within_threshold = np.sum(abs_errors <= threshold) / len(errors) * 100
            print(f"  Within ±{threshold:2d}h: {within_threshold:5.1f}%")

        # Critical TTF specific accuracy
        if critical_mask.sum() > 0:
            critical_abs_errors = np.abs(critical_y_pred - critical_y_test)
            print(f"\nCRITICAL TTF ACCURACY (≤24h):")
            for threshold in [1, 2, 3, 6, 12]:
                within_threshold = np.sum(critical_abs_errors <= threshold) / len(critical_abs_errors) * 100
                print(f"  Within ±{threshold:2d}h: {within_threshold:5.1f}%")

        # Feature importance (top 15)
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:30s}: {row['importance']:.4f}")

    def predict(self, data_dict):
        """Make single prediction with improved preprocessing"""
        if not self.is_trained:
            print("Model not trained yet")
            return None

        # Create DataFrame
        df = pd.DataFrame([data_dict])

        # Apply same preprocessing as training (with single prediction flag)
        df = self.advanced_feature_engineering(df, is_single_prediction=True)

        # Encode categorical variables
        if 'model' in df.columns and 'model' in self.label_encoders:
            try:
                df['model_encoded'] = self.label_encoders['model'].transform(df['model'])
            except ValueError:
                df['model_encoded'] = 0

        # Add polynomial features if they were used
        if self.poly_features:
            base_features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'error_count']
            available_base = [col for col in base_features if col in df.columns]
            if len(available_base) >= 2:
                poly_data = self.poly_features.transform(df[available_base])
                poly_df = pd.DataFrame(poly_data, columns=self.poly_features.get_feature_names_out(available_base), index=df.index)
                for col in poly_df.columns:
                    if col not in df.columns:
                        df[col] = poly_df[col]

        # Add missing features with default values
        for feature in self.feature_columns:
            if feature not in df.columns:
                df[feature] = 0

        # Select only the features used during training
        X = df[self.feature_columns].copy()

        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        X = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make prediction
        prediction = self.best_model.predict(X_scaled)[0]
        prediction = max(prediction, 0.1)  # Ensure positive prediction

        # Calculate confidence intervals using ensemble predictions
        ensemble_predictions = []
        for model_info in self.models.values():
            pred = model_info['model'].predict(X_scaled)[0]
            ensemble_predictions.append(max(pred, 0.1))

        ensemble_mean = np.mean(ensemble_predictions)
        ensemble_std = np.std(ensemble_predictions)

        # Create confidence interval
        confidence_interval = {
            'lower_bound': max(ensemble_mean - 1.96 * ensemble_std, 0.1),
            'upper_bound': ensemble_mean + 1.96 * ensemble_std
        }

        # Risk assessment
        risk_level = "LOW"
        if prediction <= 6:
            risk_level = "CRITICAL"
        elif prediction <= 24:
            risk_level = "HIGH"
        elif prediction <= 72:
            risk_level = "MEDIUM"

        return {
            'predicted_ttf_hours': round(prediction, 2),
            'predicted_ttf_days': round(prediction / 24, 2),
            'confidence_interval': confidence_interval,
            'risk_level': risk_level,
            'ensemble_mean': round(ensemble_mean, 2),
            'prediction_std': round(ensemble_std, 2)
        }

    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return "Model not trained yet"

        info = {
            'is_trained': self.is_trained,
            'num_features': len(self.feature_columns) if self.feature_columns else 0,
            'models_trained': list(self.models.keys()),
            'best_model': type(self.best_model).__name__ if self.best_model else None
        }

        if self.models:
            info['model_scores'] = {name: metrics['r2_score'] for name, metrics in self.models.items()}

        return info

    # Usage example
def main():
        """Example usage of the ImprovedTTFPredictor"""

        # Initialize predictor
        predictor = ImprovedTTFPredictor()

        # Train the model (replace with your actual data file path)
        print("Training model...")
        success = predictor.train_model('Advanced_TTF_Dataset.csv', target='min_ttf_hours', sample_size=100000)

        if success:
            print("\nModel training completed successfully!")

            # Get model information
            model_info = predictor.get_model_info()
            print(f"\nModel Info: {model_info}")

            # Make a prediction
            sample_data = {
                'machineID': 1,
                'datetime': '2024-01-15 10:30:00',
                'age': 45.2,
                'model': 'ModelA',
                'volt': 168.5,
                'rotate': 438.2,
                'pressure': 98.7,
                'vibration': 42.1,
                'error_count': 2
            }

            print("\nMaking prediction...")
            result = predictor.predict(sample_data)

            if result:
                print(f"\nPREDICTION RESULTS:")
                print(f"Predicted TTF: {result['predicted_ttf_hours']} hours ({result['predicted_ttf_days']} days)")
                print(f"Risk Level: {result['risk_level']}")
                print(
                    f"Confidence Interval: {result['confidence_interval']['lower_bound']:.1f} - {result['confidence_interval']['upper_bound']:.1f} hours")
                print(f"Ensemble Mean: {result['ensemble_mean']} hours")
                print(f"Prediction Std: {result['prediction_std']} hours")

        else:
            print("Model training failed!")


if __name__ == "__main__":
    main()