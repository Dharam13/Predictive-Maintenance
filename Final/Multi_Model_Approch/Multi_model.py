import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class MultiModelTTFPredictor:
    def __init__(self, model_dir='saved_models'):
        # Critical model for ≤168 hours
        self.critical_models = {}
        self.critical_scalers = {}
        self.critical_features = {}

        # Long-term model for >168 hours
        self.longterm_models = {}
        self.longterm_scalers = {}
        self.longterm_features = {}

        # Classification model to decide which model to use
        self.classifier_models = {}
        self.classifier_scalers = {}
        self.classifier_features = {}

        self.evaluation_metrics = {}
        self.is_trained = False
        self.components = ['ttf_comp1_hours', 'ttf_comp2_hours', 'ttf_comp3_hours', 'ttf_comp4_hours']
        self.model_dir = model_dir
        self.ttf_threshold = 168  # 7 days

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_models(self):
        """Save all trained models to disk"""
        if not self.is_trained:
            print("No trained models to save!")
            return False

        try:
            model_data = {
                'critical_models': self.critical_models,
                'critical_scalers': self.critical_scalers,
                'critical_features': self.critical_features,
                'longterm_models': self.longterm_models,
                'longterm_scalers': self.longterm_scalers,
                'longterm_features': self.longterm_features,
                'classifier_models': self.classifier_models,
                'classifier_scalers': self.classifier_scalers,
                'classifier_features': self.classifier_features,
                'evaluation_metrics': self.evaluation_metrics,
                'is_trained': self.is_trained,
                'components': self.components,
                'ttf_threshold': self.ttf_threshold,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            model_path = os.path.join(self.model_dir, 'multi_model_ttf_predictor.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"✅ Multi-model system saved successfully to {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False

    def load_models(self):
        """Load trained models from disk"""
        model_path = os.path.join(self.model_dir, 'multi_model_ttf_predictor.pkl')

        if not os.path.exists(model_path):
            print(f"❌ No saved multi-model found at {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.critical_models = model_data['critical_models']
            self.critical_scalers = model_data['critical_scalers']
            self.critical_features = model_data['critical_features']
            self.longterm_models = model_data['longterm_models']
            self.longterm_scalers = model_data['longterm_scalers']
            self.longterm_features = model_data['longterm_features']
            self.classifier_models = model_data['classifier_models']
            self.classifier_scalers = model_data['classifier_scalers']
            self.classifier_features = model_data['classifier_features']
            self.evaluation_metrics = model_data['evaluation_metrics']
            self.is_trained = model_data['is_trained']
            self.components = model_data['components']
            self.ttf_threshold = model_data.get('ttf_threshold', 168)

            timestamp = model_data.get('timestamp', 'Unknown')
            print(f"✅ Multi-model system loaded successfully (trained on: {timestamp})")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False

    def enhanced_feature_engineering(self, df, target_component=None):
        """Enhanced feature engineering with domain knowledge"""
        df = df.copy()

        # Handle single prediction case
        if 'machineID' not in df.columns:
            df['machineID'] = 1

        # Encode categorical variables
        if 'model' in df.columns:
            df['model'] = df['model'].astype('category').cat.codes

        # Time-based features
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['quarter'] = df['datetime'].dt.quarter
            df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
            df['is_peak_load'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)

        # Sensor health indicators
        sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
        available_sensors = [col for col in sensor_cols if col in df.columns]

        if len(available_sensors) == 4:
            # Normalized deviations from optimal ranges
            df['volt_health'] = 1 - np.abs(df['volt'] - 170) / 170
            df['rotate_health'] = 1 - np.abs(df['rotate'] - 440) / 440
            df['pressure_health'] = 1 - np.abs(df['pressure'] - 100) / 100
            df['vibration_health'] = np.maximum(0, 1 - (df['vibration'] - 35) / 35)

            # Composite health score
            df['overall_health'] = (df['volt_health'] + df['rotate_health'] +
                                    df['pressure_health'] + df['vibration_health']) / 4

            # Degradation indicators
            df['sensor_degradation'] = 1 - df['overall_health']
            df['critical_sensor_count'] = (
                    (df['volt_health'] < 0.8).astype(int) +
                    (df['rotate_health'] < 0.8).astype(int) +
                    (df['pressure_health'] < 0.8).astype(int) +
                    (df['vibration_health'] < 0.8).astype(int)
            )

        # Enhanced maintenance features
        maint_cols = [f'days_since_comp{i}_maint' for i in range(1, 5)]
        available_maint = [col for col in maint_cols if col in df.columns]

        for col in available_maint:
            if col in df.columns:
                # Non-linear transformations for long-term modeling
                df[f'{col}_sqrt'] = np.sqrt(df[col])
                df[f'{col}_log'] = np.log1p(df[col])
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_inv'] = 1 / (df[col] + 1)

                # Age-based degradation modeling
                df[f'{col}_aging_factor'] = np.exp(df[col] / 100)  # Exponential aging
                df[f'{col}_wear_level'] = np.minimum(df[col] / 365, 1.0)  # Annual wear cycle

        # Cross-component analysis for long-term predictions
        if len(available_maint) >= 2:
            df['maint_synchronization'] = df[available_maint].std(axis=1)
            df['maint_imbalance'] = df[available_maint].max(axis=1) - df[available_maint].min(axis=1)
            df['avg_maint_age'] = df[available_maint].mean(axis=1)
            df['max_maint_age'] = df[available_maint].max(axis=1)

            # Long-term degradation patterns
            df['system_wear_index'] = df['avg_maint_age'] / 30
            df['degradation_acceleration'] = df['maint_imbalance'] * df['sensor_degradation']

        # Age-related features for long-term modeling
        if 'age' in df.columns:
            df['age_normalized'] = df['age'] / 100
            df['age_decades'] = df['age'] // 10
            df['age_wear_factor'] = (df['age'] / 100) ** 1.5
            df['age_reliability'] = np.exp(-df['age'] / 200)

        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.endswith('_health') or col.endswith('_reliability'):
                df[col] = df[col].fillna(0.5)
            else:
                df[col] = df[col].fillna(df[col].median())

        df = df.fillna(0)
        return df

    def balanced_sampling_for_critical(self, df, component_col, max_samples=50000):
        """Sampling strategy optimized for critical cases (≤168h)"""
        valid_mask = ~(df[component_col].isna() | np.isinf(df[component_col]) | (df[component_col] <= 0))
        df_valid = df[valid_mask].copy()

        # Filter for critical cases only
        critical_df = df_valid[df_valid[component_col] <= self.ttf_threshold].copy()

        if len(critical_df) == 0:
            return pd.DataFrame()

        # Define bins for critical cases
        bins = [0, 2, 4, 8, 12, 24, 48, 96, self.ttf_threshold]
        labels = ['0-2h', '2-4h', '4-8h', '8-12h', '12-24h', '24-48h', '48-96h', '96-168h']

        critical_df['ttf_range'] = pd.cut(critical_df[component_col], bins=bins, labels=labels)

        # More balanced sampling for critical cases
        sampling_strategy = {
            '0-2h': 0.20,
            '2-4h': 0.18,
            '4-8h': 0.15,
            '8-12h': 0.12,
            '12-24h': 0.12,
            '24-48h': 0.10,
            '48-96h': 0.08,
            '96-168h': 0.05
        }

        sampled_dfs = []
        for range_name, proportion in sampling_strategy.items():
            range_data = critical_df[critical_df['ttf_range'] == range_name]
            target_samples = int(max_samples * proportion)

            if len(range_data) > 0:
                if len(range_data) >= target_samples:
                    sampled = range_data.sample(n=target_samples, random_state=42)
                else:
                    sampled = range_data.copy()
                sampled_dfs.append(sampled)

        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True).drop('ttf_range', axis=1)
            return result
        return pd.DataFrame()

    def balanced_sampling_for_longterm(self, df, component_col, max_samples=30000):
        """Sampling strategy optimized for long-term cases (>168h)"""
        valid_mask = ~(df[component_col].isna() | np.isinf(df[component_col]) | (df[component_col] <= 0))
        df_valid = df[valid_mask].copy()

        # Filter for long-term cases only
        longterm_df = df_valid[df_valid[component_col] > self.ttf_threshold].copy()

        if len(longterm_df) == 0:
            return pd.DataFrame()

        # Define bins for long-term cases with more granular sampling
        bins = [self.ttf_threshold, 336, 720, 1440, 2160, 4320, 8760, np.inf]
        labels = ['168-336h', '336-720h', '720-1440h', '1440-2160h', '2160-4320h', '4320-8760h', '8760h+']

        longterm_df['ttf_range'] = pd.cut(longterm_df[component_col], bins=bins, labels=labels)

        # More balanced sampling for long-term cases
        sampling_strategy = {
            '168-336h': 0.30,  # 1-2 weeks
            '336-720h': 0.25,  # 2-4 weeks
            '720-1440h': 0.20,  # 1-2 months
            '1440-2160h': 0.12,  # 2-3 months
            '2160-4320h': 0.08,  # 3-6 months
            '4320-8760h': 0.04,  # 6-12 months
            '8760h+': 0.01  # >1 year
        }

        sampled_dfs = []
        for range_name, proportion in sampling_strategy.items():
            range_data = longterm_df[longterm_df['ttf_range'] == range_name]
            target_samples = int(max_samples * proportion)

            if len(range_data) > 0:
                if len(range_data) >= target_samples:
                    sampled = range_data.sample(n=target_samples, random_state=42)
                else:
                    sampled = range_data.copy()
                sampled_dfs.append(sampled)

        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True).drop('ttf_range', axis=1)
            return result
        return pd.DataFrame()

    def prepare_features(self, df, target_component):
        """Prepare features excluding target and future information"""
        exclude_cols = [
            'datetime', 'machineID', 'failure_within_48h',
            'ttf_comp1_hours', 'ttf_comp2_hours', 'ttf_comp3_hours', 'ttf_comp4_hours'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_component].copy() if target_component in df.columns else None

        if y is not None:
            valid_mask = ~(y.isna() | np.isinf(y) | (y <= 0))
            X = X[valid_mask]
            y = y[valid_mask]

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)

        return X, y

    def create_critical_models(self):
        """Models optimized for critical cases (≤168h)"""
        return {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42
            ),
            'elastic_net': ElasticNet(
                alpha=0.05,
                l1_ratio=0.7,
                random_state=42
            )
        }

    def create_longterm_models(self):
        """Models optimized for long-term cases (>168h)"""
        return {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.02,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.9,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='log2',
                random_state=42
            ),
            'ridge': Ridge(
                alpha=10.0,
                random_state=42
            ),
            'linear': LinearRegression()
        }

    def create_classifier_models(self):
        """Models to classify critical vs long-term cases"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }

    def train_classifier(self, df, component_col):
        """Train classifier to distinguish critical vs long-term cases"""
        # Prepare data
        df_processed = self.enhanced_feature_engineering(df)
        X, y = self.prepare_features(df_processed, component_col)

        # Create binary classification target
        y_binary = (y <= self.ttf_threshold).astype(int)  # 1 for critical, 0 for long-term

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        models = self.create_classifier_models()
        trained_models = {}
        best_score = 0
        best_model = None

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            trained_models[name] = model

            if score > best_score:
                best_score = score
                best_model = name

        # Select best features (top 20)
        if best_model == 'random_forest':
            importances = trained_models[best_model].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            selected_features = feature_importance.head(20)['feature'].tolist()
        else:
            # Use all features for logistic regression
            selected_features = X.columns.tolist()[:20]

        return {
            'models': trained_models,
            'best_model': best_model,
            'scaler': scaler,
            'features': selected_features,
            'accuracy': best_score
        }

    def train_models(self, file_path, max_critical_samples=50000, max_longterm_samples=30000):
        """Train multi-model system"""
        print("🔄 Multi-Model TTF Prediction System")
        print("=" * 50)

        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape}")

        for component in self.components:
            if component not in df.columns:
                continue

            print(f"\n🔧 Training models for {component.upper()}")
            print("-" * 40)

            # 1. Train classifier
            print("📊 Training classification model...")
            classifier_info = self.train_classifier(df, component)
            self.classifier_models[component] = classifier_info
            self.classifier_scalers[component] = classifier_info['scaler']
            self.classifier_features[component] = classifier_info['features']
            print(f"   Classifier accuracy: {classifier_info['accuracy']:.3f}")

            # 2. Train critical model (≤168h)
            print("🚨 Training critical cases model...")
            critical_df = self.balanced_sampling_for_critical(df, component, max_critical_samples)
            if len(critical_df) > 0:
                critical_df = self.enhanced_feature_engineering(critical_df, component)
                X_crit, y_crit = self.prepare_features(critical_df, component)

                # Only keep critical cases
                critical_mask = y_crit <= self.ttf_threshold
                X_crit = X_crit[critical_mask]
                y_crit = y_crit[critical_mask]

                if len(y_crit) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_crit, y_crit, test_size=0.2, random_state=42
                    )

                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    models = self.create_critical_models()
                    trained_models = {}
                    predictions = {}

                    for name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        pred = model.predict(X_test_scaled)
                        pred = np.maximum(pred, 0.1)
                        trained_models[name] = model
                        predictions[name] = pred

                    # Calculate weights
                    weights = {}
                    for name, pred in predictions.items():
                        r2 = r2_score(y_test, pred)
                        weights[name] = max(0, r2)

                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v / total_weight for k, v in weights.items()}
                    else:
                        weights = {k: 1 / len(weights) for k in weights.keys()}

                    # Ensemble prediction
                    ensemble_pred = np.zeros(len(X_test))
                    for name, weight in weights.items():
                        ensemble_pred += weight * predictions[name]

                    critical_r2 = r2_score(y_test, ensemble_pred)
                    critical_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                    critical_mae = mean_absolute_error(y_test, ensemble_pred)

                    self.critical_models[component] = {
                        'models': trained_models,
                        'weights': weights,
                        'r2_score': critical_r2,
                        'rmse': critical_rmse,
                        'mae': critical_mae
                    }
                    self.critical_scalers[component] = scaler
                    self.critical_features[component] = X_crit.columns.tolist()

                    print(f"   Critical model R²: {critical_r2:.3f}, RMSE: {critical_rmse:.1f}h")

            # 3. Train long-term model (>168h)
            print("🟢 Training long-term cases model...")
            longterm_df = self.balanced_sampling_for_longterm(df, component, max_longterm_samples)
            if len(longterm_df) > 0:
                longterm_df = self.enhanced_feature_engineering(longterm_df, component)
                X_long, y_long = self.prepare_features(longterm_df, component)

                # Only keep long-term cases
                longterm_mask = y_long > self.ttf_threshold
                X_long = X_long[longterm_mask]
                y_long = y_long[longterm_mask]

                if len(y_long) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_long, y_long, test_size=0.2, random_state=42
                    )

                    scaler = StandardScaler()  # Different scaler for long-term
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    models = self.create_longterm_models()
                    trained_models = {}
                    predictions = {}

                    for name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        pred = model.predict(X_test_scaled)
                        pred = np.maximum(pred, self.ttf_threshold + 1)  # Ensure > threshold
                        trained_models[name] = model
                        predictions[name] = pred

                    # Calculate weights
                    weights = {}
                    for name, pred in predictions.items():
                        r2 = r2_score(y_test, pred)
                        weights[name] = max(0, r2)

                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v / total_weight for k, v in weights.items()}
                    else:
                        weights = {k: 1 / len(weights) for k in weights.keys()}

                    # Ensemble prediction
                    ensemble_pred = np.zeros(len(X_test))
                    for name, weight in weights.items():
                        ensemble_pred += weight * predictions[name]

                    longterm_r2 = r2_score(y_test, ensemble_pred)
                    longterm_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                    longterm_mae = mean_absolute_error(y_test, ensemble_pred)

                    self.longterm_models[component] = {
                        'models': trained_models,
                        'weights': weights,
                        'r2_score': longterm_r2,
                        'rmse': longterm_rmse,
                        'mae': longterm_mae
                    }
                    self.longterm_scalers[component] = scaler
                    self.longterm_features[component] = X_long.columns.tolist()

                    print(f"   Long-term model R²: {longterm_r2:.3f}, RMSE: {longterm_rmse:.1f}h")

        self.is_trained = True
        print(f"\n✅ Multi-model training completed!")
        self.save_models()
        return True

    def predict_all_components(self, data_dict):
        """Predict TTF using multi-model approach"""
        if not self.is_trained:
            return None

        df = pd.DataFrame([data_dict])
        predictions = {}

        for component in self.components:
            if component not in self.classifier_models:
                continue

            # Feature engineering
            df_processed = self.enhanced_feature_engineering(df, component)

            # 1. Classify as critical or long-term
            classifier_info = self.classifier_models[component]
            classifier_features = self.classifier_features[component]
            classifier_scaler = self.classifier_scalers[component]

            # Add missing features
            for feature in classifier_features:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0

            X_class = df_processed[classifier_features].fillna(0)
            X_class_scaled = classifier_scaler.transform(X_class)

            # Get classification probability
            best_classifier = classifier_info['models'][classifier_info['best_model']]
            is_critical_prob = best_classifier.predict_proba(X_class_scaled)[0][1]
            is_critical = is_critical_prob > 0.5

            # 2. Use appropriate model for prediction
            if is_critical and component in self.critical_models:
                # Use critical model
                model_info = self.critical_models[component]
                features = self.critical_features[component]
                scaler = self.critical_scalers[component]
                model_type = "Critical"
            elif not is_critical and component in self.longterm_models:
                # Use long-term model
                model_info = self.longterm_models[component]
                features = self.longterm_features[component]
                scaler = self.longterm_scalers[component]
                model_type = "Long-term"
            else:
                # Fallback to critical model if available
                if component in self.critical_models:
                    model_info = self.critical_models[component]
                    features = self.critical_features[component]
                    scaler = self.critical_scalers[component]
                    model_type = "Critical (fallback)"
                else:
                    continue

            # Add missing features
            for feature in features:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0

            X = df_processed[features].fillna(0)
            X_scaled = scaler.transform(X)

            # Ensemble prediction
            ensemble_pred = 0
            for model_name, model in model_info['models'].items():
                weight = model_info['weights'][model_name]
                pred = model.predict(X_scaled)[0]
                ensemble_pred += weight * pred

            ensemble_pred = max(ensemble_pred, 0.1)

            # Risk assessment
            if ensemble_pred <= 4:
                risk = "CRITICAL"
            elif ensemble_pred <= 12:
                risk = "HIGH"
            elif ensemble_pred <= 48:
                risk = "MEDIUM"
            elif ensemble_pred <= 168:
                risk = "LOW"
            else:
                risk = "VERY_LOW"

            predictions[component] = {
                'ttf_hours': round(ensemble_pred, 2),
                'ttf_days': round(ensemble_pred / 24, 2),
                'risk_level': risk,
                'model_used': model_type,
                'classification_confidence': round(is_critical_prob if is_critical else 1 - is_critical_prob, 3),
                'model_r2': round(model_info.get('r2_score', 0), 3),
                'model_rmse': round(model_info.get('rmse', 0), 1)
            }

        return predictions


    def predict_single_component(self, data_dict, component):
        """Predict TTF for a single component"""
        if not self.is_trained or component not in self.components:
            return None

        predictions = self.predict_all_components(data_dict)
        return predictions.get(component, None)

    def get_system_health_summary(self, data_dict):
        """Get overall system health summary"""
        predictions = self.predict_all_components(data_dict)
        if not predictions:
            return None

        # Find the component with minimum TTF
        min_ttf = float('inf')
        critical_component = None
        risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'VERY_LOW': 0}

        for comp, pred in predictions.items():
            ttf = pred['ttf_hours']
            risk = pred['risk_level']
            risk_counts[risk] += 1

            if ttf < min_ttf:
                min_ttf = ttf
                critical_component = comp

        # Overall system risk
        if risk_counts['CRITICAL'] > 0:
            system_risk = 'CRITICAL'
        elif risk_counts['HIGH'] > 0:
            system_risk = 'HIGH'
        elif risk_counts['MEDIUM'] > 0:
            system_risk = 'MEDIUM'
        elif risk_counts['LOW'] > 0:
            system_risk = 'LOW'
        else:
            system_risk = 'VERY_LOW'

        return {
            'system_ttf_hours': round(min_ttf, 2),
            'system_ttf_days': round(min_ttf / 24, 2),
            'system_risk': system_risk,
            'critical_component': critical_component,
            'risk_distribution': risk_counts,
            'components_at_risk': sum([risk_counts['CRITICAL'], risk_counts['HIGH']]),
            'total_components': len(predictions),
            'predictions': predictions
        }

    def print_evaluation_metrics(self):
        """Print evaluation metrics for all trained models"""
        if not self.is_trained:
            print("❌ No trained models to evaluate!")
            return

        print("\n📊 MODEL EVALUATION SUMMARY")
        print("=" * 60)

        for component in self.components:
            print(f"\n🔧 {component.upper()}")
            print("-" * 40)

            # Classifier metrics
            if component in self.classifier_models:
                classifier_acc = self.classifier_models[component]['accuracy']
                print(f"📊 Classifier Accuracy: {classifier_acc:.3f}")

            # Critical model metrics
            if component in self.critical_models:
                critical_info = self.critical_models[component]
                print(f"🚨 Critical Model (≤168h):")
                print(f"   R² Score: {critical_info['r2_score']:.3f}")
                print(f"   RMSE: {critical_info['rmse']:.1f} hours")
                print(f"   MAE: {critical_info['mae']:.1f} hours")

            # Long-term model metrics
            if component in self.longterm_models:
                longterm_info = self.longterm_models[component]
                print(f"🟢 Long-term Model (>168h):")
                print(f"   R² Score: {longterm_info['r2_score']:.3f}")
                print(f"   RMSE: {longterm_info['rmse']:.1f} hours")
                print(f"   MAE: {longterm_info['mae']:.1f} hours")

    def get_feature_importance(self, component=None):
        """Get feature importance for specified component(s)"""
        if not self.is_trained:
            return None

        importance_data = {}
        components_to_check = [component] if component else self.components

        for comp in components_to_check:
            if comp not in self.critical_models and comp not in self.longterm_models:
                continue

            comp_importance = {}

            # Critical model feature importance
            if comp in self.critical_models:
                critical_models = self.critical_models[comp]['models']
                if 'random_forest' in critical_models:
                    rf_model = critical_models['random_forest']
                    features = self.critical_features[comp]
                    importances = rf_model.feature_importances_
                    comp_importance['critical'] = dict(zip(features, importances))

            # Long-term model feature importance
            if comp in self.longterm_models:
                longterm_models = self.longterm_models[comp]['models']
                if 'random_forest' in longterm_models:
                    rf_model = longterm_models['random_forest']
                    features = self.longterm_features[comp]
                    importances = rf_model.feature_importances_
                    comp_importance['longterm'] = dict(zip(features, importances))

            if comp_importance:
                importance_data[comp] = comp_importance

        return importance_data

    def maintenance_recommendations(self, data_dict, threshold_hours=48):
        """Generate maintenance recommendations based on predictions"""
        predictions = self.predict_all_components(data_dict)
        if not predictions:
            return None

        recommendations = []

        for component, pred in predictions.items():
            ttf_hours = pred['ttf_hours']
            risk_level = pred['risk_level']

            if ttf_hours <= threshold_hours:
                if risk_level == 'CRITICAL':
                    priority = 'IMMEDIATE'
                    action = f"URGENT: Schedule immediate maintenance for {component}. Predicted failure in {ttf_hours:.1f} hours."
                elif risk_level == 'HIGH':
                    priority = 'HIGH'
                    action = f"Schedule maintenance for {component} within 24 hours. Predicted failure in {ttf_hours:.1f} hours."
                else:
                    priority = 'MEDIUM'
                    action = f"Plan maintenance for {component} within 48 hours. Predicted failure in {ttf_hours:.1f} hours."

                recommendations.append({
                    'component': component,
                    'priority': priority,
                    'ttf_hours': ttf_hours,
                    'risk_level': risk_level,
                    'action': action
                })

        # Sort by priority and TTF
        priority_order = {'IMMEDIATE': 0, 'HIGH': 1, 'MEDIUM': 2}
        recommendations.sort(key=lambda x: (priority_order[x['priority']], x['ttf_hours']))

        return recommendations

def example_prediction():
    """Example of how to use the MultiModelTTFPredictor"""

    # Initialize the predictor
    predictor = MultiModelTTFPredictor()

    # Try to load existing models
    if not predictor.load_models():
        print("No existing models found. Train the models first using:")
        print("predictor.train_models('your_data.csv')")
        return

    # Example machine data
    sample_data = {
        'volt': 165.2,
        'rotate': 435.8,
        'pressure': 98.5,
        'vibration': 42.1,
        'age': 85,
        'model': 'model1',
        'days_since_comp1_maint': 15,
        'days_since_comp2_maint': 8,
        'days_since_comp3_maint': 22,
        'days_since_comp4_maint': 12
    }

    # Get predictions for all components
    print("🔮 TTF PREDICTIONS")
    print("=" * 50)
    predictions = predictor.predict_all_components(sample_data)

    if predictions:
        for component, pred in predictions.items():
            print(f"\n🔧 {component.upper()}")
            print(f"   TTF: {pred['ttf_hours']} hours ({pred['ttf_days']} days)")
            print(f"   Risk Level: {pred['risk_level']}")
            print(f"   Model Used: {pred['model_used']}")
            print(f"   Confidence: {pred['classification_confidence']}")

    # Get system health summary
    print("\n🏥 SYSTEM HEALTH SUMMARY")
    print("=" * 50)
    health_summary = predictor.get_system_health_summary(sample_data)

    if health_summary:
        print(f"System TTF: {health_summary['system_ttf_hours']} hours")
        print(f"System Risk: {health_summary['system_risk']}")
        print(f"Critical Component: {health_summary['critical_component']}")
        print(
            f"Components at Risk: {health_summary['components_at_risk']}/{health_summary['total_components']}")

    # Get maintenance recommendations
    print("\n🔧 MAINTENANCE RECOMMENDATIONS")
    print("=" * 50)
    recommendations = predictor.maintenance_recommendations(sample_data)

    if recommendations:
        for rec in recommendations:
            print(f"• {rec['action']}")
    else:
        print("No immediate maintenance required.")

def train_new_model(data_path):
    """Train a new multi-model system"""
    predictor = MultiModelTTFPredictor()

    print("Starting training process...")
    success = predictor.train_models(data_path)

    if success:
        print("\n📊 Training completed! Evaluation metrics:")
        predictor.print_evaluation_metrics()
    else:
        print("Training failed!")

if __name__ == "__main__":
            #Example usage
    #example_prediction()

            #Uncomment to train new models
    train_new_model('Fixed_Advanced_TTF_Dataset.csv')