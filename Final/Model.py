import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class EnhancedComponentTTFPredictor:
    def __init__(self, model_dir='saved_models'):
        self.component_models = {}
        self.component_scalers = {}
        self.component_features = {}
        self.evaluation_metrics = {}
        self.is_trained = False
        self.components = ['ttf_comp1_hours', 'ttf_comp2_hours', 'ttf_comp3_hours', 'ttf_comp4_hours']
        self.model_dir = model_dir

        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_models(self):
        """Save trained models to disk"""
        if not self.is_trained:
            print("No trained models to save!")
            return False

        try:
            model_data = {
                'component_models': self.component_models,
                'component_scalers': self.component_scalers,
                'component_features': self.component_features,
                'evaluation_metrics': self.evaluation_metrics,
                'is_trained': self.is_trained,
                'components': self.components,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            model_path = os.path.join(self.model_dir, 'ttf_predictor_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"✅ Models saved successfully to {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False

    def load_models(self):
        """Load trained models from disk"""
        model_path = os.path.join(self.model_dir, 'ttf_predictor_model.pkl')

        if not os.path.exists(model_path):
            print(f"❌ No saved model found at {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.component_models = model_data['component_models']
            self.component_scalers = model_data['component_scalers']
            self.component_features = model_data['component_features']
            self.evaluation_metrics = model_data['evaluation_metrics']
            self.is_trained = model_data['is_trained']
            self.components = model_data['components']

            timestamp = model_data.get('timestamp', 'Unknown')
            print(f"✅ Models loaded successfully (trained on: {timestamp})")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False

    def print_comprehensive_evaluation(self):
        """Print comprehensive evaluation metrics for all components"""
        if not self.evaluation_metrics:
            print("❌ No evaluation metrics available!")
            return

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 80)

        # Overall summary table
        print("\n📋 OVERALL PERFORMANCE SUMMARY")
        print("-" * 65)
        print(f"{'Component':<12} {'R²':<8} {'RMSE(h)':<10} {'MAE(h)':<10} {'Samples':<10}")
        print("-" * 65)

        for component in self.components:
            if component in self.evaluation_metrics:
                metrics = self.evaluation_metrics[component]
                comp_num = component.replace('ttf_comp', '').replace('_hours', '')
                print(f"{'Comp ' + comp_num:<12} {metrics['r2_score']:<8.3f} "
                      f"{metrics['rmse']:<10.1f} {metrics['mae']:<10.1f} "
                      f"{metrics['total_samples']:<10}")

        # Detailed metrics for each component
        for component in self.components:
            if component not in self.evaluation_metrics:
                continue

            metrics = self.evaluation_metrics[component]
            comp_num = component.replace('ttf_comp', '').replace('_hours', '')

            print(f"\n🔧 COMPONENT {comp_num} DETAILED ANALYSIS")
            print("-" * 50)

            # Overall Performance
            print("📈 Overall Performance:")
            print(f"   • R² Score:           {metrics['r2_score']:.3f}")
            print(f"   • RMSE:              {metrics['rmse']:.1f} hours ({metrics['rmse'] / 24:.1f} days)")
            print(f"   • MAE:               {metrics['mae']:.1f} hours ({metrics['mae'] / 24:.1f} days)")
            print(f"   • Total Samples:     {metrics['total_samples']:,}")

            # Critical Cases Performance (≤24 hours)
            print(f"\n🚨 Critical Cases Performance (≤24h):")
            print(f"   • Critical R²:       {metrics['critical_r2']:.3f}")
            print(f"   • Critical MAE:      {metrics['critical_mae']:.1f} hours")
            print(f"   • Critical RMSE:     {metrics['critical_rmse']:.1f} hours")
            print(f"   • Critical Samples:  {metrics['critical_samples']:,}")

            # High Risk Cases Performance (≤48 hours)
            print(f"\n⚠️  High Risk Cases Performance (≤48h):")
            print(f"   • High Risk R²:      {metrics['high_risk_r2']:.3f}")
            print(f"   • High Risk MAE:     {metrics['high_risk_mae']:.1f} hours")
            print(f"   • High Risk RMSE:    {metrics['high_risk_rmse']:.1f} hours")
            print(f"   • High Risk Samples: {metrics['high_risk_samples']:,}")

            # Medium Risk Cases Performance (≤168 hours / 7 days)
            print(f"\n🟡 Medium Risk Cases Performance (≤168h):")
            print(f"   • Medium Risk R²:    {metrics['medium_risk_r2']:.3f}")
            print(f"   • Medium Risk MAE:   {metrics['medium_risk_mae']:.1f} hours")
            print(f"   • Medium Risk RMSE:  {metrics['medium_risk_rmse']:.1f} hours")
            print(f"   • Medium Risk Samples: {metrics['medium_risk_samples']:,}")

            # Long Term Cases Performance (>168 hours)
            print(f"\n🟢 Long Term Cases Performance (>168h):")
            print(f"   • Long Term R²:      {metrics['long_term_r2']:.3f}")
            print(f"   • Long Term MAE:     {metrics['long_term_mae']:.1f} hours")
            print(f"   • Long Term RMSE:    {metrics['long_term_rmse']:.1f} hours")
            print(f"   • Long Term Samples: {metrics['long_term_samples']:,}")

            # Model Ensemble Details
            print(f"\n🤖 Ensemble Model Weights:")
            for model_name, weight in metrics['weights'].items():
                print(f"   • {model_name.replace('_', ' ').title()}: {weight:.3f}")

    def create_survival_features(self, df, component_col):
        """Create survival analysis inspired features"""
        df = df.copy()

        # Hazard rate features
        for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
            maint_col = f'days_since_{comp}_maint'
            if maint_col in df.columns:
                # Exponential decay feature (higher risk as time increases)
                df[f'{comp}_hazard_rate'] = 1 - np.exp(-df[maint_col] / 30)

                # Weibull-inspired feature
                df[f'{comp}_weibull_feature'] = (df[maint_col] / 45) ** 1.5

                # Step function for critical periods
                df[f'{comp}_critical_period'] = (
                        (df[maint_col] > 21) & (df[maint_col] <= 35)
                ).astype(int)

        return df

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

        # Sensor health indicators with domain knowledge
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

            # Sensor interactions
            df['volt_rotate_stress'] = df['volt_health'] * df['rotate_health']
            df['pressure_vibration_stress'] = df['pressure_health'] * df['vibration_health']

        # Enhanced maintenance features
        maint_cols = [f'days_since_comp{i}_maint' for i in range(1, 5)]
        available_maint = [col for col in maint_cols if col in df.columns]

        for col in available_maint:
            if col in df.columns:
                # Non-linear transformations
                df[f'{col}_sqrt'] = np.sqrt(df[col])
                df[f'{col}_log'] = np.log1p(df[col])
                df[f'{col}_inv'] = 1 / (df[col] + 1)

                # Risk indicators
                df[f'{col}_overdue'] = (df[col] > 30).astype(int)
                df[f'{col}_critical'] = (df[col] > 45).astype(int)
                df[f'{col}_urgent'] = (df[col] > 60).astype(int)

        # Cross-component analysis
        if len(available_maint) >= 2:
            df['maint_synchronization'] = df[available_maint].std(axis=1)
            df['maint_imbalance'] = df[available_maint].max(axis=1) - df[available_maint].min(axis=1)
            df['avg_maint_risk'] = df[available_maint].mean(axis=1) / 30
            df['max_maint_risk'] = df[available_maint].max(axis=1) / 30

        # Age-related features
        if 'age' in df.columns:
            df['age_normalized'] = df['age'] / 100
            df['age_risk'] = np.minimum(df['age'] / 80, 2.0)
            df['age_exp_decay'] = np.exp(-df['age'] / 50)

        # Error pattern analysis
        if 'error_count' in df.columns:
            df['error_rate'] = np.minimum(df['error_count'] / 10, 1.0)
            df['error_severity'] = df['error_count'] ** 0.5
            df['has_errors'] = (df['error_count'] > 0).astype(int)

        # Add survival features
        df = self.create_survival_features(df, target_component)

        # Component-specific risk modeling
        if target_component:
            comp_num = target_component.replace('ttf_comp', '').replace('_hours', '')
            maint_col = f'days_since_comp{comp_num}_maint'

            if maint_col in df.columns:
                # Target component specific features
                df['target_component_risk'] = (
                        df[f'{maint_col}_sqrt'] * 0.3 +
                        df['sensor_degradation'] * 0.4 +
                        df['age_risk'] * 0.3
                )

                # Interaction with other components
                other_comps = [f'days_since_comp{i}_maint' for i in range(1, 5) if i != int(comp_num)]
                other_comps = [c for c in other_comps if c in df.columns]

                if other_comps:
                    df['other_components_avg'] = df[other_comps].mean(axis=1)
                    df['target_vs_others'] = df[maint_col] - df['other_components_avg']

        # Fill missing values intelligently
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.endswith('_health') or col.endswith('_risk'):
                df[col] = df[col].fillna(0.5)  # Neutral for health/risk scores
            else:
                df[col] = df[col].fillna(df[col].median())

        df = df.fillna(0)
        return df

    def smart_sampling(self, df, component_col, max_samples=100000):
        """Intelligent sampling strategy for TTF prediction"""
        # Remove invalid values
        valid_mask = ~(df[component_col].isna() | np.isinf(df[component_col]) | (df[component_col] <= 0))
        df_valid = df[valid_mask].copy()

        if len(df_valid) == 0:
            return pd.DataFrame()

        # Define TTF ranges with business logic
        bins = [0, 4, 8, 16, 24, 48, 96, 168, 336, 720, np.inf]
        labels = ['0-4h', '4-8h', '8-16h', '16-24h', '24-48h', '48-96h',
                  '96-168h', '168-336h', '336-720h', '720h+']

        df_valid['ttf_range'] = pd.cut(df_valid[component_col], bins=bins, labels=labels)

        # Sampling strategy focused on critical and medium-term failures
        sampling_strategy = {
            '0-4h': 0.25,  # 25% - Most critical
            '4-8h': 0.20,  # 20% - Critical
            '8-16h': 0.15,  # 15% - High risk
            '16-24h': 0.15,  # 15% - High risk
            '24-48h': 0.10,  # 10% - Medium-high risk
            '48-96h': 0.08,  # 8% - Medium risk
            '96-168h': 0.04,  # 4% - Low-medium risk
            '168-336h': 0.02,  # 2% - Low risk
            '336-720h': 0.01,  # 1% - Very low risk
            '720h+': 0.001  # 0.1% - Minimal risk
        }

        sampled_dfs = []
        for range_name, proportion in sampling_strategy.items():
            range_data = df_valid[df_valid['ttf_range'] == range_name]
            target_samples = int(max_samples * proportion)

            if len(range_data) > 0:
                if len(range_data) >= target_samples:
                    sampled = range_data.sample(n=target_samples, random_state=42)
                else:
                    # For critical cases, use all data and oversample if needed
                    if range_name in ['0-4h', '4-8h', '8-16h', '16-24h']:
                        n_repeats = max(1, target_samples // len(range_data))
                        remaining = target_samples % len(range_data)

                        sampled = pd.concat([range_data] * n_repeats, ignore_index=True)
                        if remaining > 0:
                            extra = range_data.sample(n=remaining, random_state=42, replace=True)
                            sampled = pd.concat([sampled, extra], ignore_index=True)
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

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)

        return X, y

    def advanced_feature_selection(self, X, y, max_features=25):
        """Advanced feature selection combining multiple methods"""
        # Remove low variance features
        feature_vars = X.var()
        variable_features = feature_vars[feature_vars > 1e-8].index
        X_filtered = X[variable_features]

        # Remove highly correlated features
        corr_matrix = X_filtered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]
        X_filtered = X_filtered.drop(columns=high_corr_features)

        # Gradient boosting for feature importance
        gb_selector = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_selector.fit(X_filtered, y)

        # Recursive feature elimination
        rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe_selector = RFE(rf_estimator, n_features_to_select=max_features, step=1)
        rfe_selector.fit(X_filtered, y)

        # Combine both selections
        gb_importance = pd.DataFrame({
            'feature': X_filtered.columns,
            'importance': gb_selector.feature_importances_
        }).sort_values('importance', ascending=False)

        top_gb_features = set(gb_importance.head(max_features)['feature'])
        rfe_features = set(X_filtered.columns[rfe_selector.support_])

        # Take intersection or combine intelligently
        selected_features = list(top_gb_features.intersection(rfe_features))

        # If intersection is too small, add top features from GB
        if len(selected_features) < max_features // 2:
            remaining_needed = max_features - len(selected_features)
            additional_features = [f for f in gb_importance['feature'].head(max_features)
                                   if f not in selected_features][:remaining_needed]
            selected_features.extend(additional_features)

        return selected_features[:max_features]

    def create_ensemble_model(self):
        """Create ensemble of different algorithms"""
        models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=42
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        }
        return models

    def calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics for different TTF ranges"""
        metrics = {}

        # Overall metrics
        metrics['r2_score'] = r2_score(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['total_samples'] = len(y_true)

        # Critical cases (≤24 hours)
        critical_mask = y_true <= 24
        if critical_mask.sum() > 0:
            y_crit = y_true[critical_mask]
            pred_crit = y_pred[critical_mask]

            metrics['critical_r2'] = r2_score(y_crit, pred_crit)
            metrics['critical_rmse'] = np.sqrt(mean_squared_error(y_crit, pred_crit))
            metrics['critical_mae'] = mean_absolute_error(y_crit, pred_crit)
            metrics['critical_samples'] = critical_mask.sum()
        else:
            metrics.update({'critical_r2': 0, 'critical_rmse': 0, 'critical_mae': 0, 'critical_samples': 0})

        # High risk cases (≤48 hours)
        high_risk_mask = y_true <= 48
        if high_risk_mask.sum() > 0:
            y_high = y_true[high_risk_mask]
            pred_high = y_pred[high_risk_mask]

            metrics['high_risk_r2'] = r2_score(y_high, pred_high)
            metrics['high_risk_rmse'] = np.sqrt(mean_squared_error(y_high, pred_high))
            metrics['high_risk_mae'] = mean_absolute_error(y_high, pred_high)
            metrics['high_risk_samples'] = high_risk_mask.sum()
        else:
            metrics.update({'high_risk_r2': 0, 'high_risk_rmse': 0, 'high_risk_mae': 0, 'high_risk_samples': 0})

        # Medium risk cases (≤168 hours / 7 days)
        medium_risk_mask = y_true <= 168
        if medium_risk_mask.sum() > 0:
            y_med = y_true[medium_risk_mask]
            pred_med = y_pred[medium_risk_mask]

            metrics['medium_risk_r2'] = r2_score(y_med, pred_med)
            metrics['medium_risk_rmse'] = np.sqrt(mean_squared_error(y_med, pred_med))
            metrics['medium_risk_mae'] = mean_absolute_error(y_med, pred_med)
            metrics['medium_risk_samples'] = medium_risk_mask.sum()
        else:
            metrics.update({'medium_risk_r2': 0, 'medium_risk_rmse': 0, 'medium_risk_mae': 0, 'medium_risk_samples': 0})

        # Long term cases (>168 hours)
        long_term_mask = y_true > 168
        if long_term_mask.sum() > 0:
            y_long = y_true[long_term_mask]
            pred_long = y_pred[long_term_mask]

            metrics['long_term_r2'] = r2_score(y_long, pred_long)
            metrics['long_term_rmse'] = np.sqrt(mean_squared_error(y_long, pred_long))
            metrics['long_term_mae'] = mean_absolute_error(y_long, pred_long)
            metrics['long_term_samples'] = long_term_mask.sum()
        else:
            metrics.update({'long_term_r2': 0, 'long_term_rmse': 0, 'long_term_mae': 0, 'long_term_samples': 0})

        return metrics

    def train_component_ensemble(self, X_train, y_train, X_test, y_test, component_name):
        """Train ensemble model for component"""
        models = self.create_ensemble_model()
        trained_models = {}
        predictions = {}

        # Train individual models
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred = np.maximum(pred, 0.1)  # Ensure positive predictions

            trained_models[name] = model
            predictions[name] = pred

        # Simple ensemble - weighted average based on individual performance
        weights = {}
        for name, pred in predictions.items():
            r2 = r2_score(y_test, pred)
            weights[name] = max(0, r2)  # Only positive weights

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1 / len(weights) for k in weights.keys()}

        # Create ensemble prediction
        ensemble_pred = np.zeros(len(X_test))
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]

        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(y_test, ensemble_pred)
        metrics['weights'] = weights

        return {
            'models': trained_models,
            'weights': weights,
            **metrics
        }

    def train_models(self, file_path, max_samples=80000):
        """Train enhanced models for all components"""
        print("Enhanced Component TTF Prediction Models")
        print("=" * 50)

        # Load data
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape}")

        for component in self.components:
            if component not in df.columns:
                continue

            print(f"\n🔧 Training {component.upper()}")
            print("-" * 30)

            # Smart sampling
            component_df = self.smart_sampling(df, component, max_samples)
            if len(component_df) == 0:
                continue

            # Enhanced feature engineering
            component_df = self.enhanced_feature_engineering(component_df, component)

            # Prepare features
            X, y = self.prepare_features(component_df, component)
            if len(y) == 0:
                continue

            # Advanced feature selection
            selected_features = self.advanced_feature_selection(X, y, max_features=25)
            X = X[selected_features]
            self.component_features[component] = selected_features

            # Split data with stratification
            y_bins = pd.cut(y, bins=[0, 4, 12, 24, 72, 168, np.inf], labels=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y_bins
            )

            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.component_scalers[component] = scaler

            # Train ensemble
            model_info = self.train_component_ensemble(
                X_train_scaled, y_train, X_test_scaled, y_test, component
            )
            self.component_models[component] = model_info
            self.evaluation_metrics[component] = model_info

        self.is_trained = True
        print(f"\n✅ Training completed successfully!")

        # Save models automatically after training
        self.save_models()

        return True

    def predict_all_components(self, data_dict):
        """Predict TTF for all components"""
        if not self.is_trained:
            return None

        df = pd.DataFrame([data_dict])
        predictions = {}

        for component in self.components:
            if component not in self.component_models:
                continue

            # Feature engineering
            df_processed = self.enhanced_feature_engineering(df, component)

            # Prepare features
            model_info = self.component_models[component]
            features = self.component_features[component]
            scaler = self.component_scalers[component]

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
            else:
                risk = "LOW"

            predictions[component] = {
                'ttf_hours': round(ensemble_pred, 1),
                'ttf_days': round(ensemble_pred / 24, 1),
                'risk_level': risk,
                'model_r2': round(model_info['r2_score'], 3)
            }

        return predictions


def main():
    predictor = EnhancedComponentTTFPredictor()

    # Try to load existing models first
    models_loaded = predictor.load_models()

    if not models_loaded:
        print("🔄 No existing models found. Training new models...")
        # Train new models
        success = predictor.train_models('Fixed_Advanced_TTF_Dataset.csv')
        if not success:
            print("❌ Training failed!")
            return
    else:
        print("🔄 Models loaded successfully. Skipping training...")

    # Always print comprehensive evaluation metrics
    predictor.print_comprehensive_evaluation()

    # Test prediction
    print(f"\n🔮 SAMPLE PREDICTION TEST")
    print("=" * 40)

    sample_data = {
        'age': 45.2,
        'model': 'model3',
        'volt': 168.5,
        'rotate': 438.2,
        'pressure': 98.7,
        'vibration': 42.1,
        'error_count': 2,
        'days_since_comp1_maint': 25,
        'days_since_comp2_maint': 15,
        'days_since_comp3_maint': 35,
        'days_since_comp4_maint': 8
    }

    predictions = predictor.predict_all_components(sample_data)

    if predictions:
        print(f"\n📊 Component TTF Predictions:")
        print("-" * 40)
        for comp, pred in predictions.items():
            comp_num = comp.replace('ttf_comp', '').replace('_hours', '')
            risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
            emoji = risk_emoji.get(pred['risk_level'], "⚪")

            print(f"{emoji} Component {comp_num}: {pred['ttf_hours']}h ({pred['ttf_days']}d) - {pred['risk_level']} "
                  f"(Model R²: {pred['model_r2']})")

        # Find most critical
        min_pred = min(predictions.values(), key=lambda x: x['ttf_hours'])
        critical_comp = [k for k, v in predictions.items() if v['ttf_hours'] == min_pred['ttf_hours']][0]
        comp_num = critical_comp.replace('ttf_comp', '').replace('_hours', '')

        print(f"\n🚨 Most Critical: Component {comp_num} ({min_pred['ttf_hours']}h - {min_pred['risk_level']})")

        # Maintenance recommendations
        print(f"\n🔧 MAINTENANCE RECOMMENDATIONS")
        print("-" * 40)

        critical_components = [k for k, v in predictions.items() if v['risk_level'] in ['CRITICAL', 'HIGH']]
        if critical_components:
            print("⚠️  Immediate attention required:")
            for comp in critical_components:
                comp_num = comp.replace('ttf_comp', '').replace('_hours', '')
                pred = predictions[comp]
                print(f"   • Component {comp_num}: Schedule maintenance within {pred['ttf_hours']}h")

        medium_components = [k for k, v in predictions.items() if v['risk_level'] == 'MEDIUM']
        if medium_components:
            print("\n📅 Schedule preventive maintenance:")
            for comp in medium_components:
                comp_num = comp.replace('ttf_comp', '').replace('_hours', '')
                pred = predictions[comp]
                print(f"   • Component {comp_num}: Maintenance recommended within {pred['ttf_days']} days")

        low_components = [k for k, v in predictions.items() if v['risk_level'] == 'LOW']
        if low_components:
            print(f"\n✅ Components in good condition:")
            for comp in low_components:
                comp_num = comp.replace('ttf_comp', '').replace('_hours', '')
                pred = predictions[comp]
                print(f"   • Component {comp_num}: Next maintenance in ~{pred['ttf_days']} days")


if __name__ == "__main__":
    main()