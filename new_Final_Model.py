import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class OptimizedTTFPredictor:
    def __init__(self, model_path="models/"):
        """
        Optimized TTF Predictor using RandomForest (best performing model)
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.is_trained = False

        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def advanced_feature_engineering(self, df):
        """Enhanced feature engineering for better TTF prediction"""
        print("Performing advanced feature engineering...")

        # Sort by machineID and datetime for proper time series operations
        df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)

        # Basic time features
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['day_of_month'] = df['datetime'].dt.day
            df['quarter'] = df['datetime'].dt.quarter
            df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)

            # Work shift indicators (assuming 8-hour shifts)
            df['shift'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], labels=[0, 1, 2], include_lowest=True)
            df['shift'] = df['shift'].astype(int)

        sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']

        # Group by machine for time-series features
        machine_groups = df.groupby('machineID')

        # Multi-window rolling statistics
        windows = [3, 6, 12, 24, 48]
        for window in windows:
            for col in sensor_cols:
                if col in df.columns:
                    df[f'{col}_mean_{window}'] = machine_groups[col].rolling(window, min_periods=1).mean().reset_index(
                        0, drop=True)
                    df[f'{col}_std_{window}'] = machine_groups[col].rolling(window, min_periods=1).std().reset_index(0,
                                                                                                                     drop=True)
                    df[f'{col}_max_{window}'] = machine_groups[col].rolling(window, min_periods=1).max().reset_index(0,
                                                                                                                     drop=True)
                    df[f'{col}_range_{window}'] = df[f'{col}_max_{window}'] - machine_groups[col].rolling(window,
                                                                                                          min_periods=1).min().reset_index(
                        0, drop=True)

        # Trend and slope features
        for window in [6, 12, 24]:
            for col in sensor_cols:
                if col in df.columns:
                    def calculate_slope(series):
                        if len(series) < 2:
                            return 0
                        x = np.arange(len(series))
                        slope, _ = np.polyfit(x, series, 1)
                        return slope

                    df[f'{col}_slope_{window}'] = machine_groups[col].rolling(window, min_periods=2).apply(
                        calculate_slope).reset_index(0, drop=True)

        # Volatility measures
        for window in [12, 24]:
            for col in sensor_cols:
                if col in df.columns:
                    mean_val = machine_groups[col].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
                    std_val = machine_groups[col].rolling(window, min_periods=1).std().reset_index(0, drop=True)
                    df[f'{col}_cv_{window}'] = std_val / (mean_val + 1e-8)
                    df[f'{col}_stability_{window}'] = 1 / (df[f'{col}_cv_{window}'] + 1e-8)

        # Cross-sensor relationships
        if all(col in df.columns for col in sensor_cols):
            df['volt_pressure_ratio'] = df['volt'] / (df['pressure'] + 1e-8)
            df['rotate_vibration_ratio'] = df['rotate'] / (df['vibration'] + 1e-8)

            for window in [6, 12, 24]:
                df[f'sensor_stress_{window}'] = (
                                                        df[f'volt_std_{window}'] + df[f'rotate_std_{window}'] +
                                                        df[f'pressure_std_{window}'] + df[f'vibration_std_{window}']
                                                ) / 4

        # Degradation indicators
        for col in sensor_cols:
            if col in df.columns:
                normal_value = df[col].median()
                df[f'{col}_cum_deviation'] = machine_groups[col].apply(
                    lambda x: (x - normal_value).abs().cumsum()
                ).reset_index(0, drop=True)

        # Age-related features
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['age_log'] = np.log1p(df['age'])

            for col in sensor_cols:
                if col in df.columns:
                    df[f'age_{col}_interaction'] = df['age'] * df[col]

        # Error pattern features
        if 'error_count' in df.columns:
            df['cumulative_errors'] = machine_groups['error_count'].cumsum().reset_index(0, drop=True)
            for window in [6, 12, 24]:
                df[f'error_rate_{window}'] = machine_groups['error_count'].rolling(window,
                                                                                   min_periods=1).sum().reset_index(0,
                                                                                                                    drop=True)

        # Fill any remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the predictive maintenance data"""
        print("Loading data...")
        df = pd.read_csv(file_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Convert datetime columns
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        if 'failure_datetime' in df.columns:
            df['failure_datetime'] = pd.to_datetime(df['failure_datetime'])

        # Encode categorical variables
        categorical_cols = ['model', 'comp', 'maint_comp']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # Apply advanced feature engineering
        df = self.advanced_feature_engineering(df)

        return df

    def prepare_features_target(self, df):
        """Prepare features and target variable with feature selection"""
        # Base feature columns
        base_features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'error_count']

        # Time features
        time_features = ['hour', 'day_of_week', 'month', 'day_of_month', 'quarter', 'is_weekend', 'shift']

        # Get all engineered features
        engineered_features = [col for col in df.columns if any(
            suffix in col for suffix in ['_mean_', '_std_', '_max_', '_range_', '_slope_', '_cv_', '_stability_',
                                         '_ratio', '_stress_', '_cum_', '_squared', '_log', '_interaction',
                                         '_cumulative_', '_rate_']
        )]

        # Encoded categorical features
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]

        # Combine all features
        all_features = base_features + time_features + engineered_features + encoded_features
        feature_cols = [col for col in all_features if col in df.columns]

        print(f"Total features available: {len(feature_cols)}")

        # Feature selection based on correlation with target
        if 'ttf_hours' in df.columns:
            correlations = df[feature_cols + ['ttf_hours']].corr()['ttf_hours'].abs().sort_values(ascending=False)
            top_features = correlations.drop('ttf_hours').head(50).index.tolist()
            feature_cols = top_features
            print(f"Selected top {len(feature_cols)} features based on correlation")

        self.feature_columns = feature_cols

        X = df[feature_cols].copy()
        y = df['ttf_hours'].copy() if 'ttf_hours' in df.columns else None

        # Handle any remaining missing values
        X = X.fillna(X.median())

        print(f"Feature matrix shape: {X.shape}")
        if y is not None:
            print(f"Target variable shape: {y.shape}")

        return X, y

    def train_model(self, file_path):
        """Train the optimized RandomForest TTF prediction model"""
        print("OPTIMIZED PREDICTIVE MAINTENANCE - TTF PREDICTION MODEL")
        print("=" * 70)

        if not os.path.exists(file_path):
            print(f"Error: Data file '{file_path}' not found!")
            return None

        # Load and preprocess data
        df = self.load_and_preprocess_data(file_path)
        X, y = self.prepare_features_target(df)

        if y is None:
            print("Error: Target variable 'ttf_hours' not found!")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )

        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train optimized RandomForest model
        print("\nTraining Optimized RandomForest...")
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)

        # Calculate and print detailed metrics
        self.print_model_performance(y_test, y_pred)
        self.print_feature_importance()
        self.print_summary_statistics(y)
        self.error_analysis(y_test, y_pred)

        self.is_trained = True
        self.save_model()

        return {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

    def print_model_performance(self, y_test, y_pred):
        """Print detailed model performance metrics"""
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n{'=' * 70}")
        print("MODEL PERFORMANCE METRICS")
        print(f"{'=' * 70}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f} hours")
        print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
        print(f"R² Score: {r2:.4f}")
        print(f"Model explains {r2 * 100:.2f}% of the variance in TTF")

    def error_analysis(self, y_true, y_pred):
        """Analyze prediction errors with detailed statistics"""
        print(f"\n{'=' * 70}")
        print("DETAILED ERROR ANALYSIS")
        print(f"{'=' * 70}")

        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        print(f"Error Statistics:")
        print(f"  Mean Error (Bias): {errors.mean():.2f} hours")
        print(f"  Median Error: {np.median(errors):.2f} hours")
        print(f"  Error Standard Deviation: {errors.std():.2f} hours")
        print(f"  Mean Absolute Error: {abs_errors.mean():.2f} hours")
        print(f"  Median Absolute Error: {np.median(abs_errors):.2f} hours")
        print(f"  95% of errors within: ±{np.percentile(abs_errors, 95):.2f} hours")
        print(f"  99% of errors within: ±{np.percentile(abs_errors, 99):.2f} hours")

        print(f"\nPrediction Accuracy by Time Windows:")
        # Percentage of predictions within different error bands
        for threshold in [6, 12, 24, 48, 72, 168]:  # 6h, 12h, 1d, 2d, 3d, 1w
            within_threshold = np.sum(abs_errors <= threshold) / len(errors) * 100
            print(f"  Within ±{threshold:3d} hours: {within_threshold:5.1f}% of predictions")

        # Performance by TTF ranges
        print(f"\nPerformance by TTF Ranges:")
        ttf_ranges = [(0, 24), (24, 168), (168, 720), (720, float('inf'))]
        range_names = ['Critical (<24h)', 'Short (1-7d)', 'Medium (1-30d)', 'Long (>30d)']

        for (low, high), name in zip(ttf_ranges, range_names):
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                range_mae = np.mean(abs_errors[mask])
                range_count = mask.sum()
                print(f"  {name:15s}: MAE = {range_mae:6.2f}h, Count = {range_count:5d}")

    def print_feature_importance(self):
        """Print feature importance analysis"""
        print(f"\n{'=' * 70}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'=' * 70}")

        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 15 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:35s}: {row['importance']:.4f}")

            # Group features by category
            print("\nFeature Importance by Category:")
            categories = {
                'Age Features': [f for f in self.feature_columns if 'age' in f.lower()],
                'Sensor Features': [f for f in self.feature_columns if
                                    any(s in f for s in ['volt', 'rotate', 'pressure', 'vibration'])],
                'Time Features': [f for f in self.feature_columns if
                                  any(t in f for t in ['hour', 'day', 'month', 'quarter'])],
                'Encoded Features': [f for f in self.feature_columns if 'encoded' in f],
                'Error Features': [f for f in self.feature_columns if 'error' in f]
            }

            for category, features in categories.items():
                if features:
                    cat_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
                    print(f"  {category:20s}: {cat_importance:.4f}")

    def print_summary_statistics(self, y):
        """Print comprehensive target variable statistics"""
        print(f"\n{'=' * 70}")
        print("TARGET VARIABLE STATISTICS")
        print(f"{'=' * 70}")
        print(f"Time to Failure (TTF) Distribution:")
        print(f"  Count: {len(y):,} samples")
        print(f"  Mean: {y.mean():.2f} hours ({y.mean() / 24:.1f} days)")
        print(f"  Median: {y.median():.2f} hours ({y.median() / 24:.1f} days)")
        print(f"  Standard Deviation: {y.std():.2f} hours")
        print(f"  Minimum: {y.min():.2f} hours")
        print(f"  Maximum: {y.max():.2f} hours ({y.max() / 24:.1f} days)")

        print(f"\nPercentile Distribution:")
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(y, p)
            print(f"  {p:2d}th percentile: {value:7.2f} hours ({value / 24:5.1f} days)")

    def save_model(self):
        """Save the trained model and preprocessing objects"""
        if not self.is_trained:
            print("Model is not trained yet.")
            return

        # Save all model components
        joblib.dump(self.model, os.path.join(self.model_path, 'optimized_ttf_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_path, 'optimized_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(self.model_path, 'optimized_label_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(self.model_path, 'optimized_feature_columns.pkl'))

        print(f"\n{'=' * 70}")
        print("MODEL SAVED SUCCESSFULLY")
        print(f"{'=' * 70}")
        print(f"Model files saved to: {self.model_path}")
        print("Files saved:")
        print("  - optimized_ttf_model.pkl")
        print("  - optimized_scaler.pkl")
        print("  - optimized_label_encoders.pkl")
        print("  - optimized_feature_columns.pkl")

    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model = joblib.load(os.path.join(self.model_path, 'optimized_ttf_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_path, 'optimized_scaler.pkl'))
            self.label_encoders = joblib.load(os.path.join(self.model_path, 'optimized_label_encoders.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.model_path, 'optimized_feature_columns.pkl'))

            self.is_trained = True
            print("Optimized model loaded successfully!")
            return True

        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            return False

    def predict_single(self, data_dict):
        """Predict TTF for a single data point"""
        if not self.is_trained:
            if not self.load_model():
                return None

        # Create DataFrame and preprocess
        df = pd.DataFrame([data_dict])
        df = self.preprocess_single_input(df)

        # Prepare features
        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]

        return prediction

    def preprocess_single_input(self, df):
        """Preprocess single input - simplified version"""
        # Basic time features
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['day_of_month'] = df['datetime'].dt.day
            df['quarter'] = df['datetime'].dt.quarter
            df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
            df['shift'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], labels=[0, 1, 2], include_lowest=True)
            df['shift'] = df['shift'].astype(int)

        # Encode categorical variables
        categorical_cols = ['model', 'comp', 'maint_comp']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                try:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                except ValueError:
                    df[f'{col}_encoded'] = 0

        # Add missing features with default values
        for feature in self.feature_columns:
            if feature not in df.columns:
                df[feature] = 0

        return df


def main():
    """Main execution function"""
    predictor = OptimizedTTFPredictor()

    # Check for existing models
    model_files_exist = all([
        os.path.exists(os.path.join(predictor.model_path, 'optimized_ttf_model.pkl')),
        os.path.exists(os.path.join(predictor.model_path, 'optimized_scaler.pkl')),
        os.path.exists(os.path.join(predictor.model_path, 'optimized_label_encoders.pkl')),
        os.path.exists(os.path.join(predictor.model_path, 'optimized_feature_columns.pkl'))
    ])

    if not model_files_exist:
        print("Training new optimized model...")
        data_file_path = "data/Final_with_TTF.csv"

        if os.path.exists(data_file_path):
            results = predictor.train_model(data_file_path)
            if results:
                print(f"\n{'=' * 70}")
                print("TRAINING COMPLETED SUCCESSFULLY!")
                print(f"{'=' * 70}")
                print(f"Final Model Performance:")
                print(f"  RMSE: {results['rmse']:.2f} hours")
                print(f"  MAE: {results['mae']:.2f} hours")
                print(f"  R² Score: {results['r2']:.4f}")
        else:
            print(f"Data file not found: {data_file_path}")
            return

    # Load and test model
    if predictor.load_model():
        # Test prediction
        sample_data = {
            'volt': 175.0,
            'rotate': 420.0,
            'pressure': 110.0,
            'vibration': 40.0,
            'model': 'model1',
            'age': 18,
            'error_count': 0,
            'comp': 'comp1',
            'maint_comp': 'comp1',
            'datetime': '2015-01-01 06:00:00',
            'machineID': 1
        }

        predicted_ttf = predictor.predict_single(sample_data)
        if predicted_ttf is not None:
            print(f"\n{'=' * 50}")
            print("SAMPLE PREDICTION")
            print(f"{'=' * 50}")
            print(f"Predicted TTF: {predicted_ttf:.2f} hours ({predicted_ttf / 24:.1f} days)")


if __name__ == "__main__":
    main()