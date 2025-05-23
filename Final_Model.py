import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class TTFPredictor:
    def __init__(self, model_path="models/"):
        """
        Initialize TTF Predictor

        Args:
            model_path (str): Path to save/load models
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.is_trained = False

        # Create model directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the predictive maintenance data"""
        print("Loading data...")
        df = pd.read_csv(file_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Convert datetime columns if they exist
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        if 'failure_datetime' in df.columns:
            df['failure_datetime'] = pd.to_datetime(df['failure_datetime'])

        # Feature engineering - only if datetime column exists
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month

        # Rolling statistics for sensor readings - only if machineID exists
        sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
        if 'machineID' in df.columns:
            for col in sensor_cols:
                if col in df.columns:
                    df[f'{col}_rolling_mean_3'] = df.groupby('machineID')[col].rolling(
                        window=3, min_periods=1).mean().reset_index(0, drop=True)
                    df[f'{col}_rolling_std_3'] = df.groupby('machineID')[col].rolling(
                        window=3, min_periods=1).std().reset_index(0, drop=True)

        # Encode categorical variables and save encoders
        categorical_cols = ['model', 'comp', 'maint_comp']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        return df

    def prepare_features_target(self, df):
        """Prepare features and target variable"""
        # Define feature columns
        feature_cols = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'error_count',
                        'hour', 'day_of_week', 'month']

        # Add rolling features if they exist
        rolling_cols = [col for col in df.columns if 'rolling' in col]
        feature_cols.extend(rolling_cols)

        # Add encoded categorical features
        encoded_cols = [col for col in df.columns if 'encoded' in col]
        feature_cols.extend(encoded_cols)

        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols

        X = df[feature_cols].copy()
        y = df['ttf_hours'].copy() if 'ttf_hours' in df.columns else None

        # Handle any remaining missing values
        X = X.fillna(X.mean())

        print(f"Features used: {feature_cols}")
        print(f"Feature matrix shape: {X.shape}")
        if y is not None:
            print(f"Target variable shape: {y.shape}")

        return X, y

    def train_model(self, file_path):
        """Train the Random Forest model"""
        print("PREDICTIVE MAINTENANCE - TTF PREDICTION MODEL")
        print("=" * 60)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: Data file '{file_path}' not found!")
            print("Please ensure the data file exists and the path is correct.")
            return None

        # Load and preprocess data
        df = self.load_and_preprocess_data(file_path)
        X, y = self.prepare_features_target(df)

        if y is None:
            print("Error: Target variable 'ttf_hours' not found in the dataset!")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest model
        print("\nTraining Random Forest...")
        print("=" * 50)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Feature importance analysis
        self.print_feature_importance()

        # Summary statistics
        self.print_summary_statistics(y)

        # Classification evaluation with 24-hour threshold
        self.classification_evaluation(y_test, y_pred, threshold_hours=24)

        self.is_trained = True

        # Save the trained model
        self.save_model()

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def print_feature_importance(self):
        """Print feature importance"""
        print(f"\n{'=' * 60}")
        print("FEATURE IMPORTANCE ANALYSIS - Random Forest")
        print(f"{'=' * 60}")

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:25s}: {row['importance']:.4f}")

    def print_summary_statistics(self, y):
        """Print target variable statistics"""
        print(f"\n{'=' * 60}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 60}")
        print(f"Target variable (TTF_hours) statistics:")
        print(f"Mean: {y.mean():.2f} hours")
        print(f"Median: {y.median():.2f} hours")
        print(f"Std: {y.std():.2f} hours")
        print(f"Min: {y.min():.2f} hours")
        print(f"Max: {y.max():.2f} hours")

    def classification_evaluation(self, y_test, y_pred, threshold_hours=24):
        """Convert regression problem to classification for evaluation"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

        print(f"\n{'=' * 60}")
        print("CLASSIFICATION EVALUATION")
        print(f"{'=' * 60}")
        print(f"Converting to binary classification with threshold: {threshold_hours} hours")
        print("1 = Failure within threshold, 0 = No failure within threshold")

        # Convert to binary classification
        y_test_binary = (y_test <= threshold_hours).astype(int)
        y_pred_binary = (y_pred <= threshold_hours).astype(int)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_binary, y_pred_binary, average='binary')

        print(f"\nBinary Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    def save_model(self):
        """Save the trained model and preprocessing objects"""
        if not self.is_trained:
            print("Model is not trained yet. Please train the model first.")
            return

        # Save model
        joblib.dump(self.model, os.path.join(self.model_path, 'random_forest_model.pkl'))

        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))

        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(self.model_path, 'label_encoders.pkl'))

        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(self.model_path, 'feature_columns.pkl'))

        print(f"\nModel saved successfully to {self.model_path}")

    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            # Load model
            self.model = joblib.load(os.path.join(self.model_path, 'random_forest_model.pkl'))

            # Load scaler
            self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.pkl'))

            # Load label encoders
            self.label_encoders = joblib.load(os.path.join(self.model_path, 'label_encoders.pkl'))

            # Load feature columns
            self.feature_columns = joblib.load(os.path.join(self.model_path, 'feature_columns.pkl'))

            self.is_trained = True
            print("Model loaded successfully!")
            return True

        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Please train the model first using train_model() method.")
            return False

    def predict_single(self, data_dict):
        """
        Predict TTF for a single data point

        Args:
            data_dict (dict): Dictionary containing feature values
                Example: {
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

        Returns:
            float: Predicted TTF in hours
        """
        if not self.is_trained:
            if not self.load_model():
                return None

        # Create DataFrame from input
        df = pd.DataFrame([data_dict])

        # Apply the same preprocessing as training
        df = self.preprocess_single_input(df)

        # Prepare features
        X = df[self.feature_columns].copy()
        X = X.fillna(X.mean())

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make prediction
        prediction = self.model.predict(X_scaled)[0]

        return prediction

    def predict_batch(self, data_list):
        """
        Predict TTF for multiple data points

        Args:
            data_list (list): List of dictionaries containing feature values

        Returns:
            list: List of predicted TTF values in hours
        """
        if not self.is_trained:
            if not self.load_model():
                return None

        predictions = []
        for data_dict in data_list:
            prediction = self.predict_single(data_dict)
            predictions.append(prediction)

        return predictions

    def preprocess_single_input(self, df):
        """Preprocess single input data (same as training preprocessing)"""
        # Convert datetime columns
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

            # Feature engineering
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month

        # Rolling statistics (for single point, use the current values)
        sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
        for col in sensor_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_3'] = df[col]  # Use current value as rolling mean
                df[f'{col}_rolling_std_3'] = 0.0  # Set rolling std to 0 for single point

        # Encode categorical variables using saved encoders
        categorical_cols = ['model', 'comp', 'maint_comp']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                try:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                except ValueError:
                    # Handle unseen categories by using the most frequent category
                    print(f"Warning: Unseen category in {col}. Using mode value.")
                    df[f'{col}_encoded'] = 0

        return df


# Example usage and main function
def main():
    """Main execution function"""
    # Initialize predictor
    predictor = TTFPredictor()

    # Check if model files exist
    model_files_exist = all([
        os.path.exists(os.path.join(predictor.model_path, 'random_forest_model.pkl')),
        os.path.exists(os.path.join(predictor.model_path, 'scaler.pkl')),
        os.path.exists(os.path.join(predictor.model_path, 'label_encoders.pkl')),
        os.path.exists(os.path.join(predictor.model_path, 'feature_columns.pkl'))
    ])

    if not model_files_exist:
        print("Model files not found. Training new model...")
        # Train model - UPDATE THIS PATH TO YOUR ACTUAL DATA FILE
        data_file_path = "data/Final_with_TTF.csv"  # Update this path

        # Check if data file exists
        if os.path.exists(data_file_path):
            results = predictor.train_model(data_file_path)
            if results is None:
                print("Training failed. Please check your data file.")
                return
        else:
            print(f"Data file not found: {data_file_path}")
            print("Please update the data_file_path variable with the correct path to your data file.")
            return

    # Load pre-trained model (will load either existing model or newly trained one)
    if predictor.load_model():
        # Example prediction for a single data point
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

        # Make prediction
        predicted_ttf = predictor.predict_single(sample_data)
        if predicted_ttf is not None:
            print(f"\nPredicted TTF: {predicted_ttf:.2f} hours")

            # Example batch prediction
            batch_data = [sample_data, sample_data.copy()]  # Duplicate for example
            batch_predictions = predictor.predict_batch(batch_data)
            print(f"Batch predictions: {[f'{pred:.2f}' for pred in batch_predictions]} hours")


if __name__ == "__main__":
    main()