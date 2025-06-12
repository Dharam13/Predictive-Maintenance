from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# Global variable to store model
model_package = None


def load_model():
    """Load the enhanced model package"""
    global model_package
    try:
        model_package = joblib.load('tttf_xgb_model_enhanced.pklgb_model_enhanced.pkl')
        return True
    except FileNotFoundError:
        try:
            model_package = joblib.load('tttf_xgb_model_enhanced_backup.pkl')
            return True
        except FileNotFoundError:
            return False
    except Exception:
        return False


def get_feature_info():
    """Get feature information for the frontend"""
    if not model_package:
        return None

    feature_names = model_package['feature_names']
    label_encoders = model_package['label_encoders']

    features = []
    for feature in feature_names:
        if feature in label_encoders:
            features.append({
                'name': feature,
                'type': 'categorical',
                'options': list(label_encoders[feature].classes_)
            })
        else:
            # Add guidance for numerical features
            guidance = ""
            if feature == 'volt':
                guidance = "Voltage reading"
            elif feature == 'rotate':
                guidance = "Rotation speed"
            elif feature == 'pressure':
                guidance = "Pressure reading"
            elif feature == 'vibration':
                guidance = "Vibration level"
            elif feature == 'age':
                guidance = "Machine age in years"
            elif feature == 'error_count':
                guidance = "Recent error count"
            elif 'days_since' in feature:
                component = feature.split('_')[2]
                guidance = f"Days since {component} maintenance"

            features.append({
                'name': feature,
                'type': 'numerical',
                'guidance': guidance
            })

    return features


def create_example_data():
    """Create example data for quick testing"""
    if not model_package:
        return None

    feature_names = model_package['feature_names']
    label_encoders = model_package['label_encoders']

    example_data = {}

    # Get valid model if available
    if 'model' in label_encoders:
        valid_models = list(label_encoders['model'].classes_)
        example_data['model'] = valid_models[0] if valid_models else 'model1'

    # Set example values for other features
    example_values = {
        'volt': 168.5,
        'rotate': 415.2,
        'pressure': 98.7,
        'vibration': 45.3,
        'age': 150,
        'error_count': 2,
        'days_since_comp1_maint': 15,
        'days_since_comp2_maint': 8,
        'days_since_comp3_maint': 22,
        'days_since_comp4_maint': 12
    }

    for feature in feature_names:
        if feature not in example_data:
            example_data[feature] = example_values.get(feature, 100.0)

    return example_data


def make_prediction(input_data):
    """Make prediction using the loaded model"""
    try:
        # Extract model components
        model = model_package['model']
        scaler = model_package['scaler']
        imputer = model_package['imputer']
        label_encoders = model_package['label_encoders']
        feature_names = model_package['feature_names']
        target_names = model_package['target_names']

        # Create DataFrame from input
        sample_df = pd.DataFrame([input_data])
        X_sample = sample_df[feature_names].copy()

        # Process categorical features
        categorical_features = ['model']
        for cat_feat in categorical_features:
            if cat_feat in feature_names and cat_feat in label_encoders:
                le = label_encoders[cat_feat]
                original_value = X_sample[cat_feat].iloc[0]

                # Handle unknown categories
                if pd.isna(original_value) or str(original_value) not in le.classes_:
                    X_sample[cat_feat] = 'unknown'

                # Encode
                encoded_value = le.transform([str(X_sample[cat_feat].iloc[0])])[0]
                X_sample[cat_feat] = encoded_value

        # Process numerical features
        numerical_features = [f for f in feature_names if f not in categorical_features]
        if numerical_features:
            # Handle missing values and convert to float
            for feat in numerical_features:
                value = X_sample[feat].iloc[0]
                if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'null', 'none']:
                    X_sample[feat] = np.nan
                else:
                    X_sample[feat] = float(value)

            X_sample[numerical_features] = imputer.transform(X_sample[numerical_features])

        # Scale features
        X_scaled = scaler.transform(X_sample)

        # Make prediction
        raw_prediction = model.predict(X_scaled)[0]
        prediction_floor = np.floor(raw_prediction)

        # Prepare results
        results = []
        for i, target in enumerate(target_names):
            component = target.replace('ttf_', '').replace('_weeks', '').upper()
            exact_weeks = float(raw_prediction[i])
            floor_weeks = int(prediction_floor[i])

            # Determine status
            if floor_weeks == 0:
                status = "CRITICAL"
                emoji = "🚨"
            elif floor_weeks <= 2:
                status = "URGENT"
                emoji = "🔴"
            elif floor_weeks <= 4:
                status = "SOON"
                emoji = "🟡"
            elif floor_weeks <= 8:
                status = "NORMAL"
                emoji = "🟢"
            else:
                status = "GOOD"
                emoji = "✅"

            results.append({
                'component': component,
                'exact_weeks': exact_weeks,
                'floor_weeks': floor_weeks,
                'status': status,
                'emoji': emoji
            })

        # Calculate summary
        min_weeks = int(min(prediction_floor))
        avg_weeks = float(np.mean(prediction_floor))
        critical_components = sum(1 for weeks in prediction_floor if weeks <= 2)

        # Maintenance recommendation
        if min_weeks == 0:
            recommendation = "IMMEDIATE maintenance required!"
            rec_detail = "Stop operations and inspect immediately"
        elif min_weeks <= 2:
            recommendation = "URGENT: Schedule maintenance within 2 weeks"
            rec_detail = "Prioritize this machine in maintenance schedule"
        elif min_weeks <= 4:
            recommendation = "Schedule maintenance within 1 month"
            rec_detail = "Monitor closely and prepare maintenance"
        elif min_weeks <= 8:
            recommendation = "Normal maintenance schedule (within 2 months)"
            rec_detail = "Continue regular monitoring"
        else:
            recommendation = "Machine in good condition"
            rec_detail = "Follow standard maintenance schedule"

        return {
            'success': True,
            'results': results,
            'summary': {
                'min_weeks': min_weeks,
                'avg_weeks': avg_weeks,
                'critical_components': critical_components,
                'total_components': len(target_names)
            },
            'recommendation': {
                'title': recommendation,
                'detail': rec_detail
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    features = get_feature_info()
    target_names = model_package['target_names']
    excluded_features = model_package.get('excluded_features', [])

    return jsonify({
        'success': True,
        'features': features,
        'targets': target_names,
        'excluded_features': excluded_features,
        'total_features': len(features)
    })


@app.route('/api/example-data')
def example_data():
    """Get example data for testing"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    example = create_example_data()
    return jsonify({'success': True, 'data': example})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction endpoint"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    try:
        input_data = request.json
        if not input_data:
            return jsonify({'success': False, 'error': 'No input data provided'})

        result = make_prediction(input_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route('/api/performance')
def performance():
    """Get model performance metrics"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    if 'performance_metrics' in model_package:
        metrics = model_package['performance_metrics']

        # Calculate averages
        avg_r2 = np.mean([m['Test_R2'] for m in metrics])
        avg_mape = np.mean([m['Test_MAPE'] for m in metrics])

        return jsonify({
            'success': True,
            'metrics': metrics,
            'summary': {
                'avg_r2': avg_r2,
                'avg_mape': avg_mape
            }
        })
    else:
        return jsonify({'success': False, 'error': 'Performance metrics not available'})


@app.route('/api/status')
def status():
    """Get API status"""
    model_loaded = model_package is not None
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'version': '2.0_enhanced' if model_loaded else None
    })


if __name__ == '__main__':
    print("🚀 Starting Enhanced TTTF Prediction Web API")
    print("=" * 50)

    # Load model on startup
    if load_model():
        print("✅ Enhanced model loaded successfully!")
        feature_count = len(model_package['feature_names'])
        target_count = len(model_package['target_names'])
        print(f"📊 Features: {feature_count}, Targets: {target_count}")
    else:
        print("❌ Failed to load model!")
        print("   Please ensure 'tttf_gb_model_enhanced.pkl' exists.")

    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)