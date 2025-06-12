import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("🔮 Enhanced TTTF Manual Prediction Tool")
print("=" * 60)


def load_model_info():
    """Load enhanced model and display information about required inputs"""
    try:
        # Try to load the enhanced model first
        model_package = joblib.load('tttf_gb_model_enhanced.pkl')
        print("✅ Enhanced model loaded successfully!")

    except FileNotFoundError:
        try:
            # Fallback to backup model
            model_package = joblib.load('tttf_gb_model_enhanced_backup.pkl')
            print("✅ Enhanced backup model loaded successfully!")
        except FileNotFoundError:
            print("❌ Enhanced model files not found!")
            print("   Please ensure 'tttf_gb_model_enhanced.pkl' exists.")
            print("   Run the enhanced training pipeline first.")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    feature_names = model_package['feature_names']
    target_names = model_package['target_names']
    label_encoders = model_package['label_encoders']
    excluded_features = model_package.get('excluded_features', [])

    print(f"🎯 This model predicts: {target_names}")
    print(f"🚫 Excluded features: {excluded_features}")
    print(f"📊 Required input features ({len(feature_names)}):")

    # Group features by category for better display
    sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
    operational_features = ['model', 'age', 'error_count']
    maintenance_features = ['days_since_comp1_maint', 'days_since_comp2_maint',
                            'days_since_comp3_maint', 'days_since_comp4_maint']

    print("\n  📡 Sensor Features:")
    for feature in sensor_features:
        if feature in feature_names:
            print(f"     • {feature} (numerical)")

    print("\n  ⚙️  Operational Features:")
    for feature in operational_features:
        if feature in feature_names:
            if feature in label_encoders:
                valid_categories = list(label_encoders[feature].classes_)
                print(f"     • {feature} (categorical): {valid_categories}")
            else:
                print(f"     • {feature} (numerical)")

    print("\n  🔧 Maintenance Features:")
    for feature in maintenance_features:
        if feature in feature_names:
            print(f"     • {feature} (numerical)")

    return model_package


def get_user_input(model_package):
    """Get manual input from user for all required features"""
    feature_names = model_package['feature_names']
    label_encoders = model_package['label_encoders']

    print("\n📝 Please enter values for the following features:")
    print("-" * 60)

    input_data = {}

    for feature in feature_names:
        while True:
            try:
                if feature in label_encoders:
                    # Categorical feature
                    valid_categories = list(label_encoders[feature].classes_)
                    print(f"\n🏷️  {feature}")
                    print(f"   Valid options: {valid_categories}")
                    value = input(f"   Enter {feature}: ").strip()

                    if value in valid_categories:
                        input_data[feature] = value
                        print(f"   ✅ {feature} = {value}")
                        break
                    else:
                        print(f"   ❌ Invalid value! Must be one of: {valid_categories}")

                else:
                    # Numerical feature
                    print(f"\n🔢 {feature}")

                    # Provide guidance for common features
                    if feature == 'volt':
                        print("   💡 Voltage reading (typical: 150-190 volts)")
                    elif feature == 'rotate':
                        print("   💡 Rotation speed (typical: 350-450 RPM)")
                    elif feature == 'pressure':
                        print("   💡 Pressure reading (typical: 80-130 PSI)")
                    elif feature == 'vibration':
                        print("   💡 Vibration level (typical: 30-70 units)")
                    elif feature == 'age':
                        print("   💡 Machine age in days (typical: 50-400 days)")
                    elif feature == 'error_count':
                        print("   💡 Recent error count (typical: 0-10 errors)")
                    elif 'days_since' in feature:
                        component = feature.split('_')[2]
                        print(f"   💡 Days since {component} maintenance (typical: 1-60 days)")

                    value_str = input(f"   Enter {feature}: ").strip()

                    # Handle empty input as NaN (will be imputed)
                    if value_str == '' or value_str.lower() in ['nan', 'null', 'none']:
                        value = np.nan
                        print(f"   ⚠️  Using NaN (will be imputed by model)")
                    else:
                        value = float(value_str)

                    input_data[feature] = value
                    if not pd.isna(value):
                        print(f"   ✅ {feature} = {value}")
                    else:
                        print(f"   ✅ {feature} = NaN (will be imputed)")
                    break

            except ValueError:
                print("   ❌ Invalid input! Please enter a valid number or leave empty for NaN.")
            except KeyboardInterrupt:
                print("\n\n👋 Prediction cancelled by user.")
                return None
            except Exception as e:
                print(f"   ❌ Error: {e}. Please try again.")

    return input_data


def predict_single_sample(model_package, input_data):
    """Make prediction for a single sample using enhanced model"""
    print("\n🔮 Making Prediction...")
    print("-" * 30)

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
        print("📊 Input data received")

        # Ensure correct feature order
        X_sample = sample_df[feature_names].copy()

        # Process categorical features
        categorical_features = ['model']
        for cat_feat in categorical_features:
            if cat_feat in feature_names and cat_feat in label_encoders:
                le = label_encoders[cat_feat]
                original_value = X_sample[cat_feat].iloc[0]

                # Handle unknown categories
                if pd.isna(original_value) or str(original_value) not in le.classes_:
                    print(f"   ⚠️  Unknown {cat_feat} value, using 'unknown'")
                    X_sample[cat_feat] = 'unknown'

                # Encode
                encoded_value = le.transform([str(X_sample[cat_feat].iloc[0])])[0]
                X_sample[cat_feat] = encoded_value
                print(f"   ✅ Encoded {cat_feat}: {original_value} → {encoded_value}")

        # Process numerical features (handle missing values)
        numerical_features = [f for f in feature_names if f not in categorical_features]
        if numerical_features:
            print("   🔄 Handling missing values...")
            # Count missing values before imputation
            missing_count = X_sample[numerical_features].isnull().sum().sum()
            if missing_count > 0:
                print(f"   📊 Found {missing_count} missing values (will be imputed)")

            X_sample[numerical_features] = imputer.transform(X_sample[numerical_features])
            print("   ✅ Missing values handled")

        # Scale features
        print("   🔄 Scaling features...")
        X_scaled = scaler.transform(X_sample)
        print("   ✅ Features scaled")

        # Make prediction
        print("   🔮 Generating predictions...")
        raw_prediction = model.predict(X_scaled)[0]  # Get first (and only) prediction
        prediction_floor = np.floor(raw_prediction)

        print("✅ Prediction completed successfully!")

        return raw_prediction, prediction_floor

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        print(f"📋 Error details:\n{traceback.format_exc()}")
        return None, None


def display_results(input_data, raw_prediction, prediction_floor, target_names):
    """Display the prediction results in an enhanced format"""
    print("\n" + "=" * 80)
    print("🎯 ENHANCED PREDICTION RESULTS")
    print("=" * 80)

    # Display input summary
    print("\n📊 INPUT SUMMARY:")
    print("-" * 25)
    for feature, value in input_data.items():
        if pd.isna(value):
            print(f"   {feature:<25}: NaN (imputed)")
        elif isinstance(value, float):
            print(f"   {feature:<25}: {value:.2f}")
        else:
            print(f"   {feature:<25}: {value}")

    # Display predictions with enhanced formatting
    print("\n🔮 PREDICTED TIME TO FAILURE (WEEKS):")
    print("-" * 45)
    print("Component          | Floor | Exact  | Status")
    print("-" * 55)

    for i, target in enumerate(target_names):
        component = target.replace('ttf_', '').replace('_weeks', '').upper()
        exact_weeks = raw_prediction[i]
        floor_weeks = prediction_floor[i]

        # Determine status and emoji
        if floor_weeks == 0:
            status = "🚨 CRITICAL"
        elif floor_weeks <= 2:
            status = "🔴 URGENT"
        elif floor_weeks <= 4:
            status = "🟡 SOON"
        elif floor_weeks <= 8:
            status = "🟢 NORMAL"
        else:
            status = "✅ GOOD"

        print(f"{component:<17s} | {floor_weeks:5.0f} | {exact_weeks:6.2f} | {status}")

    # Overall maintenance assessment
    min_weeks = min(prediction_floor)
    avg_weeks = np.mean(prediction_floor)
    critical_components = sum(1 for weeks in prediction_floor if weeks <= 2)

    print("\n" + "-" * 55)
    print(f"Average TTF: {avg_weeks:.1f} weeks")
    print(f"Minimum TTF: {min_weeks:.0f} weeks")
    print(f"Critical components: {critical_components}/{len(target_names)}")

    print(f"\n🚨 MAINTENANCE RECOMMENDATION:")
    print("-" * 35)
    if min_weeks == 0:
        print("   ⚠️  IMMEDIATE maintenance required!")
        print("   🔧 Stop operations and inspect immediately")
    elif min_weeks <= 2:
        print("   🔴 URGENT: Schedule maintenance within 2 weeks")
        print("   📅 Prioritize this machine in maintenance schedule")
    elif min_weeks <= 4:
        print("   🟡 Schedule maintenance within 1 month")
        print("   📊 Monitor closely and prepare maintenance")
    elif min_weeks <= 8:
        print("   🟢 Normal maintenance schedule (within 2 months)")
        print("   📈 Continue regular monitoring")
    else:
        print("   ✅ Machine in good condition")
        print("   📋 Follow standard maintenance schedule")

    # Component-specific recommendations
    print(f"\n🔧 COMPONENT-SPECIFIC PRIORITIES:")
    print("-" * 35)
    component_data = [(target.replace('ttf_', '').replace('_weeks', '').upper(),
                       prediction_floor[i], i)
                      for i, target in enumerate(target_names)]
    component_data.sort(key=lambda x: x[1])  # Sort by TTF

    for component, weeks, idx in component_data:
        priority = "HIGH" if weeks <= 2 else "MEDIUM" if weeks <= 4 else "LOW"
        print(f"   {component:<12s}: {weeks:3.0f} weeks ({priority} priority)")

    print("\n" + "=" * 80)


def create_example_data(model_package):
    """Create example data based on model requirements"""
    feature_names = model_package['feature_names']
    label_encoders = model_package['label_encoders']

    # Get valid model if available
    if 'model' in label_encoders:
        valid_models = list(label_encoders['model'].classes_)
        example_model = valid_models[0] if valid_models else 'model1'
    else:
        example_model = 'model1'

    # Create example data based on actual feature requirements
    example_data = {}

    for feature in feature_names:
        if feature == 'model':
            example_data[feature] = example_model
        elif feature == 'volt':
            example_data[feature] = 168.5
        elif feature == 'rotate':
            example_data[feature] = 415.2
        elif feature == 'pressure':
            example_data[feature] = 98.7
        elif feature == 'vibration':
            example_data[feature] = 45.3
        elif feature == 'age':
            example_data[feature] = 150
        elif feature == 'error_count':
            example_data[feature] = 2
        elif feature == 'days_since_comp1_maint':
            example_data[feature] = 15
        elif feature == 'days_since_comp2_maint':
            example_data[feature] = 8
        elif feature == 'days_since_comp3_maint':
            example_data[feature] = 22
        elif feature == 'days_since_comp4_maint':
            example_data[feature] = 12
        else:
            # Default value for any other features
            example_data[feature] = 100.0

    return example_data


def save_prediction_log(input_data, raw_prediction, prediction_floor, target_names):
    """Save prediction to a log file with enhanced format"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'model_version': 'enhanced_v2.0'
        }

        # Add input features
        log_entry.update(input_data)

        # Add predictions
        for i, target in enumerate(target_names):
            log_entry[f'{target}_floor'] = prediction_floor[i]
            log_entry[f'{target}_exact'] = raw_prediction[i]

        # Add summary statistics
        log_entry['min_ttf_weeks'] = np.min(prediction_floor)
        log_entry['avg_ttf_weeks'] = np.mean(prediction_floor)
        log_entry['max_ttf_weeks'] = np.max(prediction_floor)
        log_entry['critical_components'] = sum(1 for weeks in prediction_floor if weeks <= 2)

        # Save to CSV
        log_df = pd.DataFrame([log_entry])

        try:
            # Try to append to existing log
            existing_log = pd.read_csv('enhanced_prediction_log.csv')
            updated_log = pd.concat([existing_log, log_df], ignore_index=True)
            updated_log.to_csv('enhanced_prediction_log.csv', index=False)
            print(f"💾 Prediction appended to 'enhanced_prediction_log.csv'")
        except FileNotFoundError:
            # Create new log file
            log_df.to_csv('enhanced_prediction_log.csv', index=False)
            print(f"💾 New prediction log created: 'enhanced_prediction_log.csv'")

    except Exception as e:
        print(f"⚠️  Could not save to log: {e}")


def main():
    """Main function to run the enhanced manual prediction tool"""

    # Load model and show info
    model_package = load_model_info()
    if model_package is None:
        return

    while True:
        print("\n" + "=" * 60)
        print("🎯 ENHANCED PREDICTION OPTIONS:")
        print("1. 📝 Enter values manually")
        print("2. 🎲 Use example values (quick test)")
        print("3. 📋 Show model information")
        print("4. 📊 Show model performance")
        print("5. 👋 Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            # Manual input
            input_data = get_user_input(model_package)
            if input_data is None:
                continue

        elif choice == '2':
            # Example values
            print("\n🎲 Using example values for quick test...")
            input_data = create_example_data(model_package)

            print("📊 Example input values:")
            for feature, value in input_data.items():
                if pd.isna(value):
                    print(f"   {feature}: NaN")
                else:
                    print(f"   {feature}: {value}")

        elif choice == '3':
            # Show model information
            load_model_info()
            continue

        elif choice == '4':
            # Show model performance
            if 'performance_metrics' in model_package:
                print("\n📊 MODEL PERFORMANCE SUMMARY:")
                print("-" * 40)
                metrics_df = pd.DataFrame(model_package['performance_metrics'])
                print(metrics_df[['Target', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE']].round(4).to_string(
                    index=False))

                avg_r2 = metrics_df['Test_R2'].mean()
                avg_mape = metrics_df['Test_MAPE'].mean()
                print(f"\nOverall Performance:")
                print(f"   Average R²: {avg_r2:.4f}")
                print(f"   Average MAPE: {avg_mape:.2f}%")
            else:
                print("❌ Performance metrics not available")
            continue

        elif choice == '5':
            print("\n👋 Thank you for using the Enhanced TTTF Prediction Tool!")
            break

        else:
            print("❌ Invalid choice. Please try again.")
            continue

        # Make prediction
        raw_prediction, prediction_floor = predict_single_sample(model_package, input_data)

        if raw_prediction is not None:
            # Display results
            display_results(input_data, raw_prediction, prediction_floor,
                            model_package['target_names'])

            # Ask if user wants to save
            save_choice = input("\n💾 Save this prediction to log? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                save_prediction_log(input_data, raw_prediction, prediction_floor,
                                    model_package['target_names'])

        # Ask if user wants to continue
        continue_choice = input("\n🔄 Make another prediction? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n👋 Thank you for using the Enhanced TTTF Prediction Tool!")
            break


if __name__ == "__main__":
    main()