import pandas as pd
import numpy as np

# Load the datasets from data folder
print("Loading datasets...")
tele = pd.read_csv('data/PdM_telemetry.csv', parse_dates=['datetime'])
maint = pd.read_csv('data/PdM_maint.csv', parse_dates=['datetime'])
fail = pd.read_csv('data/PdM_failures.csv', parse_dates=['datetime'])
error = pd.read_csv('data/PdM_errors.csv', parse_dates=['datetime'])
df_mach = pd.read_csv('data/PdM_machines.csv')

print("Dataset shapes:")
print(f"Telemetry: {tele.shape}")
print(f"Maintenance: {maint.shape}")
print(f"Failures: {fail.shape}")
print(f"Errors: {error.shape}")
print(f"Machines: {df_mach.shape}")

# Convert datetime columns to datetime type (already done in read_csv, but keeping for consistency)
tele['datetime'] = pd.to_datetime(tele['datetime'])
maint['datetime'] = pd.to_datetime(maint['datetime'])
fail['datetime'] = pd.to_datetime(fail['datetime'])
error['datetime'] = pd.to_datetime(error['datetime'])

# Add error count and aggregate errors
error['error_count'] = 1
error_agg = error.groupby(['machineID', 'datetime']).size().reset_index(name='error_count')
print(f"Aggregated errors shape: {error_agg.shape}")

# Add maintenance component column
maint['maint_comp'] = maint['comp']

# Step 1: Merge telemetry with machine data
print("\nStep 1: Merging telemetry with machine data...")
df1 = pd.merge(tele, df_mach, on='machineID', how='left')
print(f"After machine merge: {df1.shape}")

# Step 2: Merge with error data using merge_asof (backward direction)
print("Step 2: Merging with error data...")
df2 = pd.merge_asof(df1.sort_values('datetime'),
                    error_agg.sort_values('datetime'),
                    by='machineID',
                    on='datetime',
                    direction='backward')
print(f"After error merge: {df2.shape}")

# Step 3: Merge with maintenance data using merge_asof (backward direction)
print("Step 3: Merging with maintenance data...")
df3 = pd.merge_asof(df2.sort_values('datetime'),
                    maint.sort_values('datetime'),
                    by='machineID',
                    on='datetime',
                    direction='backward')
print(f"After maintenance merge: {df3.shape}")

# NEW APPROACH: Get first failure timestamp for each machine BEFORE merging
print("Step 4: Processing failure data...")
print(f"Total failure records: {len(fail)}")

# Get the FIRST failure timestamp for each machine
fail_timestamp = fail.groupby('machineID')['datetime'].min().reset_index()
fail_timestamp.columns = ['machineID', 'failure_datetime']
print(f"Machines with failures: {len(fail_timestamp)}")
print(f"Sample failure timestamps:")
print(fail_timestamp.head())

# Merge failure timestamps to main dataset
df4 = pd.merge(df3, fail_timestamp, on='machineID', how='left')
print(f"After failure merge: {df4.shape}")

# Check for null values
print(f"\nNull values in dataset:")
print(df4.isnull().sum())

# Fill null values
df4['error_count'] = df4['error_count'].fillna(0)

# Calculate TTF (Time to Failure) in hours
print("\nCalculating TTF (Time to Failure)...")

# Ensure datetime columns are datetime type
df4['datetime'] = pd.to_datetime(df4['datetime'])
df4['failure_datetime'] = pd.to_datetime(df4['failure_datetime'])

# For machines that have failures, calculate time difference
mask_has_failure = df4['failure_datetime'].notna()
df4.loc[mask_has_failure, 'ttf_hours'] = (
    df4.loc[mask_has_failure, 'failure_datetime'] -
    df4.loc[mask_has_failure, 'datetime']
).dt.total_seconds() / 3600

# For machines without failures, use max timestamp as reference
maxtime = df4['datetime'].max()
print(f"Maximum timestamp in dataset: {maxtime}")

# For machines without failures, calculate time to end of observation period
mask_no_failure = df4['failure_datetime'].isna()
df4.loc[mask_no_failure, 'ttf_hours'] = (
    maxtime - df4.loc[mask_no_failure, 'datetime']
).dt.total_seconds() / 3600

# Set failure_datetime for non-failing machines to maxtime for consistency
df4.loc[mask_no_failure, 'failure_datetime'] = maxtime

# Create failure flag: 1 if failure occurs within a reasonable time window (e.g., next 24-48 hours)
# This creates a binary target for predictive modeling
prediction_window_hours = 48  # Predict failures within next 48 hours
df4['failure_within_window'] = ((df4['ttf_hours'] > 0) &
                               (df4['ttf_hours'] <= prediction_window_hours)).astype(int)

# Remove rows where TTF is negative (telemetry after failure - these shouldn't be used)
print(f"Rows before removing post-failure data: {len(df4)}")
df4 = df4[df4['ttf_hours'] >= 0].copy()
print(f"Rows after removing post-failure data: {len(df4)}")

print(f"\nFinal dataset shape: {df4.shape}")
print(f"TTF statistics:")
print(df4['ttf_hours'].describe())

print(f"\nFailure distribution:")
print(f"Records with failures within {prediction_window_hours}h: {df4['failure_within_window'].sum()}")
print(f"Records without failures within {prediction_window_hours}h: {(1-df4['failure_within_window']).sum()}")

# Verify that different machines have different failure times
print(f"\nFailure datetime distribution by machine (first 10):")
failure_check = df4[df4['failure_datetime'] != maxtime].groupby('machineID')['failure_datetime'].first().head(10)
print(failure_check)

# Check column names
print(f"\nFinal columns: {list(df4.columns)}")

# Sort by machineID and datetime for better organization
df_final = df4.sort_values(['machineID', 'datetime']).reset_index(drop=True)

# Save to CSV file
output_filename = 'data/Final_with_TTF.csv'
print(f"\nSaving processed data to {output_filename}...")
df_final.to_csv(output_filename, index=False)

print(f"Successfully saved processed dataset with {len(df_final)} records to {output_filename}")
print(f"Dataset includes TTF (Time to Failure) in hours for predictive maintenance modeling.")

# Display sample of final data
print(f"\nSample of final processed data:")
sample_cols = ['machineID', 'datetime', 'volt', 'rotate', 'pressure', 'vibration',
               'model', 'age', 'error_count', 'failure_datetime', 'ttf_hours', 'failure_within_window']
print(df_final[sample_cols].head(10))

# Show some statistics about TTF distribution by machine
print(f"\nTTF distribution by machine (sample):")
ttf_by_machine = df_final.groupby('machineID')['ttf_hours'].agg(['min', 'max', 'mean']).head(10)
print(ttf_by_machine)