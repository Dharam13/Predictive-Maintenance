import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# Convert datetime columns to datetime type
tele['datetime'] = pd.to_datetime(tele['datetime'])
maint['datetime'] = pd.to_datetime(maint['datetime'])
fail['datetime'] = pd.to_datetime(fail['datetime'])
error['datetime'] = pd.to_datetime(error['datetime'])

# Aggregate errors by machine and datetime
error_agg = error.groupby(['machineID', 'datetime']).size().reset_index(name='error_count')
print(f"Aggregated errors shape: {error_agg.shape}")

# Step 1: Merge telemetry with machine data
print("\nStep 1: Merging telemetry with machine data...")
df1 = pd.merge(tele, df_mach, on='machineID', how='left')
print(f"After machine merge: {df1.shape}")

# Step 2: Merge with error data using merge_asof (backward direction)
print("Step 2: Merging with error data...")

# Proper sorting for merge_asof: sort by 'by' column first, then by 'on' column
print("Sorting dataframes for merge_asof...")
df1_sorted = df1.sort_values(['machineID', 'datetime']).reset_index(drop=True)
error_agg_sorted = error_agg.sort_values(['machineID', 'datetime']).reset_index(drop=True)

# Verify sorting
print("Verifying sort order...")
print(
    f"df1_sorted is properly sorted: {df1_sorted.groupby('machineID')['datetime'].apply(lambda x: x.is_monotonic_increasing).all()}")
print(
    f"error_agg_sorted is properly sorted: {error_agg_sorted.groupby('machineID')['datetime'].apply(lambda x: x.is_monotonic_increasing).all()}")

# Alternative approach: Use regular merge with tolerance instead of merge_asof
print("Using alternative merge approach...")

# Create a cross join of all machine-datetime combinations from telemetry with error data
df2_list = []
for machine_id in df1_sorted['machineID'].unique():
    if machine_id % 50 == 0:
        print(f"Processing machine {machine_id} for error merge...")

    machine_tele = df1_sorted[df1_sorted['machineID'] == machine_id].copy()
    machine_error = error_agg_sorted[error_agg_sorted['machineID'] == machine_id].copy()

    if len(machine_error) == 0:
        machine_tele['error_count'] = 0
        df2_list.append(machine_tele)
        continue

    # For each telemetry record, find the most recent error count
    machine_tele['error_count'] = 0
    for idx, row in machine_tele.iterrows():
        recent_errors = machine_error[machine_error['datetime'] <= row['datetime']]
        if len(recent_errors) > 0:
            latest_error = recent_errors.loc[recent_errors['datetime'].idxmax()]
            machine_tele.loc[idx, 'error_count'] = latest_error['error_count']

    df2_list.append(machine_tele)

df2 = pd.concat(df2_list, ignore_index=True)
print(f"After error merge: {df2.shape}")

# Step 3: Advanced TTF calculation considering multiple failures and maintenance
print("Step 3: Calculating TTF with multiple failures and maintenance events...")


def calculate_ttf_advanced(df, maint_df, fail_df):
    """
    Calculate TTF considering multiple failures and maintenance events.
    Each maintenance event resets the failure timeline for that component.
    """
    results = []

    # Group by machine to process each machine separately
    unique_machines = df['machineID'].unique()
    total_machines = len(unique_machines)

    for i, machine_id in enumerate(unique_machines):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing machine {machine_id}... ({i + 1}/{total_machines})")

        # Get data for this machine
        machine_tele = df[df['machineID'] == machine_id].sort_values('datetime').copy()
        machine_maint = maint_df[maint_df['machineID'] == machine_id].sort_values('datetime')
        machine_fail = fail_df[fail_df['machineID'] == machine_id].sort_values('datetime')

        # Get unique components for this machine
        components = set()
        if len(machine_maint) > 0:
            components.update(machine_maint['comp'].unique())
        if len(machine_fail) > 0:
            components.update(machine_fail['failure'].unique())

        if len(components) == 0:
            # No maintenance or failure data for this machine
            for idx, tele_row in machine_tele.iterrows():
                row_data = tele_row.to_dict()
                row_data['failure_within_48h'] = 0
                row_data['min_ttf_hours'] = 999999  # Very large number indicating no failure expected
                results.append(row_data)
            continue

        # For each telemetry record, calculate TTF for each component
        for idx, tele_row in machine_tele.iterrows():
            current_time = tele_row['datetime']
            row_data = tele_row.to_dict()

            # Initialize TTF values for each component
            component_ttfs = {}
            has_failure_within_window = False

            for comp in components:
                # Get maintenance and failure events for this component
                comp_maint = machine_maint[machine_maint['comp'] == comp]
                comp_fail = machine_fail[machine_fail['failure'] == comp]

                # Find the most recent maintenance before current time
                recent_maint = comp_maint[comp_maint['datetime'] <= current_time]
                last_maint_time = recent_maint['datetime'].max() if len(recent_maint) > 0 else pd.NaT

                # Find the next failure after current time
                # If there was recent maintenance, only consider failures after that maintenance
                if pd.notna(last_maint_time):
                    future_failures = comp_fail[
                        (comp_fail['datetime'] > current_time) &
                        (comp_fail['datetime'] > last_maint_time)
                        ]
                else:
                    future_failures = comp_fail[comp_fail['datetime'] > current_time]

                # Calculate TTF for this component
                if len(future_failures) > 0:
                    next_failure_time = future_failures['datetime'].min()
                    ttf_hours = (next_failure_time - current_time).total_seconds() / 3600
                else:
                    # No future failure, use end of observation period
                    max_time = df['datetime'].max()
                    ttf_hours = (max_time - current_time).total_seconds() / 3600

                component_ttfs[f'ttf_{comp}_hours'] = ttf_hours

                # Check if failure is within prediction window (48 hours)
                if len(future_failures) > 0 and ttf_hours <= 48:
                    has_failure_within_window = True

            # Add component TTFs to row data
            row_data.update(component_ttfs)

            # Overall failure flag (any component failing within window)
            row_data['failure_within_48h'] = 1 if has_failure_within_window else 0

            # Minimum TTF across all components (most critical)
            if component_ttfs:
                min_ttf = min(component_ttfs.values())
                row_data['min_ttf_hours'] = max(0, min_ttf)  # Ensure non-negative
            else:
                row_data['min_ttf_hours'] = 999999

            # Add maintenance recency features
            for comp in components:
                comp_maint = machine_maint[machine_maint['comp'] == comp]
                recent_maint = comp_maint[comp_maint['datetime'] <= current_time]

                if len(recent_maint) > 0:
                    last_maint_time = recent_maint['datetime'].max()
                    days_since_maint = (current_time - last_maint_time).total_seconds() / (24 * 3600)
                    row_data[f'days_since_{comp}_maint'] = max(0, days_since_maint)  # Ensure non-negative
                else:
                    row_data[f'days_since_{comp}_maint'] = np.inf

            results.append(row_data)

    return pd.DataFrame(results)


# Apply the advanced TTF calculation
print("Calculating advanced TTF...")
df_final = calculate_ttf_advanced(df2, maint, fail)

print(f"\nFinal dataset shape: {df_final.shape}")

# Display statistics
failure_count = df_final['failure_within_48h'].sum()
total_count = len(df_final)
print(f"\nFailure prediction statistics:")
print(f"Records with failure within 48h: {failure_count}")
print(f"Records without failure within 48h: {total_count - failure_count}")
print(f"Failure rate: {failure_count / total_count:.4f}")

print(f"\nMinimum TTF statistics:")
print(df_final['min_ttf_hours'].describe())

# Component-wise TTF statistics
components = ['comp1', 'comp2', 'comp3', 'comp4']
for comp in components:
    if f'ttf_{comp}_hours' in df_final.columns:
        print(f"\n{comp} TTF statistics:")
        comp_ttf = df_final[f'ttf_{comp}_hours']
        print(f"  Mean: {comp_ttf.mean():.1f} hours")
        print(f"  Median: {comp_ttf.median():.1f} hours")
        print(f"  Failures within 48h: {(comp_ttf <= 48).sum()}")

# Maintenance recency statistics
print(f"\nMaintenance recency statistics:")
for comp in components:
    if f'days_since_{comp}_maint' in df_final.columns:
        maint_days = df_final[f'days_since_{comp}_maint']
        maint_days_finite = maint_days[maint_days != np.inf]
        if len(maint_days_finite) > 0:
            print(f"  {comp}: Mean {maint_days_finite.mean():.1f} days, "
                  f"Median {maint_days_finite.median():.1f} days since last maintenance")

# Data quality checks
print(f"\nData quality checks:")
inf_cols = [f'days_since_{comp}_maint' for comp in components if f'days_since_{comp}_maint' in df_final.columns]
if inf_cols:
    inf_count = (df_final[inf_cols] == np.inf).any(axis=1).sum()
    print(f"Records with infinite maintenance days: {inf_count}")

negative_ttf_count = (df_final['min_ttf_hours'] < 0).sum()
print(f"Records with negative TTF: {negative_ttf_count}")

# Remove records with negative TTF (shouldn't happen with new logic, but safety check)
if negative_ttf_count > 0:
    print("Removing records with negative TTF...")
    df_final = df_final[df_final['min_ttf_hours'] >= 0].copy()
    print(f"Final dataset shape after cleaning: {df_final.shape}")

# Sort by machineID and datetime
df_final = df_final.sort_values(['machineID', 'datetime']).reset_index(drop=True)

# Save to CSV file
output_filename = 'data/Advanced_TTF_Dataset.csv'
print(f"\nSaving processed data to {output_filename}...")
df_final.to_csv(output_filename, index=False)

print(f"Successfully saved processed dataset with {len(df_final)} records to {output_filename}")

# Display sample of final data
print(f"\nSample of final processed data:")
sample_cols = ['machineID', 'datetime', 'volt', 'rotate', 'pressure', 'vibration',
               'model', 'age', 'error_count', 'min_ttf_hours', 'failure_within_48h']

# Add component TTF columns if they exist
for comp in components:
    if f'ttf_{comp}_hours' in df_final.columns:
        sample_cols.append(f'ttf_{comp}_hours')
    if f'days_since_{comp}_maint' in df_final.columns:
        sample_cols.append(f'days_since_{comp}_maint')

available_cols = [col for col in sample_cols if col in df_final.columns]
print(df_final[available_cols].head(10))

print(f"\nDataset is ready for predictive maintenance modeling!")
print(f"Key features:")
print(f"- Component-specific TTF: ttf_comp1_hours, ttf_comp2_hours, etc.")
print(f"- Maintenance recency: days_since_comp1_maint, days_since_comp2_maint, etc.")
print(f"- Overall failure prediction: failure_within_48h")
print(f"- Minimum TTF across components: min_ttf_hours")