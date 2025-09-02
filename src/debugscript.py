"""
Diagnostic script to identify the exact column issue in your pipeline
Run this script first to understand what's happening with your data
"""

import pandas as pd
import numpy as np
import os

def diagnose_data_structure():
    """Diagnose the data structure to identify the column mismatch issue"""
    
    print("="*70)
    print("FLIGHT DELAY PIPELINE DIAGNOSTIC")
    print("="*70)
    
    # Check if source file exists
    source_file = 'notebook/data/flight_delay_2015_cleaned.csv'
    print(f"1. Checking source file: {source_file}")
    
    if not os.path.exists(source_file):
        print(f"❌ ERROR: Source file not found!")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Please check if the file path is correct.")
        return None
    else:
        print(f"✅ Source file found")
    
    # Read and examine the data
    print(f"\n2. Reading and examining data structure...")
    try:
        # Read a sample first
        df_sample = pd.read_csv(source_file, nrows=1000)
        print(f"✅ Successfully read {len(df_sample)} sample rows")
        print(f"   Shape: {df_sample.shape}")
        
        # Full dataset info
        df_full = pd.read_csv(source_file)
        print(f"   Full dataset shape: {df_full.shape}")
        
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        return None
    
    # 3. Column Analysis
    print(f"\n3. Column Analysis:")
    print(f"   Total columns: {len(df_full.columns)}")
    print(f"\n   All columns:")
    for i, col in enumerate(df_full.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # 4. Check for required columns
    print(f"\n4. Required Column Check:")
    required_cols = [
        'YEAR', 'MONTH', 'DAY', 'AIRLINE', 'ORIGIN_AIRPORT', 
        'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'ARRIVAL_DELAY'
    ]
    
    missing_required = []
    for col in required_cols:
        if col in df_full.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} - MISSING!")
            missing_required.append(col)
    
    if missing_required:
        print(f"\n   🚨 CRITICAL: Missing required columns: {missing_required}")
    
    # 5. Data Types Analysis
    print(f"\n5. Data Types:")
    print(df_full.dtypes.value_counts())
    
    # 6. Categorical Columns Analysis
    print(f"\n6. Potential Categorical Columns:")
    categorical_cols = df_full.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        unique_count = df_full[col].nunique()
        print(f"   {col}: {unique_count} unique values")
        if unique_count <= 10:
            sample_values = df_full[col].dropna().unique()[:5]
            print(f"      Sample values: {sample_values}")
    
    # 7. Check the specific columns that might be causing issues
    print(f"\n7. Checking Expected Categorical Columns:")
    expected_cat_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'PART_OF_DAY']
    for col in expected_cat_cols:
        if col in df_full.columns:
            unique_count = df_full[col].nunique()
            print(f"   ✅ {col}: {unique_count} unique values")
        else:
            print(f"   ❌ {col}: NOT FOUND")
    
    # 8. Check for ARRIVAL_DELAY to create target
    print(f"\n8. Target Creation Check:")
    if 'ARRIVAL_DELAY' in df_full.columns:
        print(f"   ✅ ARRIVAL_DELAY found")
        delay_stats = df_full['ARRIVAL_DELAY'].describe()
        print(f"   Delay statistics:")
        print(f"      Mean: {delay_stats['mean']:.2f}")
        print(f"      Min: {delay_stats['min']:.2f}")
        print(f"      Max: {delay_stats['max']:.2f}")
        
        # Check how many would be classified as delayed
        delayed_count = (df_full['ARRIVAL_DELAY'] >= 15).sum()
        total_count = len(df_full.dropna(subset=['ARRIVAL_DELAY']))
        delay_rate = delayed_count / total_count if total_count > 0 else 0
        print(f"      Flights delayed ≥15 min: {delayed_count:,} ({delay_rate:.1%})")
    else:
        print(f"   ❌ ARRIVAL_DELAY NOT FOUND - Cannot create target!")
    
    # 9. Missing Values Analysis
    print(f"\n9. Missing Values Analysis:")
    missing_summary = df_full.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    
    if len(missing_summary) > 0:
        print(f"   Columns with missing values:")
        for col, count in missing_summary.head(10).items():
            percentage = (count / len(df_full)) * 100
            print(f"      {col}: {count:,} ({percentage:.1f}%)")
    else:
        print(f"   ✅ No missing values found")
    
    # 10. Sample data
    print(f"\n10. Sample Data (first 3 rows):")
    print(df_full.head(3).to_string())
    
    return df_full

def simulate_transformation_issue(df):
    """Simulate the transformation to identify where it fails"""
    print(f"\n" + "="*70)
    print("SIMULATING TRANSFORMATION PIPELINE")
    print("="*70)
    
    try:
        # Step 1: Create target
        print("Step 1: Creating target variable...")
        if 'ARRIVAL_DELAY' in df.columns:
            df['DELAYED'] = (df['ARRIVAL_DELAY'] >= 15).astype(int)
            print("✅ Target variable created")
        else:
            print("❌ Cannot create target - ARRIVAL_DELAY missing")
            return
        
        # Step 2: Drop leakage columns
        print("Step 2: Dropping leakage columns...")
        leakage_cols = [
            'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'WHEELS_OFF', 'WHEELS_ON',
            'TAXI_IN', 'TAXI_OUT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
            'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
            'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DIVERTED', 'CANCELLED',
            'CANCELLATION_REASON'
        ]
        
        cols_to_drop = [col for col in leakage_cols if col in df.columns]
        print(f"   Dropping: {cols_to_drop}")
        df_processed = df.drop(columns=cols_to_drop)
        print(f"✅ Shape after dropping leakage: {df_processed.shape}")
        
        # Step 3: Check remaining columns
        print("Step 3: Analyzing remaining columns...")
        remaining_cols = list(df_processed.columns)
        print(f"   Remaining columns: {remaining_cols}")
        
        # Step 4: Identify categorical columns
        print("Step 4: Identifying categorical columns...")
        expected_cat_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'PART_OF_DAY']
        actual_cat_cols = [col for col in expected_cat_cols if col in df_processed.columns]
        
        print(f"   Expected categorical: {expected_cat_cols}")
        print(f"   Actually available: {actual_cat_cols}")
        print(f"   Missing categorical: {set(expected_cat_cols) - set(actual_cat_cols)}")
        
        if not actual_cat_cols:
            print("   🚨 WARNING: No expected categorical columns found!")
        
        # Step 5: Check if PART_OF_DAY can be created
        print("Step 5: Checking PART_OF_DAY creation...")
        if 'SCHEDULED_DEPARTURE' in df_processed.columns:
            print("   ✅ SCHEDULED_DEPARTURE available for PART_OF_DAY creation")
            # Try creating it
            sample_departures = df_processed['SCHEDULED_DEPARTURE'].dropna().head()
            print(f"   Sample departure times: {sample_departures.tolist()}")
        else:
            print("   ❌ SCHEDULED_DEPARTURE not available")
        
        print(f"\n✅ Transformation simulation completed successfully!")
        print(f"   Final shape would be: {df_processed.shape}")
        
    except Exception as e:
        print(f"❌ Transformation simulation failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")

def provide_recommendations():
    """Provide specific recommendations based on the analysis"""
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    recommendations = [
        "1. IMMEDIATE FIXES:",
        "   • Replace TargetEncoder with OneHotEncoder or LabelEncoder",
        "   • Add proper column existence checks before using TargetEncoder",
        "   • Create PART_OF_DAY feature before defining categorical columns",
        "",
        "2. ROBUST PIPELINE:",
        "   • Use the 'alternative_data_transformation.py' version I provided",
        "   • It handles missing columns gracefully",
        "   • Uses safer encoding methods",
        "",
        "3. DEBUGGING STEPS:",
        "   • Run this diagnostic script first",
        "   • Check which columns actually exist in your data",
        "   • Verify file paths are correct",
        "",
        "4. QUICK FIX:",
        "   • In get_data_transformer_object(), remove 'cols=categorical_cols' parameter",
        "   • Let TargetEncoder auto-detect columns",
        "   • Or switch to OneHotEncoder entirely"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    # Run the diagnostic
    df = diagnose_data_structure()
    
    if df is not None:
        # Simulate the transformation
        simulate_transformation_issue(df.copy())
    
    # Provide recommendations
    provide_recommendations()
    
    print(f"\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)