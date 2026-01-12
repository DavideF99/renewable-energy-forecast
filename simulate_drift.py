import pandas as pd
import numpy as np
import joblib
from src.monitoring import generate_drift_report
from src.features.build_features import create_features 

def run_simulation():
    # 1. Load data
    ref_df = pd.read_csv("data/reference_data.csv")
    ref_df['DATE_TIME'] = pd.to_datetime(ref_df['DATE_TIME'])
    
    # 2. Load the trained model
    model = joblib.load("models/solar_model.joblib") #

    # 3. Create 'Production' data with simulated drift
    prod_df = ref_df.copy()
    prod_df['IRRADIATION'] = prod_df['IRRADIATION'] * 1.3
    prod_df['DC_POWER'] = prod_df['DC_POWER'] * 0.85

    # 4. Use your existing function for Feature Engineering
    # This automatically creates lag, rolling, and cyclic features
    ref_df_feat = create_features(ref_df)
    prod_df_feat = create_features(prod_df)

    # 5. Define Feature List (Matches what the model expects)
    features = [
        'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour_sin',
       'hour_cos', 'dc_power_lag_1', 'irrad_lag_1', 'irrad_rolling_avg_1h'
    ]

    # 6. Generate Predictions using the engineered dataframes
    # We add the predictions back to the dataframes passed to the report
    ref_df_feat['prediction'] = model.predict(ref_df_feat[features])
    prod_df_feat['prediction'] = model.predict(prod_df_feat[features])

    print("📊 Predictions generated using original build_features logic.")
    
    # 7. Generate the Monitoring Report
    generate_drift_report(ref_df_feat, prod_df_feat)

if __name__ == "__main__":
    run_simulation()