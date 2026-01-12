import pandas as pd
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
from pathlib import Path
import webbrowser

def generate_drift_report(reference_df, current_df):
    """
    Generates a report using the returned evaluation object from run().
    """
    # 1. Define the Data Definition
    definition = DataDefinition(
        regression=[Regression(target="DC_POWER", prediction="prediction")]
    )

    # 2. Wrap DataFrames into Dataset objects
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=definition)

    # 3. Initialize the Report configuration
    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset()
    ])

    # 4. RUN the report and CAPTURE the result snapshot
    # In the new API, run() returns the evaluation result
    evaluation = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset
    )
    
    # 5. Save the output from the EVALUATION object, not the report config
    report_path = Path("reports/drift_report.html")
    report_path.parent.mkdir(exist_ok=True)
    evaluation.save_html(str(report_path)) # <--- Changed 'report' to 'evaluation'

    # NEW: Automatically open the report in your default browser
    webbrowser.open(f"file://{report_path.resolve()}")
    
    print(f"✅ Drift report successfully generated at: {report_path}")