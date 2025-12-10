import pandas as pd
import json
import os
import argparse
from openai import OpenAI

# --- Import your custom modules ---
# Ensure these are in the python path or same directory
try:
    from enhanced_eval import evaluate_all
except ImportError:
    print("Error: Could not import 'enhanced_eval'. Make sure enhanced_eval.py is in the same directory.")
    exit(1)

# --- Configuration ---
DATABASE_PATH = "../src/autoddg/database.json"  # Adjust path if necessary relative to where you run the script
MODEL_CONFIG = {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model_name": "llama3.1:8b",
}

def load_reference_database(path):
    """Loads the database.json to create a lookup map for descriptions."""
    if not os.path.exists(path):
        print(f"Warning: Database file not found at {path}. Reference descriptions will be empty.")
        return {}
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
        
        # Create lookup: dataset_name -> description
        name_to_desc = {
            v["dataset_name"].strip(): v.get("description", "")
            for v in db.values()
            if "dataset_name" in v
        }
        print(f"Loaded reference database with {len(name_to_desc)} entries.")
        return name_to_desc
    except Exception as e:
        print(f"Error loading database: {e}")
        return {}

def process_evaluation(input_csv_path):
    # 1. Setup Output Filename
    # Splits "data/my_results.csv" into "data/my_results" and ".csv"
    base_name, ext = os.path.splitext(input_csv_path)
    output_csv_path = f"{base_name}-eval{ext}"
    
    print(f"Processing Input: {input_csv_path}")
    print(f"Target Output:  {output_csv_path}")

    # 2. Load Data
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_path}' not found.")
        return

    # 3. Initialize Client
    print(f"Initializing LLM Client ({MODEL_CONFIG['model_name']})...")
    client = OpenAI(api_key=MODEL_CONFIG["api_key"], base_url=MODEL_CONFIG["base_url"])
    
    # 4. Map Reference Descriptions
    # (We re-map this every time to ensure we have the latest descriptions from the DB)
    print("Mapping reference descriptions from database...")
    name_to_desc = load_reference_database(DATABASE_PATH)
    
    # Clean whitespace in dataset names for better matching
    df["Dataset_Name_Clean"] = df["Dataset_Name"].astype(str).str.strip()
    
    df["Reference_Description"] = (
        df["Dataset_Name_Clean"].map(name_to_desc)
    ).fillna("")
    
    # Drop the temp clean column
    df.drop(columns=["Dataset_Name_Clean"], inplace=True)

    # Check for missing descriptions
    missing_count = (df["Reference_Description"] == "").sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} rows have missing reference descriptions (dataset name not found in DB).")

    # 5. Run Evaluation Loop
    print(f"Starting evaluation on {len(df)} rows...")
    output_rows = []

    for index, row in df.iterrows():
        # Simple progress indicator
        if index > 0 and index % 5 == 0:
            print(f"  Processed {index}/{len(df)} rows...")

        try:
            # Run your enhanced evaluation function
            # Note: We pass the row as a Series; evaluate_all expects this
            metrics = evaluate_all(
                row=row,
                client=client,
                model_name=MODEL_CONFIG["model_name"]
            )
            
            # Merge original row data + new metrics into one dict
            row_dict = row.to_dict()
            row_dict.update(metrics)
            output_rows.append(row_dict)
            
        except Exception as e:
            print(f"Error evaluating row {index} (Dataset: {row.get('Dataset_Name', 'Unknown')}): {e}")
            # Append the original row so we don't lose data, mark error
            row_dict = row.to_dict()
            row_dict['evaluation_error'] = str(e)
            output_rows.append(row_dict)

    # 6. Save Results
    metrics_df = pd.DataFrame(output_rows)
    metrics_df.to_csv(output_csv_path, index=False)
    
    print("-" * 30)
    print(f"Success! Evaluation complete.")
    print(f"Saved to: {output_csv_path}")
    print("-" * 30)

if __name__ == "__main__":
    # Argument Parser setup
    parser = argparse.ArgumentParser(description="Run AutoDDG Evaluation Pipeline on a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file (e.g., results-updated.csv)")
    
    args = parser.parse_args()
    
    process_evaluation(args.input_file)