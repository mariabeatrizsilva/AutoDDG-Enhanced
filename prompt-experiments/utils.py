# utils.py
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional

# NOTE: The constants DATASET_NAME and RESULTS_FILE will be passed 
# or defined in the notebook, so they are not defined here.

def parse_scores(raw_score_text: str) -> dict:
    """Parses the 'Metric: Score' string output into a dictionary."""
    scores = {}
    lines = raw_score_text.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                # Store the key in a normalized way (lowercase, no spaces)
                scores[key.strip().lower()] = int(value.strip())
            except ValueError:
                # Handle cases where the value isn't an integer
                pass
    return scores

def log_result(
    prompt_name: str, 
    description_type: str, 
    description: str, 
    raw_scores: str, 
    dataset_name: str,       # Pass as argument now
    file_path: str,          # Pass as argument now
    related_profile: Optional[dict] = None
) -> None:
    """Logs the results of a single test run to a CSV file."""
    
    parsed_scores = parse_scores(raw_scores)
    
    # Extract the three core metrics for CSV columns
    completeness = parsed_scores.get('completeness', 0)
    conciseness = parsed_scores.get('conciseness', 0)
    readability = parsed_scores.get('readability', 0)
    
    # Serialize the related profile dict to a JSON string if it exists
    related_profile_json = json.dumps(related_profile) if related_profile else ""
    
    # Storing the full evaluation dictionary for detail
    raw_scores_json = json.dumps(parsed_scores) 
    
    new_row = {
        'Test_ID': f"{description_type}-{datetime.now().strftime('%H%M%S')}",
        'Dataset_Name': dataset_name, # Use passed argument
        'Prompt_Type': prompt_name,
        'Description_Source': description_type,
        'Description_Text': description.replace('\n', ' '), 
        'Related_Profile_JSON': related_profile_json,
        'Completeness_Score': completeness,
        'Conciseness_Score': conciseness,
        'Readability_Score': readability,
        'Raw_Scores_JSON': raw_scores_json, 
        'Evaluation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    df_new = pd.DataFrame([new_row])
    
    # Check if file exists to decide whether to write header
    header_needed = not os.path.exists(file_path)
    
    # Append to CSV
    df_new.to_csv(file_path, mode='a', header=header_needed, index=False)
    print(f"Logged {description_type} with Prompt {prompt_name} to {file_path}")