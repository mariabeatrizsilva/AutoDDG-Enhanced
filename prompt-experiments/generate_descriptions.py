"""
Script to run related work description experiments for a single extraction strategy.
"""

# %load_ext autoreload  <-- Not needed in a standard .py script
# %autoreload 2

import os
import json
import pandas as pd
from typing import Optional, Dict
from datetime import datetime
from openai import OpenAI

# --- Import your custom modules ---
# Ensure these are in your python path
from autoddg import AutoDDG
from autoddg.evaluation import BaseEvaluator
from prompts import ALL_RELATED_WORK_PROMPTS
from utils import log_result, run_description_experiment
from cache_utils import  load_profile_from_cache

# ==========================================
# 1. USER CONFIGURATION
# ==========================================

# Choose ONE strategy: "full", "keyword", or "llm_context"
CHOSEN_STRATEGY = "llm_context" 

# Define which prompts to test
PROMPTS_TO_TEST = {
    # "V0_Original": ALL_RELATED_WORK_PROMPTS["PROMPT_V0_ORIGINAL"],
    # "V1_Revised": ALL_RELATED_WORK_PROMPTS["V1_Revised"],
    # "V2_Hybrid": ALL_RELATED_WORK_PROMPTS["V2_Hybrid"],
    # "Structured_v1": ALL_RELATED_WORK_PROMPTS["Structured_v1"],
    # "Research_longv1": ALL_RELATED_WORK_PROMPTS["Research_longv1"],
    "Research_longv2": ALL_RELATED_WORK_PROMPTS["Research_longv2"],
    # "Research_shortv1": ALL_RELATED_WORK_PROMPTS["Research_shortv1"],
}

# Resume capability: Set to a dataset ID string to skip everything before it. 
# Set to None to run from the beginning.
TARGET_START_ID = 11303604 
# Example: TARGET_START_ID = "12345"

# ==========================================
# 2. SYSTEM CONFIGURATION
# ==========================================

MODEL_CONFIG = {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model_name": "llama3.1:8b",
}

DATABASE_PATH_REL = '../src/autoddg/database.json'
RESULTS_FILE_REL = 'results-updated-1.csv'
PROFILE_CACHE_DIR = '/profile_cache'

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(script_dir, DATABASE_PATH_REL)
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, os.pardir))
RESULTS_FILE = os.path.join(script_dir, RESULTS_FILE_REL)

# ==========================================
# 3. INITIALIZATION
# ==========================================

class Eval(BaseEvaluator):
    def __init__(self, model_name: str = MODEL_CONFIG["model_name"]):
        client = OpenAI(
            api_key=MODEL_CONFIG["api_key"], 
            base_url=MODEL_CONFIG["base_url"]
        )
        super().__init__(client=client, model_name=model_name)

print("Initializing AutoDDG and Evaluator...")
client = OpenAI(api_key=MODEL_CONFIG["api_key"], base_url=MODEL_CONFIG["base_url"])
auto_ddg = AutoDDG(client=client, model_name=MODEL_CONFIG["model_name"])
auto_ddg.set_evaluator(Eval())

# Load Database
print(f"Loading database from {DATABASE_PATH}...")
with open(DATABASE_PATH, 'r') as f:
    raw_database = json.load(f)
    # Convert keys to integers if they are dataset IDs
    database = {str(k): v for k, v in raw_database.items()}

print(f"Loaded {len(database)} datasets.")
print(f"Strategy selected: {CHOSEN_STRATEGY}")
print(f"Prompts selected: {list(PROMPTS_TO_TEST.keys())}")

# ==========================================
# 4. EXPERIMENT LOOP
# ==========================================

start_processing = False if TARGET_START_ID else True

for dataset_id, dataset_info in database.items():
    dataset_id = str(dataset_id) # Ensure ID is string for comparison
    current_dataset_name = dataset_info.get('dataset_name', 'Unknown')

    # Resume Logic
    if TARGET_START_ID and dataset_id == str(TARGET_START_ID):
        start_processing = True
    
    if not start_processing:
        # print(f"Skipping {dataset_id}...") # Uncomment if you want verbose skipping
        continue

    print(f"\nProcessing Dataset ID: {dataset_id} ({current_dataset_name})")

    # 1. Attempt to load the profile from cache
    # Note: We rely on load_profile_from_cache to return None if files don't exist
    cached_profiles = load_profile_from_cache(dataset_id)
    
    if cached_profiles is not None:
        try:
            # 2. Run the experiment for the SINGLE chosen strategy
            run_description_experiment(
                dataset_id=dataset_id,
                dataset_info=dataset_info,
                auto_ddg=auto_ddg,
                PROJECT_ROOT=PROJECT_ROOT,
                RESULTS_FILE=RESULTS_FILE,
                PROMPTS_TO_TEST=PROMPTS_TO_TEST,
                load_profile_from_cache=load_profile_from_cache,
                log_result=log_result,
                generateOriginal=False,      # Set to True if you need baseline generation
                extraction_strategy=CHOSEN_STRATEGY
            )
        except Exception as e:
            print(f"ERROR: Failed to process {dataset_id}. Reason: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print(f"SKIP: Cache missing for Dataset ID {dataset_id}. (Run run_with_caching first to generate profiles)")

print(f"\nAll experiments complete for strategy '{CHOSEN_STRATEGY}'.")
print(f"Results saved to: {RESULTS_FILE}")