import pandas as pd
import sys
import os

def filter_and_save_scores(input_filename="results-eval.csv", output_filename="results-eval-scoresonly.csv"):
    """
    Reads an input CSV, filters it to keep only specified columns related to
    dataset information and evaluation scores, and saves the result to a new CSV.
    """
    # 1. Define the exact list of columns to keep, as requested by the user
    columns_to_keep = [
        'Dataset_Name',
        'Description_Source',
        'Prompt_Type',
        'bert_precision',
        'bert_recall',
        'bert_f1',
        'rouge1',
        'rouge2',
        'rougeL',
        'rougeLsum',
        'strict_coverage_overall',
        'strict_coverage_basic_info',
        'strict_coverage_data_characteristics',
        'strict_coverage_provenance',
        'strict_coverage_usage_context',
        'strict_coverage_quality_and_limitations',
        'lenient_coverage_overall',
        'lenient_coverage_basic_info',
        'lenient_coverage_data_characteristics',
        'lenient_coverage_provenance',
        'lenient_coverage_usage_context',
        'lenient_coverage_quality_and_limitations'
    ]

    print(f"Starting column filtering process...")

    # 2. Check if the input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        print("Please ensure your CSV data is saved as 'input.csv' in the same directory.")
        sys.exit(1)

    try:
        # 3. Read the entire CSV file into a pandas DataFrame
        df = pd.read_csv(input_filename)
        print(f"Successfully loaded '{input_filename}' with {len(df.columns)} columns and {len(df)} rows.")

        # 4. Select only the columns defined in columns_to_keep
        # This will raise an error if any requested column is missing, preventing silent data loss.
        df_filtered = df[columns_to_keep]

        # 5. Save the filtered DataFrame to the new CSV file
        df_filtered.to_csv(output_filename, index=False)

        print(f"\n--- SUCCESS ---")
        print(f"Filtered data saved to '{output_filename}'.")
        print(f"The new file contains {len(df_filtered.columns)} columns and {len(df_filtered)} rows.")

    except KeyError as e:
        print(f"\nError: One or more required columns were not found in the input CSV.")
        print(f"Missing column: {e}")
        print("Please verify the column names in your CSV match the required list.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    filter_and_save_scores()