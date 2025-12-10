import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. Load your data
# (Replace 'your_data.csv' with your actual file path)
# df = pd.read_csv('your_data.csv')

# For demonstration, using the data snippet you provided:
df = pd.read_csv('results-updated-eval.csv')

def parse_json_col(json_str):
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}

def parse_lengths(row):
    try:
        data = json.loads(row['Related_Profile_JSON'])
        return pd.Series([data.get('source_length', 0), data.get('full_source_length', 0)])
    except:
        return pd.Series([0, 0])

df[['source_length', 'full_source_length']] = df.apply(parse_lengths, axis=1)

# 3. Analyze "Overhead" (Source - Full)
# Filter for rows where Source is strictly larger than Full
overhead_df = df[df['source_length'] > df['full_source_length']].copy()

# Calculate the difference
overhead_df['prompt_overhead'] = overhead_df['source_length'] - overhead_df['full_source_length']

print(f"Total rows where Source > Full: {len(overhead_df)}")
print(f"Average 'Overhead' (likely prompt length): {overhead_df['prompt_overhead'].mean():.2f} characters")

# 4. View specific examples
if not overhead_df.empty:
    print("\nSample rows with high overhead:")
    print(overhead_df[['Test_ID', 'Extraction_Strategy', 'source_length', 'full_source_length', 'prompt_overhead']].head())
    
# 2. Extract Lengths again to be sure
related_data = df['Related_Profile_JSON'].apply(parse_json_col).apply(pd.Series)
df['source_length'] = related_data['source_length']
df['full_source_length'] = related_data['full_source_length']

# 3. Create a Keyword-only subset
keyword_df = df[df['Extraction_Strategy'] == 'keyword'].copy()

# Calculate the "Compression Ratio" (Keyword Length / Full Length)
# A good keyword extraction should be around 0.1 to 0.3 (10-30%)
keyword_df['ratio'] = keyword_df['source_length'] / keyword_df['full_source_length']

print(f"Total Keyword Rows: {len(keyword_df)}")
print(f"Average Ratio: {keyword_df['ratio'].mean():.2f}")

# 4. Identify the "Suspicious" rows (Ratio > 0.9 means it's basically the full text)
suspicious_rows = keyword_df[keyword_df['ratio'] > 0.9]

print(f"\nFound {len(suspicious_rows)} suspicious rows where Keyword length is >90% of Full length.")
if len(suspicious_rows) > 0:
    print("\nSample of suspicious rows:")
    print(suspicious_rows[['Test_ID', 'source_length', 'full_source_length', 'ratio']].head())

# 5. Plot distribution to visualize the problem
plt.figure(figsize=(10, 5))
sns.histplot(keyword_df['ratio'], bins=20, kde=True)
plt.title('Distribution of Keyword Extraction Ratios\n(Should be clustered near 0.1 - 0.3)')
plt.xlabel('Ratio (Source Length / Full Length)')
plt.ylabel('Count')
plt.axvline(0.9, color='red', linestyle='--', label='Suspicious Threshold (0.9)')
plt.legend()
plt.show()


# 3. Extract Source Lengths from 'Related_Profile_JSON'
related_data = df['Related_Profile_JSON'].apply(parse_json_col).apply(pd.Series)
df['source_length'] = related_data.get('source_length', 0)
df['full_source_length'] = related_data.get('full_source_length', 0)

# 4. Prepare Data for Plotting
# We group by strategy and calculate the mean to handle multiple papers per strategy
metrics = ['Completeness_Score', 'Readability_Score', 'Conciseness_Score', 'strict_coverage_overall']
grouped_df = df.groupby('Extraction_Strategy')[metrics].mean().reset_index()

# Normalize coverage to 1-10 scale so it shows up on the same chart
grouped_df['Coverage (Scaled x10)'] = grouped_df['strict_coverage_overall'] * 10

# Reshape for Seaborn
plot_df = grouped_df.melt(
    id_vars='Extraction_Strategy',
    value_vars=['Completeness_Score', 'Readability_Score', 'Conciseness_Score', 'Coverage (Scaled x10)'],
    var_name='Metric',
    value_name='Score'
)

# 5. Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x='Metric', y='Score', hue='Extraction_Strategy')

plt.title('Strategy Comparison: Human Scores vs Coverage')
plt.ylabel('Score (0-10)')
plt.ylim(0, 10)
plt.xticks(rotation=15)
plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6. Display the Summary Table (with Lengths)
# Aggregating lengths as well to see averages
summary_cols = ['source_length', 'full_source_length'] + metrics
summary_table = df.groupby('Extraction_Strategy')[summary_cols].mean()
print("Average Scores & Lengths per Strategy:")
print(summary_table)