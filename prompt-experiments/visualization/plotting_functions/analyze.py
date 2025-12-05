import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

# Create output directory
output_dir = 'correlations'
os.makedirs(output_dir, exist_ok=True)

# Load the data
results_df = pd.read_csv('results-eval.csv')
preference_df = pd.read_csv('preference_evaluations_incremental.csv')

print("=" * 80)
print("DATA OVERVIEW")
print("=" * 80)
print(f"\nResults shape: {results_df.shape}")
print(f"Preference shape: {preference_df.shape}")

print("\n" + "=" * 80)
print("RESULTS DATA STRUCTURE")
print("=" * 80)
print(results_df.head())
print("\nUnique Description Sources:", results_df['Description_Source'].unique())
print("Unique Prompt Types:", results_df['Prompt_Type'].unique())

print("\n" + "=" * 80)
print("PREFERENCE DATA STRUCTURE")
print("=" * 80)
print(preference_df.head())

# Separate vanilla and augmented results
vanilla_df = results_df[results_df['Description_Source'] == 'Vanilla_AutoDDG'].copy()
augmented_df = results_df[results_df['Description_Source'] == 'Augmented_AutoDDG'].copy()

print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)
print(f"Vanilla records: {len(vanilla_df)}")
print(f"Augmented records: {len(augmented_df)}")
print(f"Augmented prompt types: {augmented_df['Prompt_Type'].unique()}")

# Merge preference data with augmented results
# Preference data compares vanilla (A) vs augmented (B)
preference_df['Dataset_Name_Clean'] = preference_df['Dataset_Name']
augmented_df['Dataset_Name_Clean'] = augmented_df['Dataset_Name']

# Merge on dataset name and prompt type
merged_df = augmented_df.merge(
    preference_df,
    left_on=['Dataset_Name', 'Prompt_Type'],
    right_on=['Dataset_Name', 'Prompt_Type_B'],
    how='left',
    suffixes=('', '_pref')
)

print("\n" + "=" * 80)
print("MERGED DATA")
print("=" * 80)
print(f"Merged records: {len(merged_df)}")
print(f"Records with preference data: {merged_df['Preference'].notna().sum()}")

# Select metrics for correlation analysis
metrics = [
    'bert_f1',
    'rouge1',
    'rouge2',
    'rougeL',
    'strict_coverage_overall',
    'lenient_coverage_overall',
    'Completeness_Score',
    'Conciseness_Score',
    'Readability_Score',
    'Score_B'  # Augmented score from preference evaluation
]

# Create a subset with only the metrics
metrics_df = merged_df[metrics].copy()

print("\n" + "=" * 80)
print("METRICS SUMMARY STATISTICS")
print("=" * 80)
print(metrics_df.describe())

# Calculate correlation matrix
print("\n" + "=" * 80)
print("PEARSON CORRELATION MATRIX")
print("=" * 80)
correlation_matrix = metrics_df.corr(method='pearson')
print(correlation_matrix.round(3))

# Save correlation matrix as formatted table
with open(f'{output_dir}/pearson_correlation_table.txt', 'w') as f:
    f.write("PEARSON CORRELATION MATRIX\n")
    f.write("=" * 120 + "\n\n")
    f.write(correlation_matrix.to_string())

# Calculate Spearman correlation (better for non-linear relationships)
print("\n" + "=" * 80)
print("SPEARMAN CORRELATION MATRIX")
print("=" * 80)
spearman_matrix = metrics_df.corr(method='spearman')
print(spearman_matrix.round(3))

# Save Spearman correlation matrix as formatted table
with open(f'{output_dir}/spearman_correlation_table.txt', 'w') as f:
    f.write("SPEARMAN CORRELATION MATRIX\n")
    f.write("=" * 120 + "\n\n")
    f.write(spearman_matrix.to_string())

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Pearson correlation heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, ax=axes[0], 
            cbar_kws={'label': 'Correlation'})
axes[0].set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Spearman correlation heatmap
sns.heatmap(spearman_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, ax=axes[1],
            cbar_kws={'label': 'Correlation'})
axes[1].set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved correlation heatmaps to correlation_heatmaps.png")

# Analyze key relationships
print("\n" + "=" * 80)
print("KEY METRIC RELATIONSHIPS")
print("=" * 80)

key_pairs = [
    ('strict_coverage_overall', 'Readability_Score'),
    ('strict_coverage_overall', 'Conciseness_Score'),
    ('strict_coverage_overall', 'Score_B'),
    ('bert_f1', 'Score_B'),
    ('Completeness_Score', 'strict_coverage_overall'),
    ('Readability_Score', 'Score_B'),
]

# Create a list to save key relationships
key_relationships = []

for metric1, metric2 in key_pairs:
    valid_data = metrics_df[[metric1, metric2]].dropna()
    if len(valid_data) > 2:
        pearson_r, pearson_p = pearsonr(valid_data[metric1], valid_data[metric2])
        spearman_r, spearman_p = spearmanr(valid_data[metric1], valid_data[metric2])
        print(f"\n{metric1} vs {metric2}:")
        print(f"  Pearson:  r={pearson_r:.3f}, p={pearson_p:.3f}")
        print(f"  Spearman: r={spearman_r:.3f}, p={spearman_p:.3f}")
        print(f"  n={len(valid_data)}")
        
        key_relationships.append({
            'Metric_1': metric1,
            'Metric_2': metric2,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'n': len(valid_data)
        })

# Save key relationships table
key_rel_df = pd.DataFrame(key_relationships)
key_rel_df.to_csv(f'{output_dir}/key_relationships.csv', index=False)
with open(f'{output_dir}/key_relationships_table.txt', 'w') as f:
    f.write("KEY METRIC RELATIONSHIPS\n")
    f.write("=" * 120 + "\n\n")
    f.write(key_rel_df.to_string(index=False))

# Compare vanilla vs augmented
print("\n" + "=" * 80)
print("VANILLA VS AUGMENTED COMPARISON")
print("=" * 80)

comparison_metrics = [
    'bert_f1',
    'rouge1',
    'strict_coverage_overall',
    'lenient_coverage_overall',
    'Completeness_Score',
    'Conciseness_Score',
    'Readability_Score',
]

comparison_df = pd.DataFrame({
    'Metric': comparison_metrics,
    'Vanilla_Mean': [vanilla_df[m].mean() for m in comparison_metrics],
    'Augmented_Mean': [augmented_df[m].mean() for m in comparison_metrics],
})
comparison_df['Difference'] = comparison_df['Augmented_Mean'] - comparison_df['Vanilla_Mean']
comparison_df['Percent_Change'] = (comparison_df['Difference'] / comparison_df['Vanilla_Mean'] * 100)

# Add preference win rate analysis
print("\n" + "=" * 80)
print("PREFERENCE WIN RATE")
print("=" * 80)

# Count preferences for augmented (B) vs vanilla (A)
preference_counts = preference_df['Preference'].value_counts()
total_preferences = len(preference_df['Preference'].dropna())

print(f"\nTotal evaluations: {total_preferences}")
print(f"Preference counts:\n{preference_counts}")

if total_preferences > 0:
    augmented_wins = preference_counts.get('B', 0)
    vanilla_wins = preference_counts.get('A', 0)
    ties = preference_counts.get('Tie', 0) + preference_counts.get('tie', 0)
    
    win_rate_augmented = (augmented_wins / total_preferences) * 100
    win_rate_vanilla = (vanilla_wins / total_preferences) * 100
    tie_rate = (ties / total_preferences) * 100
    
    print(f"\nAugmented (B) wins: {augmented_wins} ({win_rate_augmented:.1f}%)")
    print(f"Vanilla (A) wins: {vanilla_wins} ({win_rate_vanilla:.1f}%)")
    print(f"Ties: {ties} ({tie_rate:.1f}%)")
    
    # Add to comparison dataframe
    preference_summary = pd.DataFrame({
        'Metric': ['LLM_Preference_Win_Rate'],
        'Vanilla_Mean': [win_rate_vanilla],
        'Augmented_Mean': [win_rate_augmented],
    })
    preference_summary['Difference'] = preference_summary['Augmented_Mean'] - preference_summary['Vanilla_Mean']
    preference_summary['Percent_Change'] = preference_summary['Difference']  # Already in percentage
    
    comparison_df = pd.concat([comparison_df, preference_summary], ignore_index=True)
    
    print("\nUpdated comparison table with preference win rate")

print(comparison_df.to_string(index=False))

# Identify conflicting cases - high coverage but judge preferred vanilla
print("\n" + "=" * 80)
print("CONFLICTING CASES ANALYSIS")
print("=" * 80)

# Get vanilla data for comparison
vanilla_coverage = vanilla_df.set_index('Dataset_Name')['strict_coverage_overall']

# For merged_df (augmented with preference data)
conflicting_cases = []

for idx, row in merged_df.iterrows():
    dataset_name = row['Dataset_Name']
    
    # Skip if no preference data
    if pd.isna(row['Preference']):
        continue
    
    # Get vanilla coverage for this dataset
    if dataset_name in vanilla_coverage.index:
        vanilla_cov = vanilla_coverage[dataset_name]
        augmented_cov = row['strict_coverage_overall']
        coverage_improvement = augmented_cov - vanilla_cov
        
        # Check if judge preferred vanilla despite higher coverage
        if row['Preference'] == 'A' and coverage_improvement > 0.05:  # Augmented has 5%+ better coverage but lost
            conflicting_cases.append({
                'Dataset_Name': dataset_name,
                'Prompt_Type': row['Prompt_Type'],
                'Vanilla_Coverage': vanilla_cov,
                'Augmented_Coverage': augmented_cov,
                'Coverage_Improvement': coverage_improvement,
                'Judge_Preference': 'Vanilla',
                'Score_A': row['Score_A'],
                'Score_B': row['Score_B'],
                'Completeness': row['Completeness_Score'],
                'Readability': row['Readability_Score'],
                'Conciseness': row['Conciseness_Score'],
            })
        
        # Also check opposite: judge preferred augmented despite lower/similar coverage
        elif row['Preference'] == 'B' and coverage_improvement < 0.02:  # Minimal coverage improvement but won
            conflicting_cases.append({
                'Dataset_Name': dataset_name,
                'Prompt_Type': row['Prompt_Type'],
                'Vanilla_Coverage': vanilla_cov,
                'Augmented_Coverage': augmented_cov,
                'Coverage_Improvement': coverage_improvement,
                'Judge_Preference': 'Augmented',
                'Score_A': row['Score_A'],
                'Score_B': row['Score_B'],
                'Completeness': row['Completeness_Score'],
                'Readability': row['Readability_Score'],
                'Conciseness': row['Conciseness_Score'],
            })

if conflicting_cases:
    conflicting_df = pd.DataFrame(conflicting_cases)
    
    print(f"\nFound {len(conflicting_df)} conflicting cases")
    print("\nConflicting cases breakdown:")
    print(conflicting_df['Judge_Preference'].value_counts())
    
    # Save to file
    conflicting_df.to_csv(f'{output_dir}/conflicting_cases.csv', index=False)
    
    with open(f'{output_dir}/conflicting_cases_table.txt', 'w') as f:
        f.write("CONFLICTING CASES - Coverage vs Judge Preference\n")
        f.write("=" * 120 + "\n\n")
        f.write("Cases where coverage improved but judge preferred vanilla,\n")
        f.write("or coverage barely changed but judge preferred augmented\n\n")
        f.write(conflicting_df.to_string(index=False))
    
    print("\n✓ Saved conflicting cases analysis")
    
    # Print top 3 examples to console
    print("\n" + "-" * 80)
    print("TOP 3 EXAMPLES WHERE COVERAGE IMPROVED BUT JUDGE PREFERRED VANILLA:")
    print("-" * 80)
    vanilla_preferred = conflicting_df[conflicting_df['Judge_Preference'] == 'Vanilla'].nlargest(3, 'Coverage_Improvement')
    for idx, row in vanilla_preferred.iterrows():
        print(f"\nDataset: {row['Dataset_Name']}")
        print(f"  Prompt: {row['Prompt_Type']}")
        print(f"  Coverage: {row['Vanilla_Coverage']:.3f} → {row['Augmented_Coverage']:.3f} (+{row['Coverage_Improvement']:.3f})")
        print(f"  Judge Scores: Vanilla={row['Score_A']}, Augmented={row['Score_B']}")
        print(f"  Completeness={row['Completeness']}, Readability={row['Readability']}, Conciseness={row['Conciseness']}")
else:
    print("\nNo conflicting cases found with current thresholds")

# Also create a scatter plot showing coverage vs preference scores
print("\n" + "=" * 80)
print("CREATING COVERAGE VS PREFERENCE VISUALIZATION")
print("=" * 80)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot points colored by preference
for pref, color, label in [('A', 'red', 'Preferred Vanilla'), 
                             ('B', 'blue', 'Preferred Augmented'),
                             ('Tie', 'gray', 'Tie')]:
    mask = merged_df['Preference'] == pref
    ax.scatter(merged_df[mask]['strict_coverage_overall'], 
               merged_df[mask]['Score_B'],
               c=color, label=label, alpha=0.6, s=100)

ax.set_xlabel('Coverage Score (Augmented)', fontsize=12)
ax.set_ylabel('LLM Judge Score (Augmented)', fontsize=12)
ax.set_title('Coverage vs Judge Preference Score\n(Higher coverage doesn\'t always mean higher preference)', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/coverage_vs_preference_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved coverage vs preference scatter plot")

# Save detailed results
comparison_df.to_csv(f'{output_dir}/vanilla_vs_augmented_comparison.csv', index=False)
correlation_matrix.to_csv(f'{output_dir}/pearson_correlations.csv')
spearman_matrix.to_csv(f'{output_dir}/spearman_correlations.csv')

# Save comparison as formatted table
with open(f'{output_dir}/vanilla_vs_augmented_table.txt', 'w') as f:
    f.write("VANILLA VS AUGMENTED COMPARISON\n")
    f.write("=" * 120 + "\n\n")
    f.write(comparison_df.to_string(index=False))

# Create summary statistics table
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
summary_stats = metrics_df.describe()
print(summary_stats)

summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')
with open(f'{output_dir}/summary_statistics_table.txt', 'w') as f:
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 120 + "\n\n")
    f.write(summary_stats.to_string())

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print(f"All files saved to: {output_dir}")
print("\nGenerated files:")
for file in sorted(os.listdir(output_dir)):
    print(f"  ✓ {file}")