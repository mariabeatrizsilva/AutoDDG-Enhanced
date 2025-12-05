import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
# Make plots show full prompt names instead of being cut off
plt.rcParams['figure.autolayout'] = False  # We'll use tight_layout instead

# Load the data
results_df = pd.read_csv('results-eval.csv')
preference_df = pd.read_csv('preference_evaluations_incremental.csv')

print("=" * 80)
print("ANALYSIS BY PROMPT TYPE")
print("=" * 80)

# Separate vanilla and augmented
vanilla_df = results_df[results_df['Description_Source'] == 'Vanilla_AutoDDG'].copy()
augmented_df = results_df[results_df['Description_Source'] == 'Augmented_AutoDDG'].copy()

print(f"\nVanilla records: {len(vanilla_df)}")
print(f"Augmented records: {len(augmented_df)}")
print(f"\nPrompt types found: {sorted(augmented_df['Prompt_Type'].unique())}")

# Create a mapping of prompt types to descriptions
print("\n" + "=" * 80)
print("PROMPT TYPE DESCRIPTIONS")
print("=" * 80)

prompt_descriptions = {
    'V1_Revised': 'V1 Revised - Basic augmentation with citations',
    'V2_Hybrid': 'V2 Hybrid - Combination approach',
    'Structured_v1': 'Structured v1 - Structured format',
    'Structured_v1-json': 'Structured v1 JSON - JSON-based structure',
    'Structured_v1-newchunking': 'Structured v1 New Chunking - Improved text chunking',
    'Structured_v1-llm-relevance-chunks': 'Structured v1 LLM Relevance - LLM-filtered chunks',
    'Structured_v1-llm-relevance-chunks-new-integration': 'Structured v1 New Integration - Enhanced integration',
    'Structured_v1-llm-relevance-chunks-remove-references': 'Structured v1 Remove Refs - References removed',
    'research_long': 'Research Long - Detailed research context',
    'research_short': 'Research Short - Concise research context',
    'T': 'T - Template-based',
    'research_long_v2': 'Research Long v2 - Enhanced detailed context'
}

for prompt_type in sorted(augmented_df['Prompt_Type'].unique()):
    if pd.isna(prompt_type):
        continue
    description = prompt_descriptions.get(prompt_type, f'{prompt_type} - No description available')
    count = len(augmented_df[augmented_df['Prompt_Type'] == prompt_type])
    print(f"  {prompt_type:50s} ({count:2d} samples) - {description}")

# Merge with preference data
merged_df = augmented_df.merge(
    preference_df,
    left_on=['Dataset_Name', 'Prompt_Type'],
    right_on=['Dataset_Name', 'Prompt_Type_B'],
    how='left',
    suffixes=('', '_pref')
)

print(f"\nMerged records with preference: {merged_df['Preference'].notna().sum()}")

# Define metrics to analyze
metrics = [
    'bert_f1',
    'rouge1',
    'strict_coverage_overall',
    'lenient_coverage_overall',
    'Completeness_Score',
    'Conciseness_Score',
    'Readability_Score',
]

# Calculate vanilla baseline (mean across all vanilla descriptions)
vanilla_means = vanilla_df[metrics].mean()

print("\n" + "=" * 80)
print("VANILLA BASELINE")
print("=" * 80)
print(vanilla_means.round(3))

# Group by prompt type and calculate means
print("\n" + "=" * 80)
print("METRICS BY PROMPT TYPE")
print("=" * 80)

prompt_comparison = augmented_df.groupby('Prompt_Type')[metrics].mean()
print("\n", prompt_comparison.round(3))

# Calculate improvement over vanilla for each prompt
print("\n" + "=" * 80)
print("IMPROVEMENT OVER VANILLA (by Prompt Type)")
print("=" * 80)

improvement_df = prompt_comparison.copy()
for metric in metrics:
    improvement_df[metric] = prompt_comparison[metric] - vanilla_means[metric]

print("\n", improvement_df.round(3))

# Calculate percent change
print("\n" + "=" * 80)
print("PERCENT CHANGE FROM VANILLA (by Prompt Type)")
print("=" * 80)

pct_change_df = prompt_comparison.copy()
for metric in metrics:
    pct_change_df[metric] = ((prompt_comparison[metric] - vanilla_means[metric]) / vanilla_means[metric] * 100)

print("\n", pct_change_df.round(2))

# Reset index to avoid ambiguity
pct_change_df = pct_change_df.reset_index()

# Add preference statistics by prompt type
print("\n" + "=" * 80)
print("PREFERENCE STATISTICS BY PROMPT TYPE")
print("=" * 80)

preference_stats = []
for prompt_type in merged_df['Prompt_Type'].unique():
    if pd.isna(prompt_type):
        continue
    
    prompt_prefs = merged_df[merged_df['Prompt_Type'] == prompt_type]
    
    total = prompt_prefs['Preference'].notna().sum()
    if total == 0:
        continue
    
    wins_b = (prompt_prefs['Preference'] == 'B').sum()
    wins_a = (prompt_prefs['Preference'] == 'A').sum()
    ties = (prompt_prefs['Preference'].isin(['Tie', 'tie'])).sum()
    
    avg_score_a = prompt_prefs['Score_A'].mean()
    avg_score_b = prompt_prefs['Score_B'].mean()
    
    preference_stats.append({
        'Prompt_Type': prompt_type,
        'Total_Evaluations': total,
        'Augmented_Wins': wins_b,
        'Vanilla_Wins': wins_a,
        'Ties': ties,
        'Win_Rate_%': (wins_b / total * 100) if total > 0 else 0,
        'Avg_Score_Vanilla': avg_score_a,
        'Avg_Score_Augmented': avg_score_b,
        'Score_Difference': avg_score_b - avg_score_a
    })

pref_stats_df = pd.DataFrame(preference_stats)
print("\n", pref_stats_df.to_string(index=False))

# Save results
improvement_df.to_csv('correlations/prompt_comparison_improvement.csv')
pct_change_df.to_csv('correlations/prompt_comparison_pct_change.csv', index=False)
pref_stats_df.to_csv('correlations/prompt_preference_stats.csv', index=False)

print("\n" + "=" * 80)
print("IDENTIFYING BEST PERFORMING PROMPT")
print("=" * 80)

# Rank prompts by different criteria
rankings = pd.DataFrame({
    'Prompt_Type': pct_change_df['Prompt_Type'],
    'Coverage_Improvement': pct_change_df['strict_coverage_overall'],
    'Completeness_Improvement': pct_change_df['Completeness_Score'],
    'Readability_Improvement': pct_change_df['Readability_Score'],
})

# Add preference win rate if available
if not pref_stats_df.empty:
    rankings = rankings.merge(pref_stats_df[['Prompt_Type', 'Win_Rate_%']], on='Prompt_Type', how='left')
else:
    rankings['Win_Rate_%'] = np.nan

print("\n", rankings.to_string(index=False))

# Find best prompt by different criteria
print("\n" + "-" * 80)
print("BEST PROMPTS BY CRITERION:")
print("-" * 80)
print(f"Best Coverage: {rankings.loc[rankings['Coverage_Improvement'].idxmax(), 'Prompt_Type']}")
print(f"Best Completeness: {rankings.loc[rankings['Completeness_Improvement'].idxmax(), 'Prompt_Type']}")
print(f"Best Readability: {rankings.loc[rankings['Readability_Improvement'].idxmax(), 'Prompt_Type']}")
if 'Win_Rate_%' in rankings.columns and rankings['Win_Rate_%'].notna().any():
    print(f"Best Win Rate: {rankings.loc[rankings['Win_Rate_%'].idxmax(), 'Prompt_Type']}")

# Calculate a composite score (equal weighting)
rankings['Composite_Score'] = (
    rankings['Coverage_Improvement'] + 
    rankings['Completeness_Improvement'] + 
    rankings['Readability_Improvement']
) / 3

if 'Win_Rate_%' in rankings.columns and rankings['Win_Rate_%'].notna().any():
    # Normalize win rate to same scale as improvements
    rankings['Win_Rate_Normalized'] = (rankings['Win_Rate_%'] - 50) / 10  # Convert 50-100% to 0-5 range
    rankings['Composite_Score_With_Pref'] = (
        rankings['Coverage_Improvement'] + 
        rankings['Completeness_Improvement'] + 
        rankings['Readability_Improvement'] +
        rankings['Win_Rate_Normalized']
    ) / 4

print(f"\nBest Overall (Composite): {rankings.loc[rankings['Composite_Score'].idxmax(), 'Prompt_Type']}")
if 'Composite_Score_With_Pref' in rankings.columns:
    print(f"Best Overall (with Preference): {rankings.loc[rankings['Composite_Score_With_Pref'].idxmax(), 'Prompt_Type']}")

rankings.to_csv('correlations/prompt_rankings.csv', index=False)

print("\n" + "=" * 80)
print("PROMPT TYPE SUMMARY (sorted by composite score)")
print("=" * 80)

# Create readable summary
summary_table = rankings[['Prompt_Type', 'Coverage_Improvement', 'Completeness_Improvement', 
                          'Readability_Improvement', 'Win_Rate_%', 'Composite_Score']].copy()
summary_table = summary_table.sort_values('Composite_Score', ascending=False)

# Add readable names
summary_table['Description'] = summary_table['Prompt_Type'].map(prompt_descriptions)

print("\n", summary_table.to_string(index=False))

# Save this too
summary_table.to_csv('correlations/prompt_summary_readable.csv', index=False)

# VISUALIZATION 1: Heatmap of improvements by prompt
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS...")
print("=" * 80)

fig, ax = plt.subplots(figsize=(14, 6))

# Select key metrics for heatmap
heatmap_metrics = ['strict_coverage_overall', 'Completeness_Score', 
                   'Conciseness_Score', 'Readability_Score']

# Need to set index back for heatmap
pct_change_for_heatmap = pct_change_df.set_index('Prompt_Type')
heatmap_data = pct_change_for_heatmap[heatmap_metrics].T

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Percent Change from Vanilla'},
            linewidths=1, linecolor='gray')

ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
ax.set_title('Performance by Prompt Type (% Change from Vanilla)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_yticklabels(['Coverage', 'Completeness', 'Conciseness', 'Readability'], rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

plt.tight_layout()
plt.savefig('correlations/prompt_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prompt_heatmap.png")

# BONUS VISUALIZATION: ROUGE and BERT scores by prompt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

similarity_metrics = ['bert_f1', 'rouge1']
titles = ['BERT-F1 Score', 'ROUGE-1 Score']

for idx, (metric, title) in enumerate(zip(similarity_metrics, titles)):
    data = pct_change_df.set_index('Prompt_Type')[metric]
    
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in data]
    bars = axes[idx].bar(range(len(data)), data, color=colors, alpha=0.8, 
                        edgecolor='black', linewidth=1)
    
    for i, (bar, val) in enumerate(zip(bars, data)):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{val:.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                      fontweight='bold', fontsize=9)
    
    axes[idx].set_xticks(range(len(data)))
    axes[idx].set_xticklabels(data.index, rotation=45, ha='right', fontsize=8)
    axes[idx].set_ylabel('% Change from Vanilla', fontsize=11, fontweight='bold')
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].axhline(0, color='black', linewidth=1.5)

plt.suptitle('Similarity Metrics by Prompt Type', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('correlations/prompt_similarity_scores.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prompt_similarity_scores.png")


# VISUALIZATION: Raw ROUGE and BERT scores (not percent change)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
similarity_metrics = ['bert_f1', 'rouge1']
titles = ['BERT-F1 Score (Raw)', 'ROUGE-1 Score (Raw)']

def wrap_label(label):
    """Wrap label at first underscore AND after 3rd hyphen if they exist"""
    if len(label) <= 15:
        return label
    
    # First split at underscore
    if '_' in label:
        label = label.replace('_', '_\n', 1)
    
    # Then split after 3rd hyphen if it exists (in each part if already split)
    if label.count('-') >= 3:
        parts = label.split('\n')  # Handle if already split by underscore
        wrapped_parts = []
        for part in parts:
            if part.count('-') >= 3:
                hyphen_parts = part.split('-')
                wrapped_parts.append('-'.join(hyphen_parts[:3]) + '-\n' + '-'.join(hyphen_parts[3:]))
            else:
                wrapped_parts.append(part)
        label = '\n'.join(wrapped_parts)
    
    return label

for idx, (metric, title) in enumerate(zip(similarity_metrics, titles)):
    # Get vanilla baseline
    vanilla_score = vanilla_means[metric]
    # Get augmented scores by prompt
    augmented_scores = pct_change_df.set_index('Prompt_Type')[metric]
    # Combine into one dataframe for plotting
    all_scores = pd.Series({'Vanilla': vanilla_score})
    all_scores = pd.concat([all_scores, augmented_scores])
    
    # Color: vanilla is blue, augmented prompts are various colors
    colors = ['#3498db'] + ['#e67e22'] * len(augmented_scores)
    bars = axes[idx].bar(range(len(all_scores)), all_scores, color=colors, alpha=0.85, 
                         edgecolor='black', linewidth=1)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, all_scores)):
        axes[idx].text(bar.get_x() + bar.get_width()/2., val,
                      f'{val:.3f}', ha='center', va='bottom',
                      fontweight='bold', fontsize=9)
    
    # Wrap long labels
    wrapped_labels = [wrap_label(label) for label in all_scores.index]
    
    # Set both ticks and labels together
    axes[idx].set_xticks(range(len(wrapped_labels)))
    axes[idx].set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=8)
    axes[idx].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    # Add horizontal line at vanilla baseline
    axes[idx].axhline(vanilla_score, color='#3498db', linestyle='--', 
                     linewidth=1.5, alpha=0.5, label='Vanilla Baseline')
    axes[idx].legend()
plt.suptitle('Raw Similarity Scores by Prompt Type', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('correlations/prompt_raw_similarity_scores-new.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prompt_raw_similarity_scores.png")

# VISUALIZATION 2: Bar chart comparing prompts on key metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
key_metrics = ['lenient_coverage_overall', 'Completeness_Score', 
               'Readability_Score', 'Win_Rate_%']
titles = ['Coverage Improvement', 'Completeness Improvement', 
          'Readability Improvement', 'LLM Win Rate']
for idx, (metric, title) in enumerate(zip(key_metrics, titles)):
    if metric == 'Win_Rate_%':
        if not pref_stats_df.empty:
            data = pref_stats_df.set_index('Prompt_Type')['Win_Rate_%']
            axes[idx].axhline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            ylabel = 'Win Rate (%)'
        else:
            axes[idx].text(0.5, 0.5, 'No preference data', ha='center', va='center', 
                          transform=axes[idx].transAxes, fontsize=12)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            continue
    else:
        data = pct_change_df.set_index('Prompt_Type')[metric]
        ylabel = '% Change from Vanilla'
    
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in data]
    bars = axes[idx].bar(range(len(data)), data, color=colors, alpha=0.8, 
                        edgecolor='black', linewidth=1)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, data)):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{val:.1f}', ha='center', va='bottom' if height > 0 else 'top',
                      fontweight='bold', fontsize=9)
    
    # Wrap long labels
    wrapped_labels = [wrap_label(label) for label in data.index]
    
    # Set both ticks and labels together
    axes[idx].set_xticks(range(len(wrapped_labels)))
    axes[idx].set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=8)
    axes[idx].set_ylabel(ylabel, fontsize=10, fontweight='bold')
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].axhline(0, color='black', linewidth=1)
plt.suptitle('Prompt Type Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlations/prompt_comparison_bars-l-new.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prompt_comparison_bars-l-new.png")

# VISUALIZATION 3: Radar chart for best vs worst prompt
if len(pct_change_df) >= 2:
    from math import pi
    
    # Get best and worst prompts by composite score
    best_idx = rankings['Composite_Score'].idxmax()
    worst_idx = rankings['Composite_Score'].idxmin()
    best_prompt = rankings.loc[best_idx, 'Prompt_Type']
    worst_prompt = rankings.loc[worst_idx, 'Prompt_Type']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Coverage', 'Completeness', 'Conciseness', 'Readability']
    metrics_radar = ['lenient_coverage_overall', 'Completeness_Score', 
                     'Conciseness_Score', 'Readability_Score']
    
    # Get values for best and worst - use pct_change_df rows not index lookup
    best_row = pct_change_df[pct_change_df['Prompt_Type'] == best_prompt].iloc[0]
    worst_row = pct_change_df[pct_change_df['Prompt_Type'] == worst_prompt].iloc[0]
    
    best_values = [best_row[m] for m in metrics_radar]
    worst_values = [worst_row[m] for m in metrics_radar]
    
    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    best_values += best_values[:1]
    worst_values += worst_values[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, best_values, 'o-', linewidth=2, label=f'Best: {best_prompt}', color='#27ae60')
    ax.fill(angles, best_values, alpha=0.25, color='#27ae60')
    ax.plot(angles, worst_values, 'o-', linewidth=2, label=f'Worst: {worst_prompt}', color='#e74c3c')
    ax.fill(angles, worst_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(min(min(best_values), min(worst_values)) - 2, max(max(best_values), max(worst_values)) + 2)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Best vs Worst Prompt Performance\n(% Improvement over Vanilla)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('correlations/prompt_radar_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: prompt_radar_chart.png")

print("\n" + "=" * 80)
print("SUMMARY SAVED TO:")
print("=" * 80)
print("  ✓ correlations/prompt_comparison_improvement.csv")
print("  ✓ correlations/prompt_comparison_pct_change.csv")
print("  ✓ correlations/prompt_preference_stats.csv")
print("  ✓ correlations/prompt_rankings.csv")
print("  ✓ correlations/prompt_summary_readable.csv")
print("  ✓ correlations/prompt_heatmap.png")
print("  ✓ correlations/prompt_comparison_bars.png")
print("  ✓ correlations/prompt_radar_chart-l.png")

print("\n" + "=" * 80)
print("KEY INSIGHTS FOR POSTER:")
print("=" * 80)
print("\n1. Check if ALL prompts show coverage improvement or just some")
print("2. Identify the best-performing prompt for your 'optimal case' analysis")
print("3. See if the paradox (coverage up, preference flat) is consistent across prompts")
print("4. Use this to argue: 'Effect is robust across prompts' OR 'Specific strategies work better'")