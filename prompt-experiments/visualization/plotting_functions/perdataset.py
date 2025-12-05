import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap

# Read the data
df = pd.read_csv('results-eval.csv')

# Set the style
sns.set_style("whitegrid")

# Function to wrap text
def wrap_labels(labels, width=20):
    """Wrap long labels to specified width"""
    return [textwrap.fill(label, width) for label in labels]

# Group by dataset and calculate average scores across all prompt types
dataset_avg = df.groupby('dataset_id').agg({
    'rougeL': 'mean',
    'bert_f1': 'mean',
    'rouge1': 'mean',
    'rouge2': 'mean',
    'bert_precision': 'mean',
    'bert_recall': 'mean',
    'lenient_coverage_overall': 'mean',
    'lenient_coverage_basic_info': 'mean',
    'lenient_coverage_data_characteristics': 'mean',
    'lenient_coverage_provenance': 'mean',
    'lenient_coverage_usage_context': 'mean',
    'lenient_coverage_quality_and_limitations': 'mean',
    'Completeness_Score': 'mean',
    'Conciseness_Score': 'mean',
    'Readability_Score': 'mean'
}).reset_index()

# Sort by BERT F1 for better visualization
dataset_avg = dataset_avg.sort_values('bert_f1', ascending=True)

# Wrap dataset names
wrapped_labels = wrap_labels(dataset_avg['dataset_id'].tolist(), width=20)

# Create comprehensive figure
fig = plt.figure(figsize=(24, 20))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. ROUGE-L by Dataset
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.barh(range(len(dataset_avg)), dataset_avg['rougeL'], 
                 color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(dataset_avg)))
ax1.set_yticklabels(wrapped_labels, fontsize=8)
ax1.set_xlabel('Average ROUGE-L Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax1.set_title('Average ROUGE-L Score by Dataset', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars1, dataset_avg['rougeL'])):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontweight='bold', fontsize=7)

# 2. BERT F1 by Dataset
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(range(len(dataset_avg)), dataset_avg['bert_f1'], 
                 color='coral', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(dataset_avg)))
ax2.set_yticklabels(wrapped_labels, fontsize=8)
ax2.set_xlabel('Average BERT F1 Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax2.set_title('Average BERT F1 Score by Dataset', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars2, dataset_avg['bert_f1'])):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontweight='bold', fontsize=7)

# 3. Lenient Coverage Overall by Dataset
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.barh(range(len(dataset_avg)), dataset_avg['lenient_coverage_overall'], 
                 color='mediumseagreen', alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(dataset_avg)))
ax3.set_yticklabels(wrapped_labels, fontsize=8)
ax3.set_xlabel('Average Coverage Score', fontsize=11, fontweight='bold')
ax3.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax3.set_title('Lenient Coverage (Overall) by Dataset', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars3, dataset_avg['lenient_coverage_overall'])):
    ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontweight='bold', fontsize=7)

# 4. Combined ROUGE metrics by Dataset
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(dataset_avg))
width = 0.25

bars4a = ax4.barh(x - width, dataset_avg['rouge1'], width, 
                  label='ROUGE-1', color='#1f77b4', alpha=0.7, edgecolor='black')
bars4b = ax4.barh(x, dataset_avg['rouge2'], width, 
                  label='ROUGE-2', color='#ff7f0e', alpha=0.7, edgecolor='black')
bars4c = ax4.barh(x + width, dataset_avg['rougeL'], width, 
                  label='ROUGE-L', color='#2ca02c', alpha=0.7, edgecolor='black')

ax4.set_yticks(x)
ax4.set_yticklabels(wrapped_labels, fontsize=8)
ax4.set_xlabel('Average Score', fontsize=11, fontweight='bold')
ax4.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax4.set_title('ROUGE Metrics Comparison by Dataset', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(axis='x', alpha=0.3)

# 5. BERT Precision vs Recall by Dataset
ax5 = fig.add_subplot(gs[1, 1])
bars5a = ax5.barh(x - width/2, dataset_avg['bert_precision'], width, 
                  label='Precision', color='#9467bd', alpha=0.7, edgecolor='black')
bars5b = ax5.barh(x + width/2, dataset_avg['bert_recall'], width, 
                  label='Recall', color='#8c564b', alpha=0.7, edgecolor='black')

ax5.set_yticks(x)
ax5.set_yticklabels(wrapped_labels, fontsize=8)
ax5.set_xlabel('Average Score', fontsize=11, fontweight='bold')
ax5.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax5.set_title('BERT Precision vs Recall by Dataset', fontsize=12, fontweight='bold')
ax5.legend(loc='lower right', fontsize=9)
ax5.grid(axis='x', alpha=0.3)

# 6. Quality Metrics: Completeness, Conciseness, Readability
ax6 = fig.add_subplot(gs[1, 2])
width_quality = 0.25

bars6a = ax6.barh(x - width_quality, dataset_avg['Completeness_Score'], width_quality, 
                  label='Completeness', color='#e74c3c', alpha=0.7, edgecolor='black')
bars6b = ax6.barh(x, dataset_avg['Conciseness_Score'], width_quality, 
                  label='Conciseness', color='#3498db', alpha=0.7, edgecolor='black')
bars6c = ax6.barh(x + width_quality, dataset_avg['Readability_Score'], width_quality, 
                  label='Readability', color='#2ecc71', alpha=0.7, edgecolor='black')

ax6.set_yticks(x)
ax6.set_yticklabels(wrapped_labels, fontsize=8)
ax6.set_xlabel('Average Score (1-10)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax6.set_title('Quality Metrics by Dataset', fontsize=12, fontweight='bold')
ax6.legend(loc='lower right', fontsize=9)
ax6.grid(axis='x', alpha=0.3)

# 7. Lenient Coverage Components - Stacked
ax7 = fig.add_subplot(gs[2, 0])
coverage_components = ['lenient_coverage_basic_info', 
                       'lenient_coverage_data_characteristics',
                       'lenient_coverage_provenance', 
                       'lenient_coverage_usage_context',
                       'lenient_coverage_quality_and_limitations']
component_labels = ['Basic Info', 'Data Char.', 'Provenance', 'Usage', 'Quality/Limits']
colors_coverage = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

left = np.zeros(len(dataset_avg))
for comp, label, color in zip(coverage_components, component_labels, colors_coverage):
    ax7.barh(range(len(dataset_avg)), dataset_avg[comp], left=left,
             label=label, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    left += dataset_avg[comp].values

ax7.set_yticks(range(len(dataset_avg)))
ax7.set_yticklabels(wrapped_labels, fontsize=8)
ax7.set_xlabel('Coverage Score', fontsize=11, fontweight='bold')
ax7.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax7.set_title('Lenient Coverage Components (Stacked)', fontsize=12, fontweight='bold')
ax7.legend(loc='lower right', fontsize=8)
ax7.grid(axis='x', alpha=0.3)

# 8. Overall Quality Score (Average of all three quality metrics)
ax8 = fig.add_subplot(gs[2, 1])
dataset_avg['Overall_Quality'] = (dataset_avg['Completeness_Score'] + 
                                   dataset_avg['Conciseness_Score'] + 
                                   dataset_avg['Readability_Score']) / 3

bars8 = ax8.barh(range(len(dataset_avg)), dataset_avg['Overall_Quality'], 
                 color='#16a085', alpha=0.7, edgecolor='black')
ax8.set_yticks(range(len(dataset_avg)))
ax8.set_yticklabels(wrapped_labels, fontsize=8)
ax8.set_xlabel('Average Quality Score (1-10)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Dataset', fontsize=11, fontweight='bold')
ax8.set_title('Overall Quality Score by Dataset\n(Avg of Completeness, Conciseness, Readability)', 
              fontsize=12, fontweight='bold')
ax8.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars8, dataset_avg['Overall_Quality'])):
    ax8.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}', va='center', fontweight='bold', fontsize=7)

# 9. Heatmap of all metrics
ax9 = fig.add_subplot(gs[2, 2])
metrics_for_heatmap = ['bert_f1', 'rougeL', 'lenient_coverage_overall', 
                       'Completeness_Score', 'Conciseness_Score', 'Readability_Score']
heatmap_data = dataset_avg[metrics_for_heatmap].T

# Normalize scores to 0-1 range for better comparison
heatmap_data_norm = heatmap_data.copy()
heatmap_data_norm.loc['Completeness_Score'] = heatmap_data_norm.loc['Completeness_Score'] / 10
heatmap_data_norm.loc['Conciseness_Score'] = heatmap_data_norm.loc['Conciseness_Score'] / 10
heatmap_data_norm.loc['Readability_Score'] = heatmap_data_norm.loc['Readability_Score'] / 10

im = ax9.imshow(heatmap_data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax9.set_xticks(range(len(dataset_avg)))
ax9.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=7)
ax9.set_yticks(range(len(metrics_for_heatmap)))
ax9.set_yticklabels(['BERT F1', 'ROUGE-L', 'Coverage', 'Completeness', 'Conciseness', 'Readability'], 
                    fontsize=9)
ax9.set_title('Normalized Metrics Heatmap\n(All scores scaled 0-1)', fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax9)
cbar.set_label('Normalized Score', fontsize=10, fontweight='bold')

# Add values to heatmap
for i in range(len(metrics_for_heatmap)):
    for j in range(len(dataset_avg)):
        text = ax9.text(j, i, f'{heatmap_data_norm.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=6, fontweight='bold')

# 10. Detailed Lenient Coverage Breakdown - Grouped bars
ax10 = fig.add_subplot(gs[3, :])
x_pos = np.arange(len(dataset_avg))
bar_width = 0.15

for i, (comp, label, color) in enumerate(zip(coverage_components, component_labels, colors_coverage)):
    offset = (i - 2) * bar_width
    ax10.bar(x_pos + offset, dataset_avg[comp], bar_width,
            label=label, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

ax10.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax10.set_ylabel('Coverage Score', fontsize=12, fontweight='bold')
ax10.set_title('Lenient Coverage Detailed Breakdown by Dataset', fontsize=14, fontweight='bold')
ax10.set_xticks(x_pos)
ax10.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=8)
ax10.legend(loc='upper left', fontsize=10, ncol=5)
ax10.grid(axis='y', alpha=0.3)
ax10.set_ylim(0, 1.1)

plt.savefig('comprehensive_performance_by_dataset.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics
print("\n" + "="*80)
print("COMPREHENSIVE PERFORMANCE SUMMARY BY DATASET")
print("="*80 + "\n")
print(dataset_avg.to_string(index=False))

# Count number of prompt types per dataset
prompt_counts = df.groupby('dataset_id').size()
print("\n" + "="*80)
print("NUMBER OF PROMPT TYPE VARIATIONS PER DATASET")
print("="*80)
print(prompt_counts)

# Show which datasets performed best/worst across different metrics
print("\n" + "="*80)
print("BEST PERFORMING DATASETS (by BERT F1)")
print("="*80)
print(dataset_avg.nlargest(5, 'bert_f1')[['dataset_id', 'bert_f1', 'rougeL', 
                                           'lenient_coverage_overall', 'Overall_Quality']])

print("\n" + "="*80)
print("WORST PERFORMING DATASETS (by BERT F1)")
print("="*80)
print(dataset_avg.nsmallest(5, 'bert_f1')[['dataset_id', 'bert_f1', 'rougeL', 
                                            'lenient_coverage_overall', 'Overall_Quality']])

print("\n" + "="*80)
print("BEST COVERAGE DATASETS")
print("="*80)
print(dataset_avg.nlargest(5, 'lenient_coverage_overall')[['dataset_id', 'lenient_coverage_overall', 
                                                             'bert_f1', 'rougeL', 'Overall_Quality']])

print("\n" + "="*80)
print("WORST COVERAGE DATASETS")
print("="*80)
print(dataset_avg.nsmallest(5, 'lenient_coverage_overall')[['dataset_id', 'lenient_coverage_overall', 
                                                              'bert_f1', 'rougeL', 'Overall_Quality']])

print("\n" + "="*80)
print("HIGHEST QUALITY DATASETS (Completeness + Conciseness + Readability)")
print("="*80)
print(dataset_avg.nlargest(5, 'Overall_Quality')[['dataset_id', 'Overall_Quality', 
                                                    'Completeness_Score', 'Conciseness_Score', 
                                                    'Readability_Score']])

print("\n" + "="*80)
print("LOWEST QUALITY DATASETS")
print("="*80)
print(dataset_avg.nsmallest(5, 'Overall_Quality')[['dataset_id', 'Overall_Quality', 
                                                     'Completeness_Score', 'Conciseness_Score', 
                                                     'Readability_Score']])

# Additional: Show variance across prompt types for each dataset
print("\n" + "="*80)
print("SCORE VARIANCE ACROSS PROMPT TYPES BY DATASET")
print("="*80)
variance_df = df.groupby('dataset_id').agg({
    'rougeL': ['mean', 'std'],
    'bert_f1': ['mean', 'std'],
    'lenient_coverage_overall': ['mean', 'std'],
    'Completeness_Score': ['mean', 'std'],
    'Conciseness_Score': ['mean', 'std'],
    'Readability_Score': ['mean', 'std']
}).round(4)
variance_df.columns = ['ROUGE-L Mean', 'ROUGE-L Std', 'BERT F1 Mean', 'BERT F1 Std', 
                       'Coverage Mean', 'Coverage Std', 'Complete Mean', 'Complete Std',
                       'Concise Mean', 'Concise Std', 'Readable Mean', 'Readable Std']
print(variance_df)

# Correlation analysis
print("\n" + "="*80)
print("CORRELATION BETWEEN ALL METRICS")
print("="*80)
corr_cols = ['rougeL', 'bert_f1', 'lenient_coverage_overall', 
             'Completeness_Score', 'Conciseness_Score', 'Readability_Score']
correlation = dataset_avg[corr_cols].corr()
print(correlation.round(3))

# Summary statistics for quality metrics
print("\n" + "="*80)
print("QUALITY METRICS SUMMARY STATISTICS")
print("="*80)
quality_summary = dataset_avg[['Completeness_Score', 'Conciseness_Score', 
                                'Readability_Score', 'Overall_Quality']].describe()
print(quality_summary.round(2))