import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('correlations/vanilla_vs_augmented_comparison.csv')

print("Loaded data:")
print(df)

# Create nice metric names
nice_names = {
    'bert_f1': 'BERT-F1\nSimilarity',
    'rouge1': 'ROUGE-1\nSimilarity',
    'strict_coverage_overall': 'Coverage\n(Strict)',
    'lenient_coverage_overall': 'Coverage\n(Lenient)',
    'Completeness_Score': 'Completeness',
    'Conciseness_Score': 'Conciseness',
    'Readability_Score': 'Readability',
    'LLM_Preference_Win_Rate': 'LLM Judge\nPreference'
}

df['Nice_Name'] = df['Metric'].map(nice_names)

# Color scheme
vanilla_color = '#3498db'  # Blue
augmented_color = '#e67e22'  # Orange
positive_color = '#27ae60'  # Green
negative_color = '#e74c3c'  # Red

### VISUALIZATION 1: Side-by-side bar chart with percent change
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(df))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, df['Vanilla_Mean'], width, label='Vanilla', 
               color=vanilla_color, alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, df['Augmented_Mean'], width, label='Augmented',
               color=augmented_color, alpha=0.85, edgecolor='black', linewidth=1.2)

# Add percent change as text above bars
for i, (idx, row) in enumerate(df.iterrows()):
    pct = row['Percent_Change']
    y_pos = max(row['Vanilla_Mean'], row['Augmented_Mean']) * 1.08
    
    # Color based on positive/negative
    color = positive_color if pct > 0 else negative_color
    sign = '+' if pct > 0 else ''
    
    ax.text(i, y_pos, f'{sign}{pct:.1f}%', 
            ha='center', va='bottom', fontweight='bold', 
            fontsize=12, color=color)

ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Vanilla vs Augmented Descriptions: Metric Comparison', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['Nice_Name'], fontsize=11)
ax.legend(fontsize=13, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('viz1_sidebyside_bars.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: viz1_sidebyside_bars.png")

### VISUALIZATION 2: Diverging bar chart (percent change only)
fig, ax = plt.subplots(figsize=(10, 8))

# Sort by percent change
df_sorted = df.sort_values('Percent_Change')

colors = [positive_color if x > 0 else negative_color for x in df_sorted['Percent_Change']]

bars = ax.barh(df_sorted['Nice_Name'], df_sorted['Percent_Change'], 
               color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

# Add value labels
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    pct = row['Percent_Change']
    sign = '+' if pct > 0 else ''
    x_pos = pct + (0.5 if pct > 0 else -0.5)
    ax.text(x_pos, i, f'{sign}{pct:.1f}%', 
            va='center', ha='left' if pct > 0 else 'right',
            fontsize=12, fontweight='bold')

ax.axvline(0, color='black', linewidth=2, linestyle='-')
ax.set_xlabel('Percent Change (Augmented vs Vanilla)', fontsize=14, fontweight='bold')
ax.set_title('Impact of Citation Augmentation on Dataset Descriptions', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add annotations
ax.text(0.02, 0.98, '← Vanilla Better', transform=ax.transAxes, 
        fontsize=11, va='top', ha='left', style='italic', color='gray')
ax.text(0.98, 0.98, 'Augmented Better →', transform=ax.transAxes,
        fontsize=11, va='top', ha='right', style='italic', color='gray')

plt.tight_layout()
plt.savefig('viz2_diverging_bars.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz2_diverging_bars.png")

### VISUALIZATION 3: Grouped metrics highlighting key insight
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

width = 0.35

# Group 1: Coverage metrics (THE WIN)
coverage_mask = df['Metric'].str.contains('coverage')
coverage_data = df[coverage_mask]
x1 = np.arange(len(coverage_data))
axes[0].bar(x1 - width/2, coverage_data['Vanilla_Mean'], width, 
            label='Vanilla', color=vanilla_color, alpha=0.85, edgecolor='black', linewidth=1)
axes[0].bar(x1 + width/2, coverage_data['Augmented_Mean'], width,
            label='Augmented', color=augmented_color, alpha=0.85, edgecolor='black', linewidth=1)
axes[0].set_title('Coverage Metrics\n✓ Augmented +6-10%', fontsize=13, fontweight='bold', color=positive_color)
axes[0].set_xticks(x1)
axes[0].set_xticklabels(['Strict', 'Lenient'], fontsize=11)
axes[0].legend(fontsize=10)
axes[0].set_ylabel('Score', fontsize=11, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Group 2: Quality scores (ALSO WIN)
quality_mask = df['Metric'].str.contains('Score') & ~df['Metric'].str.contains('Win_Rate')
quality_data = df[quality_mask]
x2 = np.arange(len(quality_data))
axes[1].bar(x2 - width/2, quality_data['Vanilla_Mean'], width,
            label='Vanilla', color=vanilla_color, alpha=0.85, edgecolor='black', linewidth=1)
axes[1].bar(x2 + width/2, quality_data['Augmented_Mean'], width,
            label='Augmented', color=augmented_color, alpha=0.85, edgecolor='black', linewidth=1)
axes[1].set_title('Quality Ratings\n✓ Augmented +1-5%', fontsize=13, fontweight='bold', color=positive_color)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(['Complete', 'Concise', 'Readable'], fontsize=10)
axes[1].legend(fontsize=10)
axes[1].set_ylim([7, 9.5])
axes[1].grid(axis='y', alpha=0.3)

# Group 3: Preference (THE PARADOX!)
pref_data = df[df['Metric'] == 'LLM_Preference_Win_Rate']
axes[2].bar([0, 1], [pref_data['Vanilla_Mean'].values[0], pref_data['Augmented_Mean'].values[0]], 
            color=[vanilla_color, augmented_color], alpha=0.85, width=0.6, 
            edgecolor='black', linewidth=1)
axes[2].set_title('LLM Judge Preference\n✗ Nearly Equal (~50/50)', fontsize=13, fontweight='bold', color=negative_color)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Vanilla', 'Augmented'], fontsize=11)
axes[2].set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
axes[2].set_ylim([0, 100])
axes[2].axhline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
axes[2].grid(axis='y', alpha=0.3)

# Add percentage values on bars
for i, val in enumerate([pref_data['Vanilla_Mean'].values[0], pref_data['Augmented_Mean'].values[0]]):
    axes[2].text(i, val + 2, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)

fig.suptitle('The Coverage-Preference Paradox: Better Coverage ≠ Better Preference', 
             fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('viz3_grouped_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz3_grouped_metrics.png")

### VISUALIZATION 4: Clean table visualization (good for posters)
fig, ax = plt.subplots(figsize=(11, 7))
ax.axis('tight')
ax.axis('off')

# Prepare data
display_df = df[['Metric', 'Vanilla_Mean', 'Augmented_Mean', 'Percent_Change']].copy()

# Format numbers
display_df['Vanilla_Mean'] = display_df['Vanilla_Mean'].apply(lambda x: f'{x:.2f}')
display_df['Augmented_Mean'] = display_df['Augmented_Mean'].apply(lambda x: f'{x:.2f}')
display_df['Percent_Change'] = display_df['Percent_Change'].apply(lambda x: f'{x:+.1f}%')

# Rename columns
display_df.columns = ['Metric', 'Vanilla', 'Augmented', 'Change']

# Create table
table = ax.table(cellText=display_df.values, 
                colLabels=display_df.columns,
                cellLoc='center', 
                loc='center',
                colWidths=[0.35, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(len(display_df.columns)):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# Color code percent change cells
for i in range(1, len(display_df) + 1):
    pct_val = df.iloc[i-1]['Percent_Change']
    
    # Color the change column
    if pct_val > 0:
        table[(i, 3)].set_facecolor('#d5f4e6')  # Light green
        table[(i, 3)].set_text_props(weight='bold', color=positive_color)
    else:
        table[(i, 3)].set_facecolor('#fadbd8')  # Light red
        table[(i, 3)].set_text_props(weight='bold', color=negative_color)
    
    # Alternate row colors
    if i % 2 == 0:
        for j in range(3):
            table[(i, j)].set_facecolor('#f8f9fa')

plt.title('Vanilla vs Augmented: Complete Metrics Summary', 
          fontsize=15, fontweight='bold', pad=20)
plt.savefig('viz4_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz4_table.png")

print("\n" + "="*60)
print("All visualizations created successfully!")
print("="*60)
print("\nRecommendation for poster:")
print("  • Use viz3_grouped_metrics.png - Shows the paradox clearly")
print("  • Or viz2_diverging_bars.png - Clean, easy to read")
print("  • Table (viz4) good as supplementary")