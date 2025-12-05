import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the data
df = pd.read_csv('correlations/vanilla_vs_augmented_comparison.csv')

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Prepare display data with proper precision
display_data = []
for idx, row in df.iterrows():
    metric = row['Metric']
    
    # Format based on metric type - use more precision so math is visible
    if 'Win_Rate' in metric:
        # Win rate is already a percentage
        vanilla_str = f"{row['Vanilla_Mean']:.2f}"
        augmented_str = f"{row['Augmented_Mean']:.2f}"
        diff_str = f"{row['Difference']:+.2f}"
    elif 'Score' in metric:
        # Scores on 1-10 scale
        vanilla_str = f"{row['Vanilla_Mean']:.2f}"
        augmented_str = f"{row['Augmented_Mean']:.2f}"
        diff_str = f"{row['Difference']:+.2f}"
    else:
        # BERT, ROUGE, coverage on 0-1 scale - need 4 decimals to see the difference
        vanilla_str = f"{row['Vanilla_Mean']:.4f}"
        augmented_str = f"{row['Augmented_Mean']:.4f}"
        diff_str = f"{row['Difference']:+.4f}"
    
    # Change with both absolute and percent
    pct_change = row['Percent_Change']
    change_str = f"{diff_str} ({pct_change:+.1f}%)"
    
    display_data.append([metric, vanilla_str, augmented_str, change_str, pct_change])

# Create DataFrame for table
table_df = pd.DataFrame(display_data, columns=['Metric', 'Vanilla', 'Augmented', 'Change', 'pct_val'])

# Nice metric names
nice_names = {
    'bert_f1': 'BERT-F1',
    'rouge1': 'ROUGE-1',
    'strict_coverage_overall': 'Coverage (Strict)',
    'lenient_coverage_overall': 'Coverage (Lenient)',
    'Completeness_Score': 'Completeness Score',
    'Conciseness_Score': 'Conciseness Score',
    'Readability_Score': 'Readability Score',
    'LLM_Preference_Win_Rate': 'LLM Preference Win Rate'
}

table_df['Metric'] = table_df['Metric'].map(nice_names)

# Create table without the pct_val column (just for coloring)
table_data = table_df[['Metric', 'Vanilla', 'Augmented', 'Change']].values

# Create table
table = ax.table(cellText=table_data,
                colLabels=['Metric', 'Vanilla', 'Augmented', 'Change'],
                cellLoc='center',
                loc='center',
                colWidths=[0.35, 0.15, 0.15, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 3)

# Style header row
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(weight='bold', color='white', fontsize=14)

# Style data rows
for i in range(1, len(table_df) + 1):
    pct_val = table_df.iloc[i-1]['pct_val']
    
    # Color the change column based on positive/negative
    change_cell = table[(i, 3)]
    if pct_val > 0:
        change_cell.set_facecolor('#d5f4e6')  # Light green
        change_cell.set_text_props(weight='bold', color='#27ae60', fontsize=13)
    else:
        change_cell.set_facecolor('#fadbd8')  # Light red
        change_cell.set_text_props(weight='bold', color='#e74c3c', fontsize=13)
    
    # Alternate row colors for readability
    if i % 2 == 0:
        bg_color = '#f8f9fa'
    else:
        bg_color = 'white'
    
    for j in range(3):  # First 3 columns
        table[(i, j)].set_facecolor(bg_color)
        table[(i, j)].set_text_props(fontsize=13)

# Add title
plt.title('Vanilla vs Augmented Dataset Descriptions: Performance Comparison', 
          fontsize=16, fontweight='bold', pad=20)

# Add legend
green_patch = mpatches.Patch(color='#d5f4e6', label='Improvement')
red_patch = mpatches.Patch(color='#fadbd8', label='Decline')
plt.legend(handles=[green_patch, red_patch], loc='upper left', 
          bbox_to_anchor=(0, -0.05), ncol=2, frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig('vanilla_vs_augmented_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: vanilla_vs_augmented_table.png")

# Also print to console for verification
print("\n" + "=" * 80)
print("VANILLA VS AUGMENTED COMPARISON")
print("=" * 80)
print(table_df[['Metric', 'Vanilla', 'Augmented', 'Change']].to_string(index=False))

# Save as CSV too with proper formatting
output_csv = pd.DataFrame({
    'Metric': df['Metric'].map(nice_names),
    'Vanilla': df['Vanilla_Mean'].apply(lambda x: f"{x:.4f}"),
    'Augmented': df['Augmented_Mean'].apply(lambda x: f"{x:.4f}"),
    'Difference': df['Difference'].apply(lambda x: f"{x:.4f}"),
    'Percent_Change': df['Percent_Change'].apply(lambda x: f"{x:+.2f}%")
})
output_csv.to_csv('vanilla_vs_augmented_formatted.csv', index=False)
print("\n✓ Saved: vanilla_vs_augmented_formatted.csv")