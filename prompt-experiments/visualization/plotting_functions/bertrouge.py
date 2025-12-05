import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
df = pd.read_csv('results-eval.csv')

# Set the style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. ROUGE-L scores by Prompt Type (including Vanilla)
ax1 = axes[0, 0]
prompt_types = df['Prompt_Type'].fillna('Vanilla')
x_pos = np.arange(len(df))
colors = ['#1f77b4' if pt == 'Vanilla' else '#ff7f0e' if pt == 'V1_Revised' else '#2ca02c' 
          for pt in prompt_types]

bars1 = ax1.bar(x_pos, df['rougeL'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Test ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('ROUGE-L Score', fontsize=12, fontweight='bold')
ax1.set_title('ROUGE-L Scores by Prompt Type', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['Description_Source'], rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1f77b4', label='Vanilla'),
                   Patch(facecolor='#ff7f0e', label='V1_Revised'),
                   Patch(facecolor='#2ca02c', label='V2_Hybrid')]
ax1.legend(handles=legend_elements, loc='upper right')

# 2. BERT F1 scores by Prompt Type (including Vanilla)
ax2 = axes[0, 1]
bars2 = ax2.bar(x_pos, df['bert_f1'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Test ID', fontsize=12, fontweight='bold')
ax2.set_ylabel('BERT F1 Score', fontsize=12, fontweight='bold')
ax2.set_title('BERT F1 Scores by Prompt Type', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df['Description_Source'], rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)
ax2.legend(handles=legend_elements, loc='upper right')

# 3. Average ROUGE-L by Prompt Type
ax3 = axes[1, 0]
df_grouped = df.copy()
df_grouped['Prompt_Type'] = df_grouped['Prompt_Type'].fillna('Vanilla')
avg_rougeL = df_grouped.groupby('Prompt_Type')['rougeL'].mean().sort_values(ascending=False)

bars3 = ax3.bar(range(len(avg_rougeL)), avg_rougeL.values, 
                color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.7, edgecolor='black')
ax3.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average ROUGE-L Score', fontsize=12, fontweight='bold')
ax3.set_title('Average ROUGE-L Score by Prompt Type', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(avg_rougeL)))
ax3.set_xticklabels(avg_rougeL.index, rotation=0)
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars3, avg_rougeL.values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# 4. Average BERT F1 by Prompt Type
ax4 = axes[1, 1]
avg_bert_f1 = df_grouped.groupby('Prompt_Type')['bert_f1'].mean().sort_values(ascending=False)

bars4 = ax4.bar(range(len(avg_bert_f1)), avg_bert_f1.values, 
                color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.7, edgecolor='black')
ax4.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
ax4.set_ylabel('Average BERT F1 Score', fontsize=12, fontweight='bold')
ax4.set_title('Average BERT F1 Score by Prompt Type', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(avg_bert_f1)))
ax4.set_xticklabels(avg_bert_f1.index, rotation=0)
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars4, avg_bert_f1.values)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('rouge_bert_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===\n")
print("Average ROUGE-L by Prompt Type:")
print(avg_rougeL)
print("\nAverage BERT F1 by Prompt Type:")
print(avg_bert_f1)

# Additional comparison table
print("\n=== Detailed Comparison ===")
summary_df = df_grouped.groupby('Prompt_Type').agg({
    'rougeL': ['mean', 'std', 'min', 'max'],
    'bert_f1': ['mean', 'std', 'min', 'max']
}).round(4)
print(summary_df)