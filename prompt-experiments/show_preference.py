import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_preference_data(file_path='preference_evaluations_incremental.csv'):
    """
    Loads the preference evaluation data and generates three key visualizations.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run the evaluation script first.")
        return

    # Data Cleaning and Preparation
    df['Score_A'] = pd.to_numeric(df['Score_A'], errors='coerce')
    df['Score_B'] = pd.to_numeric(df['Score_B'], errors='coerce')
    df_clean = df.dropna(subset=['Score_A', 'Score_B']).copy()

    print(f"Total valid comparisons analyzed: {len(df_clean)}")
    print("\n--- Preference Counts ---")
    print(df_clean['Preference'].value_counts())

    # --- Plot 1: Overall Preference Distribution (Bar Chart) ---
    def plot_overall_preference(df):
        """Shows the percentage breakdown of A, B, and Tie across all runs."""
        preference_counts = df['Preference'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
        preference_counts.columns = ['Preference', 'Percentage']

        order = ['B', 'A', 'Tie']
        preference_counts = preference_counts.set_index('Preference').reindex(order).reset_index()
        preference_counts.fillna(0, inplace=True) 

        plt.figure(figsize=(7, 5))
        sns.barplot(x='Preference', y='Percentage', data=preference_counts, palette={'B': '#1f77b4', 'A': '#ff7f0e', 'Tie': '#2ca02c'})

        plt.title('1. Overall LLM Preference Distribution (B vs. Baseline A)', fontsize=14)
        plt.ylabel('Percentage of Comparisons', fontsize=12)
        plt.xlabel('Preferred Description', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('plot_1_overall_preference.png')
        plt.close()

    # --- Plot 2: Average Score Comparison by Prompt Type (Grouped Bar Chart) ---
    def plot_average_scores(df):
        """Compares average numerical scores for A and B, grouped by prompt type."""
        score_comparison = df.groupby('Prompt_Type_B').agg(
            Avg_Score_B=('Score_B', 'mean'),
            Avg_Score_A=('Score_A', 'mean')
        ).reset_index()

        score_comparison_long = score_comparison.melt(
            id_vars='Prompt_Type_B', 
            value_vars=['Avg_Score_A', 'Avg_Score_B'],
            var_name='Method', 
            value_name='Average Score (1-5)'
        )

        score_comparison_long['Method'] = score_comparison_long['Method'].replace({
            'Avg_Score_A': 'Baseline (A)', 
            'Avg_Score_B': 'Augmented (B)'
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Prompt_Type_B', 
            y='Average Score (1-5)', 
            hue='Method', 
            data=score_comparison_long, 
            palette={'Baseline (A)': '#ff7f0e', 'Augmented (B)': '#1f77b4'}
        )
        plt.title('2. Average Researcher Utility Score (1-5) by Prompt Type', fontsize=14)
        plt.ylabel('Average Score', fontsize=12)
        plt.xlabel('Augmented Prompt Type', fontsize=12)
        plt.ylim(0, 5.2)
        plt.legend(title='Method')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plot_2_average_scores.png')
        plt.close()

    # --- Plot 3: Preference Breakdown by Prompt Type (Stacked Bar Chart) ---
    def plot_stacked_preference(df):
        """Shows the distribution of A, B, and Tie preference for each prompt type."""
        preference_by_prompt = df.groupby('Prompt_Type_B')['Preference'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

        preference_by_prompt['Preference'] = pd.Categorical(
            preference_by_prompt['Preference'], 
            categories=['B', 'Tie', 'A'], 
            ordered=True
        )
        
        preference_by_prompt = preference_by_prompt.sort_values(['Prompt_Type_B', 'Preference'], ascending=[True, False])

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=preference_by_prompt,
            x='Prompt_Type_B',
            y='Percentage',
            hue='Preference',
            hue_order=['A', 'Tie', 'B'], # Order A, Tie, B to stack B on top for better visualization
            palette={'B': '#1f77b4', 'A': '#ff7f0e', 'Tie': '#2ca02c'},
            dodge=False
        )

        plt.title('3. LLM Preference Breakdown by Augmented Prompt Type', fontsize=14)
        plt.ylabel('Percentage', fontsize=12)
        plt.xlabel('Augmented Prompt Type', fontsize=12)
        plt.ylim(0, 100)
        plt.legend(title='Preference', labels=['Baseline (A)', 'Tie', 'Augmented (B)'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plot_3_stacked_preference.png')
        plt.close()

    # Execute all plots
    plot_overall_preference(df_clean)
    plot_average_scores(df_clean)
    plot_stacked_preference(df_clean)
    
    print("\nVisualizations saved as: plot_1_overall_preference.png, plot_2_average_scores.png, and plot_3_stacked_preference.png")

if __name__ == '__main__':
    analyze_preference_data()