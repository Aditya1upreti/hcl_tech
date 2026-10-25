#!/usr/bin/env python
"""
notebooks/eda.py - EDA script for RAG Document Q&A System

Loads dataset.csv, cleans and checks it, saves cleaned_dataset.csv,
and creates at least 4 saved visualisations in /figures:
1. retrieval_score distribution by doc_type
2. answer_quality_score by query_type (boxplot)
3. word_count histogram
4. correlation heatmap between retrieval_score and answer_quality_score

Adds a short written insight under each chart.
"""

import os
import csv
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET_PATH = Path("dataset.csv")
CLEANED_DATASET_PATH = Path("cleaned_dataset.csv")
FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("results")

# ---------------------------------------------------------------------------
# LOAD AND CLEAN DATASET
# ---------------------------------------------------------------------------

def load_dataset(path=DATASET_PATH):
    """Load dataset.csv and return a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    return df


def check_nulls(df):
    """Check for null values in the dataset."""
    null_counts = df.isnull().sum()
    print("\n=== Null values per column ===")
    for col, count in null_counts.items():
        if count > 0:
            pct = count / len(df) * 100
            print(f"  {col}: {count} nulls ({pct:.1f}%)")
    return null_counts


def check_duplicates(df):
    """Check for duplicate rows."""
    dupes = df.duplicated().sum()
    print(f"\n=== Duplicate rows: {dupes} ===")
    return dupes


def check_dtypes(df):
    """Check and report dtypes."""
    print("\n=== Data types ===")
    print(df.dtypes)
    return df.dtypes


def clean_data(df):
    """Clean the dataset: fill nulls, remove duplicates, fix dtypes."""
    original_len = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Removed duplicates. {original_len - len(df)} rows removed.")
    
    # Fill numeric nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill string nulls with "unknown"
    string_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in string_cols:
        df[col] = df[col].fillna("unknown")
    
    print(f"Cleaned dataset: {len(df)} rows (was {original_len})")
    return df


# ---------------------------------------------------------------------------
# VISUALISATION 1: retrieval_score distribution by doc_type
# ---------------------------------------------------------------------------

def plot_retrieval_score_by_doc_type(df, output_path=None):
    """Create bar chart of retrieval_score distribution by doc_type."""
    if output_path is None:
        output_path = FIGURES_DIR / "retrieval_score_by_doc_type.png"
    
    plt.figure(figsize=(10, 6))
    
    # Order doc_types for consistent display
    doc_types = df['doc_type'].value_counts().index.tolist()
    
    # Use matplotlib directly for bar chart
    avg_scores = df.groupby('doc_type')['retrieval_score'].mean()
    colors = plt.cm.viridis(np.linspace(0, 1, len(avg_scores)))
    
    bars = plt.bar(avg_scores.index, avg_scores.values, color=colors, edgecolor='black')
    plt.title('Retrieval Score Distribution by Document Type', fontsize=14, fontweight='bold')
    plt.xlabel('Document Type', fontsize=12)
    plt.ylabel('Average Retrieval Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for bar, val in zip(bars, avg_scores.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)
    
    # Add insight as text
    best_type = avg_scores.index[0]
    worst_type = avg_scores.index[-1]
    best_score = avg_scores.iloc[0]
    worst_score = avg_scores.iloc[-1]
    insight_text = f"Insight: {best_type} documents have the highest average retrieval score ({best_score:.3f}), "
    insight_text += f"while {worst_type} documents have the lowest ({worst_score:.3f})."
    plt.figtext(0.5, 0.01, insight_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# VISUALISATION 2: answer_quality_score by query_type (boxplot)
# ---------------------------------------------------------------------------

def plot_answer_quality_by_query_type(df, output_path=None):
    """Create boxplot of answer_quality_score by query_type."""
    if output_path is None:
        output_path = FIGURES_DIR / "answer_quality_by_query_type.png"
    
    plt.figure(figsize=(10, 6))
    
    # Use matplotlib directly for boxplot
    query_types = df['query_type'].unique().tolist()
    query_data = [df[df['query_type'] == qt]['answer_quality_score'].values 
                  for qt in query_types]
    
    plt.boxplot(query_data, labels=query_types, patch_artist=True)
    plt.title('Answer Quality Score by Query Type', fontsize=14, fontweight='bold')
    plt.xlabel('Query Type', fontsize=12)
    plt.ylabel('Answer Quality Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Compute insights
    medians = df.groupby('query_type')['answer_quality_score'].median().sort_values(ascending=False)
    best_query = medians.index[0]
    worst_query = medians.index[-1]
    best_median = medians.iloc[0]
    worst_median = medians.iloc[-1]
    
    insight_text = f"Insight: {best_query} queries have the highest median answer quality ({best_median:.2f}), "
    insight_text += f"while {worst_query} queries have the lowest ({worst_median:.2f})."
    plt.figtext(0.5, 0.01, insight_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# VISUALISATION 3: word_count histogram
# ---------------------------------------------------------------------------

def plot_word_count_histogram(df, output_path=None):
    """Create histogram of word_count distribution."""
    if output_path is None:
        output_path = FIGURES_DIR / "word_count_histogram.png"
    
    plt.figure(figsize=(10, 6))
    
    # Filter out unreasonable values
    word_counts = df['word_count']
    valid_counts = word_counts[(word_counts > 0) & (word_counts < 2000)]
    
    # Use matplotlib histogram
    plt.hist(valid_counts, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title('Word Count Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Compute insights
    mean_wc = round(valid_counts.mean(), 1)
    median_wc = round(valid_counts.median(), 1)
    
    insight_text = f"Insight: Mean word count = {mean_wc}, Median = {median_wc}. "
    insight_text += f"Most documents have {int(median_wc)} words or fewer."
    plt.figtext(0.5, 0.01, insight_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# VISUALISATION 4: correlation heatmap between retrieval_score and answer_quality_score
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, output_path=None):
    """Create heatmap of correlation between retrieval_score and answer_quality_score."""
    if output_path is None:
        output_path = FIGURES_DIR / "correlation_heatmap.png"
    
    plt.figure(figsize=(8, 6))
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Use matplotlib imshow for heatmap
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap: retrieval_score vs answer_quality_score', 
              fontsize=14, fontweight='bold')
    
    # Add tick labels
    tick_names = list(corr_matrix.columns)
    plt.xticks(range(len(tick_names)), tick_names, rotation=45, ha='left')
    plt.yticks(range(len(tick_names)), tick_names)
    
    # Add correlation values as text
    for i in range(len(tick_names)):
        for j in range(len(tick_names)):
            text = plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    # Compute correlation value and p-value
    r = round(float(corr_matrix.loc['retrieval_score', 'answer_quality_score']), 3)
    p_val = round(float(stats.pearsonr(df['retrieval_score'], df['answer_quality_score'])[1]), 4)
    
    # Determine direction
    if r > 0:
        direction = "positive"
    elif r < 0:
        direction = "negative"
    else:
        direction = "no"
    
    # Add insight as text
    insight_text = f"Insight: Pearson correlation r = {r} ({direction}, p = {p_val}). "
    if r > 0:
        insight_text += "Higher retrieval scores tend to correlate with higher answer quality scores."
    elif r < 0:
        insight_text += "Higher retrieval scores are associated with lower answer quality scores, suggesting the need for improved ranking or re-ranking strategies."
    else:
        insight_text += "There is no correlation between retrieval scores and answer quality scores."
    plt.figtext(0.5, 0.01, insight_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# GENERATE WRITTEN INSIGHTS
# ---------------------------------------------------------------------------

def generate_insights(df):
    """Generate written insights from the EDA data."""
    insights = []
    
    # 1. retrieval_score by doc_type
    avg_by_type = df.groupby('doc_type')['retrieval_score'].mean().sort_values(ascending=False)
    best_type = avg_by_type.index[0]
    worst_type = avg_by_type.index[-1]
    insights.append(
        f"1. Document type analysis: {best_type} documents achieve the highest average retrieval "
        f"score ({avg_by_type.iloc[0]:.3f}), while {worst_type} documents have the lowest "
        f"({avg_by_type.iloc[-1]:.3f}). This suggests varying effectiveness of chunking/embedding "
        f"across document formats."
    )
    
    # 2. answer_quality_score by query_type
    medians = df.groupby('query_type')['answer_quality_score'].median()
    best_query = medians.index[0]
    worst_query = medians.index[-1]
    insights.append(
        f"2. Query type analysis: {best_query} queries achieve the highest median answer quality "
        f"({medians.iloc[0]:.2f}), while {worst_query} queries have the lowest "
        f"({medians.iloc[-1]:.2f}). Query type significantly impacts answer quality."
    )
    
    # 3. word_count distribution
    mean_wc = round(df['word_count'].mean(), 1)
    median_wc = round(df['word_count'].median(), 1)
    insights.append(
        f"3. Document length analysis: Average document length is {mean_wc} words (median: {median_wc} words). "
        f"Shorter documents may be easier to retrieve accurately, while very long documents "
        f"may require better chunking strategies."
    )
    
    # 4. correlation between retrieval and answer quality
    r, p_val = stats.pearsonr(df['retrieval_score'], df['answer_quality_score'])
    direction = "positive" if r > 0 else "negative" if r < 0 else "no"
    insight_4 = f"4. Retrieval-quality correlation: There is a {direction} correlation (r = {r:.3f}, p = {p_val:.4f}) "
    insight_4 += f"between retrieval scores and answer quality scores. "
    if r > 0:
        insight_4 += "Higher retrieval scores tend to correlate with higher answer quality."
    elif r < 0:
        insight_4 += "Higher retrieval scores are associated with lower answer quality, suggesting the need for improved ranking or re-ranking strategies."
    else:
        insight_4 += "There is no correlation between retrieval scores and answer quality scores."
    insights.append(insight_4)
    
    # 5. query type distribution
    query_counts = df['query_type'].value_counts()
    insights.append(
        f"5. Query type distribution: The dataset contains {len(df)} questions across {len(query_counts)} query types. "
        f"Most common: {query_counts.index[0]} ({query_counts.iloc[0]} questions), least common: "
        f"{query_counts.index[-1]} ({query_counts.iloc[-1]} questions)."
    )
    
    return insights


# ---------------------------------------------------------------------------
# MAIN EDA FUNCTION
# ---------------------------------------------------------------------------

def main():
    """Run the complete EDA pipeline."""
    print("=" * 60)
    print("RAG Document Q&A System - Exploratory Data Analysis")
    print("=" * 60)
    
    # Ensure output directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(DATASET_PATH)
    
    # Basic checks
    check_nulls(df)
    check_duplicates(df)
    check_dtypes(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save cleaned dataset
    df_clean.to_csv(CLEANED_DATASET_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {CLEANED_DATASET_PATH}")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating visualisations...")
    print("=" * 60)
    
    plot1 = plot_retrieval_score_by_doc_type(df_clean)
    plot2 = plot_answer_quality_by_query_type(df_clean)
    plot3 = plot_word_count_histogram(df_clean)
    plot4 = plot_correlation_heatmap(df_clean)
    
    print(f"\nSaved 4 figures to: {FIGURES_DIR}/")
    print(f"  1. {plot1.name}")
    print(f"  2. {plot2.name}")
    print(f"  3. {plot3.name}")
    print(f"  4. {plot4.name}")
    
    # Generate insights
    print("\n" + "=" * 60)
    print("Generating written insights...")
    print("=" * 60)
    insights = generate_insights(df_clean)
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight}")
    
    # Save insights to file
    insights_path = OUTPUT_DIR / "eda_insights.txt"
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write("# EDA Insights\n\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n\n")
    print(f"\nSaved insights to: {insights_path}")
    
    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()