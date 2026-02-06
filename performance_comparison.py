"""
Performance comparison and benchmarking for all tree + LSH combinations.
Measures build times, query times, memory usage, and generates visualizations.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys

from lsh import create_lsh_index
from combined_queries import (
    query_kdtree_lsh,
    query_quadtree_lsh,
    query_rangetree_lsh,
    query_rtree_lsh
)


def measure_build_times(trees: Dict, df: pd.DataFrame, 
                       text_attribute: str = 'production_company_names',
                       num_perm: int = 128) -> Dict[str, Dict[str, float]]:
    """
    Measure build times for tree structures and LSH indices.
    
    Args:
        trees: Dictionary of built trees with their build times
        df: DataFrame for LSH indexing
        text_attribute: Column to build LSH index on
        num_perm: Number of permutations for MinHash
        
    Returns:
        Dictionary of build time metrics
    """
    print("\n" + "=" * 80)
    print("  MEASURING BUILD TIMES")
    print("=" * 80)
    
    results = {}
    
    # Tree build times are already measured
    for tree_name, tree_obj in trees.items():
        if hasattr(tree_obj, 'build_time'):
            tree_time = tree_obj.build_time
        else:
            tree_time = 0.0  # Fallback if not available
        
        # Measure LSH build time
        print(f"\nBuilding LSH index for {tree_name}...")
        start = time.time()
        lsh_index, minhash_dict, df_valid = create_lsh_index(df, text_attribute, num_perm)
        lsh_time = time.time() - start
        
        results[tree_name] = {
            'tree_build_time': tree_time,
            'lsh_build_time': lsh_time,
            'total_build_time': tree_time + lsh_time,
            'lsh_index_size': len(minhash_dict)
        }
        
        print(f"  LSH build time: {lsh_time:.4f}s")
        print(f"  LSH index size: {len(minhash_dict)} items")
    
    return results


def measure_query_times(trees: Dict, data: np.ndarray, df: pd.DataFrame,
                       query_text: str = "Warner Bros",
                       text_attribute: str = 'production_company_names',
                       num_queries: int = 5,
                       top_k: int = 10) -> Dict[str, List[float]]:
    """
    Measure query times for combined tree + LSH queries.
    
    Args:
        trees: Dictionary of built trees
        data: Numerical data array
        df: DataFrame
        query_text: Text to query for
        text_attribute: Column to search
        num_queries: Number of queries to average over
        top_k: Number of top results
        
    Returns:
        Dictionary of query time lists
    """
    print("\n" + "=" * 80)
    print("  MEASURING QUERY TIMES")
    print("=" * 80)
    print(f"\nRunning {num_queries} queries per method...")
    
    spatial_filters = {
        'popularity': (3, 10),
        'vote_average': (5, 8),
        'runtime': (60, 150)
    }
    
    metadata_filters = {
        'release_date': ('2000-01-01', '2020-12-31'),
        'origin_country': ['US', 'GB'],
        'original_language': 'en'
    }
    
    results = {
        'kdtree': [],
        'quadtree': [],
        'range_tree': [],
        'rtree': []
    }
    
    query_functions = {
        'kdtree': query_kdtree_lsh,
        'quadtree': query_quadtree_lsh,
        'range_tree': query_rangetree_lsh,
        'rtree': query_rtree_lsh
    }
    
    for method_name, query_func in query_functions.items():
        print(f"\n{method_name}:")
        for i in range(num_queries):
            try:
                _, _, query_time = query_func(
                    trees[method_name], data, df,
                    spatial_filters, text_attribute, query_text,
                    metadata_filters, top_k
                )
                results[method_name].append(query_time)
                print(f"  Query {i+1}: {query_time:.4f}s")
            except Exception as e:
                print(f"  Query {i+1}: Error - {e}")
                results[method_name].append(np.nan)
    
    return results


def calculate_memory_estimates(trees: Dict, df: pd.DataFrame) -> Dict[str, int]:
    """
    Estimate memory usage for each structure.
    
    Args:
        trees: Dictionary of built trees
        df: DataFrame for size estimation
        
    Returns:
        Dictionary of memory estimates in bytes
    """
    memory_estimates = {}
    
    for tree_name, tree_obj in trees.items():
        # Use tree size attribute if available, otherwise estimate based on data
        if hasattr(tree_obj, 'size'):
            tree_size = tree_obj.size
        else:
            # Conservative estimate: assume at least 1 node per data point
            tree_size = len(df)
        
        # Different memory multipliers for different structures
        if tree_name == 'kdtree':
            bytes_per_node = 100
        elif tree_name == 'quadtree':
            bytes_per_node = 120
        elif tree_name == 'range_tree':
            bytes_per_node = 200
        else:  # rtree
            bytes_per_node = 150
        
        tree_memory = tree_size * bytes_per_node
        
        # LSH memory estimate (MinHash + index)
        # Approximately 128 integers per item (num_perm) + overhead
        lsh_memory = len(df) * 128 * 4  # 4 bytes per int
        
        memory_estimates[tree_name] = {
            'tree_memory': tree_memory,
            'lsh_memory': lsh_memory,
            'total_memory': tree_memory + lsh_memory
        }
    
    return memory_estimates


def generate_comparison_table(build_results: Dict, query_results: Dict, 
                             memory_estimates: Dict) -> pd.DataFrame:
    """
    Generate a comprehensive comparison table.
    
    Args:
        build_results: Build time measurements
        query_results: Query time measurements
        memory_estimates: Memory usage estimates
        
    Returns:
        DataFrame with comparison metrics
    """
    data = []
    
    for method in ['kdtree', 'quadtree', 'range_tree', 'rtree']:
        row = {
            'Method': f"{method.replace('_', ' ').title()} + LSH",
            'Tree Build (s)': build_results[method]['tree_build_time'],
            'LSH Build (s)': build_results[method]['lsh_build_time'],
            'Total Build (s)': build_results[method]['total_build_time'],
            'Avg Query (s)': np.nanmean(query_results[method]) if query_results[method] else 0,
            'Min Query (s)': np.nanmin(query_results[method]) if query_results[method] else 0,
            'Max Query (s)': np.nanmax(query_results[method]) if query_results[method] else 0,
            'Memory (KB)': memory_estimates[method]['total_memory'] / 1024
        }
        data.append(row)
    
    return pd.DataFrame(data)


def create_visualizations(comparison_df: pd.DataFrame, output_prefix: str = 'performance'):
    """
    Create comparison visualizations.
    
    Args:
        comparison_df: Comparison DataFrame
        output_prefix: Prefix for output files
    """
    print("\n" + "=" * 80)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Build time comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = comparison_df['Method'].str.replace(' + LSH', '')
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_df['Tree Build (s)'], width, label='Tree Build', alpha=0.8)
    ax1.bar(x + width/2, comparison_df['LSH Build (s)'], width, label='LSH Build', alpha=0.8)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Build Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Figure 2: Query time comparison
    ax2.bar(methods, comparison_df['Avg Query (s)'], alpha=0.8)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Average Query Time Comparison')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig(f'{output_prefix}_build_query_times.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {output_prefix}_build_query_times.png")
    except Exception as e:
        print(f"Could not save figure: {e}")
    plt.close()
    
    # Figure 3: Memory usage comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(methods, comparison_df['Memory (KB)'], alpha=0.8, color='coral')
    ax.set_xlabel('Method')
    ax.set_ylabel('Memory Usage (KB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig(f'{output_prefix}_memory.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {output_prefix}_memory.png")
    except Exception as e:
        print(f"Could not save figure: {e}")
    plt.close()


def run_performance_comparison(trees: Dict, build_times: Dict, data: np.ndarray, 
                              df: pd.DataFrame, output_csv: str = 'performance_results.csv'):
    """
    Run complete performance comparison and generate report.
    
    Args:
        trees: Dictionary of built tree structures
        build_times: Dictionary of tree build times
        data: Numerical data array
        df: DataFrame
        output_csv: Output CSV file name
    """
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Add build times to tree objects
    for tree_name, tree_obj in trees.items():
        if tree_name in build_times:
            tree_obj.build_time = build_times[tree_name]
    
    # Measure build times (including LSH)
    build_results = measure_build_times(trees, df, 'production_company_names')
    
    # Measure query times
    query_results = measure_query_times(trees, data, df, num_queries=5)
    
    # Calculate memory estimates
    memory_estimates = calculate_memory_estimates(trees, df)
    
    # Generate comparison table
    comparison_df = generate_comparison_table(build_results, query_results, memory_estimates)
    
    # Display results
    print("\n" + "=" * 80)
    print("  PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print()
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    try:
        comparison_df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to {output_csv}")
    except Exception as e:
        print(f"\nCould not save CSV: {e}")
    
    # Generate visualizations
    create_visualizations(comparison_df)
    
    # Print summary insights
    print("\n" + "=" * 80)
    print("  KEY INSIGHTS")
    print("=" * 80)
    
    # Check if we have valid data
    if len(comparison_df) == 0 or comparison_df['Total Build (s)'].isna().all():
        print("\nInsufficient data for insights")
        return comparison_df
    
    fastest_build = comparison_df.loc[comparison_df['Total Build (s)'].idxmin(), 'Method']
    fastest_query = comparison_df.loc[comparison_df['Avg Query (s)'].idxmin(), 'Method']
    lowest_memory = comparison_df.loc[comparison_df['Memory (KB)'].idxmin(), 'Method']
    
    print(f"\n✓ Fastest Build Time: {fastest_build}")
    print(f"✓ Fastest Query Time: {fastest_query}")
    print(f"✓ Lowest Memory Usage: {lowest_memory}")
    
    print("\nNote: All methods use the same LSH implementation for phase 2 similarity.")
    print("Performance differences primarily come from the spatial filtering phase (phase 1).")
    
    return comparison_df
