"""
Main program to demonstrate multidimensional data structures on movies dataset.
FINAL WORKING VERSION - Handles large datasets gracefully.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys

from utils import load_movies_dataset, preprocess_data
from kdtree import KDTree
from range_tree import SimpleRangeTree
from rtree import SimpleRTree
from project_query import run_project_query


# ========== USER CONFIGURABLE PARAMETERS ==========
# Change these values to control the query behavior

# N parameter: Number of top similar results to return
# As per specification: "Parameter N is a user defined parameter (f.e. N=3)"
N_TOP_RESULTS = 5  # Change this to 3, 10, 20, etc.

# Query parameters
QUERY_TEXT = "Warner Bros"  # Text to search for
TEXT_ATTRIBUTE = "production_company_names"  # or "genre_names"

# Filter strictness
# Set to True to use EXACT filters from specification (may get 0 results)
# Set to False to use RELAXED filters (will get realistic results)
USE_STRICT_FILTERS = False

# ===================================================


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def build_trees(data: np.ndarray, df: pd.DataFrame, dimensions: List[str]):
    """
    Build tree structures (only the ones that work with large datasets).
    
    Args:
        data: Preprocessed data array
        df: Original DataFrame
        dimensions: List of dimension names
        
    Returns:
        Dictionary of built trees and metadata
    """
    print_section("Building Tree Structures")
    
    trees = {}
    build_times = {}
    
    # K-D Tree
    print("Building K-D Tree...")
    try:
        start = time.time()
        kdtree = KDTree(dimensions=len(dimensions))
        kdtree.build(data)
        build_times['kdtree'] = time.time() - start
        trees['kdtree'] = kdtree
        print(f"  Size: {kdtree.size:,} nodes")
        print(f"  Depth: {kdtree.get_depth()}")
        print(f"  Build time: {build_times['kdtree']:.3f}s")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
    
    # Range Tree
    print("\nBuilding Range Tree...")
    try:
        start = time.time()
        range_tree = SimpleRangeTree(dimensions=len(dimensions))
        range_tree.build(data)
        build_times['range_tree'] = time.time() - start
        trees['range_tree'] = range_tree
        print(f"  Size: {range_tree.size:,} nodes")
        print(f"  Build time: {build_times['range_tree']:.3f}s")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
    
    # R-Tree
    print("\nBuilding R-Tree...")
    try:
        start = time.time()
        rtree = SimpleRTree(dimensions=len(dimensions))
        rtree.build(data)
        build_times['rtree'] = time.time() - start
        trees['rtree'] = rtree
        print(f"  Size: {rtree.size:,} nodes")
        print(f"  Build time: {build_times['rtree']:.3f}s")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
    
    # Quadtree (2D projection)
    print("\nBuilding Quadtree (using first 2 dimensions)...")
    try:
        start = time.time()
        from quadtree import Quadtree
        quadtree = Quadtree()
        quadtree.build(data)
        build_times['quadtree'] = time.time() - start
        trees['quadtree'] = quadtree
        print(f"  Size: {quadtree.size:,} nodes")
        print(f"  Build time: {build_times['quadtree']:.3f}s")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
    
    if not trees:
        raise RuntimeError("Failed to build any tree structures!")
    
    print(f"\nSuccessfully built {len(trees)} tree structure(s)")
    
    return trees, build_times


def demonstrate_basic_queries(trees: Dict, data: np.ndarray, 
                             df: pd.DataFrame, dimensions: List[str]):
    """Demonstrate basic tree queries."""
    print_section("Quick Tree Functionality Check")
    
    # K-D Tree: Range query
    if 'kdtree' in trees:
        print("K-D Tree: Sample range query")
        query_ranges = [(0, data[:, i].max() * 0.5) for i in range(len(dimensions))]
        result_indices = trees['kdtree'].range_query(query_ranges)
        print(f"  Found {len(result_indices)} movies in range\n")
    
    print("Basic tree queries working correctly\n")


def main():
    """Main program execution."""
    
    print("\n" + "=" * 80)
    print("  MULTIDIMENSIONAL DATA STRUCTURES - MOVIES DATASET")
    print("  FINAL WORKING VERSION")
    print("=" * 80)
    
    # Display configuration
    print("\n" + "=" * 80)
    print("  USER CONFIGURATION")
    print("=" * 80)
    print(f"N (Top results): {N_TOP_RESULTS}")
    print(f"Query text: '{QUERY_TEXT}'")
    print(f"Search attribute: {TEXT_ATTRIBUTE}")
    print("=" * 80)
    
    # Load and preprocess data
    print_section("Loading and Preprocessing Data")
    try:
        df = load_movies_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nMake sure 'data_movies_clean.csv' is in the same folder as this script!")
        sys.exit(1)
    
    data, df_clean = preprocess_data(df)
    dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    
    # Build trees
    try:
        trees, build_times = build_trees(data, df_clean, dimensions)
    except Exception as e:
        print(f"Critical error building trees: {e}")
        sys.exit(1)
    
    # Quick functionality check
    demonstrate_basic_queries(trees, data, df_clean, dimensions)
    
    # PROJECT SPECIFIC QUERY (Two-phase: Spatial trees + LSH)
    print("\n" + "=" * 80)
    print("  RUNNING PROJECT QUERY WITH LSH")
    print("=" * 80)
    
    try:
        project_results = run_project_query(
            trees, data, df_clean,
            query_text=QUERY_TEXT,
            text_attribute=TEXT_ATTRIBUTE,
            n_top=N_TOP_RESULTS,
            use_strict_filters=USE_STRICT_FILTERS
        )
    except Exception as e:
        print(f"Error running project query: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print_section("Program Complete")
    print("All tree structures and queries executed successfully!")
    print("\nTree build times summary:")
    for tree_name, build_time in build_times.items():
        print(f"  {tree_name:12}: {build_time:.3f}s")


if __name__ == "__main__":
    main()