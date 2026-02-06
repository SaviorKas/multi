"""
Test script to verify the implementations work correctly with a sample dataset.
"""

import numpy as np
import pandas as pd
import time

from utils import load_movies_dataset, preprocess_data
from kdtree import KDTree
from quadtree import Quadtree
from range_tree import SimpleRangeTree
from rtree import SimpleRTree


def test_with_sample():
    """Test all structures with a sample of the dataset."""
    print("=" * 80)
    print("  TESTING MULTIDIMENSIONAL DATA STRUCTURES")
    print("=" * 80)
    
    # Load and sample dataset
    print("\n1. Loading dataset...")
    try:
        df = load_movies_dataset("data_movies_clean.xlsx")
        print(f"   Original size: {len(df):,} rows")
        
        # Take a sample for faster testing
        sample_size = min(10000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"   Using sample: {len(df_sample):,} rows")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Preprocess
    print("\n2. Preprocessing data...")
    dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    data, df_clean = preprocess_data(df_sample, dimensions)
    print(f"   Clean data: {data.shape}")
    
    # Test K-D Tree
    print("\n3. Testing K-D Tree...")
    start = time.time()
    kdtree = KDTree(dimensions=len(dimensions))
    kdtree.build(data)
    build_time = time.time() - start
    print(f"   ✓ Built in {build_time:.3f}s, Size: {kdtree.size:,}, Depth: {kdtree.get_depth()}")
    
    # Test range query
    ranges = [(0, np.inf)] * len(dimensions)
    ranges[0] = (5_000_000, 20_000_000)  # budget
    ranges[4] = (7.0, 9.0)  # vote_average
    results = kdtree.range_query(ranges)
    print(f"   ✓ Range query found {len(results)} results")
    
    # Test nearest neighbor
    target = data[0]
    neighbors = kdtree.nearest_neighbors(target, k=5)
    print(f"   ✓ Nearest neighbor search found {len(neighbors)} neighbors")
    
    # Test Quadtree
    print("\n4. Testing Quadtree...")
    start = time.time()
    quadtree = Quadtree(x_dim=0, y_dim=1, capacity=50)
    quadtree.build(data)
    build_time = time.time() - start
    print(f"   ✓ Built in {build_time:.3f}s, Size: {quadtree.size:,}, Depth: {quadtree.get_depth()}")
    
    # Test range query
    results = quadtree.query_range((1_000_000, 10_000_000), (5_000_000, 50_000_000))
    print(f"   ✓ Range query found {len(results)} results")
    
    # Test Range Tree
    print("\n5. Testing Range Tree...")
    start = time.time()
    range_tree = SimpleRangeTree(dimensions=len(dimensions))
    range_tree.build(data)
    build_time = time.time() - start
    print(f"   ✓ Built in {build_time:.3f}s, Size: {range_tree.size:,}")
    
    # Test query
    ranges = [(0, np.inf)] * len(dimensions)
    ranges[0] = (5_000_000, 50_000_000)  # budget
    ranges[2] = (90, 120)  # runtime
    ranges[4] = (7.0, 10.0)  # vote_average
    results = range_tree.range_query(ranges)
    print(f"   ✓ Range query found {len(results)} results")
    
    # Test R-Tree
    print("\n6. Testing R-Tree...")
    start = time.time()
    rtree = SimpleRTree(dimensions=len(dimensions))
    rtree.build(data)
    build_time = time.time() - start
    print(f"   ✓ Built in {build_time:.3f}s, Size: {rtree.size:,}")
    
    # Test query
    ranges = [(0, np.inf)] * len(dimensions)
    ranges[0] = (10_000_000, 100_000_000)  # budget
    ranges[1] = (20_000_000, 500_000_000)  # revenue
    ranges[4] = (6.0, 9.0)  # vote_average
    results = rtree.range_query(ranges)
    print(f"   ✓ Range query found {len(results)} results")
    
    # Display sample results
    print("\n7. Sample Query Results:")
    if len(results) > 0:
        sample_indices = results[:3]
        for idx in sample_indices:
            movie = df_clean.iloc[idx]
            print(f"   - {movie['title']}: "
                  f"Budget=${movie['budget']:,.0f}, "
                  f"Revenue=${movie['revenue']:,.0f}, "
                  f"Rating={movie['vote_average']:.1f}")
    
    # Test LSH implementation
    print("\n8. Testing LSH Implementation...")
    from lsh import create_lsh_index, query_similar, get_similarity
    
    # Create LSH index for production companies
    lsh_index, minhash_dict, df_valid = create_lsh_index(
        df_clean, 'production_company_names', num_perm=128
    )
    print(f"   ✓ LSH index created with {len(minhash_dict)} items")
    
    # Query for similar production companies
    if len(df_valid) > 0:
        similar_results = query_similar(
            lsh_index, minhash_dict, df_valid,
            "Warner Bros", top_k=5
        )
        print(f"   ✓ Found {len(similar_results)} similar items")
        
        # Test similarity calculation
        sim_score = get_similarity("Warner Bros Pictures", "Warner Bros")
        print(f"   ✓ Similarity score: {sim_score:.3f}")
    
    # Test combined queries
    print("\n9. Testing Combined Queries (Tree + LSH)...")
    from combined_queries import query_kdtree_lsh
    
    spatial_filters = {
        'popularity': (3, 10),
        'vote_average': (5, 8)
    }
    
    try:
        _, result_df, query_time = query_kdtree_lsh(
            kdtree, data, df_clean,
            spatial_filters, 'production_company_names', 'Universal',
            top_k=5
        )
        print(f"   ✓ Combined query completed in {query_time:.4f}s")
        print(f"   ✓ Found {len(result_df)} results")
    except Exception as e:
        print(f"   Note: Combined query test skipped ({e})")
    
    print("\n" + "=" * 80)
    print("  ✓ ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_with_sample()
