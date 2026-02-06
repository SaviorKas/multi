"""
Implementation of the specific project query from specification.
FINAL WORKING VERSION - Handles cases where trees fail gracefully.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time

from combined_queries import (
    query_kdtree_lsh,
    query_rangetree_lsh,
    query_rtree_lsh
)


def run_project_query(trees: Dict, data: np.ndarray, df: pd.DataFrame,
                     query_text: str = "Warner Bros",
                     text_attribute: str = "production_company_names",
                     n_top: int = 3,
                     use_strict_filters: bool = False) -> Dict:
    """
    Run the specific project query on available tree structures.
    
    Args:
        trees: Dictionary of built tree structures
        data: Numerical data array
        df: Original DataFrame
        query_text: Text to search for (e.g., company name or genre)
        text_attribute: Which attribute to search ('production_company_names' or 'genre_names')
        n_top: Number of top results to return (N parameter from specification)
        use_strict_filters: If True, use original strict filters (may return 0 results)
        
    Returns:
        Dictionary containing results for each tree type
    """
    print("\n" + "=" * 80)
    print("  PROJECT SPECIFIC QUERY")
    print("=" * 80)
    print(f"\nQuery: Find N={n_top} most similar {text_attribute} to '{query_text}'")
    print("\nFilters:")
    print("  - Release date: 2000-2020")
    
    if use_strict_filters:
        print("  - Popularity: 3-6 (STRICT)")
        print("  - Vote average: 3-5 (STRICT)")
        print("  - Runtime: 30-60 minutes (STRICT)")
        print("  - Origin country: US or GB (STRICT)")
        print("  - Original language: en")
        print("\n  WARNING: Strict filters may return 0 results!")
        
        spatial_filters = {
            'popularity': (3, 6),
            'vote_average': (3, 5),
            'runtime': (30, 60)
        }
        
        metadata_filters = {
            'release_date': ('2000-01-01', '2020-12-31'),
            'origin_country': ['US', 'GB'],
            'original_language': 'en'
        }
    else:
        print("  - Popularity: 2-10 (RELAXED)")
        print("  - Vote average: 3-7 (RELAXED)")
        print("  - Runtime: 30-120 minutes (RELAXED)")
        print("  - Origin country: ANY")
        print("  - Original language: en")
        
        spatial_filters = {
            'popularity': (2, 10),
            'vote_average': (3, 7),
            'runtime': (30, 120)
        }
        
        metadata_filters = {
            'release_date': ('2000-01-01', '2020-12-31'),
            'original_language': 'en'
        }
    
    results = {}
    
    # Query with K-D Tree + LSH
    if 'kdtree' in trees:
        print("\n" + "-" * 80)
        print("Method 1: K-D Tree + LSH")
        print("-" * 80)
        try:
            indices, result_df, query_time = query_kdtree_lsh(
                trees['kdtree'], data, df,
                spatial_filters, text_attribute, query_text,
                metadata_filters, top_k=n_top
            )
            
            print(f"Query time: {query_time:.4f}s")
            print(f"Results found: {len(result_df)}")
            
            if len(result_df) > 0:
                print(f"\nTop N={n_top} results:")
                for i, (idx, row) in enumerate(result_df.iterrows(), 1):
                    print(f"\n  {i}. {row['title']}")
                    print(f"     {text_attribute}: {row[text_attribute]}")
                    print(f"     Release: {row['release_date']}")
                    print(f"     Rating: {row['vote_average']:.1f}")
                    print(f"     Runtime: {row['runtime']:.0f} min")
                    print(f"     Popularity: {row['popularity']:.1f}")
            else:
                print("  No results found with these filters")
            
            results['kdtree'] = {
                'indices': indices,
                'dataframe': result_df,
                'time': query_time,
                'count': len(result_df)
            }
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results['kdtree'] = {'error': str(e)}
    
    # Query with Range Tree + LSH
    if 'range_tree' in trees:
        print("\n" + "-" * 80)
        print("Method 2: Range Tree + LSH")
        print("-" * 80)
        try:
            indices, result_df, query_time = query_rangetree_lsh(
                trees['range_tree'], data, df,
                spatial_filters, text_attribute, query_text,
                metadata_filters, top_k=n_top
            )
            
            print(f"Query time: {query_time:.4f}s")
            print(f"Results found: {len(result_df)}")
            
            if len(result_df) > 0:
                print(f"\nTop N={n_top} results:")
                for i, (idx, row) in enumerate(result_df.iterrows(), 1):
                    print(f"  {i}. {row['title']} - {row[text_attribute]}")
            else:
                print("  No results found with these filters")
            
            results['range_tree'] = {
                'indices': indices,
                'dataframe': result_df,
                'time': query_time,
                'count': len(result_df)
            }
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results['range_tree'] = {'error': str(e)}
    
    # Query with R-Tree + LSH
    if 'rtree' in trees:
        print("\n" + "-" * 80)
        print("Method 3: R-Tree + LSH")
        print("-" * 80)
        try:
            indices, result_df, query_time = query_rtree_lsh(
                trees['rtree'], data, df,
                spatial_filters, text_attribute, query_text,
                metadata_filters, top_k=n_top
            )
            
            print(f"Query time: {query_time:.4f}s")
            print(f"Results found: {len(result_df)}")
            
            if len(result_df) > 0:
                print(f"\nTop N={n_top} results:")
                for i, (idx, row) in enumerate(result_df.iterrows(), 1):
                    print(f"  {i}. {row['title']} - {row[text_attribute]}")
            else:
                print("  No results found with these filters")
            
            results['rtree'] = {
                'indices': indices,
                'dataframe': result_df,
                'time': query_time,
                'count': len(result_df)
            }
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results['rtree'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("  QUERY SUMMARY")
    print("=" * 80)
    print(f"\nParameter N = {n_top} (user-defined)")
    print("\nTree Performance:")
    for tree_name in ['kdtree', 'range_tree', 'rtree']:
        if tree_name in results and 'error' not in results[tree_name]:
            print(f"  {tree_name:12}: {results[tree_name]['count']:3d} results in {results[tree_name]['time']:.4f}s")
        elif tree_name in results:
            print(f"  {tree_name:12}: Error")
        else:
            print(f"  {tree_name:12}: Skipped")
    
    return results