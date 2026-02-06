"""
Combined query system: Two-phase filtering using spatial trees and LSH similarity.
Phase 1: Filter by numerical attributes using tree structures
Phase 2: Apply LSH for textual similarity on filtered results
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from datasketch import MinHashLSH

from lsh import create_lsh_index, query_similar
from utils import filter_by_metadata


def query_kdtree_lsh(kdtree, data: np.ndarray, df: pd.DataFrame,
                     spatial_filters: Dict[str, Tuple[float, float]],
                     text_attribute: str,
                     query_text: str,
                     metadata_filters: Optional[Dict] = None,
                     top_k: int = 10,
                     num_perm: int = 128) -> Tuple[List[int], pd.DataFrame, float]:
    """
    Combined query using K-D Tree for spatial filtering and LSH for similarity.
    
    Args:
        kdtree: Built K-D Tree
        data: Numerical data array
        df: Original DataFrame
        spatial_filters: Dict mapping dimension names to (min, max) ranges
        text_attribute: Column name for textual similarity (e.g., 'production_company_names')
        query_text: Text to find similar items to
        metadata_filters: Optional dict of categorical filters
        top_k: Number of top similar results to return
        num_perm: Number of permutations for MinHash
        
    Returns:
        Tuple of (result_indices, result_df, total_time)
    """
    import time
    start_time = time.time()
    
    # Phase 1: Spatial filtering with K-D Tree
    dimension_names = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    ranges = []
    for dim in dimension_names:
        if dim in spatial_filters:
            ranges.append(spatial_filters[dim])
        else:
            ranges.append((0, np.inf))
    
    spatial_results = kdtree.range_query(ranges)
    
    if not spatial_results:
        return [], pd.DataFrame(), time.time() - start_time
    
    # Filter DataFrame by spatial results
    df_filtered = df.iloc[spatial_results].copy()
    
    # Apply metadata filters if provided
    if metadata_filters:
        df_filtered = filter_by_metadata(df_filtered, metadata_filters)
    
    if len(df_filtered) == 0:
        return [], df_filtered, time.time() - start_time
    
    # Phase 2: LSH similarity on filtered results
    lsh_index, minhash_dict, df_valid = create_lsh_index(df_filtered, text_attribute, num_perm, verbose=False)
    
    if len(df_valid) == 0:
        return [], df_valid, time.time() - start_time
    
    similar_results = query_similar(lsh_index, minhash_dict, df_valid, query_text, top_k, num_perm)
    
    # Get actual indices from similar results
    result_indices = [idx for idx, _ in similar_results]
    result_df = df_valid.loc[result_indices].copy()
    
    total_time = time.time() - start_time
    return result_indices, result_df, total_time


def query_quadtree_lsh(quadtree, data: np.ndarray, df: pd.DataFrame,
                       spatial_filters: Dict[str, Tuple[float, float]],
                       text_attribute: str,
                       query_text: str,
                       metadata_filters: Optional[Dict] = None,
                       top_k: int = 10,
                       num_perm: int = 128) -> Tuple[List[int], pd.DataFrame, float]:
    """
    Combined query using Quadtree for spatial filtering and LSH for similarity.
    
    Note: Quadtree uses only 2D (budget and revenue by default)
    
    Args:
        quadtree: Built Quadtree
        data: Numerical data array
        df: Original DataFrame
        spatial_filters: Dict with 'budget' and 'revenue' ranges
        text_attribute: Column name for textual similarity
        query_text: Text to find similar items to
        metadata_filters: Optional dict of categorical filters
        top_k: Number of top similar results to return
        num_perm: Number of permutations for MinHash
        
    Returns:
        Tuple of (result_indices, result_df, total_time)
    """
    import time
    start_time = time.time()
    
    # Phase 1: Spatial filtering with Quadtree (2D only)
    x_range = spatial_filters.get('budget', (0, np.inf))
    y_range = spatial_filters.get('revenue', (0, np.inf))
    
    spatial_results = quadtree.query_range(x_range, y_range)
    
    if not spatial_results:
        return [], pd.DataFrame(), time.time() - start_time
    
    # Filter DataFrame by spatial results
    df_filtered = df.iloc[spatial_results].copy()
    
    # Apply additional numerical filters manually (Quadtree only handles 2D)
    for dim in ['runtime', 'popularity', 'vote_average', 'vote_count']:
        if dim in spatial_filters:
            min_val, max_val = spatial_filters[dim]
            df_filtered = df_filtered[
                (df_filtered[dim] >= min_val) & (df_filtered[dim] <= max_val)
            ]
    
    # Apply metadata filters if provided
    if metadata_filters:
        df_filtered = filter_by_metadata(df_filtered, metadata_filters)
    
    if len(df_filtered) == 0:
        return [], df_filtered, time.time() - start_time
    
    # Phase 2: LSH similarity on filtered results
    lsh_index, minhash_dict, df_valid = create_lsh_index(df_filtered, text_attribute, num_perm, verbose=False)
    
    if len(df_valid) == 0:
        return [], df_valid, time.time() - start_time
    
    similar_results = query_similar(lsh_index, minhash_dict, df_valid, query_text, top_k, num_perm)
    
    # Get actual indices from similar results
    result_indices = [idx for idx, _ in similar_results]
    result_df = df_valid.loc[result_indices].copy()
    
    total_time = time.time() - start_time
    return result_indices, result_df, total_time


def query_rangetree_lsh(range_tree, data: np.ndarray, df: pd.DataFrame,
                        spatial_filters: Dict[str, Tuple[float, float]],
                        text_attribute: str,
                        query_text: str,
                        metadata_filters: Optional[Dict] = None,
                        top_k: int = 10,
                        num_perm: int = 128) -> Tuple[List[int], pd.DataFrame, float]:
    """
    Combined query using Range Tree for spatial filtering and LSH for similarity.
    
    Args:
        range_tree: Built Range Tree
        data: Numerical data array
        df: Original DataFrame
        spatial_filters: Dict mapping dimension names to (min, max) ranges
        text_attribute: Column name for textual similarity
        query_text: Text to find similar items to
        metadata_filters: Optional dict of categorical filters
        top_k: Number of top similar results to return
        num_perm: Number of permutations for MinHash
        
    Returns:
        Tuple of (result_indices, result_df, total_time)
    """
    import time
    start_time = time.time()
    
    # Phase 1: Spatial filtering with Range Tree
    dimension_names = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    ranges = []
    for dim in dimension_names:
        if dim in spatial_filters:
            ranges.append(spatial_filters[dim])
        else:
            ranges.append((0, np.inf))
    
    spatial_results = range_tree.range_query(ranges)
    
    if not spatial_results:
        return [], pd.DataFrame(), time.time() - start_time
    
    # Filter DataFrame by spatial results
    df_filtered = df.iloc[spatial_results].copy()
    
    # Apply metadata filters if provided
    if metadata_filters:
        df_filtered = filter_by_metadata(df_filtered, metadata_filters)
    
    if len(df_filtered) == 0:
        return [], df_filtered, time.time() - start_time
    
    # Phase 2: LSH similarity on filtered results
    lsh_index, minhash_dict, df_valid = create_lsh_index(df_filtered, text_attribute, num_perm, verbose=False)
    
    if len(df_valid) == 0:
        return [], df_valid, time.time() - start_time
    
    similar_results = query_similar(lsh_index, minhash_dict, df_valid, query_text, top_k, num_perm)
    
    # Get actual indices from similar results
    result_indices = [idx for idx, _ in similar_results]
    result_df = df_valid.loc[result_indices].copy()
    
    total_time = time.time() - start_time
    return result_indices, result_df, total_time


def query_rtree_lsh(rtree, data: np.ndarray, df: pd.DataFrame,
                    spatial_filters: Dict[str, Tuple[float, float]],
                    text_attribute: str,
                    query_text: str,
                    metadata_filters: Optional[Dict] = None,
                    top_k: int = 10,
                    num_perm: int = 128) -> Tuple[List[int], pd.DataFrame, float]:
    """
    Combined query using R-Tree for spatial filtering and LSH for similarity.
    
    Args:
        rtree: Built R-Tree
        data: Numerical data array
        df: Original DataFrame
        spatial_filters: Dict mapping dimension names to (min, max) ranges
        text_attribute: Column name for textual similarity
        query_text: Text to find similar items to
        metadata_filters: Optional dict of categorical filters
        top_k: Number of top similar results to return
        num_perm: Number of permutations for MinHash
        
    Returns:
        Tuple of (result_indices, result_df, total_time)
    """
    import time
    start_time = time.time()
    
    # Phase 1: Spatial filtering with R-Tree
    dimension_names = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    ranges = []
    for dim in dimension_names:
        if dim in spatial_filters:
            ranges.append(spatial_filters[dim])
        else:
            ranges.append((0, np.inf))
    
    spatial_results = rtree.range_query(ranges)
    
    if not spatial_results:
        return [], pd.DataFrame(), time.time() - start_time
    
    # Filter DataFrame by spatial results
    df_filtered = df.iloc[spatial_results].copy()
    
    # Apply metadata filters if provided
    if metadata_filters:
        df_filtered = filter_by_metadata(df_filtered, metadata_filters)
    
    if len(df_filtered) == 0:
        return [], df_filtered, time.time() - start_time
    
    # Phase 2: LSH similarity on filtered results
    lsh_index, minhash_dict, df_valid = create_lsh_index(df_filtered, text_attribute, num_perm, verbose=False)
    
    if len(df_valid) == 0:
        return [], df_valid, time.time() - start_time
    
    similar_results = query_similar(lsh_index, minhash_dict, df_valid, query_text, top_k, num_perm)
    
    # Get actual indices from similar results
    result_indices = [idx for idx, _ in similar_results]
    result_df = df_valid.loc[result_indices].copy()
    
    total_time = time.time() - start_time
    return result_indices, result_df, total_time
