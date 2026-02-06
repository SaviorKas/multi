"""
LSH (Locality-Sensitive Hashing) implementation for textual similarity queries on movie attributes.
Uses MinHash for efficient set-based similarity on production companies and genres.
"""

import ast
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from datasketch import MinHash, MinHashLSH


def parse_list_string(value: Union[str, list]) -> List[str]:
    """
    Parse a string representation of a list into an actual list.
    
    Args:
        value: String representation of list or actual list
        
    Returns:
        Parsed list of strings
    """
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == '' or value == '[]':
        return []
    
    try:
        # Try to parse as Python literal (for strings like "['a', 'b']")
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        # If parsing fails, treat as single string
        return [str(value)]


def tokenize_text(text: Union[str, List[str]]) -> set:
    """
    Tokenize text into a set of tokens for similarity comparison.
    
    Args:
        text: Text string or list of strings to tokenize
        
    Returns:
        Set of tokens (lowercased words)
    """
    if isinstance(text, list):
        # Join list elements with space
        text = ' '.join(str(item) for item in text)
    
    if not text or pd.isna(text):
        return set()
    
    # Tokenize: lowercase, split on non-alphanumeric, filter short words
    tokens = text.lower().replace(',', ' ').replace('[', '').replace(']', '').replace("'", '').split()
    return set(token for token in tokens if len(token) > 1)


def create_minhash(tokens: set, num_perm: int = 128) -> MinHash:
    """
    Create a MinHash object from a set of tokens.
    
    Args:
        tokens: Set of string tokens
        num_perm: Number of permutations for MinHash (higher = more accurate)
        
    Returns:
        MinHash object
    """
    mh = MinHash(num_perm=num_perm)
    for token in tokens:
        mh.update(token.encode('utf-8'))
    return mh


def create_lsh_index(df: pd.DataFrame, text_column: str, num_perm: int = 128,
                     threshold: float = 0.5, verbose: bool = True) -> Tuple[MinHashLSH, Dict[int, MinHash], pd.DataFrame]:
    """
    Build an LSH index for textual similarity on a DataFrame column.
    
    Args:
        df: DataFrame containing the text data
        text_column: Name of the column containing text to index
        num_perm: Number of permutations for MinHash (default: 128)
        threshold: Jaccard similarity threshold for LSH (default: 0.5)
        verbose: Whether to print progress messages (default: True)
        
    Returns:
        Tuple of (lsh_index, minhash_dict, filtered_df)
        - lsh_index: MinHashLSH index for similarity queries
        - minhash_dict: Dictionary mapping indices to MinHash objects
        - filtered_df: DataFrame with only valid (non-empty) entries
    """
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhash_dict = {}
    
    # Filter to only rows with non-empty text
    valid_indices = []
    
    for idx, row in df.iterrows():
        # Parse and tokenize the text column
        text_value = row[text_column]
        parsed_list = parse_list_string(text_value)
        tokens = tokenize_text(parsed_list)
        
        # Only add if we have tokens
        if tokens:
            mh = create_minhash(tokens, num_perm)
            minhash_dict[idx] = mh
            lsh.insert(str(idx), mh)
            valid_indices.append(idx)
    
    # Create filtered DataFrame with valid indices
    df_filtered = df.loc[valid_indices].copy()
    
    if verbose:
        print(f"Created LSH index for '{text_column}' with {len(minhash_dict)} items")
    
    return lsh, minhash_dict, df_filtered


def query_similar(lsh_index: MinHashLSH, minhash_dict: Dict[int, MinHash],
                 df: pd.DataFrame, query_text: Union[str, List[str]], 
                 top_k: int = 10, num_perm: int = 128) -> List[Tuple[int, float]]:
    """
    Find top-K most similar items to a query text using LSH.
    
    Args:
        lsh_index: MinHashLSH index
        minhash_dict: Dictionary mapping indices to MinHash objects
        df: DataFrame containing the data
        query_text: Query text or list of strings
        top_k: Number of top results to return
        num_perm: Number of permutations (must match index)
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity (descending)
    """
    # Tokenize query
    tokens = tokenize_text(query_text)
    
    if not tokens:
        return []
    
    # Create MinHash for query
    query_mh = create_minhash(tokens, num_perm)
    
    # Query LSH index for candidates
    candidates = lsh_index.query(query_mh)
    
    # Calculate exact Jaccard similarity for candidates
    similarities = []
    for candidate_key in candidates:
        candidate_idx = int(candidate_key)
        if candidate_idx in minhash_dict:
            candidate_mh = minhash_dict[candidate_idx]
            similarity = query_mh.jaccard(candidate_mh)
            similarities.append((candidate_idx, similarity))
    
    # Sort by similarity (descending) and return top-K
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def get_similarity(text1: Union[str, List[str]], text2: Union[str, List[str]], 
                  num_perm: int = 128) -> float:
    """
    Calculate Jaccard similarity between two texts using MinHash.
    
    Args:
        text1: First text or list of strings
        text2: Second text or list of strings
        num_perm: Number of permutations for MinHash
        
    Returns:
        Estimated Jaccard similarity (0.0 to 1.0)
    """
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    mh1 = create_minhash(tokens1, num_perm)
    mh2 = create_minhash(tokens2, num_perm)
    
    return mh1.jaccard(mh2)


def query_similar_by_index(lsh_index: MinHashLSH, minhash_dict: Dict[int, MinHash],
                           df: pd.DataFrame, query_idx: int, 
                           top_k: int = 10) -> List[Tuple[int, float]]:
    """
    Find top-K most similar items to a specific item in the dataset.
    
    Args:
        lsh_index: MinHashLSH index
        minhash_dict: Dictionary mapping indices to MinHash objects
        df: DataFrame containing the data
        query_idx: Index of the query item in the DataFrame
        top_k: Number of top results to return
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity (descending)
    """
    if query_idx not in minhash_dict:
        return []
    
    query_mh = minhash_dict[query_idx]
    
    # Query LSH index
    candidates = lsh_index.query(query_mh)
    
    # Calculate exact similarities
    similarities = []
    for candidate_key in candidates:
        candidate_idx = int(candidate_key)
        if candidate_idx in minhash_dict and candidate_idx != query_idx:
            candidate_mh = minhash_dict[candidate_idx]
            similarity = query_mh.jaccard(candidate_mh)
            similarities.append((candidate_idx, similarity))
    
    # Sort by similarity (descending) and return top-K
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
