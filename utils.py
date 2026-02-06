"""
Utility functions for loading and preprocessing the movies dataset.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def load_movies_dataset(filepath: str = "data_movies_clean.csv") -> pd.DataFrame:
    """
    Load the movies dataset from Excel or CSV file.
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        DataFrame containing the movies dataset
    """
    try:
        df = pd.read_csv(filepath)
                    
        print(f"Loaded {len(df):,} movies from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def preprocess_data(df: pd.DataFrame, 
                    dimensions: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocess the movies dataset for tree structures.
    
    Args:
        df: Raw movies DataFrame
        dimensions: List of numerical columns to use (default: all numerical dimensions)
        
    Returns:
        Tuple of (processed_data_array, cleaned_dataframe)
    """
    if dimensions is None:
        dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle missing values - fill with median for numerical columns
    for dim in dimensions:
        if dim in df_clean.columns:
            median_val = df_clean[dim].median()
            df_clean[dim] = df_clean[dim].fillna(median_val)
            # Replace any infinity values
            df_clean[dim] = df_clean[dim].replace([np.inf, -np.inf], median_val)
    
    # Filter out rows with all zeros in dimensions
    mask = (df_clean[dimensions] != 0).any(axis=1)
    df_clean = df_clean[mask].reset_index(drop=True)
    
    # Extract numerical data as numpy array
    data_array = df_clean[dimensions].values.astype(np.float64)
    
    print(f"Preprocessed {len(df_clean):,} valid movies")
    print(f"Dimensions used: {dimensions}")
    
    return data_array, df_clean


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data to [0, 1] range for each dimension.
    
    Args:
        data: Raw data array
        
    Returns:
        Tuple of (normalized_data, min_values, max_values)
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    normalized = (data - min_vals) / range_vals
    
    return normalized, min_vals, max_vals


def denormalize_data(normalized_data: np.ndarray, 
                    min_vals: np.ndarray, 
                    max_vals: np.ndarray) -> np.ndarray:
    """
    Convert normalized data back to original scale.
    
    Args:
        normalized_data: Normalized data array
        min_vals: Minimum values for each dimension
        max_vals: Maximum values for each dimension
        
    Returns:
        Denormalized data array
    """
    range_vals = max_vals - min_vals
    return normalized_data * range_vals + min_vals


def get_sample_queries() -> dict:
    """
    Return example query configurations for demonstration.
    
    Returns:
        Dictionary of example queries
    """
    queries = {
        'budget_range': {
            'description': 'Movies with budget $5M-$20M and rating 7-9',
            'budget': (5_000_000, 20_000_000),
            'vote_average': (7.0, 9.0)
        },
        'popular_movies': {
            'description': 'Popular movies with high revenue',
            'popularity': (50, 500),
            'revenue': (100_000_000, 1_000_000_000)
        },
        'runtime_range': {
            'description': 'Movies between 90-120 minutes with good ratings',
            'runtime': (90, 120),
            'vote_average': (6.5, 10.0)
        }
    }
    return queries


def filter_by_metadata(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Filter DataFrame by categorical/metadata attributes.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of filter criteria:
            - 'release_date': tuple of (start_date, end_date) as strings
            - 'origin_country': list of country codes or single code
            - 'original_language': language code string or list
            - Other categorical columns: value or list of values
            
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    for column, criteria in filters.items():
        if column not in df_filtered.columns:
            continue
            
        if column == 'release_date':
            # Handle date range filtering
            start_date, end_date = criteria
            df_filtered['release_date'] = pd.to_datetime(df_filtered['release_date'], errors='coerce')
            df_filtered = df_filtered[
                (df_filtered['release_date'] >= start_date) & 
                (df_filtered['release_date'] <= end_date)
            ]
        elif column == 'origin_country':
            # Handle list column filtering (country can be in list)
            if isinstance(criteria, str):
                criteria = [criteria]
            # Filter rows where any of the criteria countries are in the origin_country list
            mask = df_filtered[column].apply(
                lambda x: any(country in str(x) for country in criteria) if pd.notna(x) else False
            )
            df_filtered = df_filtered[mask]
        elif isinstance(criteria, list):
            # Multiple values (OR condition)
            df_filtered = df_filtered[df_filtered[column].isin(criteria)]
        else:
            # Single value
            df_filtered = df_filtered[df_filtered[column] == criteria]
    
    return df_filtered.reset_index(drop=True)


def parse_list_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Parse a column containing string representations of lists.
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to parse
        
    Returns:
        DataFrame with parsed column
    """
    import ast
    
    def safe_parse(value):
        """Safely parse list string."""
        if pd.isna(value) or value == '' or value == '[]':
            return []
        if isinstance(value, list):
            return value
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except (ValueError, SyntaxError):
            return [value]
    
    df_copy = df.copy()
    if column_name in df_copy.columns:
        df_copy[column_name] = df_copy[column_name].apply(safe_parse)
    
    return df_copy


def prepare_text_for_lsh(text: str) -> str:
    """
    Prepare text for LSH by cleaning and tokenizing.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text == '':
        return ""
    
    # Convert to lowercase, remove special characters
    text = str(text).lower()
    text = text.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
    text = text.replace(',', ' ').replace('  ', ' ')
    
    return text.strip()
