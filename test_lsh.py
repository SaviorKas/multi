import pandas as pd
from combined_queries import query_kdtree_lsh
from kdtree import KDTree
import numpy as np
from utils import load_movies_dataset, preprocess_data

# Load data
df = pd.read_csv('data_movies_clean.csv')
data, df_clean = preprocess_data(df)  # Only 2 values returned!
dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']

# Build tree
kdtree = KDTree(dimensions=len(dimensions))
kdtree.build(data)

# Test with looser threshold
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

# Try searching for just "Warner" instead
indices, result_df, query_time = query_kdtree_lsh(
    kdtree, data, df_clean,
    spatial_filters, 'production_company_names', 'Warner',
    metadata_filters, top_k=5
)

print(f"Found {len(result_df)} results")
if len(result_df) > 0:
    print(result_df[['title', 'production_company_names']].head())
else:
    print("No results found - filters might be too strict")