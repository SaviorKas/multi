import pandas as pd
from combined_queries import query_kdtree_lsh
from kdtree import KDTree
from utils import preprocess_data

# Load data
df = pd.read_csv('data_movies_clean.csv')
data, df_clean = preprocess_data(df)
dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']

# Build tree
print("Building K-D Tree...")
kdtree = KDTree(dimensions=len(dimensions))
kdtree.build(data)

print("\n" + "="*80)
print("TESTING LSH with RELAXED FILTERS")
print("="*80)

# More realistic filters for Warner Bros movies
spatial_filters = {
    'popularity': (2, 10),      # Relaxed from 3-6
    'vote_average': (3, 7),     # Relaxed from 3-5
    'runtime': (30, 120)        # Relaxed from 30-60
}

metadata_filters = {
    'release_date': ('2000-01-01', '2020-12-31'),
    'original_language': 'en'
    # Removed origin_country filter - it's too strict!
}

print("\nSearching for movies similar to 'Warner' with:")
print(f"  - Popularity: 2-10")
print(f"  - Vote average: 3-7")
print(f"  - Runtime: 30-120 min")
print(f"  - Release: 2000-2020")
print(f"  - Language: en")

indices, result_df, query_time = query_kdtree_lsh(
    kdtree, data, df_clean,
    spatial_filters, 'production_company_names', 'Warner',
    metadata_filters, top_k=5
)

print(f"\n✅ Query time: {query_time:.4f}s")
print(f"✅ Results found: {len(result_df)}")

if len(result_df) > 0:
    print("\n🎬 Top similar production companies:")
    for i, (idx, row) in enumerate(result_df.iterrows(), 1):
        print(f"\n{i}. {row['title']}")
        print(f"   Company: {row['production_company_names']}")
        print(f"   Release: {row['release_date']}, Rating: {row['vote_average']:.1f}, Runtime: {row['runtime']:.0f}min")
        print(f"   Popularity: {row['popularity']:.1f}")
else:
    print("\n❌ Still no results - LSH threshold might be too strict")