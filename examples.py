"""
Example usage script showing how to use each data structure independently.
"""

import numpy as np
from utils import load_movies_dataset, preprocess_data
from kdtree import KDTree
from quadtree import Quadtree
from range_tree import SimpleRangeTree
from rtree import SimpleRTree


# Load and preprocess data once at the module level
print("Loading dataset (this may take a moment)...")
df = load_movies_dataset("data_movies_clean.xlsx")
df_sample = df.sample(n=5000, random_state=42).reset_index(drop=True)
dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
data, df_clean = preprocess_data(df_sample, dimensions)
print(f"Loaded {len(df_clean):,} movies for examples\n")


def example_kdtree():
    """Example: Using K-D Tree for nearest neighbor search."""
    print("\n" + "="*70)
    print("K-D TREE EXAMPLE: Finding Similar Movies")
    print("="*70)
    
    # Build K-D Tree
    kdtree = KDTree(dimensions=len(dimensions))
    kdtree.build(data)
    print(f"\nBuilt K-D Tree with {kdtree.size:,} nodes (depth: {kdtree.get_depth()})")
    
    # Find similar movies to "The Matrix" (if it exists in sample)
    # For demo, we'll just use the first movie
    target_idx = 0
    target_movie = df_clean.iloc[target_idx]
    
    print(f"\nFinding movies similar to: {target_movie['title']}")
    print(f"  Budget: ${target_movie['budget']:,.0f}")
    print(f"  Revenue: ${target_movie['revenue']:,.0f}")
    print(f"  Rating: {target_movie['vote_average']:.1f}")
    
    # Get 5 nearest neighbors
    neighbors = kdtree.nearest_neighbors(data[target_idx], k=6)
    
    print(f"\nTop 5 similar movies:")
    for i, (idx, distance) in enumerate(neighbors[1:6]):  # Skip first (itself)
        movie = df_clean.iloc[idx]
        print(f"\n{i+1}. {movie['title']} (similarity score: {1000/distance:.2f})")
        print(f"   Budget: ${movie['budget']:,.0f}")
        print(f"   Revenue: ${movie['revenue']:,.0f}")
        print(f"   Rating: {movie['vote_average']:.1f}")


def example_quadtree():
    """Example: Using Quadtree for spatial queries."""
    print("\n" + "="*70)
    print("QUADTREE EXAMPLE: Budget vs Revenue Analysis")
    print("="*70)
    
    # Build Quadtree (budget vs revenue)
    quadtree = Quadtree(x_dim=0, y_dim=1, capacity=50)
    quadtree.build(data)
    print(f"\nBuilt Quadtree with {quadtree.size:,} nodes (depth: {quadtree.get_depth()})")
    
    # Find movies in sweet spot: moderate budget, high revenue
    print("\nSearching for movies with:")
    print("  Budget: $5M - $30M")
    print("  Revenue: $50M - $500M")
    
    results = quadtree.query_range(
        x_range=(5_000_000, 30_000_000),
        y_range=(50_000_000, 500_000_000)
    )
    
    print(f"\nFound {len(results)} movies in this profitable range:")
    for idx in results[:5]:
        movie = df_clean.iloc[idx]
        roi = (movie['revenue'] / movie['budget'] - 1) * 100 if movie['budget'] > 0 else 0
        print(f"\n  {movie['title']}")
        print(f"    Budget: ${movie['budget']:,.0f}")
        print(f"    Revenue: ${movie['revenue']:,.0f}")
        print(f"    ROI: {roi:.1f}%")


def example_rangetree():
    """Example: Using Range Tree for multi-dimensional filtering."""
    print("\n" + "="*70)
    print("RANGE TREE EXAMPLE: Finding Perfect Movies")
    print("="*70)
    
    # Build Range Tree
    range_tree = SimpleRangeTree(dimensions=len(dimensions))
    range_tree.build(data)
    print(f"\nBuilt Range Tree with {range_tree.size:,} nodes")
    
    # Find high-quality movies with specific criteria
    print("\nSearching for high-quality movies with:")
    print("  Budget: $10M - $50M (mid-budget)")
    print("  Runtime: 90-150 minutes (standard length)")
    print("  Rating: 7.5-10 (highly rated)")
    print("  Vote count: 100+ (well-reviewed)")
    
    ranges = [
        (10_000_000, 50_000_000),  # budget
        (0, np.inf),                # revenue (any)
        (90, 150),                  # runtime
        (0, np.inf),                # popularity (any)
        (7.5, 10.0),                # vote_average
        (100, np.inf)               # vote_count
    ]
    
    results = range_tree.range_query(ranges)
    
    print(f"\nFound {len(results)} movies matching all criteria:")
    for idx in results[:5]:
        movie = df_clean.iloc[idx]
        print(f"\n  {movie['title']}")
        print(f"    Budget: ${movie['budget']:,.0f}")
        print(f"    Runtime: {movie['runtime']:.0f} min")
        print(f"    Rating: {movie['vote_average']:.1f} ({movie['vote_count']:.0f} votes)")


def example_rtree():
    """Example: Using R-Tree for bounding box queries."""
    print("\n" + "="*70)
    print("R-TREE EXAMPLE: Blockbuster Analysis")
    print("="*70)
    
    # Build R-Tree
    rtree = SimpleRTree(dimensions=len(dimensions))
    rtree.build(data)
    print(f"\nBuilt R-Tree with {rtree.size:,} nodes")
    
    # Find blockbuster movies
    print("\nSearching for blockbuster movies with:")
    print("  Budget: $50M+ (big budget)")
    print("  Revenue: $100M+ (box office success)")
    print("  Popularity: 10+ (widely known)")
    print("  Rating: 6+ (audience approved)")
    
    ranges = [
        (50_000_000, np.inf),      # budget
        (100_000_000, np.inf),     # revenue
        (0, np.inf),                # runtime (any)
        (10, np.inf),               # popularity
        (6.0, 10.0),                # vote_average
        (0, np.inf)                 # vote_count (any)
    ]
    
    results = rtree.range_query(ranges)
    
    print(f"\nFound {len(results)} blockbuster movies:")
    for idx in results[:5]:
        movie = df_clean.iloc[idx]
        print(f"\n  {movie['title']}")
        print(f"    Budget: ${movie['budget']:,.0f}")
        print(f"    Revenue: ${movie['revenue']:,.0f}")
        print(f"    Popularity: {movie['popularity']:.1f}")
        print(f"    Rating: {movie['vote_average']:.1f}")


if __name__ == "__main__":
    print("="*70)
    print("MULTIDIMENSIONAL DATA STRUCTURES - USAGE EXAMPLES")
    print("="*70)
    
    # Run all examples
    try:
        example_kdtree()
    except Exception as e:
        print(f"K-D Tree example error: {e}")
    
    try:
        example_quadtree()
    except Exception as e:
        print(f"Quadtree example error: {e}")
    
    try:
        example_rangetree()
    except Exception as e:
        print(f"Range Tree example error: {e}")
    
    try:
        example_rtree()
    except Exception as e:
        print(f"R-Tree example error: {e}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
