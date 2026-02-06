"""
Unit tests for multidimensional data structures.
Run with: python -m pytest test_unit.py (requires pytest)
Or: python test_unit.py (standalone)
"""

import numpy as np
import sys


def test_kdtree():
    """Test K-D Tree basic functionality."""
    from kdtree import KDTree
    
    # Create small test dataset
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0]
    ])
    
    # Build tree
    kdtree = KDTree(dimensions=3)
    kdtree.build(data)
    
    assert kdtree.size == 5, f"Expected size 5, got {kdtree.size}"
    assert kdtree.get_depth() > 0, "Tree depth should be positive"
    
    # Test range query
    ranges = [(0, 5), (0, 5), (0, 5)]
    results = kdtree.range_query(ranges)
    assert len(results) > 0, "Range query should return results"
    
    # Test nearest neighbor
    target = data[0]
    neighbors = kdtree.nearest_neighbors(target, k=3)
    assert len(neighbors) == 3, f"Expected 3 neighbors, got {len(neighbors)}"
    assert neighbors[0][0] == 0, "First neighbor should be itself"
    
    print("✓ K-D Tree tests passed")


def test_quadtree():
    """Test Quadtree basic functionality."""
    from quadtree import Quadtree
    
    # Create small test dataset
    data = np.array([
        [1.0, 2.0],
        [4.0, 5.0],
        [7.0, 8.0],
        [2.0, 3.0],
        [5.0, 6.0]
    ])
    
    # Build tree
    quadtree = Quadtree(x_dim=0, y_dim=1, capacity=2)
    quadtree.build(data)
    
    assert quadtree.size == 5, f"Expected size 5, got {quadtree.size}"
    assert quadtree.get_depth() > 0, "Tree depth should be positive"
    
    # Test range query
    results = quadtree.query_range((0, 5), (0, 5))
    assert len(results) > 0, "Range query should return results"
    
    # Test point query
    results = quadtree.query_point(1.0, 2.0, tolerance=0.1)
    assert len(results) > 0, "Point query should return results"
    
    print("✓ Quadtree tests passed")


def test_rangetree():
    """Test Range Tree basic functionality."""
    from range_tree import SimpleRangeTree
    
    # Create small test dataset
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0]
    ])
    
    # Build tree
    range_tree = SimpleRangeTree(dimensions=3)
    range_tree.build(data)
    
    assert range_tree.size == 5, f"Expected size 5, got {range_tree.size}"
    
    # Test range query
    ranges = [(0, 5), (0, 5), (0, 5)]
    results = range_tree.range_query(ranges)
    assert len(results) > 0, "Range query should return results"
    
    # Test with specific range
    ranges = [(1, 3), (2, 4), (3, 5)]
    results = range_tree.range_query(ranges)
    assert isinstance(results, list), "Results should be a list"
    
    print("✓ Range Tree tests passed")


def test_rtree():
    """Test R-Tree basic functionality."""
    from rtree import SimpleRTree
    
    # Create small test dataset
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0]
    ])
    
    # Build tree
    rtree = SimpleRTree(dimensions=3)
    rtree.build(data)
    
    assert rtree.size == 5, f"Expected size 5, got {rtree.size}"
    
    # Test range query
    ranges = [(0, 5), (0, 5), (0, 5)]
    results = rtree.range_query(ranges)
    assert len(results) > 0, "Range query should return results"
    
    # Test with specific range
    ranges = [(3, 8), (4, 9), (5, 10)]
    results = rtree.range_query(ranges)
    assert isinstance(results, list), "Results should be a list"
    
    print("✓ R-Tree tests passed")


def test_utils():
    """Test utility functions."""
    from utils import normalize_data, denormalize_data
    
    # Test normalization
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    normalized, min_vals, max_vals = normalize_data(data)
    
    assert normalized.shape == data.shape, "Shape should be preserved"
    assert np.all(normalized >= 0) and np.all(normalized <= 1), "Values should be in [0, 1]"
    
    # Test denormalization
    denormalized = denormalize_data(normalized, min_vals, max_vals)
    assert np.allclose(data, denormalized), "Denormalization should restore original values"
    
    print("✓ Utility function tests passed")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("Running Unit Tests")
    print("="*70)
    
    tests = [
        ("K-D Tree", test_kdtree),
        ("Quadtree", test_quadtree),
        ("Range Tree", test_rangetree),
        ("R-Tree", test_rtree),
        ("Utils", test_utils)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\nTesting {name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} tests failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
