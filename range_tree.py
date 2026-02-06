"""
Range Tree implementation for efficient orthogonal range queries.
"""

import numpy as np
from typing import List, Tuple, Optional
import bisect


class RangeTreeNode:
    """Node in a Range Tree."""
    
    def __init__(self, value: float, index: int, dim: int):
        """
        Initialize a Range Tree node.
        
        Args:
            value: Value in the current dimension
            index: Original index in dataset
            dim: Current dimension
        """
        self.value = value
        self.index = index
        self.dim = dim
        self.left = None
        self.right = None
        self.associated_structure = None  # For next dimension


class RangeTree:
    """
    Range Tree for efficient multidimensional orthogonal range queries.
    """
    
    def __init__(self, dimensions: int):
        """
        Initialize a Range Tree.
        
        Args:
            dimensions: Number of dimensions
        """
        self.dimensions = dimensions
        self.root = None
        self.size = 0
        self.dimension_names = [f"dim_{i}" for i in range(dimensions)]
    
    def build(self, points: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the Range Tree from a set of points.
        
        Args:
            points: Array of shape (n_points, n_dimensions)
            indices: Optional array of original indices
        """
        if indices is None:
            indices = np.arange(len(points))
        
        self.root = self._build_recursive(points, indices, dim=0)
        self.size = len(points)
    
    def _build_recursive(self, points: np.ndarray, indices: np.ndarray, 
                        dim: int) -> Optional[RangeTreeNode]:
        """
        Recursively build the Range Tree.
        
        Args:
            points: Points for this subtree
            indices: Corresponding indices
            dim: Current dimension
            
        Returns:
            Root node of subtree
        """
        if len(points) == 0:
            return None
        
        # Sort by current dimension
        sorted_order = np.argsort(points[:, dim])
        sorted_points = points[sorted_order]
        sorted_indices = indices[sorted_order]
        
        # Find median
        median_idx = len(sorted_points) // 2
        median_value = sorted_points[median_idx, dim]
        
        # Create node
        node = RangeTreeNode(median_value, sorted_indices[median_idx], dim)
        
        # Build associated structure for next dimension if not last
        if dim < self.dimensions - 1:
            node.associated_structure = self._build_1d_tree(sorted_points, sorted_indices, dim + 1)
        
        # Recursively build left and right subtrees
        node.left = self._build_recursive(sorted_points[:median_idx], 
                                          sorted_indices[:median_idx], dim)
        node.right = self._build_recursive(sorted_points[median_idx + 1:], 
                                           sorted_indices[median_idx + 1:], dim)
        
        return node
    
    def _build_1d_tree(self, points: np.ndarray, indices: np.ndarray, 
                      dim: int) -> List[Tuple[float, int]]:
        """
        Build a 1D sorted array for a specific dimension.
        
        Args:
            points: Points to include
            indices: Corresponding indices
            dim: Dimension to sort by
            
        Returns:
            Sorted list of (value, index) tuples
        """
        values_with_indices = [(points[i, dim], indices[i]) for i in range(len(points))]
        values_with_indices.sort(key=lambda x: x[0])
        return values_with_indices
    
    def range_query(self, ranges: List[Tuple[float, float]]) -> List[int]:
        """
        Find all points within the specified ranges.
        
        Args:
            ranges: List of (min, max) tuples for each dimension
            
        Returns:
            List of indices of points in range
        """
        if len(ranges) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} ranges, got {len(ranges)}")
        
        results = set()
        self._range_query_recursive(self.root, ranges, 0, results)
        return list(results)
    
    def _range_query_recursive(self, node: Optional[RangeTreeNode], 
                               ranges: List[Tuple[float, float]], 
                               dim: int, results: set):
        """
        Recursively perform range query.
        
        Args:
            node: Current node
            ranges: Range constraints
            dim: Current dimension
            results: Set to accumulate results
        """
        if node is None:
            return
        
        min_val, max_val = ranges[dim]
        
        # Check if this node's value is in range for current dimension
        if min_val <= node.value <= max_val:
            # If this is the last dimension, check the point
            if dim == self.dimensions - 1:
                results.add(node.index)
            else:
                # Query associated structure for remaining dimensions
                self._query_associated(node.associated_structure, ranges, dim + 1, results)
        
        # Recurse on children based on range
        if min_val <= node.value:
            self._range_query_recursive(node.left, ranges, dim, results)
        if max_val >= node.value:
            self._range_query_recursive(node.right, ranges, dim, results)
    
    def _query_associated(self, structure: Optional[List[Tuple[float, int]]], 
                         ranges: List[Tuple[float, float]], 
                         dim: int, results: set):
        """
        Query the associated 1D structure.
        
        Args:
            structure: Sorted list of (value, index) tuples
            ranges: Range constraints
            dim: Current dimension
            results: Set to accumulate results
        """
        if structure is None:
            return
        
        min_val, max_val = ranges[dim]
        
        # Binary search for range
        for value, idx in structure:
            if value < min_val:
                continue
            if value > max_val:
                break
            
            # If last dimension, add to results
            if dim == self.dimensions - 1:
                results.add(idx)
    
    def get_depth(self) -> int:
        """Get the depth of the tree."""
        return self._get_depth_recursive(self.root)
    
    def _get_depth_recursive(self, node: Optional[RangeTreeNode]) -> int:
        """Recursively calculate tree depth."""
        if node is None:
            return 0
        return 1 + max(self._get_depth_recursive(node.left),
                      self._get_depth_recursive(node.right))


class SimpleRangeTree:
    """
    Simplified Range Tree implementation using sorted arrays for each dimension.
    This is more memory-efficient and easier to understand.
    """
    
    def __init__(self, dimensions: int):
        """
        Initialize a Simple Range Tree.
        
        Args:
            dimensions: Number of dimensions
        """
        self.dimensions = dimensions
        self.sorted_dims = []  # List of sorted arrays, one per dimension
        self.size = 0
    
    def build(self, points: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the Range Tree by creating sorted arrays for each dimension.
        
        Args:
            points: Array of shape (n_points, n_dimensions)
            indices: Optional array of original indices
        """
        if indices is None:
            indices = np.arange(len(points))
        
        self.sorted_dims = []
        
        # For each dimension, create a sorted array with (value, index) pairs
        for dim in range(self.dimensions):
            dim_data = [(points[i, dim], indices[i]) for i in range(len(points))]
            dim_data.sort(key=lambda x: x[0])
            self.sorted_dims.append(dim_data)
        
        self.size = len(points)
    
    def range_query(self, ranges: List[Tuple[float, float]]) -> List[int]:
        """
        Find all points within the specified ranges using intersection.
        
        Args:
            ranges: List of (min, max) tuples for each dimension
            
        Returns:
            List of indices of points in range
        """
        if len(ranges) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} ranges, got {len(ranges)}")
        
        # Get candidates from first dimension
        candidates = self._query_dimension(0, ranges[0])
        
        # Intersect with other dimensions
        for dim in range(1, self.dimensions):
            if not candidates:
                break
            dim_results = self._query_dimension(dim, ranges[dim])
            candidates = candidates.intersection(dim_results)
        
        return list(candidates)
    
    def _query_dimension(self, dim: int, range_val: Tuple[float, float]) -> set:
        """
        Query a single dimension.
        
        Args:
            dim: Dimension index
            range_val: (min, max) range
            
        Returns:
            Set of indices in range for this dimension
        """
        min_val, max_val = range_val
        results = set()
        
        # Binary search for start position
        sorted_data = self.sorted_dims[dim]
        start_idx = bisect.bisect_left([x[0] for x in sorted_data], min_val)
        
        # Collect all values in range
        for i in range(start_idx, len(sorted_data)):
            value, idx = sorted_data[i]
            if value > max_val:
                break
            results.add(idx)
        
        return results
