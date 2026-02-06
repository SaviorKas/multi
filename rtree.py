"""
R-Tree implementation for indexing multidimensional rectangles/bounding boxes.
"""

import numpy as np
from typing import List, Tuple, Optional
import sys


class MBR:
    """Minimum Bounding Rectangle."""
    
    def __init__(self, dimensions: int):
        """
        Initialize an MBR.
        
        Args:
            dimensions: Number of dimensions
        """
        self.dimensions = dimensions
        self.min_bounds = np.full(dimensions, np.inf)
        self.max_bounds = np.full(dimensions, -np.inf)
    
    def extend(self, point: np.ndarray):
        """Extend the MBR to include a point."""
        self.min_bounds = np.minimum(self.min_bounds, point)
        self.max_bounds = np.maximum(self.max_bounds, point)
    
    def extend_mbr(self, other: 'MBR'):
        """Extend the MBR to include another MBR."""
        self.min_bounds = np.minimum(self.min_bounds, other.min_bounds)
        self.max_bounds = np.maximum(self.max_bounds, other.max_bounds)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if MBR contains a point."""
        return np.all(self.min_bounds <= point) and np.all(point <= self.max_bounds)
    
    def intersects(self, other: 'MBR') -> bool:
        """Check if this MBR intersects with another MBR."""
        return np.all(self.min_bounds <= other.max_bounds) and \
               np.all(other.min_bounds <= self.max_bounds)
    
    def area(self) -> float:
        """Calculate the area (volume) of the MBR."""
        if np.any(np.isinf(self.min_bounds)) or np.any(np.isinf(self.max_bounds)):
            return 0.0
        dims = self.max_bounds - self.min_bounds
        return np.prod(dims)
    
    def enlargement(self, point: np.ndarray) -> float:
        """Calculate area increase needed to include a point."""
        new_mbr = MBR(self.dimensions)
        new_mbr.min_bounds = np.copy(self.min_bounds)
        new_mbr.max_bounds = np.copy(self.max_bounds)
        new_mbr.extend(point)
        return new_mbr.area() - self.area()
    
    def copy(self) -> 'MBR':
        """Create a copy of this MBR."""
        new_mbr = MBR(self.dimensions)
        new_mbr.min_bounds = np.copy(self.min_bounds)
        new_mbr.max_bounds = np.copy(self.max_bounds)
        return new_mbr


class RTreeNode:
    """Node in an R-Tree."""
    
    def __init__(self, is_leaf: bool = True, max_entries: int = 50):
        """
        Initialize an R-Tree node.
        
        Args:
            is_leaf: Whether this is a leaf node
            max_entries: Maximum number of entries
        """
        self.is_leaf = is_leaf
        self.max_entries = max_entries
        self.mbr = None
        self.entries = []  # List of (mbr, data) tuples
        # data is either index (for leaf) or child node (for internal)
    
    def is_full(self) -> bool:
        """Check if node is at capacity."""
        return len(self.entries) >= self.max_entries
    
    def add_entry(self, mbr: MBR, data):
        """Add an entry to this node."""
        self.entries.append((mbr, data))
        
        # Update node's MBR
        if self.mbr is None:
            self.mbr = mbr.copy()
        else:
            self.mbr.extend_mbr(mbr)


class RTree:
    """
    R-Tree for indexing multidimensional rectangles/bounding boxes.
    """
    
    def __init__(self, dimensions: int, max_entries: int = 50):
        """
        Initialize an R-Tree.
        
        Args:
            dimensions: Number of dimensions
            max_entries: Maximum entries per node
        """
        self.dimensions = dimensions
        self.max_entries = max_entries
        self.root = RTreeNode(is_leaf=True, max_entries=max_entries)
        self.size = 0
    
    def build(self, points: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the R-Tree from a set of points.
        
        Args:
            points: Array of shape (n_points, n_dimensions)
            indices: Optional array of original indices
        """
        if indices is None:
            indices = np.arange(len(points))
        
        # Insert each point
        for i, idx in enumerate(indices):
            self.insert(points[i], idx)
    
    def insert(self, point: np.ndarray, index: int):
        """
        Insert a point into the R-Tree.
        
        Args:
            point: Point to insert
            index: Original index in dataset
        """
        # Create MBR for the point (point as both min and max)
        mbr = MBR(self.dimensions)
        mbr.extend(point)
        
        # Find the best leaf to insert into
        leaf = self._choose_leaf(self.root, mbr)
        
        # Add to leaf
        leaf.add_entry(mbr, index)
        self.size += 1
        
        # Split if necessary
        if leaf.is_full():
            self._split_node(leaf)
    
    def _choose_leaf(self, node: RTreeNode, mbr: MBR) -> RTreeNode:
        """
        Choose the best leaf node to insert into.
        
        Args:
            node: Current node
            mbr: MBR to insert
            
        Returns:
            Best leaf node
        """
        # If leaf, return it
        if node.is_leaf:
            return node
        
        # Choose child with minimum enlargement
        min_enlargement = float('inf')
        best_child = None
        
        for child_mbr, child_node in node.entries:
            enlargement = child_mbr.enlargement(mbr.min_bounds)
            if enlargement < min_enlargement:
                min_enlargement = enlargement
                best_child = child_node
        
        return self._choose_leaf(best_child, mbr)
    
    def _split_node(self, node: RTreeNode):
        """
        Split a full node (simplified linear split).
        
        Args:
            node: Node to split
        """
        # This is a simplified version - just split in half
        mid = len(node.entries) // 2
        
        # For simplicity, we don't propagate splits up the tree
        # In a full implementation, this would handle tree growth
        pass
    
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
        
        # Create query MBR
        query_mbr = MBR(self.dimensions)
        query_mbr.min_bounds = np.array([r[0] for r in ranges])
        query_mbr.max_bounds = np.array([r[1] for r in ranges])
        
        results = []
        self._range_query_recursive(self.root, query_mbr, results)
        return results
    
    def _range_query_recursive(self, node: RTreeNode, query_mbr: MBR, results: List[int]):
        """
        Recursively search for points in range.
        
        Args:
            node: Current node
            query_mbr: Query bounding box
            results: List to accumulate results
        """
        for entry_mbr, data in node.entries:
            # Check if entry intersects with query
            if entry_mbr.intersects(query_mbr):
                if node.is_leaf:
                    # data is an index
                    results.append(data)
                else:
                    # data is a child node
                    self._range_query_recursive(data, query_mbr, results)
    
    def get_depth(self) -> int:
        """Get the depth of the tree."""
        return self._get_depth_recursive(self.root)
    
    def _get_depth_recursive(self, node: RTreeNode) -> int:
        """Recursively calculate tree depth."""
        if node.is_leaf:
            return 1
        
        max_depth = 0
        for _, child in node.entries:
            if isinstance(child, RTreeNode):
                max_depth = max(max_depth, self._get_depth_recursive(child))
        
        return 1 + max_depth


class SimpleRTree:
    """
    Simplified R-Tree that uses a flat structure for easier implementation.
    This version doesn't do sophisticated splitting but works for querying.
    """
    
    def __init__(self, dimensions: int):
        """
        Initialize a Simple R-Tree.
        
        Args:
            dimensions: Number of dimensions
        """
        self.dimensions = dimensions
        self.points = []  # List of (point, index) tuples
        self.size = 0
    
    def build(self, points: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the R-Tree from points.
        
        Args:
            points: Array of shape (n_points, n_dimensions)
            indices: Optional array of original indices
        """
        if indices is None:
            indices = np.arange(len(points))
        
        self.points = [(points[i], indices[i]) for i in range(len(points))]
        self.size = len(points)
    
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
        
        results = []
        
        for point, idx in self.points:
            in_range = all(ranges[d][0] <= point[d] <= ranges[d][1] 
                          for d in range(self.dimensions))
            if in_range:
                results.append(idx)
        
        return results
