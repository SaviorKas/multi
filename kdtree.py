"""
K-D Tree implementation for multidimensional point queries and nearest neighbor searches.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from collections import namedtuple


class KDNode:
    """Node in a K-D Tree."""
    
    def __init__(self, point: np.ndarray, index: int, axis: int):
        """
        Initialize a K-D Tree node.
        
        Args:
            point: The data point (feature vector)
            index: Original index in the dataset
            axis: The splitting dimension for this node
        """
        self.point = point
        self.index = index
        self.axis = axis
        self.left = None
        self.right = None


class KDTree:
    """
    K-D Tree for efficient multidimensional point queries and nearest neighbor searches.
    """
    
    def __init__(self, dimensions: int):
        """
        Initialize an empty K-D Tree.
        
        Args:
            dimensions: Number of dimensions in the data
        """
        self.dimensions = dimensions
        self.root = None
        self.size = 0
    
    def build(self, points: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the K-D Tree from a set of points.
        
        Args:
            points: Array of shape (n_points, n_dimensions)
            indices: Optional array of original indices
        """
        if indices is None:
            indices = np.arange(len(points))
        
        self.root = self._build_recursive(points, indices, depth=0)
        self.size = len(points)
    
    def _build_recursive(self, points: np.ndarray, indices: np.ndarray, depth: int) -> Optional[KDNode]:
        """
        Recursively build the K-D Tree.
        
        Args:
            points: Points to add to this subtree
            indices: Corresponding original indices
            depth: Current depth in the tree
            
        Returns:
            Root node of the subtree
        """
        if len(points) == 0:
            return None
        
        # Select axis based on depth
        axis = depth % self.dimensions
        
        # Sort points by the current axis
        sorted_indices = np.argsort(points[:, axis])
        median_idx = len(points) // 2
        
        # Create node with median point
        median_pos = sorted_indices[median_idx]
        node = KDNode(points[median_pos], indices[median_pos], axis)
        
        # Recursively build left and right subtrees
        left_mask = sorted_indices[:median_idx]
        right_mask = sorted_indices[median_idx + 1:]
        
        node.left = self._build_recursive(points[left_mask], indices[left_mask], depth + 1)
        node.right = self._build_recursive(points[right_mask], indices[right_mask], depth + 1)
        
        return node
    
    def range_query(self, ranges: List[Tuple[float, float]]) -> List[int]:
        """
        Find all points within the specified ranges.
        
        Args:
            ranges: List of (min, max) tuples for each dimension
            
        Returns:
            List of indices of points within the range
        """
        if len(ranges) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} ranges, got {len(ranges)}")
        
        results = []
        self._range_query_recursive(self.root, ranges, results)
        return results
    
    def _range_query_recursive(self, node: Optional[KDNode], 
                               ranges: List[Tuple[float, float]], 
                               results: List[int]):
        """
        Recursively search for points in range.
        
        Args:
            node: Current node
            ranges: Range constraints
            results: List to accumulate results
        """
        if node is None:
            return
        
        # Check if current point is in range
        in_range = all(ranges[i][0] <= node.point[i] <= ranges[i][1] 
                      for i in range(self.dimensions))
        
        if in_range:
            results.append(node.index)
        
        # Determine which subtrees to search
        axis = node.axis
        if ranges[axis][0] <= node.point[axis]:
            self._range_query_recursive(node.left, ranges, results)
        if ranges[axis][1] >= node.point[axis]:
            self._range_query_recursive(node.right, ranges, results)
    
    def nearest_neighbors(self, target: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to the target point.
        
        Args:
            target: Target point
            k: Number of neighbors to find
            
        Returns:
            List of (index, distance) tuples for k nearest neighbors
        """
        if len(target) != self.dimensions:
            raise ValueError(f"Target must have {self.dimensions} dimensions")
        
        # Use a max heap to keep track of k nearest neighbors
        best = []
        self._nearest_recursive(self.root, target, k, best)
        
        # Sort by distance
        best.sort(key=lambda x: x[1])
        return best[:k]
    
    def _nearest_recursive(self, node: Optional[KDNode], target: np.ndarray, 
                          k: int, best: List[Tuple[int, float]]):
        """
        Recursively search for nearest neighbors.
        
        Args:
            node: Current node
            target: Target point
            k: Number of neighbors needed
            best: List to accumulate best candidates
        """
        if node is None:
            return
        
        # Calculate distance to current node
        dist = np.linalg.norm(node.point - target)
        
        # Update best list
        if len(best) < k:
            best.append((node.index, dist))
            best.sort(key=lambda x: x[1], reverse=True)
        elif dist < best[0][1]:
            best[0] = (node.index, dist)
            best.sort(key=lambda x: x[1], reverse=True)
        
        # Determine which branch to search first
        axis = node.axis
        if target[axis] < node.point[axis]:
            near_node = node.left
            far_node = node.right
        else:
            near_node = node.right
            far_node = node.left
        
        # Search near branch
        self._nearest_recursive(near_node, target, k, best)
        
        # Check if we need to search far branch
        if len(best) < k or abs(target[axis] - node.point[axis]) < best[0][1]:
            self._nearest_recursive(far_node, target, k, best)
    
    def get_depth(self) -> int:
        """Get the depth of the tree."""
        return self._get_depth_recursive(self.root)
    
    def _get_depth_recursive(self, node: Optional[KDNode]) -> int:
        """Recursively calculate tree depth."""
        if node is None:
            return 0
        return 1 + max(self._get_depth_recursive(node.left), 
                      self._get_depth_recursive(node.right))
