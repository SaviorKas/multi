"""
Quadtree implementation for 2D spatial indexing.
"""

import numpy as np
from typing import List, Tuple, Optional


class QuadtreeNode:
    """Node in a Quadtree."""
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, capacity: int = 50):
        """
        Initialize a Quadtree node.
        
        Args:
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate
            y_min: Minimum y coordinate
            y_max: Maximum y coordinate
            capacity: Maximum number of points before splitting
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.capacity = capacity
        
        self.points = []  # List of (x, y, index) tuples
        self.is_divided = False
        
        # Children nodes (NW, NE, SW, SE)
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None
    
    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within this node's bounds."""
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)
    
    def intersects(self, x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
        """Check if a range intersects with this node's bounds."""
        return not (x_max < self.x_min or x_min > self.x_max or
                   y_max < self.y_min or y_min > self.y_max)
    
    def subdivide(self):
        """Split this node into four quadrants."""
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        
        # Create four children
        self.nw = QuadtreeNode(self.x_min, x_mid, y_mid, self.y_max, self.capacity)
        self.ne = QuadtreeNode(x_mid, self.x_max, y_mid, self.y_max, self.capacity)
        self.sw = QuadtreeNode(self.x_min, x_mid, self.y_min, y_mid, self.capacity)
        self.se = QuadtreeNode(x_mid, self.x_max, self.y_min, y_mid, self.capacity)
        
        self.is_divided = True
        
        # Redistribute existing points to children
        for point in self.points:
            self._insert_to_child(point)
        
        # Clear points from this node
        self.points = []
    
    def _insert_to_child(self, point: Tuple[float, float, int]) -> bool:
        """Insert a point into the appropriate child."""
        x, y, idx = point
        
        if self.nw.contains(x, y):
            return self.nw.insert(x, y, idx)
        elif self.ne.contains(x, y):
            return self.ne.insert(x, y, idx)
        elif self.sw.contains(x, y):
            return self.sw.insert(x, y, idx)
        elif self.se.contains(x, y):
            return self.se.insert(x, y, idx)
        
        return False
    
    def insert(self, x: float, y: float, index: int) -> bool:
        """
        Insert a point into the quadtree.
        
        Args:
            x: X coordinate
            y: Y coordinate
            index: Original index in dataset
            
        Returns:
            True if insertion successful
        """
        # Check if point is in bounds
        if not self.contains(x, y):
            return False
        
        # If node has capacity and is not divided, add point
        if not self.is_divided and len(self.points) < self.capacity:
            self.points.append((x, y, index))
            return True
        
        # If node is at capacity, subdivide
        if not self.is_divided:
            self.subdivide()
        
        # Insert into appropriate child
        return self._insert_to_child((x, y, index))
    
    def query_range(self, x_min: float, x_max: float, 
                   y_min: float, y_max: float) -> List[int]:
        """
        Find all points within the specified range.
        
        Args:
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate
            y_min: Minimum y coordinate
            y_max: Maximum y coordinate
            
        Returns:
            List of indices of points in range
        """
        results = []
        
        # If range doesn't intersect this node, return empty
        if not self.intersects(x_min, x_max, y_min, y_max):
            return results
        
        # Check points in this node
        for x, y, idx in self.points:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                results.append(idx)
        
        # Recursively check children
        if self.is_divided:
            results.extend(self.nw.query_range(x_min, x_max, y_min, y_max))
            results.extend(self.ne.query_range(x_min, x_max, y_min, y_max))
            results.extend(self.sw.query_range(x_min, x_max, y_min, y_max))
            results.extend(self.se.query_range(x_min, x_max, y_min, y_max))
        
        return results
    
    def query_point(self, x: float, y: float, tolerance: float = 0.0) -> List[int]:
        """
        Find points at or near a specific location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            tolerance: Distance tolerance
            
        Returns:
            List of indices of points at this location
        """
        return self.query_range(x - tolerance, x + tolerance, 
                               y - tolerance, y + tolerance)


class Quadtree:
    """
    Quadtree for 2D spatial indexing.
    """
    
    def __init__(self, x_dim: int = 0, y_dim: int = 1, capacity: int = 50):
        """
        Initialize a Quadtree.
        
        Args:
            x_dim: Index of dimension to use for x-axis
            y_dim: Index of dimension to use for y-axis
            capacity: Maximum points per node before splitting
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.capacity = capacity
        self.root = None
        self.size = 0
    
    def build(self, points: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the Quadtree from a set of points.
        
        Args:
            points: Array of shape (n_points, n_dimensions)
            indices: Optional array of original indices
        """
        if indices is None:
            indices = np.arange(len(points))
        
        # Extract 2D coordinates
        x_coords = points[:, self.x_dim]
        y_coords = points[:, self.y_dim]
        
        # Determine bounds with small padding
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add small padding to ensure all points fit
        x_padding = (x_max - x_min) * 0.01
        y_padding = (y_max - y_min) * 0.01
        
        self.root = QuadtreeNode(x_min - x_padding, x_max + x_padding,
                                y_min - y_padding, y_max + y_padding,
                                self.capacity)
        
        # Insert all points
        for i, idx in enumerate(indices):
            self.root.insert(x_coords[i], y_coords[i], idx)
            self.size += 1
    
    def query_range(self, x_range: Tuple[float, float], 
                   y_range: Tuple[float, float]) -> List[int]:
        """
        Find all points within the specified 2D range.
        
        Args:
            x_range: (min, max) for x dimension
            y_range: (min, max) for y dimension
            
        Returns:
            List of indices of points in range
        """
        if self.root is None:
            return []
        
        return self.root.query_range(x_range[0], x_range[1], 
                                     y_range[0], y_range[1])
    
    def query_point(self, x: float, y: float, tolerance: float = 0.0) -> List[int]:
        """
        Find points at or near a specific 2D location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            tolerance: Distance tolerance
            
        Returns:
            List of indices of points at this location
        """
        if self.root is None:
            return []
        
        return self.root.query_point(x, y, tolerance)
    
    def get_depth(self) -> int:
        """Get the maximum depth of the tree."""
        return self._get_depth_recursive(self.root)
    
    def _get_depth_recursive(self, node: Optional[QuadtreeNode]) -> int:
        """Recursively calculate tree depth."""
        if node is None or not node.is_divided:
            return 1
        
        return 1 + max(
            self._get_depth_recursive(node.nw),
            self._get_depth_recursive(node.ne),
            self._get_depth_recursive(node.sw),
            self._get_depth_recursive(node.se)
        )
