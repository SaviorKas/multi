"""
Quadtree implementation for 2D spatial indexing.
"""

import numpy as np
from typing import List, Tuple, Optional


class QuadtreeNode:
    """Node in a Quadtree."""
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, 
                 capacity: int = 50, depth: int = 0, max_depth: int = 25):
        """
        Initialize a Quadtree node.
        
        Args:
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate
            y_min: Minimum y coordinate
            y_max: Maximum y coordinate
            capacity: Maximum number of points before splitting
            depth: Current depth in the tree (0 = root)
            max_depth: Maximum allowed depth to prevent infinite recursion
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.capacity = capacity
        self.depth = depth
        self.max_depth = max_depth
        
        self.points = []  # List of (x, y, index, full_data) tuples - now stores full 6D data
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
        # Prevent infinite recursion - stop at max depth
        if self.depth >= self.max_depth:
            return
        
        # Check if all points are at the same location
        if len(self.points) > 1:
            first_x, first_y = self.points[0][0], self.points[0][1]
            all_same = all(abs(p[0] - first_x) < 1e-10 and abs(p[1] - first_y) < 1e-10 
                          for p in self.points)
            if all_same:
                # Can't split identical points, keep them here
                return
        
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        
        # Create four children with incremented depth
        self.nw = QuadtreeNode(self.x_min, x_mid, y_mid, self.y_max, 
                               self.capacity, self.depth + 1, self.max_depth)
        self.ne = QuadtreeNode(x_mid, self.x_max, y_mid, self.y_max, 
                               self.capacity, self.depth + 1, self.max_depth)
        self.sw = QuadtreeNode(self.x_min, x_mid, self.y_min, y_mid, 
                               self.capacity, self.depth + 1, self.max_depth)
        self.se = QuadtreeNode(x_mid, self.x_max, self.y_min, y_mid, 
                               self.capacity, self.depth + 1, self.max_depth)
        
        self.is_divided = True
        
        # Redistribute existing points to children
        for point in self.points:
            self._insert_to_child(point)
        
        # Clear points from this node
        self.points = []
    
    def _insert_to_child(self, point: Tuple) -> bool:
        """Insert a point into the appropriate child."""
        x, y, idx, full_data = point
        
        if self.nw.contains(x, y):
            return self.nw.insert(x, y, idx, full_data)
        elif self.ne.contains(x, y):
            return self.ne.insert(x, y, idx, full_data)
        elif self.sw.contains(x, y):
            return self.sw.insert(x, y, idx, full_data)
        elif self.se.contains(x, y):
            return self.se.insert(x, y, idx, full_data)
        
        return False
    
    def insert(self, x: float, y: float, index: int, full_data: Optional[np.ndarray] = None) -> bool:
        """
        Insert a point into the quadtree.
        
        Args:
            x: X coordinate (first dimension)
            y: Y coordinate (second dimension)
            index: Original index in dataset
            full_data: Full 6D data point (optional, for filtering other dimensions)
            
        Returns:
            True if insertion successful
        """
        if not self.contains(x, y):
            return False
        
        # If this node is not divided
        if not self.is_divided:
            # If we have room, add the point
            if len(self.points) < self.capacity:
                self.points.append((x, y, index, full_data))
                return True
            
            # If at max depth, just add it even if over capacity
            if self.depth >= self.max_depth:
                self.points.append((x, y, index, full_data))
                return True
            
            # Otherwise, subdivide and redistribute
            self.subdivide()
            
            # If subdivision failed (all points identical), just add it
            if not self.is_divided:
                self.points.append((x, y, index, full_data))
                return True
        
        # If divided, insert into appropriate child
        return self._insert_to_child((x, y, index, full_data))
    
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
        for x, y, idx, full_data in self.points:
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
    
    def __init__(self, x_dim: int = 0, y_dim: int = 1, capacity: int = 50, max_depth: int = 25):
        """
        Initialize a Quadtree.
        
        Args:
            x_dim: Index of dimension to use for x-axis
            y_dim: Index of dimension to use for y-axis
            capacity: Maximum points per node before splitting
            max_depth: Maximum tree depth to prevent infinite recursion
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.capacity = capacity
        self.max_depth = max_depth
        self.root = None
        self.size = 0
    
    def build(self, points: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build the Quadtree from a set of points.
        
        Args:
            points: Array of shape (n_points, n_dimensions) - stores full 6D but indexes on 2D
            indices: Optional array of original indices
        """
        if indices is None:
            indices = np.arange(len(points))
        
        # Extract 2D coordinates for spatial indexing (budget and revenue)
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
                                self.capacity, 0, self.max_depth)
        
        # Insert all points with full 6D data
        for i, idx in enumerate(indices):
            full_data = points[i] if points.shape[1] > 2 else None
            self.root.insert(x_coords[i], y_coords[i], idx, full_data)
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
