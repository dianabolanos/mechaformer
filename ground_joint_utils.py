import numpy as np
import json
import os

class GroundJointNormalizer:
    def __init__(self, bsi_dict_path='BSIdict_468.json'):
        """Initialize with BSI dictionary for ground joint identification"""
        self.bsi_dict_path = bsi_dict_path
        self.bsi_dict = self._load_bsi_dict()
    
    def _load_bsi_dict(self):
        """Load BSI dictionary from file"""
        if os.path.exists(self.bsi_dict_path):
            with open(self.bsi_dict_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"BSI dictionary not found at {self.bsi_dict_path}")
    
    def get_ground_joint_indices(self, mechanism_type):
        """Get ground joint indices from BSI matrix B first row"""
        if mechanism_type not in self.bsi_dict:
            raise ValueError(f"Unknown mechanism type: {mechanism_type}")
        
        b_matrix = self.bsi_dict[mechanism_type]['B']
        first_row = b_matrix[0]
        
        # Find indices where value is 1 (ground joints)
        ground_indices = [i for i, val in enumerate(first_row) if val == 1]
        
        return ground_indices
    
    def compute_normalization_transform(self, coords, ground_indices):
        """
        Compute transformation matrix that maps:
        - First ground joint to (0, 0)
        - Second ground joint to (1, 0)
        - The mechanism is scaled to have a ground link length of 1
        
        Returns: transform_matrix
        """
        if len(ground_indices) < 2:
            raise ValueError(f"Need at least 2 ground joints, got {len(ground_indices)}")
        
        # Convert coords to array of points
        num_points = len(coords) // 2
        points = np.array(coords).reshape(num_points, 2)
        
        # Get ground joint positions
        p1 = points[ground_indices[0]]  # First ground joint
        p2 = points[ground_indices[1]]  # Second ground joint
        
        # Step 1: Translation to move first ground joint to origin
        translation = -p1
        
        # Step 2: After translation, second ground joint is at p2 - p1
        p2_translated = p2 - p1
        
        # Calculate rotation to align second ground joint with positive x-axis
        distance = np.linalg.norm(p2_translated)
        if distance < 1e-10:
            raise ValueError("Ground joints are too close together")
        
        # Rotation angle to align with x-axis
        angle = np.arctan2(p2_translated[1], p2_translated[0])
        cos_angle = np.cos(-angle)
        sin_angle = np.sin(-angle)
        
        # Step 3: Scale factor to make ground link length = 1
        scale = 1.0 / distance
        
        # Combined transformation matrix: scale * rotation * translation
        # The transformation is applied as: p' = scale * R * (p + translation)
        # In matrix form: [x'] = [scale*R | scale*R*translation] [x]
        #                 [y']                                    [y]
        #                                                         [1]
        
        # Build the 2x3 affine transformation matrix
        # Standard rotation matrix is [[cos, -sin], [sin, cos]]
        transform_matrix = np.array([
            [scale * cos_angle, scale * -sin_angle, scale * (cos_angle * translation[0] - sin_angle * translation[1])],
            [scale * sin_angle, scale * cos_angle, scale * (sin_angle * translation[0] + cos_angle * translation[1])]
        ])
        
        return transform_matrix
    
    def apply_transform(self, points, transform_matrix):
        """Apply affine transformation to points"""
        if len(points) == 0:
            return points
        
        # Ensure points is 2D array
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        
        # Convert to homogeneous coordinates
        ones = np.ones((len(points), 1))
        homogeneous_points = np.hstack([points, ones])
        
        # Apply transformation
        transformed_points = (transform_matrix @ homogeneous_points.T).T
        
        return transformed_points
    
    def normalize_mechanism(self, coords, mechanism_type, curve_points=None):
        """
        Normalize mechanism by placing ground joints at (0,0) and (1,0)
        
        Args:
            coords: List of coordinates [x1, y1, x2, y2, ...]
            mechanism_type: Type of mechanism (e.g., 'RRRR')
            curve_points: Optional curve points to transform
            
        Returns: {
            'normalized_coords': Normalized coordinates (ground joints removed)
            'transform_matrix': Transformation matrix used
            'ground_indices': Indices of ground joints
            'normalized_curve': Transformed curve points (if provided)
            'is_ternary': Whether this has ternary ground joints
        }
        """
        # Get ground joint indices
        ground_indices = self.get_ground_joint_indices(mechanism_type)
        
        # Compute transformation
        transform_matrix = self.compute_normalization_transform(coords, ground_indices)
        
        # Apply transformation to all points
        num_points = len(coords) // 2
        points = np.array(coords).reshape(num_points, 2)
        transformed_points = self.apply_transform(points, transform_matrix)
        
        # Determine if we have ternary ground joints
        is_ternary = len(ground_indices) >= 3
        
        # Create list of indices to keep (exclude ground joints)
        if is_ternary:
            # For ternary joints, keep the third ground joint
            indices_to_remove = ground_indices[:2]
            indices_to_keep = [i for i in range(num_points) if i not in indices_to_remove]
        else:
            # For binary joints, remove all ground joints
            indices_to_remove = ground_indices
            indices_to_keep = [i for i in range(num_points) if i not in indices_to_remove]
        
        # Extract non-ground joint coordinates
        normalized_points = transformed_points[indices_to_keep]
        normalized_coords = normalized_points.flatten().tolist()
        
        # Transform curve points if provided
        normalized_curve = None
        if curve_points is not None:
            normalized_curve = self.apply_transform(curve_points, transform_matrix)
        
        return {
            'normalized_coords': normalized_coords,
            'transform_matrix': transform_matrix.tolist(),
            'ground_indices': ground_indices,
            'normalized_curve': normalized_curve,
            'is_ternary': is_ternary,
            'removed_indices': indices_to_remove
        }
    
    def denormalize_mechanism(self, normalized_coords, mechanism_type, transform_matrix, removed_indices=None):
        """
        Inverse operation: add ground joints back and apply inverse transformation
        
        Args:
            normalized_coords: Normalized coordinates (without ground joints)
            mechanism_type: Type of mechanism
            transform_matrix: Original transformation matrix
            removed_indices: Indices where ground joints were removed
            
        Returns: Original coordinates with ground joints
        """
        transform_matrix = np.array(transform_matrix)
        ground_indices = self.get_ground_joint_indices(mechanism_type)
        
        # Determine which indices were removed
        if removed_indices is None:
            is_ternary = len(ground_indices) >= 3
            if is_ternary:
                removed_indices = ground_indices[:2]
            else:
                removed_indices = ground_indices
        
        # Convert normalized coords to points array
        normalized_points = np.array(normalized_coords).reshape(-1, 2)
        
        # Calculate total number of points after adding back ground joints
        total_points = len(normalized_points) + len(removed_indices)
        
        # Create array to hold all points in their correct positions
        full_points = np.zeros((total_points, 2))
        
        # First, place the ground joints at their fixed normalized positions
        ground_joint_map = {}
        for i, idx in enumerate(sorted(removed_indices)):
            if idx == removed_indices[0]:  # First ground joint
                full_points[idx] = [0.0, 0.0]
                ground_joint_map[idx] = True
            elif idx == removed_indices[1]:  # Second ground joint  
                full_points[idx] = [1.0, 0.0]
                ground_joint_map[idx] = True
        
        # Now place all other points in their correct positions
        norm_idx = 0
        for i in range(total_points):
            if i not in ground_joint_map:
                full_points[i] = normalized_points[norm_idx]
                norm_idx += 1
        
        # Apply inverse transformation
        # For affine transform [A | b], inverse is [A^-1 | -A^-1 * b]
        A = transform_matrix[:, :2]
        b = transform_matrix[:, 2]
        
        A_inv = np.linalg.inv(A)
        b_inv = -A_inv @ b
        
        inverse_transform = np.zeros((2, 3))
        inverse_transform[:, :2] = A_inv
        inverse_transform[:, 2] = b_inv
        
        # Apply inverse transformation
        original_points = self.apply_transform(full_points, inverse_transform)
        
        return original_points.flatten().tolist()