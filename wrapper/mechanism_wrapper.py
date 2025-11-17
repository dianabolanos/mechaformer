from .mechanism_core import *
import numpy as np
import json
import requests

class MechanismWrapper:
    """
    Wrapper for mechanism analysis and animation capabilities.
    
    Provides API for:
    - Parsing coordinate strings and mechanism types
    - Server communication for simulation data
    - BSI topology analysis
    - Mechanism classification
    - Animation and static plot generation
    - Data export and analysis
    """
    
    def __init__(self, coord_string=None, server_url="http://localhost:4001/simulation"):
        """
        Initialize the MechanismWrapper.
        
        Args:
            coord_string (str, optional): Coordinate string to parse
            server_url (str): URL for the simulation server
        """
        self.server_url = server_url
        self.coord_string = coord_string
        self.mech_type = None
        self.coord_params = None
        self.poses = None
        self.simulation_result = None
        
        # Initialize analysis components
        self.bsi_analyzer = BSITopologyAnalyzer(BSI_DICT)
        self.classifier = MechanismClassifier()
        self.drawer = MechanismDrawer()
        self.animator = MechanismAnimator()
        
        # Analysis results cache
        self._bsi_info = None
        self._mechanism_info = None
        self._core_joints = None
        
        # Parse coordinate string if provided
        if coord_string:
            self.parse_coordinate_string(coord_string)
    
    def parse_coordinate_string(self, coord_string):
        """
        Parse a coordinate string to extract mechanism type and parameters.
        
        Args:
            coord_string (str): The coordinate string to parse
            
        Returns:
            dict: Parsed information including mechanism type and coordinates
        """
        self.coord_string = coord_string
        parts = coord_string.split('_')
        parts = [part for part in parts if part != '']
        
        mech_type = None
        coord_end_index = 0
        
        # Find where coordinates end and mechanism type begins
        for i, part in enumerate(parts):
            try:
                float(part)
                coord_end_index = i + 1
            except ValueError:
                mech_type = part
                break
        
        if mech_type is None:
            coords = [float(part) for part in parts[:10]]
            mech_type = "Null"
        else:
            coords = [float(part) for part in parts[:coord_end_index]]
        
        self.mech_type = mech_type
        
        # Create coordinate pairs
        num_points = len(coords) // 2
        x = np.array([coords[i*2] for i in range(num_points)])
        y = np.array([coords[i*2 + 1] for i in range(num_points)])
        self.coord_params = np.stack((x, y), axis=1)
        
        return {
            'mechanism_type': self.mech_type,
            'coordinates': self.coord_params,
            'num_points': num_points,
            'raw_coords': coords
        }
    
    def set_parameters(self, coord_params, mech_type):
        """
        Manually set mechanism parameters.
        
        Args:
            coord_params (np.array): Coordinate parameters as (n, 2) array
            mech_type (str): Mechanism type identifier
        """
        self.coord_params = np.array(coord_params)
        self.mech_type = mech_type
        self._clear_cache()
    
    def simulate(self, speed_scale=1, steps=360, relative_tolerance=1e-12, 
                driving_element=1, start_angle=0, end_angle=360, timeout=10):
        """
        Run mechanism simulation on the server.
        
        Args:
            speed_scale (float): Speed scaling factor
            steps (int): Number of simulation steps
            relative_tolerance (float): Numerical tolerance
            driving_element (int): Which element drives the mechanism
            start_angle (float): Starting angle in degrees
            end_angle (float): Ending angle in degrees
            timeout (int): Request timeout in seconds
            
        Returns:
            dict: Simulation results including poses
        """
        if self.coord_params is None or self.mech_type is None:
            raise ValueError("No mechanism parameters set. Use parse_coordinate_string() or set_parameters() first.")
        
        headers = {"Content-Type": "application/json"}
        data = [{
            "params": self.coord_params.tolist(),
            "type": self.mech_type,
            "speedScale": speed_scale,
            "steps": steps,
            "relativeTolerance": relative_tolerance,
            "drivingElement": driving_element,
            "startAngle": start_angle,
            "endAngle": end_angle,
        }]
        
        try:
            response = requests.post(self.server_url, headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and 'poses' in result[0]:
                    self.poses = np.array(result[0]['poses'])
                    self.simulation_result = result[0]
                    self._clear_cache()
                    return self.simulation_result
                else:
                    raise ValueError("Server returned empty or invalid response")
            else:
                raise ConnectionError(f"Server returned error status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Server request timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Could not connect to simulation server")
    
    def get_bsi_info(self):
        """
        Get BSI topology information for the mechanism.
        
        Returns:
            dict: BSI topology data including matrices and analysis
        """
        if self._bsi_info is None:
            if self.mech_type is None:
                raise ValueError("No mechanism type set")
            self._bsi_info = self.bsi_analyzer.get_mechanism_info(self.mech_type)
        return self._bsi_info
    
    def get_bsi_topology(self):
        """
        Get raw BSI topology matrices.
        
        Returns:
            dict: Raw BSI matrices (B, T, S, I, c)
        """
        if self.mech_type is None:
            raise ValueError("No mechanism type set")
        return self.bsi_analyzer.get_mechanism_topology(self.mech_type)
    
    def get_bsi_connections(self):
        """
        Get BSI connection data for visualization.
        
        Returns:
            list: List of connections as (point1, point2, link_type) tuples
        """
        if self.mech_type is None:
            raise ValueError("No mechanism type set")
        return self.bsi_analyzer.extract_connections(self.mech_type)
    
    def get_mechanism_info(self):
        """
        Get comprehensive mechanism classification and information.
        
        Returns:
            dict: Complete mechanism analysis including BSI data
        """
        if self._mechanism_info is None:
            if self.poses is None:
                raise ValueError("No simulation data available. Run simulate() first.")
            joint_count = len(self.poses[0]) if len(self.poses) > 0 else 0
            self._mechanism_info = self.classifier.classify_mechanism(self.mech_type, joint_count)
        return self._mechanism_info
    
    def get_type(self):
        """
        Get mechanism type.
        
        Returns:
            str: Mechanism type identifier
        """
        return self.mech_type
    
    def get_family(self):
        """
        Get mechanism family (4-bar, 6-bar, etc.).
        
        Returns:
            str: Mechanism family
        """
        info = self.get_mechanism_info()
        return info.get('family', 'Unknown')
    
    def get_subtype(self):
        """
        Get mechanism subtype.
        
        Returns:
            str: Mechanism subtype
        """
        info = self.get_mechanism_info()
        return info.get('subtype', 'Unknown')
    
    def get_poses(self):
        """
        Get simulation pose data.
        
        Returns:
            np.array: Array of poses, shape (n_steps, n_joints, 2)
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        return self.poses
    
    def get_joint_count(self):
        """
        Get number of joints in the mechanism.
        
        Returns:
            int: Number of joints
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        return len(self.poses[0]) if len(self.poses) > 0 else 0
    
    def get_step_count(self):
        """
        Get number of simulation steps.
        
        Returns:
            int: Number of simulation steps
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        return len(self.poses)
    
    def get_core_joints(self):
        """
        Get core mechanism joints (filtered from auxiliary joints).
        
        Returns:
            list: List of core joint indices
        """
        if self._core_joints is None:
            if self.poses is None:
                raise ValueError("No simulation data available. Run simulate() first.")
            self._core_joints = self.animator._identify_core_mechanism_joints(self.poses, self.mech_type)
        return self._core_joints
    
    def get_joint_trajectories(self, joint_indices=None):
        """
        Get trajectory data for specific joints.
        
        Args:
            joint_indices (list, optional): Specific joint indices to get trajectories for
            
        Returns:
            dict: Joint trajectories as {joint_idx: [(x, y), ...]}
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        if joint_indices is None:
            joint_indices = range(len(self.poses[0]))
        
        trajectories = {}
        for joint_idx in joint_indices:
            if joint_idx < len(self.poses[0]):
                trajectory = [(pose[joint_idx][0], pose[joint_idx][1]) for pose in self.poses]
                trajectories[joint_idx] = trajectory
        
        return trajectories
    
    def get_coupler_trajectory(self):
        """
        Get coupler point trajectory if available.
        
        Returns:
            list: Coupler point trajectory as [(x, y), ...] or None
        """
        bsi_info = self.get_bsi_info()
        coupler_point = bsi_info.get('coupler_point', -1)
        
        if coupler_point >= 0 and self.poses is not None:
            core_joints = self.get_core_joints()
            if coupler_point in core_joints:
                filtered_idx = core_joints.index(coupler_point)
                # Get filtered poses with only core joints
                filtered_poses = []
                for pose in self.poses:
                    filtered_pose = [pose[i] for i in core_joints if i < len(pose)]
                    filtered_poses.append(filtered_pose)
                
                if filtered_idx < len(filtered_poses[0]):
                    return [(pose[filtered_idx][0], pose[filtered_idx][1]) for pose in filtered_poses]
        
        return None
    
    def create_animation(self, filename=None, **kwargs):
        """
        Create mechanism animation.
        
        Args:
            filename (str, optional): Output filename for animation
            **kwargs: Additional animation parameters
            
        Returns:
            str: Filename of created animation
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        # Store original method to restore later
        original_method = self.animator.animate_mechanism
        
        # Override to capture filename
        created_filename = None
        def capture_filename(*args, **inner_kwargs):
            nonlocal created_filename
            result = original_method(*args, **inner_kwargs)
            # Extract filename from print statements or create default
            if filename:
                created_filename = filename
            else:
                created_filename = f'mechanism_animation_robust_{self.mech_type}_traced.mp4'
            return result
        
        self.animator.animate_mechanism = capture_filename
        try:
            self.animator.animate_mechanism(self.poses, self.mech_type)
        finally:
            self.animator.animate_mechanism = original_method
        
        return created_filename
    
    def create_static_plots(self, filename=None):
        """
        Create static mechanism plots.
        
        Args:
            filename (str, optional): Output filename for plots
            
        Returns:
            str: Filename of created plots
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        self.animator.create_static_plots(self.poses, self.mech_type)
        
        if filename:
            return filename
        else:
            return f'mechanism_static_plots_bsi_{self.mech_type}.png'
    
    def draw_mechanism_frame(self, frame_index=0, ax=None, **kwargs):
        """
        Draw a single frame of the mechanism.
        
        Args:
            frame_index (int): Frame index to draw
            ax (matplotlib.axes, optional): Axes to draw on
            **kwargs: Additional drawing parameters
            
        Returns:
            matplotlib.axes: The axes object used for drawing
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        if frame_index >= len(self.poses):
            raise IndexError(f"Frame index {frame_index} exceeds available frames {len(self.poses)}")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get filtered poses for core joints
        core_joints = self.get_core_joints()
        filtered_pose = [self.poses[frame_index][i] for i in core_joints if i < len(self.poses[frame_index])]
        
        # Get mechanism info
        mech_info = self.get_mechanism_info()
        
        # Draw the mechanism
        self.drawer.draw_mechanism(ax, filtered_pose, mech_info, **kwargs)
        
        # Setup plot
        self.animator._setup_plot_limits(ax, [filtered_pose])
        ax.set_title(f'{mech_info["description"]} - Frame {frame_index + 1}')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def validate_bsi_data(self):
        """
        Validate BSI data consistency.
        
        Returns:
            tuple: (is_valid, issues_list)
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        return self.bsi_analyzer.validate_bsi_data(self.mech_type, len(self.poses[0]))
    
    def export_data(self, format='json'):
        """
        Export mechanism data in various formats.
        
        Args:
            format (str): Export format ('json', 'csv', 'numpy')
            
        Returns:
            dict or str: Exported data
        """
        if self.poses is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        data = {
            'mechanism_type': self.mech_type,
            'coordinate_parameters': self.coord_params.tolist(),
            'poses': self.poses.tolist(),
            'bsi_info': self.get_bsi_info(),
            'mechanism_info': self.get_mechanism_info(),
            'core_joints': self.get_core_joints(),
            'joint_trajectories': self.get_joint_trajectories()
        }
        
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'csv':
            # Export poses as CSV
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            header = ['step']
            for i in range(len(self.poses[0])):
                header.extend([f'joint_{i}_x', f'joint_{i}_y'])
            writer.writerow(header)
            
            # Data rows
            for step, pose in enumerate(self.poses):
                row = [step]
                for joint in pose:
                    row.extend([joint[0], joint[1]])
                writer.writerow(row)
            
            return output.getvalue()
        elif format == 'numpy':
            return data
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_summary(self):
        """
        Get a comprehensive summary of the mechanism.
        
        Returns:
            dict: Complete mechanism summary
        """
        summary = {
            'mechanism_type': self.mech_type,
            'coordinate_string': self.coord_string,
        }
        
        if self.coord_params is not None:
            summary['coordinate_parameters'] = self.coord_params.tolist()
            summary['num_coordinate_points'] = len(self.coord_params)
        
        if self.poses is not None:
            summary['simulation_steps'] = len(self.poses)
            summary['joint_count'] = len(self.poses[0])
            summary['core_joints'] = self.get_core_joints()
        
        try:
            bsi_info = self.get_bsi_info()
            summary['bsi_analysis'] = bsi_info
        except:
            pass
        
        try:
            mech_info = self.get_mechanism_info()
            summary['classification'] = {
                'family': mech_info.get('family'),
                'subtype': mech_info.get('subtype'),
                'description': mech_info.get('description')
            }
        except:
            pass
        
        try:
            is_valid, issues = self.validate_bsi_data()
            summary['bsi_validation'] = {
                'is_valid': is_valid,
                'issues': issues
            }
        except:
            pass
        
        return summary
    
    def _clear_cache(self):
        """Clear cached analysis results."""
        self._bsi_info = None
        self._mechanism_info = None
        self._core_joints = None
    
    def __repr__(self):
        """String representation of the wrapper."""
        if self.mech_type:
            return f"MechanismWrapper(type='{self.mech_type}', joints={self.get_joint_count() if self.poses is not None else 'N/A'})"
        else:
            return "MechanismWrapper(uninitialized)"
    
    def get_joint_roles_from_b_matrix(self):
        """
        Extract joint roles from B matrix for proper visualization labeling.
        
        Returns:
            dict: Joint roles categorized by their function in the mechanism
        """
        if self.mech_type is None:
            raise ValueError("No mechanism type set")
        
        return self.bsi_analyzer.get_joint_roles_from_b_matrix(self.mech_type)
    
    def get_link_descriptions(self):
        """
        Get descriptions of each link based on B matrix.
        
        Returns:
            dict: Link descriptions with joint connectivity
        """
        if self.mech_type is None:
            raise ValueError("No mechanism type set")
            
        return self.bsi_analyzer.get_link_descriptions(self.mech_type)
    
    def print_mechanism_structure(self):
        """
        Print a detailed breakdown of the mechanism structure from B matrix.
        """
        if self.mech_type is None:
            raise ValueError("No mechanism type set")
        
        self.bsi_analyzer.print_mechanism_structure(self.mech_type)


# Convenience function to create wrapper from coordinate string
def create_mechanism_wrapper(coord_string, server_url="http://localhost:4001/simulation"):
    """
    Convenience function to create and initialize a MechanismWrapper.
    
    Args:
        coord_string (str): Coordinate string to parse
        server_url (str): URL for the simulation server
        
    Returns:
        MechanismWrapper: Initialized wrapper instance
    """
    return MechanismWrapper(coord_string, server_url)
