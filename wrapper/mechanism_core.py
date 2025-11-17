import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os

# Load BSIdict topology information
def load_bsi_dict(filename="BSIdict_468.json"):
    """Load BSI topology dictionary"""
    try:
        # Make path relative to this module's location, not current working directory
        if not os.path.isabs(filename):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(module_dir, filename)
        
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found, using default topologies")
        return {}

# Global BSI dictionary
BSI_DICT = load_bsi_dict()

class BSITopologyAnalyzer:
    """Analyze BSI topology matrices with variable sizes"""
    
    def __init__(self, bsi_dict):
        self.bsi_dict = bsi_dict
    
    def get_mechanism_topology(self, mech_type):
        """Get topology information from BSI dictionary
        
        BSI Matrix Structure (based on paper definition):
        - B matrix: Each row represents a link, each column represents a point of interest (joint or coupler point)
        - S matrix: Defines slot constraints for prismatic joints (empty for pure revolute mechanisms)
        - I matrix: Defines actuators with point indices and type (0=rotary, 1=linear)
        - First link (row 0) is always the ground link with fixed joints
        """
        if mech_type not in self.bsi_dict:
            print(f"Warning: {mech_type} not found in BSI dictionary")
            return None
            
        mech_data = self.bsi_dict[mech_type]
        topology = {
            'B_matrix': mech_data.get('B', []),    # Binary links: rows=links, cols=points
            'T_matrix': mech_data.get('T', []),    # Ternary links: rows=links, cols=points
            'S_matrix': mech_data.get('S', []),    # Slot constraints (prismatic joints)
            'I_matrix': mech_data.get('I', []),    # Actuator matrix 
            'c_vector': mech_data.get('c', [])     # Coupler point vector
        }
        
        # # Print matrix dimensions for debugging
        # if topology['B_matrix']:
        #     print(f"BSI B matrix: {len(topology['B_matrix'])} links × {len(topology['B_matrix'][0])} points")
        # if topology['T_matrix']:
        #     print(f"BSI T matrix: {len(topology['T_matrix'])} links × {len(topology['T_matrix'][0])} points")
        # if topology['S_matrix']:
        #     print(f"BSI S matrix: {len(topology['S_matrix'])} prismatic joint constraints")
        # if topology['c_vector']:
        #     print(f"BSI c vector: {len(topology['c_vector'])} points of interest")
            
        return topology
    
    def extract_connections(self, mech_type):
        """Extract all connections (rigid links and prismatic joints) from BSI matrices
        
        Connection Extraction Logic:
        - B matrix: Each row (link) connects all points marked with 1 in that row
        - S matrix: Each row defines a prismatic joint constraint [fixed_point1, sliding_point, fixed_point2]
        - Ground link is always the first row (link 0) in B matrix
        """
        topology = self.get_mechanism_topology(mech_type)
        if not topology:
            return []
        
        connections = []
        
        # Extract rigid link connections from B matrix (binary and ternary links)
        # Each row represents a link, each column represents a point of interest
        if topology['B_matrix']:
            for link_idx, link_row in enumerate(topology['B_matrix']):
                # Find all points belonging to this link (marked with 1)
                link_points = [i for i, belongs in enumerate(link_row) if belongs == 1]
                
                if len(link_points) >= 2:
                    # Determine link type - first link (row 0) is always ground
                    if link_idx == 0:
                        link_type = 'ground'
                        #print(f"  Ground link (link {link_idx}): points {link_points}")
                    elif len(link_points) == 2:
                        link_type = 'binary'
                        #print(f"  Binary link (link {link_idx}): points {link_points}")
                    else:  # len(link_points) >= 3
                        link_type = 'ternary'
                        #print(f"  Ternary link (link {link_idx}): points {link_points}")
                    
                    # Create connections between all pairs of points in this link
                    for i in range(len(link_points)):
                        for j in range(i + 1, len(link_points)):
                            point1, point2 = link_points[i], link_points[j]
                            connections.append((point1, point2, link_type))
        
        # Extract rigid link connections from T matrix (ternary links)
        if topology['T_matrix']:
            for link_idx, link_row in enumerate(topology['T_matrix']):
                # Find all points belonging to this ternary link
                link_points = [i for i, belongs in enumerate(link_row) if belongs == 1]
                
                if len(link_points) >= 3:
                    #print(f"  Ternary link (T{link_idx}): points {link_points}")
                    # Create connections between all pairs of points in this ternary link
                    for i in range(len(link_points)):
                        for j in range(i + 1, len(link_points)):
                            point1, point2 = link_points[i], link_points[j]
                            connections.append((point1, point2, 'ternary'))
        
        # Extract prismatic joint connections from S matrix (slot constraints)
        if topology['S_matrix']:
            for slot_idx, slot_constraint in enumerate(topology['S_matrix']):
                if len(slot_constraint) >= 3:
                    # S matrix format: [fixed_point1, sliding_point, fixed_point2] 
                    # where sliding_point slides along line between fixed_point1 and fixed_point2
                    fixed_point1, sliding_point, fixed_point2 = slot_constraint[:3]
                    
                    # Validate indices are reasonable (basic sanity check)
                    max_expected_points = 20  # Generous upper bound for any mechanism
                    if all(0 <= point < max_expected_points for point in [fixed_point1, sliding_point, fixed_point2]):
                        # Add prismatic joint connections
                        # The sliding point connects to both fixed points via prismatic joints
                        connections.append((fixed_point1, sliding_point, 'prismatic'))
                        connections.append((sliding_point, fixed_point2, 'prismatic'))
                        
                        #print(f"  Prismatic constraint {slot_idx}: point {sliding_point} slides between points {fixed_point1} and {fixed_point2}")
                    #else:
                        #print(f"  Warning: Invalid prismatic constraint indices {slot_constraint[:3]} - skipping")
        
        #print(f"Total connections extracted for {mech_type}: {len(connections)}")
        return connections
    
    
    def get_mechanism_info(self, mech_type):
        """Get comprehensive mechanism information from BSI data"""
        topology = self.get_mechanism_topology(mech_type)
        if not topology:
            return {}
        
        # Get comprehensive joint roles analysis (eliminates redundant B matrix parsing)
        joint_roles = self.get_joint_roles_from_b_matrix(mech_type)
        
        # Count binary and ternary links from B matrix
        num_binary_links_from_B = 0
        num_ternary_links_from_B = 0
        
        if topology['B_matrix']:
            for link_idx, link_row in enumerate(topology['B_matrix']):
                link_points = [i for i, belongs in enumerate(link_row) if belongs == 1]
                if len(link_points) >= 2:
                    if len(link_points) == 2:
                        num_binary_links_from_B += 1
                    elif len(link_points) >= 3:
                        num_ternary_links_from_B += 1
        
        # Count ternary links from T matrix
        num_ternary_links_from_t = len(topology['T_matrix']) if topology['T_matrix'] else 0
            
        info = {
            'num_binary_links': num_binary_links_from_B,
            'num_ternary_links': num_ternary_links_from_B + num_ternary_links_from_t,
            'num_prismatic_joints': len(topology['S_matrix']) if topology['S_matrix'] else 0,
            'num_points': len(topology['c_vector']) if topology['c_vector'] else 0,
            'ground_joints': joint_roles.get('ground_joints', []),  # Use joint_roles result
            'coupler_point': joint_roles.get('coupler_point', -1),  # Use joint_roles result
            'actuator_info': None,
            'prismatic_constraints': topology['S_matrix'] if topology['S_matrix'] else []
        }
        
        # Get actuator information
        if topology['I_matrix'] and len(topology['I_matrix']) > 0:
            actuator = topology['I_matrix'][0]
            if len(actuator) >= 4:
                info['actuator_info'] = {
                    'points': actuator[:3],
                    'type': 'rotary' if actuator[3] == 0 else 'linear'
                }
        
        #print(f"Mechanism info for {mech_type}: {info}")
        return info

    def validate_bsi_data(self, mech_type, pose_count):
        """Validate BSI data consistency against actual pose data to prevent runtime errors"""
        topology = self.get_mechanism_topology(mech_type)
        if not topology:
            return False, "No BSI topology data found"
        
        issues = []
        
        # Check B matrix consistency
        if topology['B_matrix']:
            max_b_point = 0
            for row_idx, row in enumerate(topology['B_matrix']):
                for col_idx, value in enumerate(row):
                    if value == 1:
                        max_b_point = max(max_b_point, col_idx)
            
            if max_b_point >= pose_count:
                issues.append(f"B matrix references point {max_b_point}, but only {pose_count} points available")
        
        # Check T matrix consistency
        if topology['T_matrix']:
            max_t_point = 0
            for row_idx, row in enumerate(topology['T_matrix']):
                for col_idx, value in enumerate(row):
                    if value == 1:
                        max_t_point = max(max_t_point, col_idx)
            
            if max_t_point >= pose_count:
                issues.append(f"T matrix references point {max_t_point}, but only {pose_count} points available")
        
        # Check S matrix (prismatic constraints) consistency
        if topology['S_matrix']:
            for constraint_idx, constraint in enumerate(topology['S_matrix']):
                if len(constraint) >= 3:
                    max_s_point = max(constraint[:3])
                    if max_s_point >= pose_count:
                        issues.append(f"S matrix constraint {constraint_idx} references point {max_s_point}, but only {pose_count} points available")
        
        # Check c vector consistency
        if topology['c_vector']:
            for point_idx, is_coupler in enumerate(topology['c_vector']):
                if is_coupler == 1 and point_idx >= pose_count:
                    issues.append(f"Coupler point {point_idx} exceeds available points {pose_count}")
        
        is_valid = len(issues) == 0
        return is_valid, issues

    def get_joint_roles_from_b_matrix(self, mech_type):
        """
        Extract joint roles from B matrix for proper visualization labeling.
        
        Args:
            mech_type (str): Mechanism type identifier
            
        Returns:
            dict: Joint roles categorized by their function in the mechanism
        """
        # Get BSI topology
        topology = self.get_mechanism_topology(mech_type)
        if not topology or not topology['B_matrix']:
            return {}
        
        b_matrix = topology['B_matrix']
        c_vector = topology['c_vector']
        
        # Initialize role categories
        joint_roles = {
            'ground_links': {},      # Link index -> [joint1, joint2, ...]
            'moving_links': {},      # Link index -> [joint1, joint2, ...]
            'ground_joints': [],     # Joints that are part of ground links
            'moving_joints': [],     # Joints that are not ground or coupler
            'coupler_point': -1,     # The coupler point
            'joint_connectivity': {} # Joint index -> [link1, link2, ...]
        }
        
        # Analyze each link (row) in B matrix
        for link_idx, link_row in enumerate(b_matrix):
            # Find all joints (points) belonging to this link
            link_joints = [j for j, belongs in enumerate(link_row) if belongs == 1]
            
            if len(link_joints) >= 2:
                if link_idx == 0:  # First link is always ground
                    joint_roles['ground_links'][link_idx] = link_joints
                    joint_roles['ground_joints'].extend(link_joints)
                else:
                    joint_roles['moving_links'][link_idx] = link_joints
        
        # Remove duplicates from ground joints
        joint_roles['ground_joints'] = list(set(joint_roles['ground_joints']))
        
        # Analyze joint connectivity (which links each joint belongs to)
        num_joints = len(b_matrix[0])
        for joint_idx in range(num_joints):
            connected_links = []
            for link_idx, link_row in enumerate(b_matrix):
                if link_row[joint_idx] == 1:
                    connected_links.append(link_idx)
            joint_roles['joint_connectivity'][joint_idx] = connected_links
        
        # Identify coupler point from c_vector
        if c_vector:
            for i, is_coupler in enumerate(c_vector):
                if is_coupler == 1:
                    joint_roles['coupler_point'] = i
                    break
        
        # Categorize moving joints (not ground, not coupler)
        for joint_idx in range(num_joints):
            if (joint_idx not in joint_roles['ground_joints'] and 
                joint_idx != joint_roles['coupler_point']):
                joint_roles['moving_joints'].append(joint_idx)
        
        return joint_roles
    
    def get_link_descriptions(self, mech_type):
        """
        Get human-readable descriptions of each link based on B matrix.
        
        Args:
            mech_type (str): Mechanism type identifier
            
        Returns:
            dict: Link descriptions with joint connectivity
        """
        joint_roles = self.get_joint_roles_from_b_matrix(mech_type)
        descriptions = {}
        
        # Ground links
        for link_idx, joints in joint_roles['ground_links'].items():
            descriptions[f'Link {link_idx}'] = {
                'type': 'Ground Link',
                'joints': joints,
                'description': f"Ground link connecting joints {joints}"
            }
        
        # Moving links
        for link_idx, joints in joint_roles['moving_links'].items():
            link_type = 'Binary' if len(joints) == 2 else 'Ternary' if len(joints) == 3 else f'{len(joints)}-way'
            descriptions[f'Link {link_idx}'] = {
                'type': f'{link_type} Link',
                'joints': joints,
                'description': f"{link_type} moving link connecting joints {joints}"
            }
        
        return descriptions
    
    def print_mechanism_structure(self, mech_type):
        """
        Print a detailed breakdown of the mechanism structure from B matrix.
        
        Args:
            mech_type (str): Mechanism type identifier
        """
        print(f"=== MECHANISM STRUCTURE FOR {mech_type} ===\n")
        
        joint_roles = self.get_joint_roles_from_b_matrix(mech_type)
        link_descriptions = self.get_link_descriptions(mech_type)
        
        print("LINKS:")
        for link_name, info in link_descriptions.items():
            print(f"  {link_name}: {info['description']}")
        
        print(f"\nJOINT ROLES:")
        print(f"  Ground joints: {joint_roles['ground_joints']}")
        print(f"  Moving joints: {joint_roles['moving_joints']}")
        print(f"  Coupler point: {joint_roles['coupler_point']}")
        
        print(f"\nJOINT CONNECTIVITY:")
        for joint_idx, links in joint_roles['joint_connectivity'].items():
            joint_type = "Ground" if joint_idx in joint_roles['ground_joints'] else "Moving"
            if joint_idx == joint_roles['coupler_point']:
                joint_type = "Coupler"
            print(f"  Joint {joint_idx} ({joint_type}): Connected to links {links}")

class MechanismClassifier:
    def __init__(self):
        self.bsi_analyzer = BSITopologyAnalyzer(BSI_DICT)
    
    def classify_mechanism(self, mech_type, joint_count):
        """Classify mechanism using BSI topology for 4-bar and 6-bar mechanisms only"""
        #print(f"Classifying mechanism: type='{mech_type}', joint_count={joint_count}")
        
        # Get BSI information
        bsi_info = self.bsi_analyzer.get_mechanism_info(mech_type)
        bsi_connections = self.bsi_analyzer.extract_connections(mech_type)
        
        # Determine mechanism family from BSI data (4-bar or 6-bar only)
        if bsi_info:
            total_links = bsi_info['num_binary_links'] + bsi_info['num_ternary_links']
            num_points = bsi_info['num_points']
            num_prismatic = bsi_info['num_prismatic_joints']
            
            #print(f"BSI analysis: {total_links} total links, {num_points} points, {num_prismatic} prismatic joints")
            
            if total_links <= 4:
                family = '4-bar'
            elif total_links <= 6:
                family = '6-bar'
            else:
                print(f"Warning: {total_links} links detected - only 4-bar and 6-bar mechanisms supported")
                family = '6-bar'  # Default to 6-bar for unsupported mechanisms
        else:
            # Fallback classification for 4-bar and 6-bar only
            if 'Watt' in mech_type or 'Steph' in mech_type:
                family = '6-bar'
            elif joint_count <= 5:
                family = '4-bar'
            else:
                family = '6-bar'  # Default to 6-bar for larger mechanisms
        
        # Determine subtype based on mechanism characteristics
        if 'Watt' in mech_type:
            subtype = 'Watt'
        elif 'Steph' in mech_type:
            subtype = 'Stephenson'
        elif bsi_info and bsi_info.get('num_prismatic_joints', 0) > 0:
            # Generic prismatic mechanism classification
            subtype = f"Prismatic-{family}"
        elif 'RRRR' in mech_type:
            subtype = 'RRRR'
        else:
            subtype = mech_type
            
        return {
            'family': family,
            'subtype': subtype,
            'joint_count': joint_count,
            'bsi_info': bsi_info,
            'bsi_connections': bsi_connections,
            'description': f'{mech_type} {family} linkage'
        }

class MechanismDrawer:
    def __init__(self):
        # Modern, clean color palette matching the reference image
        self.colors = {
            'binary': '#A0A0A0',      # Medium-dark gray for links
            'ternary': '#A0A0A0',     # Medium-dark gray for ternary edges
            'ground': '#000000',      # Black for ground
            'coupler': '#0009b2',     # Blue for emphasis
            'connector': '#A0A0A0',   # Medium-dark gray
            'prismatic': '#0009b2'    # Blue for prismatic
        }
        
        # Fill colors for ternary links - matching the reference image
        self.fill_colors = {
            'ternary': '#9b9a98',     # Exact ternary fill color
            'ground': '#9b9a98',      # Ground fill color
            'binary': '#FFFFFF',      # White
            'prismatic': '#E8F5E9'    # Light mint green for prismatic
        }
        
        # More subtle joint styling
        self.joint_style = {
            'size': 5,               # Smaller joints
            'edge_width': 0.5,       # Thinner edges
            'edge_color': '#808080'  # Medium gray edges
        }
        
        # Clean, modern link styling
        self.link_style = {
            'width_scale': 1.2,      # Slightly thicker lines
            'alpha': 0.8,            # Slight transparency
            'cap_style': 'round',    # Rounded line caps
            'join_style': 'round',   # Rounded line joins
            'prismatic_width': 4     # Width of prismatic slots
        }
    
    def set_aesthetic_style(self, style='pastel'):
        """Set the aesthetic style for mechanism visualization
        
        Args:
            style (str): Style preset ('pastel', 'vibrant', 'minimal', 'dark')
        """
        if style == 'pastel':
            # Already set as default
            pass
        elif style == 'vibrant':
            self.colors.update({
                'binary': '#FF6B6B',      # Coral red
                'ternary': '#4ECDC4',     # Turquoise
                'ground': '#45B7D1',      # Blue
                'coupler': '#96CEB4',     # Mint green
                'connector': '#FFEAA7',   # Light yellow
                'prismatic': '#DDA0DD'    # Plum
            })
        elif style == 'minimal':
            self.colors.update({
                'binary': '#2C3E50',      # Dark blue-gray
                'ternary': '#34495E',     # Darker blue-gray
                'ground': '#7F8C8D',      # Medium gray
                'coupler': '#E74C3C',     # Red accent
                'connector': '#95A5A6',   # Light gray
                'prismatic': '#F39C12'    # Orange
            })
        elif style == 'dark':
            self.colors.update({
                'binary': '#BB86FC',      # Light purple
                'ternary': '#03DAC6',     # Teal
                'ground': '#CF6679',      # Pink
                'coupler': '#FFB74D',     # Orange
                'connector': '#90A4AE',   # Blue gray
                'prismatic': '#81C784'    # Light green
            })
    
    def draw_mechanism(self, ax, pose, mech_info, alpha=1.0, show_labels=True):
        """Draw mechanism using BSI connection data"""
        if 'bsi_connections' in mech_info:
            # Fill ternary links first (lowest z-order)
            self._fill_ternary_links(ax, pose, mech_info['bsi_connections'], alpha)
            
            # Draw prismatic slots next
            self._draw_prismatic_slots(ax, pose, mech_info, alpha)
            
            # Draw regular connections on top
            return self._draw_from_bsi(ax, pose, mech_info, alpha, show_labels)

    def _compute_prismatic_paths(self, poses, mech_type):
        """Compute the motion paths of prismatic joints"""
        bsi_info = self.classifier.bsi_analyzer.get_mechanism_info(mech_type)
        prismatic_constraints = bsi_info.get('prismatic_constraints', [])
        paths = {}
        
        for constraint in prismatic_constraints:
            if len(constraint) >= 3:
                sliding_point = constraint[1]
                if sliding_point < len(poses[0]):
                    # Collect all positions of the sliding point
                    path = []
                    for pose in poses:
                        path.append((pose[sliding_point][0], pose[sliding_point][1]))
                    paths[sliding_point] = path
        
        return paths

    def _draw_prismatic_slots(self, ax, pose, mech_info, alpha):
        """Draw prismatic joints with slots only along motion paths"""
        bsi_info = mech_info.get('bsi_info', {})
        prismatic_constraints = bsi_info.get('prismatic_constraints', [])
        
        for constraint in prismatic_constraints:
            if len(constraint) >= 3:
                fixed_point1, sliding_point, fixed_point2 = constraint[:3]
                
                if all(0 <= p < len(pose) for p in [fixed_point1, sliding_point, fixed_point2]):
                    try:
                        p1 = pose[fixed_point1]
                        sliding_p = pose[sliding_point]
                        p2 = pose[fixed_point2]
                        
                        # Draw the actual motion path of the sliding joint
                        if hasattr(self, 'prismatic_paths') and sliding_point in self.prismatic_paths:
                            path = self.prismatic_paths[sliding_point]
                            path_x = [p[0] for p in path]
                            path_y = [p[1] for p in path]
                            
                            # Draw the slot along the motion path
                            ax.plot(path_x, path_y,
                                   color=self.colors['prismatic'],
                                   linewidth=self.link_style['prismatic_width'],
                                   alpha=alpha * 0.3,  # Very transparent
                                   solid_capstyle='round',
                                   zorder=2)
                        
                        # Draw connecting lines (structure of the mechanism)
                        ax.plot([p1[0], sliding_p[0]], [p1[1], sliding_p[1]],
                               color=self.colors['binary'],  # Use regular link color
                               linewidth=4,
                               alpha=alpha * 0.8,
                               solid_capstyle='round',
                               solid_joinstyle='round',
                               zorder=3)
                        
                        ax.plot([sliding_p[0], p2[0]], [sliding_p[1], p2[1]],
                               color=self.colors['binary'],  # Use regular link color
                               linewidth=4,
                               alpha=alpha * 0.8,
                               solid_capstyle='round',
                               solid_joinstyle='round',
                               zorder=3)
                        
                        # Draw the sliding point
                        ax.plot(sliding_p[0], sliding_p[1], 'o',
                               color='#D3D3D3',  # Light gray
                               markeredgecolor=self.colors['prismatic'],
                               markeredgewidth=0.5,
                               markersize=5,
                               zorder=4)
                            
                    except (IndexError, TypeError) as e:
                        print(f"Error drawing prismatic slot: {e}")

    def _draw_from_bsi(self, ax, pose, mech_info, alpha, show_labels):
        """Draw using BSI connection data with proper prismatic joint handling"""
        connections = mech_info['bsi_connections']
        links = []
        
        # Draw regular connections (not prismatic)
        for connection in connections:
            if len(connection) >= 3:
                point1, point2, link_type = connection[:3]
                
                # Skip prismatic connections as they're handled by _draw_prismatic_slots
                if link_type != 'prismatic':
                    if point1 < len(pose) and point2 < len(pose) and point1 >= 0 and point2 >= 0:
                        color = self.colors.get(link_type, self.colors['connector'])
                        linewidth = 4 * self.link_style['width_scale']  # Consistent width for all links
                        
                        try:
                            link = ax.plot([pose[point1][0], pose[point2][0]], 
                                         [pose[point1][1], pose[point2][1]],
                                         color=color, 
                                         alpha=alpha * self.link_style['alpha'], 
                                         linewidth=linewidth,
                                         solid_capstyle=self.link_style['cap_style'],
                                         solid_joinstyle=self.link_style['join_style'],
                                         zorder=3)[0]
                            links.append(link)
                        except (IndexError, TypeError) as e:
                            print(f"Error drawing connection: {e}")
        
        return links

    def _fill_ternary_links(self, ax, pose, connections, alpha):
        """Fill ternary links as polygons using existing connection data"""
        # Group connections by link type and collect points for each ternary link
        ternary_links = {}
        
        for connection in connections:
            if len(connection) >= 3:
                point1, point2, link_type = connection[:3]
                
                # Only process ternary links (not ground links) to match reference image
                if link_type == 'ternary':
                    if link_type not in ternary_links:
                        ternary_links[link_type] = set()
                    ternary_links[link_type].add(point1)
                    ternary_links[link_type].add(point2)
        
        # Fill each ternary link as a polygon
        for link_type, points in ternary_links.items():
            points_list = list(points)
            
            # Only fill if we have exactly 3 points for clean triangles
            if len(points_list) == 3 and all(0 <= p < len(pose) for p in points_list):
                try:
                    # Get coordinates
                    polygon_x = [pose[p][0] for p in points_list]
                    polygon_y = [pose[p][1] for p in points_list]
                    
                    # Fill with the specified color
                    fill_color = '#9b9a98'  # Exact color specified
                    fill_alpha = 0.5  # More transparent fill to match reference
                    
                    # Fill the polygon with soft edges
                    ax.fill(polygon_x, polygon_y, 
                           color=fill_color, 
                           alpha=fill_alpha, 
                           edgecolor='none',
                           zorder=1)
                    
                except (IndexError, TypeError) as e:
                    print(f"Error filling ternary link polygon: {e}")

class MechanismAnimator:
    def __init__(self):
        self.classifier = MechanismClassifier()
        self.drawer = MechanismDrawer()
    
    def animate_mechanism(self, poses, mech_type, target_curve=None):
        """Create animation for 4-bar or 6-bar mechanisms using BSI topology with robust error handling"""
        if len(poses) == 0:
            return
            
        # Store target curve for animation
        self.target_curve = target_curve
        
        # Pre-compute prismatic joint paths
        self.prismatic_paths = self._compute_prismatic_paths(poses, mech_type)
            
        # Create figure with clean white background
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Validate BSI data consistency before proceeding
        is_valid, issues = self.classifier.bsi_analyzer.validate_bsi_data(mech_type, len(poses[0]))
        if not is_valid:
            print("BSI validation issues detected:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Identify core mechanism joints (filter out extreme auxiliary joints)
        core_joints = self._identify_core_mechanism_joints(poses, mech_type)
        
        # Filter poses to include only core joints
        filtered_poses = []
        for pose in poses:
            filtered_pose = [pose[i] for i in core_joints if i < len(pose)]
            filtered_poses.append(filtered_pose)
        
        if not filtered_poses or not filtered_poses[0]:
            print("No valid core joints found for animation")
            return
            
       #print(f"Filtered to {len(filtered_poses[0])} core joints: {core_joints}")
        
        mech_info = self.classifier.classify_mechanism(mech_type, len(filtered_poses[0]))
        
        # Pre-compute coupler point trajectory for tracing with validation
        coupler_trajectory = []
        bsi_info = mech_info.get('bsi_info', {})
        coupler_point_idx = bsi_info.get('coupler_point', -1)
        
        if coupler_point_idx >= 0:
            # Find the corresponding index in the filtered poses
            filtered_coupler_idx = -1
            if coupler_point_idx < len(core_joints):
                try:
                    filtered_coupler_idx = core_joints.index(coupler_point_idx) if coupler_point_idx in core_joints else -1
                except ValueError:
                    filtered_coupler_idx = -1
            
            if filtered_coupler_idx >= 0 and filtered_coupler_idx < len(filtered_poses[0]):
                try:
                    for pose in filtered_poses:
                        coupler_trajectory.append([pose[filtered_coupler_idx][0], pose[filtered_coupler_idx][1]])
                    #print(f"Coupler point tracing enabled for point {coupler_point_idx} (filtered index {filtered_coupler_idx})")
                except (IndexError, TypeError) as e:
                    print(f"Error computing coupler trajectory: {e}")
                    coupler_trajectory = []
            else:
                print(f"Coupler point {coupler_point_idx} not found in filtered joints {core_joints}")
        else:
            print("No coupler point defined for tracing")
        
        # Create animation
        def animate(frame):
            ax.clear()
            
            # Set clean white background
            ax.set_facecolor('white')
            
            # Draw mechanism at current frame
            try:
                pose = filtered_poses[frame]
                links = self.drawer.draw_mechanism(ax, pose, mech_info, alpha=1.0)
                
                # Minimal joint drawing - only essential points
                for i, joint in enumerate(pose):
                    if i in bsi_info.get('ground_joints', []):
                        # Ground joints as small gray dots
                        ax.plot(joint[0], joint[1], 'o',
                               color='#A8A8A8',
                               markersize=6,
                               alpha=0.8)
                
                # Draw coupler trajectory with subtle styling
                if coupler_trajectory and len(coupler_trajectory) > frame:
                    if frame > 0:
                        current_traj_x = [point[0] for point in coupler_trajectory[:frame+1]]
                        current_traj_y = [point[1] for point in coupler_trajectory[:frame+1]]
                        ax.plot(current_traj_x, current_traj_y,
                               color='#0009b2',
                               alpha=0.4,
                               linewidth=1,
                               linestyle='-',
                               zorder=1)
                
                # Draw target curve with appropriate styling
                if hasattr(self, 'target_curve') and self.target_curve is not None:
                    target_x = [point[0] for point in self.target_curve]
                    target_y = [point[1] for point in self.target_curve]
                    ax.plot(target_x, target_y,
                           color='#0009b2',  # Blue
                           alpha=0.6,
                           linewidth=1.5,
                           linestyle='--',
                           label='Target Curve',
                           zorder=2)
                
            except Exception as e:
                print(f"Error in animation frame {frame}: {e}")
            
            # Clean, minimal plot styling
            self._setup_plot_limits(ax, poses)
            
            # Remove title and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove all spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove grid
            ax.grid(False)
            
            return links if 'links' in locals() else []
        
        # Create animation
        try:
            anim = animation.FuncAnimation(fig, animate, frames=len(filtered_poses), 
                                         interval=50, blit=False, repeat=True)
            
            # Save animation
            filename = f'mechanism_animation_robust_{mech_type}_traced.mp4'
            anim.save(filename, writer='ffmpeg', fps=20, bitrate=1800)
            plt.show()
            #print(f"BSI-based animation saved as '{filename}'")
            
                
        except Exception as e:
            print(f"Animation creation failed: {e}")
            print("Creating static plot as fallback...")
            self.create_static_plots(poses, mech_type)
    
    def _setup_plot_limits(self, ax, poses):
        """Setup plot limits filtering out extreme coordinates"""
        all_x = []
        all_y = []
        
        # Collect coordinates but filter out extreme outliers
        for pose in poses:
            for joint in pose:
                # Simple range filter - only include reasonable coordinates
                if abs(joint[0]) < 100 and abs(joint[1]) < 100:
                    all_x.append(joint[0])
                    all_y.append(joint[1])
        
        if not all_x or not all_y:
            all_x = [0]
            all_y = [0]
            print("Warning: All joints filtered out, using default range")
        
        margin = 2.0
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        
        # print(f"Plot limits: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f})")
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        #ax.set_aspect('equal')
        
        # Aesthetic axis styling
        ax.set_xlabel('X Position', fontsize=12, color='#34495E')
        ax.set_ylabel('Y Position', fontsize=12, color='#34495E')
        ax.tick_params(colors='#7F8C8D', labelsize=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
    
    def _draw_joints(self, ax, pose, mech_info):
        """Draw joint markers with aesthetic BSI information"""
        bsi_info = mech_info.get('bsi_info', {})
        
        # Draw all joints as aesthetic circles with soft edges
        for i, joint in enumerate(pose):
            ax.plot(joint[0], joint[1], 'o', 
                   color='#E0E0E0',  # Light gray
                   markersize=8,
                   markeredgecolor='#808080',  # Medium gray edge
                   markeredgewidth=1,
                   alpha=0.8)
            ax.annotate(f'J{i}', (joint[0], joint[1]), xytext=(6, 6), 
                       textcoords='offset points', fontsize=9, 
                       color='#505050', alpha=0.8)
        
        # Highlight ground joints with aesthetic squares
        ground_joints = bsi_info.get('ground_joints', [])
        for joint_idx in ground_joints:
            if joint_idx < len(pose):
                ax.plot(pose[joint_idx][0], pose[joint_idx][1], 
                       's', 
                       color=self.colors['ground'], 
                       markersize=self.joint_style['size'],
                       markeredgecolor=self.joint_style['edge_color'],
                       markeredgewidth=self.joint_style['edge_width'],
                       alpha=0.9)
        
        # Highlight coupler point with prominent aesthetic circle
        coupler_point = bsi_info.get('coupler_point', -1)
        if coupler_point >= 0 and coupler_point < len(pose):
            ax.plot(pose[coupler_point][0], pose[coupler_point][1], 
                   'o', 
                   color=self.colors['coupler'], 
                   markersize=self.joint_style['size'] + 4,
                   #markeredgecolor=self.joint_style['edge_color'],
                   #markeredgewidth=self.joint_style['edge_width'],
                   alpha=0.9)
    
    def create_static_plots(self, poses, mech_type):
        """Create static plots for 4-bar or 6-bar mechanisms using BSI topology"""
        if len(poses) == 0:
            return
            
        # Identify core mechanism joints
        core_joints = self._identify_core_mechanism_joints(poses, mech_type)
        
        # Filter poses to include only core joints
        filtered_poses = []
        for pose in poses:
            filtered_pose = [pose[i] for i in core_joints if i < len(pose)]
            filtered_poses.append(filtered_pose)
            
        if not filtered_poses:
            print("No valid core joints found for static plots")
            return
            
        mech_info = self.classifier.classify_mechanism(mech_type, len(filtered_poses[0]))
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All joint trajectories
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for joint_idx in range(len(filtered_poses[0])):
            x_traj = [pose[joint_idx][0] for pose in filtered_poses]
            y_traj = [pose[joint_idx][1] for pose in filtered_poses]
            ax1.plot(x_traj, y_traj, color=colors[joint_idx % len(colors)], 
                    linewidth=2, label=f'Joint {core_joints[joint_idx]}')
        
        ax1.set_title(f'{mech_info["description"]} - Joint Trajectories')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Mechanism configurations at different time steps
        time_steps = [0, len(filtered_poses)//4, len(filtered_poses)//2, 3*len(filtered_poses)//4, len(filtered_poses)-1]
        alphas = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        for i, (step, alpha) in enumerate(zip(time_steps, alphas)):
            pose = filtered_poses[step]
            links = self.drawer.draw_mechanism(ax2, pose, mech_info, alpha=alpha)
            
            # Draw joints for this configuration
            for j, joint in enumerate(pose):
                ax2.plot(joint[0], joint[1], 'ko', markersize=3, alpha=alpha)
        
        ax2.set_title(f'{mech_info["description"]} - Multiple Configurations')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'mechanism_static_plots_bsi_{mech_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _identify_core_mechanism_joints(self, poses, mech_type):
        """Identify core mechanism joints for 4-bar and 6-bar mechanisms using BSI information with robust prismatic support"""
        if len(poses) == 0:
            return []
            
        # Get BSI information to guide core joint selection
        bsi_info = self.classifier.bsi_analyzer.get_mechanism_info(mech_type)
        
        # Get first pose to analyze joint positions
        first_pose = poses[0]
        
        # Define mechanism-specific joint counts based on BSI data
        mechanism_point_counts = {
            'RRRR': 5,    # 4 revolute joints + 1 coupler point
            'RRRP': 6,    # 3 revolute + 1 prismatic + 2 points for sliding
            'RRPR': 7,    # 2 revolute + 1 prismatic + 4 points for complex sliding  
            'PRPR': 5     # 2 prismatic joints + 3 support points
        }
        
        # Get expected number of points for this mechanism type
        expected_points = mechanism_point_counts.get(mech_type, len(first_pose))
        
        #print(f"Mechanism {mech_type}: Expected {expected_points} points, Got {len(first_pose)} points")
        
        # Filter joints by reasonable coordinate range first
        reasonable_joints = []
        for i, joint in enumerate(first_pose):
            if abs(joint[0]) < 1000 and abs(joint[1]) < 1000:  # Increased range for prismatic mechanisms
                reasonable_joints.append(i)
        
        #print(f"Reasonable joints (within ±1000 units): {reasonable_joints}")
        
        # Robust joint selection based on mechanism type and BSI data
        if bsi_info and bsi_info.get('num_points', 0) > 0:
            # Use BSI-defined number of points as the authoritative count
            target_joints = min(bsi_info['num_points'], len(reasonable_joints))
            
            # For prismatic mechanisms, validate against known constraints
            if bsi_info.get('num_prismatic_joints', 0) > 0:
                #print(f"Prismatic mechanism detected: {bsi_info['num_prismatic_joints']} prismatic joints")
                
                # Validate prismatic constraints against available joints
                prismatic_constraints = bsi_info.get('prismatic_constraints', [])
                max_constraint_index = 0
                
                for constraint in prismatic_constraints:
                    if len(constraint) >= 3:
                        max_index = max(constraint[:3])
                        max_constraint_index = max(max_constraint_index, max_index)
                
                # Ensure we have enough joints to satisfy all constraints
                if max_constraint_index >= len(reasonable_joints):
                    print(f"Warning: Max constraint index {max_constraint_index} exceeds available joints {len(reasonable_joints)}")
                    # Use all reasonable joints to avoid index errors
                    core_joints = reasonable_joints
                else:
                    # Use the minimum of: BSI points, constraint requirements, or available joints
                    target_joints = min(target_joints, max_constraint_index + 1, len(reasonable_joints))
                    core_joints = reasonable_joints[:target_joints]
                
                #print(f"Prismatic mechanism: using {len(core_joints)} joints from BSI constraints")
            else:
                # Non-prismatic mechanism - use BSI count directly
                core_joints = reasonable_joints[:target_joints]
                #print(f"Non-prismatic mechanism: using {target_joints} joints from BSI")
        else:
            # Fallback selection when BSI data is unavailable
            print("No BSI data available, using fallback selection")
            
            if mech_type in mechanism_point_counts:
                # Use known mechanism point counts
                target_joints = min(mechanism_point_counts[mech_type], len(reasonable_joints))
                core_joints = reasonable_joints[:target_joints]
                #print(f"Using known mechanism count: {target_joints} joints for {mech_type}")
            else:
                # Generic fallback
                if any(joint_type in mech_type for joint_type in ['PRPR', 'RRRP', 'RRPR']):
                    # For prismatic mechanisms, be more conservative
                    core_joints = reasonable_joints[:min(7, len(reasonable_joints))]
                    print(f"Prismatic fallback: using {len(core_joints)} joints")
                elif 'Steph' in mech_type or 'Watt' in mech_type:
                    core_joints = reasonable_joints[:min(8, len(reasonable_joints))]
                else:
                    # Default 4-bar
                    core_joints = reasonable_joints[:min(5, len(reasonable_joints))]
        
        # Final validation - ensure core_joints is not empty
        if not core_joints:
            print("Warning: No core joints identified, using first 4 joints as fallback")
            core_joints = list(range(min(4, len(first_pose))))
        
        #print(f"Final selected core joints: {core_joints}")
        return core_joints