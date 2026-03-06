import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.optimize import minimize

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference_mechanism import MechanismInference
from wrapper.mechanism_wrapper import MechanismWrapper
from wrapper.mechanism_core import MechanismAnimator
from config import Config
from scipy.interpolate import splprep, splev, BSpline
import pickle
from tslearn.metrics import dtw_path


# ============================================================================
# AFFINE TRANSFORMATION UTILITIES
# ============================================================================

def rotate_points(points, angle_degrees):
    """Rotate points about origin by given angle in degrees"""
    angle = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=np.float32)
    return points @ rotation_matrix


def translate_points(points, tx, ty):
    """Translate points by (tx, ty)"""
    return points + np.array([tx, ty], dtype=np.float32)


def apply_affine_transform(points, angle_degrees, tx, ty):
    """Apply rotation then translation to points"""
    # Convert to numpy if tensor
    if hasattr(points, 'numpy'):
        points = points.numpy()
    points = np.array(points, dtype=np.float32)
    
    # Apply rotation first, then translation
    rotated = rotate_points(points, angle_degrees)
    transformed = translate_points(rotated, tx, ty)
    return transformed


def inverse_affine_transform(points, angle_degrees, tx, ty):
    """Apply inverse transformation: -translation then -rotation"""
    if hasattr(points, 'numpy'):
        points = points.numpy()
    points = np.array(points, dtype=np.float32)
    
    # Inverse: first undo translation, then undo rotation
    untranslated = translate_points(points, -tx, -ty)
    unrotated = rotate_points(untranslated, -angle_degrees)
    return unrotated


# ============================================================================
# DATA LOADING
# ============================================================================

def load_processed_data(pickle_path=None):
    """Load processed data from pickle file"""
    if pickle_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(script_dir))
        pickle_path = os.path.join(root_dir, 'processed_data.pkl')
    
    print(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['processed_data'])} samples")
    return data


def create_validation_split(dataset, val_split=0.1, seed=42):
    """Create stratified validation split"""
    np.random.seed(seed)
    
    mechanism_groups = {}
    for idx, sample in enumerate(dataset['processed_data']):
        mech_type = sample["mechanism_type"]
        if mech_type not in mechanism_groups:
            mechanism_groups[mech_type] = []
        mechanism_groups[mech_type].append(idx)
    
    train_indices, val_indices = [], []
    for mech_type, indices in mechanism_groups.items():
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    
    np.random.shuffle(val_indices)
    return np.array(train_indices), np.array(val_indices)


def get_sample(data, val_indices, val_index):
    """Get a validation sample by index"""
    actual_index = val_indices[val_index]
    return data['processed_data'][actual_index]


# ============================================================================
# CURVE UTILITIES
# ============================================================================

def reconstruct_curve_from_control_points(control_points, num_points=200):
    """Reconstruct curve from B-spline control points"""
    control_points = np.array(control_points, dtype=float)
    degree = 3
    n = len(control_points)
    
    if n <= degree:
        return None
    
    knot_vector = np.concatenate((
        np.zeros(degree),
        np.linspace(0, 1, n - degree + 1),
        np.ones(degree)
    ))
    
    cx, cy = control_points[:, 0], control_points[:, 1]
    bspline_x = BSpline(knot_vector, cx, degree)
    bspline_y = BSpline(knot_vector, cy, degree)
    
    t_vals = np.linspace(0, 1, num_points)
    return np.vstack([bspline_x(t_vals), bspline_y(t_vals)]).T


def compute_dtw_distance(curve1, curve2):
    """Compute DTW distance between two curves"""
    try:
        _, distance = dtw_path(curve1, curve2)
        return distance if not (np.isnan(distance) or np.isinf(distance)) else float('inf')
    except:
        return float('inf')


# ============================================================================
# MECHANISM GENERATION
# ============================================================================

def generate_mechanism_from_curve(inference, control_points, temperature=0.001):
    """Generate mechanism parameters from control points"""
    result = inference.generate_mechanism_params(
        control_points,
        temperature=temperature,
        top_k=5,
        process_curve=False
    )
    return result


def simulate_mechanism(mechanism_params, mechanism_type):
    """Simulate mechanism and return coupler trajectory"""
    # Determine bar type
    if mechanism_type in ['RRRR', 'PRPR', 'RRPR', 'RRRP', 'RPPR', 'RRPP']:
        bar_type = '4bar'
        coords_to_use = mechanism_params[:5]
    elif mechanism_type.startswith('Steph') or mechanism_type.startswith('Watt'):
        bar_type = '6bar'
        coords_to_use = mechanism_params
    else:
        return None, None, None
    
    # Build coordinate string
    try:
        flat_coords = [str(round(float(num), 2)) for pair in coords_to_use for num in pair]
    except (ValueError, TypeError):
        return None, None, None
    
    coords = "_" + "_".join(flat_coords) + "_" + mechanism_type
    
    # Simulate
    wrapper = MechanismWrapper(coords)
    wrapper.simulate(speed_scale=1.0, steps=200, relative_tolerance=0.1,
                     driving_element=1, start_angle=0, end_angle=360)
    
    if len(wrapper.poses) <= 1:
        return None, None, None
    
    coupler_traj = wrapper.get_coupler_trajectory()
    if not coupler_traj or len(coupler_traj) == 0:
        return None, None, None
    
    return np.array(coupler_traj), wrapper, bar_type


# ============================================================================
# AFFINE SEARCH - FAST VERSION
# ============================================================================

def search_best_affine_transform(inference, control_points, 
                                  angles=None, translations=None,
                                  temperature=0.001, verbose=True):
    """
    Search for the best affine transformation that produces a valid mechanism.
    
    Args:
        inference: MechanismInference instance
        control_points: Original control points
        angles: List of rotation angles to try (degrees)
        translations: List of (tx, ty) translation pairs to try
        temperature: Model temperature
        verbose: Print progress
    
    Returns:
        dict with best result information
    """
    if angles is None:
        # Default: 4 rotations for speed
        angles = [0, 90, 180, 270]
    
    if translations is None:
        # Default: small translation grid
        translations = [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
    
    # Convert control points to numpy
    if hasattr(control_points, 'numpy'):
        control_points = control_points.numpy()
    control_points = np.array(control_points, dtype=np.float32)
    
    # Generate original target curve
    target_curve = reconstruct_curve_from_control_points(control_points)
    if target_curve is None:
        return None
    
    best_result = {
        'dtw_distance': float('inf'),
        'angle': None,
        'translation': None,
        'mechanism_type': None,
        'mechanism_params': None,
        'coupler_trajectory': None,
        'target_curve': target_curve,
        'wrapper': None,
        'bar_type': None
    }
    
    total_attempts = len(angles) * len(translations)
    attempt = 0
    successful = 0
    
    if verbose:
        print(f"Searching {len(angles)} rotations × {len(translations)} translations = {total_attempts} combinations")
    
    start_time = time.time()
    
    for angle in angles:
        for (tx, ty) in translations:
            attempt += 1
            
            # Apply affine transformation to control points
            transformed_cp = apply_affine_transform(control_points, angle, tx, ty)
            
            # Generate mechanism
            try:
                result = generate_mechanism_from_curve(inference, transformed_cp, temperature)
                mech_type = result.get("type")
                mech_params = result.get("params", [])
                
                if not mech_type or len(mech_params) < 3:
                    continue
                
                # Simulate mechanism
                coupler_traj, wrapper, bar_type = simulate_mechanism(mech_params, mech_type)
                
                if coupler_traj is None:
                    continue
                
                successful += 1
                
                # Transform coupler trajectory back to original space
                coupler_traj_original = inverse_affine_transform(coupler_traj, angle, tx, ty)
                
                # Compute DTW distance to original target
                dtw_dist = compute_dtw_distance(target_curve, coupler_traj_original)
                
                if verbose and attempt % 5 == 0:
                    print(f"  [{attempt}/{total_attempts}] angle={angle}°, t=({tx:.1f},{ty:.1f}) → DTW={dtw_dist:.4f}")
                
                # Update best if improved
                if dtw_dist < best_result['dtw_distance']:
                    # Also compute transformed target curve for animation
                    transformed_target = apply_affine_transform(target_curve, angle, tx, ty)
                    best_result.update({
                        'dtw_distance': dtw_dist,
                        'angle': angle,
                        'translation': (tx, ty),
                        'mechanism_type': mech_type,
                        'mechanism_params': mech_params,
                        'coupler_trajectory': coupler_traj_original,  # Original space (for comparison plot)
                        'transformed_coupler_trajectory': coupler_traj,  # Transformed space (matches mechanism)
                        'transformed_target_curve': transformed_target,  # Transformed space (for animation)
                        'wrapper': wrapper,
                        'bar_type': bar_type
                    })
                    
            except Exception as e:
                if verbose:
                    print(f"  Error at angle={angle}, t=({tx},{ty}): {e}")
                continue
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nSearch complete in {elapsed:.2f}s")
        print(f"  Successful simulations: {successful}/{total_attempts}")
        if best_result['angle'] is not None:
            print(f"  Best: angle={best_result['angle']}°, t={best_result['translation']}, DTW={best_result['dtw_distance']:.4f}")
    
    return best_result if best_result['angle'] is not None else None


# ============================================================================
# L-BFGS OPTIMIZATION
# ============================================================================

def get_normalization_params(curve):
    """Get normalization parameters from a curve"""
    curve_array = np.array(curve).squeeze()
    mean = np.mean(curve_array, axis=0)
    centered = curve_array - mean
    var_x = np.var(centered[:, 0])
    var_y = np.var(centered[:, 1])
    rms_var = np.sqrt(var_x + var_y)
    return mean, rms_var if rms_var != 0 else 1.0


def apply_normalization(curve, mean, rms_var):
    """Apply normalization to a curve"""
    curve_array = np.array(curve).squeeze()
    centered = curve_array - mean
    normalized = centered / rms_var
    return normalized


def normalize_curves(coupler_traj, target_traj):
    """Normalize both curves using target trajectory parameters"""
    mean, rms_var = get_normalization_params(target_traj)
    normalized_target = apply_normalization(target_traj, mean, rms_var)
    normalized_coupler = apply_normalization(coupler_traj, mean, rms_var)
    return normalized_coupler, normalized_target, mean, rms_var


def optimization_routine(dtw_distance, coupler_traj, coords, b_spline_trajectory, mean, rms_var):
    """
    Optimize mechanism coordinates using L-BFGS-B to minimize DTW distance.
    
    Args:
        dtw_distance: Initial DTW distance
        coupler_traj: Initial normalized coupler trajectory
        coords: String containing mechanism coordinates and type (e.g. "_x1_y1_x2_y2_..._TYPE")
        b_spline_trajectory: Target normalized trajectory to match
        mean: Mean used for normalization
        rms_var: RMS variance used for normalization
        
    Returns:
        tuple: (optimized coordinates string, final DTW distance, final coupler trajectory)
    """
    # Extract coordinates and mechanism type from coords string
    parts = coords.strip('_').split('_')
    mech_type = parts[-1]  # Last part is mechanism type
    all_coords = np.array([float(x) for x in parts[:-1]])  # All coordinates
    
    # Identify which coordinates to optimize (excluding fixed points)
    fixed_indices = [0, 1, 4, 5]  # Indices of fixed coordinates (0,0) and (1,0)
    variable_indices = [i for i in range(len(all_coords)) if i not in fixed_indices]
    coord_values = all_coords[variable_indices]  # Only the coordinates we'll optimize
    
    def objective_function(x):
        """Objective function for optimization: DTW distance between normalized trajectories"""
        # Reconstruct full coordinate array with fixed points
        current_full_coords = all_coords.copy()
        current_full_coords[variable_indices] = x
        
        # Reconstruct coords string with current parameters
        current_coords = '_' + '_'.join(f"{val:.3f}" for val in current_full_coords) + f'_{mech_type}'
        
        # Create mechanism and simulate
        wrapper = MechanismWrapper(current_coords)
        wrapper.simulate()
        current_traj = np.array(wrapper.get_coupler_trajectory())
        
        if len(current_traj) == 0:
            return 1e6  # Return large value if simulation fails
        
        # Normalize current trajectory using the same parameters
        current_traj_norm = apply_normalization(current_traj, mean, rms_var)
        
        # Calculate DTW distance between normalized trajectories
        _, current_dtw = dtw_path(b_spline_trajectory, current_traj_norm)
        return float(current_dtw)
    
    # Set bounds for coordinates (±50% of initial values to allow more exploration)
    bounds = []
    for val in coord_values:
        bound_range = max(abs(val) * 0.5, 0.5)  # At least ±0.5 range
        bounds.append((val - bound_range, val + bound_range))
    
    # Run L-BFGS-B optimization with early stopping
    result = minimize(
        objective_function,
        coord_values,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 50,        # Reduced from 100
            'maxfun': 100,        # Limit function evaluations
            'ftol': 1e-6,         # Slightly relaxed from 1e-8
            'gtol': 1e-6,         # Slightly relaxed from 1e-8
            'eps': 1e-3,          # Keep the same step size
            'disp': False         # Disable progress output
        }
    )
    
    # Reconstruct final coordinates with fixed points
    final_full_coords = all_coords.copy()
    final_full_coords[variable_indices] = result.x
    
    # Construct final coordinates string
    final_coords = '_' + '_'.join(f"{val:.3f}" for val in final_full_coords) + f'_{mech_type}'
    
    # Get final trajectory
    wrapper = MechanismWrapper(final_coords)
    wrapper.simulate()
    final_traj = np.array(wrapper.get_coupler_trajectory())
    
    # Normalize final trajectory
    final_traj_norm = apply_normalization(final_traj, mean, rms_var)
    
    # Calculate final DTW distance using normalized trajectories
    _, final_dtw = dtw_path(b_spline_trajectory, final_traj_norm)
    
    return final_coords, final_dtw, final_traj, b_spline_trajectory



# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(target_curve, coupler_trajectory, title="Mechanism vs Target", save_path=None):
    """Plot target curve vs mechanism output"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(target_curve[:, 0], target_curve[:, 1], 'r-', 
            linewidth=2, label='Target Curve', alpha=0.8)
    ax.plot(coupler_trajectory[:, 0], coupler_trajectory[:, 1], 'b--', 
            linewidth=2, label='Mechanism Output', alpha=0.8)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()
    plt.close()


def transform_poses_back(poses, angle, translation):
    """Transform all mechanism poses back to original space"""
    tx, ty = translation
    transformed_poses = []
    for pose in poses:
        # Apply inverse: first undo translation, then undo rotation
        pose_array = np.array(pose)
        transformed_pose = inverse_affine_transform(pose_array, angle, tx, ty)
        transformed_poses.append(transformed_pose)
    return transformed_poses


def create_animation(wrapper, animator, target_curve, coupler_traj, poses, save_path=None):
    """Create animation showing mechanism tracing the target curve
    
    Args:
        wrapper: MechanismWrapper (for mechanism info)
        animator: MechanismAnimator
        target_curve: Target curve points (original space)
        coupler_traj: Coupler trajectory (original space)
        poses: Mechanism poses (already transformed to original space)
        save_path: Optional path to save animation
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate plot limits including mechanism poses
    all_x, all_y = [], []
    
    # Add curve and trajectory points
    all_x.extend(target_curve[:, 0])
    all_y.extend(target_curve[:, 1])
    all_x.extend(coupler_traj[:, 0])
    all_y.extend(coupler_traj[:, 1])
    
    # Add mechanism pose points
    for pose in poses:
        for point in pose[:5]:  # First 5 points for 4-bar mechanisms
            all_x.append(point[0])
            all_y.append(point[1])
    
    margin = 1.0
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    
    def animate_frame(frame):
        ax.clear()
        
        if frame < len(poses):
            animator.drawer.draw_mechanism(ax, poses[frame], wrapper.get_mechanism_info())
        
        ax.plot(target_curve[:, 0], target_curve[:, 1], 'r:', 
                linewidth=2, alpha=0.8, label='Target Curve')
        
        ax.plot(coupler_traj[:, 0], coupler_traj[:, 1], 'b--', 
                linewidth=2, alpha=0.3, label='Mechanism Path')
        
        if frame > 0:
            ax.plot(coupler_traj[:frame+1, 0], coupler_traj[:frame+1, 1], 'b-', 
                    linewidth=3, alpha=0.8)
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='upper right')
        ax.set_title(f'Frame {frame+1}/{len(poses)}')
    
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(poses),
                                    interval=50, blit=False, repeat=True)
    
    if save_path:
        writer = animation.FFMpegWriter(fps=30, bitrate=3000, codec='h264')
        anim.save(save_path, writer=writer, dpi=150)
        print(f"Saved animation to: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return anim


# ============================================================================
# MAIN DEMO
# ============================================================================

def run_affine_search_demo(sample_index=0, output_dir='motion_synthesis_output', run_optimization=True):
    """
    Demo: Find best affine transformation for mechanism generation,
    then optionally optimize with L-BFGS.
    
    Args:
        sample_index: Index in validation set to use
        output_dir: Directory for output files
        run_optimization: Whether to run L-BFGS optimization on best result
    """
    print("=" * 60)
    print("AFFINE TRANSFORMATION MECHANISM SYNTHESIS")
    print("=" * 60)
    
    # Load data
    data = load_processed_data()
    _, val_indices = create_validation_split(data)
    print(f"Validation set: {len(val_indices)} samples")
    
    # Get sample
    if sample_index >= len(val_indices):
        print(f"Error: sample_index {sample_index} out of range (max: {len(val_indices)-1})")
        return None
    
    sample = get_sample(data, val_indices, sample_index)
    control_points = np.array(sample['control_points'], dtype=np.float32)
    
    print(f"\nSample {sample_index}:")
    print(f"  Mechanism type: {sample['mechanism_type']}")
    print(f"  Control points: {len(control_points)}")
    
    # Initialize model (change to project root for relative paths)
    print("\nLoading model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    inference = MechanismInference(Config.MODEL_SAVE_PATH, Config.VOCAB_SAVE_PATH)
    
    # Define search space (fast version)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 rotations
    translations = [
        (0, 0),
        (0.3, 0), (-0.3, 0),
        (0, 0.3), (0, -0.3),
        (0.3, 0.3), (-0.3, -0.3)
    ]  # 7 translations
    
    # Search for best transformation
    print("\nSearching for best affine transformation...")
    result = search_best_affine_transform(
        inference, 
        control_points,
        angles=angles,
        translations=translations,
        temperature=0.001,
        verbose=True
    )
    
    if result is None:
        print("\n❌ No valid mechanism found!")
        return None
    
    # Print results
    print("\n" + "=" * 60)
    print("AFFINE SEARCH RESULTS")
    print("=" * 60)
    print(f"Best Transformation:")
    print(f"  Rotation: {result['angle']}°")
    print(f"  Translation: ({result['translation'][0]:.2f}, {result['translation'][1]:.2f})")
    print(f"  DTW Distance: {result['dtw_distance']:.4f}")
    print(f"  Mechanism Type: {result['mechanism_type']} ({result['bar_type']})")
    
    # Build mechanism coords string for optimization
    mech_params = result['mechanism_params']
    mech_type = result['mechanism_type']
    if result['bar_type'] == '4bar':
        coords_to_use = mech_params[:5]
    else:
        coords_to_use = mech_params
    flat_coords = [str(round(float(num), 2)) for pair in coords_to_use for num in pair]
    coords_string = "_" + "_".join(flat_coords) + "_" + mech_type
    
    # Run L-BFGS optimization on best result (following optimization_full-study.py logic)
    if run_optimization:
        print("\n" + "=" * 60)
        print("L-BFGS OPTIMIZATION")
        print("=" * 60)
        
        # Get the coupler trajectory in mechanism space (not transformed back)
        coupler_traj = result['transformed_coupler_trajectory']
        
        # Get transformed target curve (B-spline trajectory in mechanism space)
        b_spline_trajectory = result['transformed_target_curve']
        
        # Normalize both trajectories using B-spline parameters (exactly like optimization_full-study.py)
        normalized_coupler, normalized_bspline, mean, rms_var = normalize_curves(coupler_traj, b_spline_trajectory)
        
        # Calculate initial DTW distance using normalized trajectories
        _, initial_dtw = dtw_path(normalized_bspline, normalized_coupler)
        
        print(f"  Initial DTW (normalized): {initial_dtw:.4f}")
        print(f"  Coords: {coords_string}")
        
        # Run optimization with normalized trajectories (using optimization_routine)
        final_coords, final_dtw, final_traj, _ = optimization_routine(
            initial_dtw, 
            normalized_coupler, 
            coords_string, 
            normalized_bspline, 
            mean, 
            rms_var
        )
        
        improvement = initial_dtw - final_dtw
        print(f"  Final DTW (normalized): {final_dtw:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        
        if improvement > 0:
            print(f"\n✓ Optimization improved DTW by {improvement:.4f}")
            
            # Update result with optimized mechanism
            result['optimized_coords'] = final_coords
            result['optimized_dtw'] = final_dtw
            result['pre_optimization_dtw'] = initial_dtw
            
            # Transform optimized trajectory back to original space
            opt_traj_original = inverse_affine_transform(
                final_traj, result['angle'], result['translation'][0], result['translation'][1]
            )
            
            # Update DTW in original space (unnormalized)
            result['optimized_coupler_trajectory'] = opt_traj_original
            result['optimized_dtw_original'] = compute_dtw_distance(result['target_curve'], opt_traj_original)
            
            # Create wrapper for animation
            wrapper_opt = MechanismWrapper(final_coords)
            wrapper_opt.simulate(steps=200)
            result['wrapper_optimized'] = wrapper_opt
            
            print(f"  Original space DTW: {result['dtw_distance']:.4f} → {result['optimized_dtw_original']:.4f}")
        else:
            print("\n  No improvement from optimization")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot comparison (use optimized if available)
    plot_path = os.path.join(output_dir, f'comparison_sample_{sample_index}.png')
    
    if run_optimization and 'optimized_coupler_trajectory' in result:
        # Plot both original and optimized
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Before optimization
        ax = axes[0]
        ax.plot(result['target_curve'][:, 0], result['target_curve'][:, 1], 'r-', 
                linewidth=2, label='Target Curve', alpha=0.8)
        ax.plot(result['coupler_trajectory'][:, 0], result['coupler_trajectory'][:, 1], 'b--', 
                linewidth=2, label='Before Optimization', alpha=0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Before L-BFGS (DTW={result['dtw_distance']:.4f})")
        
        # After optimization
        ax = axes[1]
        ax.plot(result['target_curve'][:, 0], result['target_curve'][:, 1], 'r-', 
                linewidth=2, label='Target Curve', alpha=0.8)
        ax.plot(result['optimized_coupler_trajectory'][:, 0], result['optimized_coupler_trajectory'][:, 1], 'g--', 
                linewidth=2, label='After Optimization', alpha=0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"After L-BFGS (DTW={result['optimized_dtw_original']:.4f})")
        
        plt.suptitle(f"Sample {sample_index}: {result['mechanism_type']}\n"
                     f"Rotation={result['angle']}°, Translation={result['translation']}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {plot_path}")
        plt.show()
        plt.close()
    else:
        plot_comparison(
            result['target_curve'],
            result['coupler_trajectory'],
            title=f"Sample {sample_index}: {result['mechanism_type']} (DTW={result['dtw_distance']:.4f})\n"
                  f"Rotation={result['angle']}°, Translation={result['translation']}",
            save_path=plot_path
        )
    
    # Create animation (transform mechanism back to original curve space)
    anim_path = os.path.join(output_dir, f'animation_sample_{sample_index}.mp4')
    animator = MechanismAnimator()
    
    # Use optimized mechanism if available
    if run_optimization and 'wrapper_optimized' in result:
        print("\nUsing optimized mechanism for animation...")
        wrapper_for_anim = result['wrapper_optimized']
        coupler_traj_for_anim = result['optimized_coupler_trajectory']
        final_dtw = result['optimized_dtw_original']
    else:
        wrapper_for_anim = result['wrapper']
        coupler_traj_for_anim = result['coupler_trajectory']
        final_dtw = result['dtw_distance']
    
    # Transform mechanism poses back to original space
    original_space_poses = transform_poses_back(
        wrapper_for_anim.poses, 
        result['angle'], 
        result['translation']
    )
    
    create_animation(
        wrapper_for_anim, 
        animator, 
        result['target_curve'],  # Original user curve
        coupler_traj_for_anim,  # Coupler trajectory (optimized or not)
        original_space_poses,  # Mechanism poses transformed back to original space
        save_path=anim_path
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Mechanism Type: {result['mechanism_type']}")
    print(f"  Affine Search DTW: {result['dtw_distance']:.4f}")
    if run_optimization and 'optimized_dtw_original' in result:
        print(f"  After L-BFGS DTW: {result['optimized_dtw_original']:.4f}")
        improvement = result['dtw_distance'] - result['optimized_dtw_original']
        print(f"  Total Improvement: {improvement:.4f} ({100*improvement/result['dtw_distance']:.1f}%)")
    
    return result


def main():
    """Main entry point"""
    import random
    import torch
    
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Configuration
    SAMPLE_INDEX = 84636  # Change to try different samples
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    
    # Run demo
    result = run_affine_search_demo(SAMPLE_INDEX, OUTPUT_DIR)
    
    if result:
        print(f"\n✅ Demo complete! Check output in: {OUTPUT_DIR}")
    else:
        print("\n❌ Demo failed.")


if __name__ == "__main__":
    main()# generate two four bars from two separate curves
# connect two mechanisms in global space (where they are generated in their original space)
# only drive one rerun simulation as eight bar


