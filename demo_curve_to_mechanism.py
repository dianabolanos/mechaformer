import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wrapper.mechanism_core import MechanismAnimator, MechanismClassifier
from wrapper.mechanism_wrapper import MechanismWrapper
import matplotlib.animation as animation
from inference_mechanism import MechanismInference
from config import Config
import pickle
from scipy.interpolate import splprep, splev


def load_processed_data(pickle_path='processed_data.pkl'):
    """Load processed data from pickle file"""
    print("Loading processed data...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['processed_data'])} samples")
    print(f"  Vocabulary size: {data['vocab_size']}")
    return data


def create_validation_split(dataset, val_split=0.1, seed=42):
    """Create stratified train/validation split ensuring all mechanism types are represented"""
    np.random.seed(seed)
    
    # Group samples by mechanism type
    mechanism_groups = {}
    for idx in range(len(dataset['processed_data'])):
        sample = dataset['processed_data'][idx]
        mech_type = sample["mechanism_type"]
        if mech_type not in mechanism_groups:
            mechanism_groups[mech_type] = []
        mechanism_groups[mech_type].append(idx)
    
    print(f"Found {len(mechanism_groups)} mechanism types:")
    for mech_type, indices in mechanism_groups.items():
        print(f"  {mech_type}: {len(indices)} samples")
    
    # Create stratified split
    train_indices = []
    val_indices = []
    
    for mech_type, indices in mechanism_groups.items():
        # Shuffle indices for this mechanism type
        np.random.shuffle(indices)
        
        # Calculate validation size for this mechanism type
        n_samples = len(indices)
        n_val = max(1, int(n_samples * val_split))  # Ensure at least 1 validation sample
        
        # Split indices
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
        
        print(f"  {mech_type}: {n_samples - n_val} train, {n_val} validation")
    
    # Shuffle the final lists
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    print(f"Total: {len(train_indices)} train, {len(val_indices)} validation samples")
    
    return np.array(train_indices), np.array(val_indices)


def get_validation_sample_by_index(data, validation_indices, val_index):
    """Get a sample from the validation set by its index in the validation set"""
    if val_index >= len(validation_indices):
        raise ValueError(f"Validation index {val_index} is out of range. Validation set has {len(validation_indices)} samples.")
    
    # Get the actual dataset index from the validation indices
    actual_index = validation_indices[val_index]
    sample = data['processed_data'][actual_index]
    
    print(f"Selected validation sample {val_index} (dataset index {actual_index})")
    print(f"Mechanism type: {sample['mechanism_type']}")
    print(f"Bar type: {sample['bar_type']}")
    
    return sample


def reconstruct_curve_from_control_points(control_points, num_points=360):
    """Reconstruct curve from B-spline control points"""
    try:
        # Ensure control points are float arrays
        control_points = np.array(control_points, dtype=float)
        
        # Create B-spline from control points
        tck, u = splprep([control_points[:, 0], control_points[:, 1]], 
                         k=min(3, len(control_points)-1), s=0)
        
        # Evaluate B-spline at uniform parameter values
        u_new = np.linspace(0, 1, num_points)
        curve_points = splev(u_new, tck)

        return np.column_stack(curve_points)
    except Exception as e:
        print(f"Failed to reconstruct curve: {e}")
        return None


def get_plot_limits(poses, curve_points, coupler_traj=None):
    """Calculate fixed plot limits that encompass both mechanism and curve"""
    all_x = []
    all_y = []
    
    # Add mechanism points
    for pose in poses:
        for point in pose[:5, :]:   # only add the first 5 points for 4 bar mechanisms
            all_x.append(point[0])
            all_y.append(point[1])
    
    # Add curve points
    all_x.extend(curve_points[:, 0])
    all_y.extend(curve_points[:, 1])
    
    # Add coupler trajectory if available
    if coupler_traj is not None:
        all_x.extend(coupler_traj[:, 0])
        all_y.extend(coupler_traj[:, 1])
    
    # Calculate limits with margin
    margin = 1.0
    x_min = min(all_x) - margin
    x_max = max(all_x) + margin
    y_min = min(all_y) - margin
    y_max = max(all_y) + margin
    
    return [x_min, x_max, y_min, y_max]


def create_animation(wrapper, animator, target_curve, coupler_traj, plot_limits):
    """Create and return the animation showing mechanism tracing the target curve"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate_frame(frame):
        ax.clear()
        
        # Draw the mechanism at current frame
        if frame < len(wrapper.poses):
            animator.drawer.draw_mechanism(ax, wrapper.poses[frame], wrapper.get_mechanism_info())
        
        # Draw the target curve (input to the model)
        ax.plot(target_curve[:, 0], target_curve[:, 1], linestyle=':', color='red', 
                linewidth=2, alpha=0.8, label='Target Curve (Input)')
        
        # Draw coupler trajectory (mechanism output)
        if coupler_traj is not None:
            # Draw complete trajectory as faint line
            ax.plot(coupler_traj[:, 0], coupler_traj[:, 1], '--', color='blue', 
                    linewidth=2, alpha=0.3, label='Generated Mechanism Path')
            # Draw trajectory up to current frame with emphasis
            if frame > 0:
                ax.plot(coupler_traj[:frame+1, 0], coupler_traj[:frame+1, 1], '-', 
                        color='blue', linewidth=4, alpha=0.8)
        
        # Setup plot
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(plot_limits[0], plot_limits[1])
        ax.set_ylim(plot_limits[2], plot_limits[3])
        ax.legend(loc='upper right')
        ax.set_title(f'AI-Generated Mechanism Tracing Target Curve - Frame {frame+1}/{len(wrapper.poses)}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(wrapper.poses), 
                                 interval=50, blit=False, repeat=True)
    return anim


def generate_mechanism_demo(sample_index=0, output_dir='demo_output'):
    """
    Demo function that generates a mechanism from a target curve and creates an animation.
    
    Args:
        sample_index: Index of validation sample to use (0 to validation_size-1)
        output_dir: Directory to save the output video
    """
    print("=" * 60)
    print("CURVE-TO-MECHANISM GENERATION DEMO")
    print("=" * 60)
    
    # Load data and create validation split
    data = load_processed_data('processed_data.pkl')
    train_indices, val_indices = create_validation_split(data, val_split=0.1, seed=42)
    
    print(f"\nValidation set contains {len(val_indices)} samples")
    print(f"Using sample index: {sample_index}")
    
    # Get the validation sample
    try:
        sample = get_validation_sample_by_index(data, val_indices, sample_index)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Please choose a validation sample index between 0 and {len(val_indices)-1}")
        return None
    
    # Extract target curve from sample
    control_points = sample['control_points']
    target_curve = reconstruct_curve_from_control_points(control_points)
    
    print(f"\nTarget curve information:")
    print(f"  Control points shape: {np.array(control_points).shape}")
    print(f"  Reconstructed curve shape: {target_curve.shape}")
    print(f"  Curve range: X=[{np.min(target_curve[:, 0]):.3f}, {np.max(target_curve[:, 0]):.3f}], "
          f"Y=[{np.min(target_curve[:, 1]):.3f}, {np.max(target_curve[:, 1]):.3f}]")
    
    # Initialize the model
    print(f"\nInitializing model...")
    inference = MechanismInference(Config.MODEL_SAVE_PATH, Config.VOCAB_SAVE_PATH)
    
    # Generate mechanism parameters from the target curve
    print(f"Generating mechanism from target curve...")
    mechanism_result = inference.generate_mechanism_params(
        control_points, 
        temperature=0.001,  # Low temperature for more deterministic results
        top_k=5,
        process_curve=False  # Use control points as-is
    )
    
    mechanism_params = mechanism_result["params"]
    mechanism_type = mechanism_result["type"]
    
    print(f"\nGenerated mechanism:")
    print(f"  Type: {mechanism_type}")
    print(f"  Number of joints: {len(mechanism_params)}")
    print(f"  Joint coordinates: {mechanism_params}")
    
    # Determine bar type for display
    if mechanism_type in ['RRRR', 'PRPR', 'RRPR', 'RRRP', 'RPPR', 'RRPP']:
        bar_type = '4bar'
    else:
        bar_type = '6bar'
    
    print(f"  Bar type: {bar_type}")
    
    # Create mechanism simulation
    print(f"\nSimulating mechanism...")
    
    # Create coordinate string for the mechanism wrapper
    flat_coords = [str(round(float(num), 2)) for pair in mechanism_params for num in pair]
    coords = "_" + "_".join(flat_coords) + "_" + mechanism_type
    
    # Initialize and simulate the mechanism
    wrapper = MechanismWrapper(coords)
    wrapper.simulate(speed_scale=1.0, steps=360, relative_tolerance=0.1, 
                    driving_element=1, start_angle=0, end_angle=360)
    
    # Check simulation success
    if len(wrapper.poses) <= 1:
        print(f"❌ Mechanism simulation failed - only {len(wrapper.poses)} poses generated")
        return None
    
    print(f"✅ Simulation successful - {len(wrapper.poses)} poses generated")
    
    # Get coupler trajectory (the path traced by the mechanism)
    coupler_traj = np.array(wrapper.get_coupler_trajectory())
    if len(coupler_traj) == 0:
        print("❌ No coupler trajectory generated")
        return None
    
    print(f"✅ Coupler trajectory generated with {len(coupler_traj)} points")
    
    # Create animation
    print(f"\nCreating animation...")
    animator = MechanismAnimator()
    plot_limits = get_plot_limits(wrapper.poses, target_curve, coupler_traj)
    
    anim = create_animation(wrapper, animator, target_curve, coupler_traj, plot_limits)
    
    # Save video
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'demo_{bar_type}_{mechanism_type}.mp4')
    
    print(f"Saving animation to: {output_filename}")
    
    writer = animation.FFMpegWriter(
        fps=30,
        metadata=dict(artist='Mechformer Mechanism Generator'),
        bitrate=3000,
        codec='h264'
    )
    
    anim.save(output_filename, writer=writer, dpi=150)
    
    # Verify file creation
    if os.path.exists(output_filename):
        file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
        print(f"✅ Animation saved successfully!")
        print(f"  File: {output_filename}")
        print(f"  Size: {file_size:.1f} MB")
    else:
        print(f"❌ Failed to save animation")
        return None
    
    # Close figure to free memory
    plt.close()
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Input: Target curve with {len(target_curve)} points")
    print(f"🤖 Mechformer Generated: {mechanism_type} mechanism with {len(mechanism_params)} joints")
    print(f"📹 Output: Animation showing mechanism tracing the target curve")
    print(f"📁 Saved to: {output_filename}")
    print("=" * 60)
    
    return {
        'video_path': output_filename,
        'mechanism_type': mechanism_type,
        'bar_type': bar_type,
        'target_curve': target_curve,
        'coupler_trajectory': coupler_traj,
        'mechanism_params': mechanism_params
    }


def main():
    """
    Main demo function. Modify SAMPLE_INDEX to try different curves.
    """
    # ==============================================
    # DEMO CONFIGURATION
    # ==============================================
    SAMPLE_INDEX = 365   # Try different values: 0, 1, 2, 3, ... up to validation set size
    OUTPUT_DIR = 'demo_output'
    # ==============================================
    
    # Set random seeds for reproducibility
    import random
    import torch
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Run the demo
    result = generate_mechanism_demo(SAMPLE_INDEX, OUTPUT_DIR)
    
    if result:
        print(f"\n🎉 Demo completed! Check out the video: {result['video_path']}")
        print(f"\n💡 Tip: Try changing SAMPLE_INDEX to see different examples!")
    else:
        print(f"\n❌ Demo failed. Please check the error messages above.")


if __name__ == "__main__":
    main() 