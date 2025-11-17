import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add mechanism paths to Python path
# Get the correct path to the mechaformer-supp directory
script_dir = os.path.dirname(os.path.abspath(__file__))
mechanism_root = os.path.dirname(script_dir)
if mechanism_root not in sys.path:
    sys.path.append(mechanism_root)

from wrapper.mechanism_core import MechanismAnimator
from wrapper.mechanism_wrapper import MechanismWrapper
import matplotlib.animation as animation
from inference_mechanism import MechanismInference
from config import Config
import glob
import random
import json
import requests
from tslearn.metrics import dtw, dtw_path
from torch.utils.data import Subset, DataLoader, Dataset
import pickle
from scipy.interpolate import BSpline
import torch
import time
import gc
from collections import defaultdict
import gzip
import multiprocessing
from functools import partial
import tqdm
import psutil
import datetime



class MechanismDataset(Dataset):
    def __init__(self, processed_data_path=None):
        if processed_data_path is None:
            # Default to processed_data.pkl in the parent directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            processed_data_path = os.path.join(parent_dir, 'processed_data.pkl')
        self.processed_data_path = processed_data_path


        # Load pre-processed data
        self._load_processed_data()

    def _load_processed_data(self):
        """Load pre-processed data from pickle file"""
        
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(f"Processed data file not found: {self.processed_data_path}")
        
        with open(self.processed_data_path, "rb") as f:
            data = pickle.load(f)
        
        # Load all necessary attributes
        self.processed_data = data["processed_data"]
        self.vocab = data["vocab"]
        self.id_to_token = data["id_to_token"]
        self.vocab_size = data["vocab_size"]
        self.mechanism_types = set(data["mechanism_types"])
        self.max_seq_len = data["max_seq_len"]
        
    def _count_mechanism_instances(self):
        """Count the number of instances for each mechanism type"""
        mechanism_counts = {}
        
        for sample in self.processed_data:
            mechanism_type = sample["mechanism_type"]
            mechanism_counts[mechanism_type] = mechanism_counts.get(mechanism_type, 0) + 1
        
        return mechanism_counts

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data[idx]

        # Return control points as input and DSL sequence as target
        return {
            "control_points": torch.FloatTensor(sample["control_points"]),
            "dsl_sequence": torch.LongTensor(sample["dsl_ids"]),
            "mechanism_type": sample["mechanism_type"],
        }


def create_validation_split(dataset, val_split=0.1, seed=42):
    """Create stratified train/validation split ensuring all mechanism types are represented"""
    np.random.seed(seed)
    
    # Group samples by mechanism type
    mechanism_groups = {}
    for idx in range(len(dataset)):
        sample = dataset.processed_data[idx]
        mech_type = sample["mechanism_type"]
        if mech_type not in mechanism_groups:
            mechanism_groups[mech_type] = []
        mechanism_groups[mech_type].append(idx)
    
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
    
    # Shuffle the final lists
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    print(f"Total: {len(train_indices)} train, {len(val_indices)} validation samples")
    
    return np.array(train_indices), np.array(val_indices)


def get_validation_dataset():
    """Get the exact same validation dataset used in training"""
    
    # Get the correct path to processed_data.pkl
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(script_dir)
    processed_data_path = os.path.join(mechanism_root, 'processed_data.pkl')
    
    # Load dataset with identical parameters to training
    dataset = MechanismDataset(processed_data_path=processed_data_path)

    
    # Create split with same parameters
    _, val_indices = create_validation_split(
        dataset, 
        val_split=0.1,  # Same as training default
        seed=42         # Same as training default
    )
    
    # Create validation subset
    from torch.utils.data import Subset
    val_dataset = Subset(dataset, val_indices)
    
    return val_dataset, dataset.vocab, dataset.id_to_token, dataset.vocab_size




def rotate_control_points(control_points, angle_degrees):
    """Rotate control points about origin
    
    Args:
        control_points: numpy array or torch tensor of shape (N, 2)
        angle_degrees: rotation angle in degrees
    
    Returns:
        Rotated points in same format as input
    """
    # Convert angle to radians
    angle = np.radians(angle_degrees)
    
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    
    # Convert to numpy if it's a tensor, preserving dtype
    if hasattr(control_points, 'numpy'):
        points_np = control_points.numpy()
    else:
        points_np = control_points
    
    # Perform rotation
    rotated_points = points_np @ rotation_matrix
    
    # Convert back to tensor if original was a tensor
    if hasattr(control_points, 'numpy'):
        import torch
        return torch.FloatTensor(rotated_points)
    else:
        return rotated_points


def evaluate_single_sample(sample_idx, validation_dataset, inference, angle=0, temperature=0.001):
    """Evaluate a single validation sample
    
    Args:
        sample_idx: Index of the sample to evaluate
        validation_dataset: Dataset containing the samples
        inference: MechanismInference instance
        angle: Rotation angle in degrees
        temperature: Temperature for model sampling (default: 0.8)
    
    Returns:
        tuple: (results dict or None, error message or None)
    """
    
    try:
        #print(f"\nStarting sample {sample_idx} angle {angle}°")
        sys.stdout.flush()  # Force the print to show immediately
        
        control_points = validation_dataset[sample_idx]["control_points"]
        control_points = rotate_control_points(control_points, angle)

        # Convert to NumPy array if it's a tensor
        if hasattr(control_points, 'numpy'):
            control_points_np = control_points.numpy()
        else:
            control_points_np = control_points
        
        #print("Generated control points")
        
        # Generate original B-spline trajectory for comparison
        degree = 3
        n = len(control_points_np)
        
        if n <= degree:
            return None, f"Not enough control points: {n} <= {degree}"
        
        knot_vector = np.concatenate((
            np.zeros(degree),
            np.linspace(0, 1, n - degree + 1),
            np.ones(degree)
        ))

        # Separate x and y
        cx = control_points_np[:, 0]
        cy = control_points_np[:, 1]
        bspline_x = BSpline(knot_vector, cx, degree)
        bspline_y = BSpline(knot_vector, cy, degree)

        # Evaluate the spline
        t_vals = np.linspace(0, 1, 200)
        bspline_trajectory = np.vstack([bspline_x(t_vals), bspline_y(t_vals)]).T
        
        #print("Generated B-spline trajectory")
        
        all_results = {
            'failed_count': 0,
            "dtw_distance": 0,
            "bspline_trajectory": [],
            "coords": [],
            "coupler_traj": [],
            "val_sample_idx": sample_idx,
            "angle": angle
        }
                
        #print("Generating mechanism parameters...")
        # Generate model output using the same method as curve_to_mech_new.py
        mechanism_params = inference.generate_mechanism_params(
            control_points, 
            temperature=temperature,
            top_k=5,  # Using same top_k as curve_to_mech_new.py
            mech_type=None,  # Let model decide mechanism type
            process_curve=False  # Use control points as-is
        )

        mech_type = mechanism_params["type"]
        points = mechanism_params.get("params", [])  # Safely get points with default empty list
        
        #print(f"Generated mechanism type: {mech_type}")
        
        if not mech_type or not points or len(points) < 3:
            all_results['failed_count'] += 1
            return all_results, "Invalid mechanism type or insufficient points"
        
        # Determine bar type
        if mech_type in ["RRRR", "PRPR", "RRRP", "RRPR", "RPPR", "RRPP"]:
            bar_type = "4bar"  
        elif mech_type.startswith("Steph") or mech_type.startswith("Watt"):
            bar_type = "6bar"
        else:
            # Skip 8-bar mechanisms or invalid types
            all_results['failed_count'] += 1
            return all_results, "Unsupported mechanism type"

        #print(f"Mechanism bar type: {bar_type}")

        # Apply condition: only take the first 5 coordinate pairs if bar_type is 4bar
        coords_to_use = points[:5] if bar_type == "4bar" else points
        
        # Validate coordinates are numeric
        try:
            flat_coords = [str(round(float(num), 2)) for pair in coords_to_use for num in pair]
        except (ValueError, TypeError):
            all_results['failed_count'] += 1
            return all_results, "Invalid coordinate values"
            
        coords = "_" + "_".join(flat_coords) + "_" + mech_type

        #print("Simulating mechanism...")
        # Initialize mechanism wrapper
        wrapper = MechanismWrapper(coords)
        wrapper.simulate()
        
        # Check if mechanism is valid (need more than 1 pose)
        if len(wrapper.poses) <= 1:
            all_results['failed_count'] += 1
            return all_results, "Invalid mechanism: insufficient poses"

        # Get coupler trajectory
        coupler_traj = wrapper.get_coupler_trajectory()
        if not coupler_traj or len(coupler_traj) == 0:
            all_results['failed_count'] += 1
            return all_results, "Failed to get coupler trajectory"
            
        coupler_traj = np.array(coupler_traj)
        #print("Generated coupler trajectory")

        # Calculate DTW distance
        #print("Calculating DTW distance...")
        optimal_path, dtw_distance = dtw_path(bspline_trajectory, coupler_traj)
        
        # Validate DTW result
        if np.isnan(dtw_distance) or np.isinf(dtw_distance):
            all_results['failed_count'] += 1
            return all_results, "Invalid DTW distance"
        
        all_results['dtw_distance'] = dtw_distance
        all_results['bspline_trajectory'].append(bspline_trajectory.tolist() if hasattr(bspline_trajectory, 'tolist') else bspline_trajectory)
        all_results['coords'] = coords
        all_results['coupler_traj'].append(coupler_traj.tolist())
        all_results['val_sample_idx'] = sample_idx
        all_results['angle'] = angle

        #print(f"Evaluation complete. DTW distance: {dtw_distance}")
        return all_results, None

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None, str(e)




# Calculate fixed plot limits that encompass both mechanism and curve
def get_plot_limits(curve_points, coupler_traj=None):
    """Calculate plot limits that encompass both the curve and mechanism trajectory
    
    Args:
        curve_points: List of points for the curve
        coupler_traj: List of points for the coupler trajectory (optional)
    """
    curve_points = np.array(curve_points)
    if coupler_traj is not None:
        coupler_traj = np.array(coupler_traj)
    
    all_x = []
    all_y = []
    
    # Add curve points
    all_x.extend(curve_points[:, 0])
    all_y.extend(curve_points[:, 1])
    
    # Add coupler trajectory if available
    if coupler_traj is not None:
        all_x.extend(coupler_traj[:, 0])
        all_y.extend(coupler_traj[:, 1])
    
    # Calculate limits with margin
    margin = 2.0
    x_min = min(all_x) - margin
    x_max = max(all_x) + margin
    y_min = min(all_y) - margin
    y_max = max(all_y) + margin
    
    #print(f"Calculated plot limits (filtered): X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")

    # x_min = -1
    # x_max = 1
    # y_min = -1
    # y_max = 1

    return [x_min, x_max, y_min, y_max]


def plot_results(results_dir):
    """Plot all results from the results directory overlaid on a single plot"""
    # Load all results files
    results_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    
    if not results_files:
        print("No result files found in", results_dir)
        return
        
    print(f"Found {len(results_files)} result files")
    
    # Sort files by angle to ensure consistent ordering
    results_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Create a single figure
    plt.figure(figsize=(12, 8))
    
    # Track plot limits
    all_x_min, all_x_max = float('inf'), float('-inf')
    all_y_min, all_y_max = float('inf'), float('-inf')
    
    # Define a color cycle
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results_files)))
    
    # Plot all curves
    for idx, file in enumerate(results_files):
        try:
            with open(file, "r") as f:
                results = json.load(f)
            
            # Extract data
            bspline_traj = np.array(results['bspline_trajectory'][0])
            coupler_traj = np.array(results['coupler_traj'][0])
            dtw_distance = results['dtw_distance']
            angle = int(file.split('_')[-1].split('.')[0])
            
            # Update plot limits
            all_x_min = min(all_x_min, bspline_traj[:, 0].min(), coupler_traj[:, 0].min())
            all_x_max = max(all_x_max, bspline_traj[:, 0].max(), coupler_traj[:, 0].max())
            all_y_min = min(all_y_min, bspline_traj[:, 1].min(), coupler_traj[:, 1].min())
            all_y_max = max(all_y_max, bspline_traj[:, 1].max(), coupler_traj[:, 1].max())
            
            # Plot with same color for each angle pair
            color = colors[idx]
            plt.plot(bspline_traj[:, 0], bspline_traj[:, 1], '-', color=color, label=f'{angle}° Target')
            plt.plot(coupler_traj[:, 0], coupler_traj[:, 1], '--', color=color, label=f'{angle}° Mechanism (DTW: {dtw_distance:.4f})')
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Add margin to limits
    margin = 2.0
    plt.xlim(all_x_min - margin, all_x_max + margin)
    plt.ylim(all_y_min - margin, all_y_max + margin)
    
    # Add grid and labels
    plt.grid(True)
    plt.title('Mechanism Output vs Target Curve at Different Angles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(results_dir, "combined_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {plot_file}")
    plt.close()


def process_sample_batch(sample_indices, validation_dataset, model_path, vocab_path, angles, temperature, results_dir):
    """Process a batch of samples in parallel, storing ALL results for each angle"""
    # Initialize model in the worker process
    inference = MechanismInference(model_path, vocab_path)
    
    batch_results = {}
    for sample_idx in sample_indices:
        sample_results = {
            'sample_idx': sample_idx,
            'angles': {},  # Store results for each angle
            'best_overall': {
                'dtw': float('inf'),
                'angle': None,
                'mechanism_type': None,
                'coords': None,
                'coupler_traj': None,
                'bspline_trajectory': None
            }
        }
        
        # Process each angle
        for angle in angles:
            all_results, error = evaluate_single_sample(
                sample_idx, 
                validation_dataset, 
                inference, 
                angle=angle,
                temperature=temperature
            )
            
            # Store results for this angle
            if error:
                sample_results['angles'][str(angle)] = {
                    'error': error,
                    'angle': angle
                }
            else:
                sample_results['angles'][str(angle)] = {
                    'dtw': all_results['dtw_distance'],
                    'coords': all_results['coords'],
                    'mechanism_type': all_results['coords'].split('_')[-1],
                    'coupler_traj': all_results['coupler_traj'],
                    'bspline_trajectory': all_results['bspline_trajectory'],
                    'angle': angle
                }
                
                # Update best overall if this is better
                if all_results['dtw_distance'] < sample_results['best_overall']['dtw']:
                    sample_results['best_overall'].update({
                        'dtw': all_results['dtw_distance'],
                        'angle': angle,
                        'mechanism_type': all_results['coords'].split('_')[-1],
                        'coords': all_results['coords'],
                        'coupler_traj': all_results['coupler_traj'],
                        'bspline_trajectory': all_results['bspline_trajectory']
                    })
        
        # Only store sample if we have any successful results
        if any('dtw' in angle_data for angle_data in sample_results['angles'].values()):
            batch_results[str(sample_idx)] = sample_results
            
    return batch_results

def print_batch_stats(batch_results, batch_num, total_batches, start_time):
    """Print statistics for the current batch"""
    if not batch_results:
        print(f"\nBatch {batch_num}/{total_batches}: No successful results")
        return
    
    # Count mechanism types and successes per angle
    mech_types = defaultdict(int)
    angle_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    
    for sample_result in batch_results.values():
        # Count best mechanism type
        if sample_result['best_overall']['mechanism_type']:
            mech_types[sample_result['best_overall']['mechanism_type']] += 1
        
        # Count successes and failures per angle
        for angle, angle_data in sample_result['angles'].items():
            angle_stats[angle]['total'] += 1
            if 'dtw' in angle_data:  # Successful run
                angle_stats[angle]['success'] += 1
    
    # Calculate time estimates
    elapsed_time = time.time() - start_time
    avg_time_per_batch = elapsed_time / batch_num
    remaining_batches = total_batches - batch_num
    estimated_remaining_time = remaining_batches * avg_time_per_batch
    
    # Get memory usage
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Print detailed statistics
    print(f"\nBatch {batch_num}/{total_batches} Statistics:")
    print(f"├── Success Rate: {len(batch_results)}/{len(batch_results)} samples")
    print(f"├── Angle Statistics:")
    for angle in sorted(angle_stats.keys()):
        stats = angle_stats[angle]
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"│   ├── {angle}°: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    print(f"├── Best Mechanism Types:")
    for mech_type, count in sorted(mech_types.items()):
        print(f"│   └── {mech_type}: {count}")
    print(f"├── Time:")
    print(f"│   ├── Elapsed: {datetime.timedelta(seconds=int(elapsed_time))}")
    print(f"│   └── Estimated Remaining: {datetime.timedelta(seconds=int(estimated_remaining_time))}")
    print(f"└── Memory Usage: {memory_usage:.1f} MB")

def print_cumulative_stats(all_results):
    """Print cumulative statistics for all processed samples"""
    if not all_results:
        print("\nNo results to report yet")
        return
    
    # Count mechanism types and collect angle statistics
    mech_types = defaultdict(int)
    angle_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'dtws': []})
    
    for sample_result in all_results.values():
        # Count best mechanism type
        if sample_result['best_overall']['mechanism_type']:
            mech_types[sample_result['best_overall']['mechanism_type']] += 1
        
        # Collect stats for each angle
        for angle, angle_data in sample_result['angles'].items():
            angle_stats[angle]['total'] += 1
            if 'dtw' in angle_data:  # Successful run
                angle_stats[angle]['success'] += 1
                angle_stats[angle]['dtws'].append(angle_data['dtw'])
    
    # Print overall statistics
    print("\nCumulative Statistics:")
    print(f"├── Total Samples Processed: {len(all_results)}")
    print(f"├── Angle Performance:")
    for angle in sorted(angle_stats.keys()):
        stats = angle_stats[angle]
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_dtw = np.mean(stats['dtws']) if stats['dtws'] else 0
        print(f"│   ├── {angle}°:")
        print(f"│   │   ├── Success Rate: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        print(f"│   │   └── Average DTW: {avg_dtw:.4f}")
    print(f"└── Mechanism Type Distribution:")
    for mech_type, count in sorted(mech_types.items()):
        percentage = (count / len(all_results)) * 100
        print(f"    └── {mech_type}: {count} ({percentage:.1f}%)")

def main():
    """
    Process samples in parallel batches for faster processing
    """
    print("Starting main function...")
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(mechanism_root, Config.MODEL_SAVE_PATH)
    vocab_path = os.path.join(mechanism_root, Config.VOCAB_SAVE_PATH)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(script_dir, "results_rotation")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    print("Loading validation dataset...")
    validation_dataset, vocab, id_to_token, vocab_size = get_validation_dataset()
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    temperature = 0.1

    n_validation_samples = 1000  # Process n samples
    
    # Calculate optimal batch size based on CPU cores
    num_processes = 7  # Leave one CPU free
    batch_size = 70  # Smaller batch size for more frequent updates
    print(f"\nSystem info:")
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    print(f"Using {num_processes} processes")
    print(f"Batch size: {batch_size} samples per batch")
    
    # Generate random sample indices
    np.random.seed(42)  # Fixed seed for reproducibility
    total_samples = len(validation_dataset)
    random_indices = [int(i) for i in np.random.choice(total_samples, n_validation_samples, replace=False)]
    
    # Split indices into batches
    batches = [random_indices[i:i + batch_size] for i in range(0, len(random_indices), batch_size)]
    print(f"Split {n_validation_samples} samples into {len(batches)} batches")
    
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create partial function with fixed arguments
    process_batch = partial(
        process_sample_batch,
        validation_dataset=validation_dataset,
        model_path=model_path,
        vocab_path=vocab_path,
        angles=angles,
        temperature=temperature,
        results_dir=results_dir
    )
    
    # Process batches with progress bar
    print(f"\nProcessing {n_validation_samples} samples in {len(batches)} batches...")
    all_best_results = {}
    start_time = time.time()
    
    try:
        for batch_num, batch_results in enumerate(tqdm.tqdm(pool.imap(process_batch, batches, chunksize=1), total=len(batches)), 1):
            # Update results
            all_best_results.update(batch_results)
            
            # Print batch and cumulative statistics
            print_batch_stats(batch_results, batch_num, len(batches), start_time)
            print_cumulative_stats(all_best_results)
            
            # Save intermediate results
            temp_save_path = os.path.join(results_dir, "all_best_results_temp.pkl")
            with open(temp_save_path, "wb") as f:
                pickle.dump(all_best_results, f)
    
    finally:
        pool.close()
        pool.join()
    
    # Save final results
    final_save_path = os.path.join(results_dir, "all_best_results.pkl")
    with open(final_save_path, "wb") as f:
        pickle.dump(all_best_results, f)
    
    # Print final statistics
    print("\nProcessing Complete!")
    print(f"Results saved to {final_save_path}")
    print("\nFinal Statistics:")
    print_cumulative_stats(all_best_results)
    print(f"\nTotal processing time: {datetime.timedelta(seconds=int(time.time() - start_time))}")

if __name__ == "__main__":
    main()
        