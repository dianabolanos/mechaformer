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
from wrapper.mechanism_core import MechanismAnimator, MechanismClassifier
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
from scipy.interpolate import splprep, splev
import multiprocessing
from functools import partial
import tqdm
import datetime 
import psutil
from requests.exceptions import RequestException

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



def evaluate_single_sample(sample_idx, validation_dataset, inference, mechanism_type, temperature=0.001):
    """Evaluate a single validation sample using different mechanisms
    """
    
    try:
        control_points = validation_dataset[sample_idx]["control_points"]

        # Convert to NumPy array if it's a tensor
        if hasattr(control_points, 'numpy'):
            control_points_np = control_points.numpy()
        else:
            control_points_np = control_points   
        
        # Generate original B-spline trajectory for comparison
        degree = 3
        n = len(control_points_np)
        
        if n <= degree:
            return None, f"Not enough control points: {n} <= {degree}", None
        
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
        
        # Initialize results dictionary with attempt-specific data
        attempt_data = {
            "sample_idx": sample_idx,
            "mechanism_type": mechanism_type,
            "bspline_trajectory": bspline_trajectory.tolist() if hasattr(bspline_trajectory, 'tolist') else bspline_trajectory,
            "control_points": control_points_np.tolist() if hasattr(control_points_np, 'tolist') else control_points_np,
            "success": False,
            "error": None,
            "dtw_distance": None,
            "coords": None,
            "coupler_trajectory": None
        }
                
        # Generate model output
        try:
            mechanism_params = inference.generate_mechanism_params(
                control_points, 
                temperature=temperature,
                top_k=5,
                mech_type=mechanism_type,
                process_curve=False
            )
        except Exception as e:
            attempt_data["error"] = f"Model inference failed: {str(e)}"
            return None, attempt_data["error"], attempt_data

        mech_type = mechanism_params["type"]
        points = mechanism_params.get("params", [])
        
        if not mech_type or not points or len(points) < 3:
            attempt_data["error"] = "Invalid mechanism type or insufficient points"
            return None, attempt_data["error"], attempt_data
        
        # Determine bar type
        if mech_type in ["RRRR", "PRPR", "RRRP", "RRPR", "RPPR", "RRPP"]:
            bar_type = "4bar"  
        elif mech_type.startswith("Steph") or mech_type.startswith("Watt"):
            bar_type = "6bar"
        else:
            attempt_data["error"] = "Unsupported mechanism type"
            return None, attempt_data["error"], attempt_data

        # Apply condition: only take the first 5 coordinate pairs if bar_type is 4bar
        coords_to_use = points[:5] if bar_type == "4bar" else points
        
        # Validate coordinates are numeric
        try:
            flat_coords = [str(round(float(num), 2)) for pair in coords_to_use for num in pair]
        except (ValueError, TypeError):
            attempt_data["error"] = "Invalid coordinate values"
            return None, attempt_data["error"], attempt_data
            
        coords = "_" + "_".join(flat_coords) + "_" + mech_type
        attempt_data["coords"] = coords

        # Initialize mechanism wrapper
        wrapper = MechanismWrapper(coords)
        wrapper.simulate()
        
        # Check if mechanism is valid (need more than 1 pose)
        if len(wrapper.poses) <= 1:
            attempt_data["error"] = "Invalid mechanism: insufficient poses"
            return None, attempt_data["error"], attempt_data

        # Get coupler trajectory
        coupler_traj = wrapper.get_coupler_trajectory()
        if not coupler_traj or len(coupler_traj) == 0:
            attempt_data["error"] = "Failed to get coupler trajectory"
            return None, attempt_data["error"], attempt_data
            
        coupler_traj = np.array(coupler_traj)
        attempt_data["coupler_trajectory"] = coupler_traj.tolist()

        # Calculate DTW distance
        optimal_path, dtw_distance = dtw_path(bspline_trajectory, coupler_traj)
        
        # Validate DTW result
        if np.isnan(dtw_distance) or np.isinf(dtw_distance):
            attempt_data["error"] = "Invalid DTW distance"
            return None, attempt_data["error"], attempt_data
        
        # Update successful attempt data
        attempt_data["success"] = True
        attempt_data["dtw_distance"] = float(dtw_distance)

        return attempt_data, None, attempt_data

    except Exception as e:
        if 'attempt_data' not in locals():
            attempt_data = {
                "sample_idx": sample_idx,
                "mechanism_type": mechanism_type,
                "success": False,
                "error": str(e)
            }
        else:
            attempt_data["error"] = str(e)
        return None, str(e), attempt_data


def process_sample_batch(sample_indices, validation_dataset, model_path, vocab_path, mechanism_types, temperature, results_dir):
    """Process a batch of samples in parallel"""
    try:
        # Initialize model in the worker process
        inference = MechanismInference(model_path, vocab_path)
        
        batch_results = {}
        for sample_idx in sample_indices:
            best_dtw = float('inf')
            best_results = None
            best_mechanism_type = None
            
            # Track all attempts for this sample
            sample_results = {
                'attempts': {},  # mechanism_type -> attempt_data
                'best_dtw': None,
                'best_mechanism_type': None
            }
            
            for mechanism_type in mechanism_types:
                attempt_data, error, _ = evaluate_single_sample(
                    sample_idx, 
                    validation_dataset, 
                    inference, 
                    mechanism_type=mechanism_type,
                    temperature=temperature
                )
                
                # Store attempt data regardless of success/failure
                sample_results['attempts'][mechanism_type] = attempt_data
                
                if not error and attempt_data and attempt_data['dtw_distance'] < best_dtw:
                    best_dtw = attempt_data['dtw_distance']
                    best_results = attempt_data
                    best_mechanism_type = mechanism_type
            
            if best_results is not None:
                sample_results['best_dtw'] = best_dtw
                sample_results['best_mechanism_type'] = best_mechanism_type
                
            # Store all results for this sample
            batch_results[str(sample_idx)] = sample_results
                
        return batch_results
    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        return {}


def print_batch_stats(batch_results, batch_num, total_batches, start_time):
    """Print statistics for the current batch"""
    if not batch_results:
        print(f"\nBatch {batch_num}/{total_batches}: No results")
        return
    
    # Count successful and failed attempts
    success_by_type = defaultdict(int)
    failure_by_type = defaultdict(int)
    failure_reasons = defaultdict(lambda: defaultdict(int))
    total_attempts = 0
    successful_samples = 0
    
    for sample_results in batch_results.values():
        if sample_results['best_mechanism_type'] is not None:
            successful_samples += 1
            
        for mech_type, attempt in sample_results['attempts'].items():
            total_attempts += 1
            if attempt is None:
                failure_by_type[mech_type] += 1
                failure_reasons[mech_type]["Unknown error (attempt is None)"] += 1
                continue
                
            if attempt.get('success', False):
                success_by_type[mech_type] += 1
            else:
                failure_by_type[mech_type] += 1
                error_msg = attempt.get('error', "Unknown error")
                failure_reasons[mech_type][error_msg] += 1
    
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
    print(f"├── Samples with Valid Results: {successful_samples}/{len(batch_results)}")
    print(f"├── Total Attempts: {total_attempts}")
    print(f"├── Success by Mechanism Type:")
    for mech_type in sorted(success_by_type.keys()):
        success_count = success_by_type[mech_type]
        fail_count = failure_by_type[mech_type]
        total = success_count + fail_count
        success_rate = (success_count / total) * 100 if total > 0 else 0
        print(f"│   └── {mech_type}: {success_count}/{total} ({success_rate:.1f}%)")
    print(f"├── Failure Reasons by Type:")
    for mech_type in sorted(failure_reasons.keys()):
        if failure_reasons[mech_type]:
            print(f"│   ├── {mech_type}:")
            for reason, count in failure_reasons[mech_type].items():
                print(f"│   │   └── {reason}: {count}")
    print(f"├── Time:")
    print(f"│   ├── Elapsed: {datetime.timedelta(seconds=int(elapsed_time))}")
    print(f"│   └── Estimated Remaining: {datetime.timedelta(seconds=int(estimated_remaining_time))}")
    print(f"└── Memory Usage: {memory_usage:.1f} MB")

def print_cumulative_stats(all_best_results):
    """Print cumulative statistics for all processed samples"""
    if not all_best_results:
        print("\nNo results to report yet")
        return
    
    # Count successful and failed attempts
    success_by_type = defaultdict(int)
    failure_by_type = defaultdict(int)
    failure_reasons = defaultdict(lambda: defaultdict(int))
    total_attempts = 0
    successful_samples = 0
    
    for sample_results in all_best_results.values():
        if sample_results['best_mechanism_type'] is not None:
            successful_samples += 1
            
        for mech_type, attempt in sample_results['attempts'].items():
            total_attempts += 1
            if attempt is None:
                failure_by_type[mech_type] += 1
                failure_reasons[mech_type]["Unknown error (attempt is None)"] += 1
                continue
                
            if attempt.get('success', False):
                success_by_type[mech_type] += 1
            else:
                failure_by_type[mech_type] += 1
                error_msg = attempt.get('error', "Unknown error")
                failure_reasons[mech_type][error_msg] += 1

    # Print overall statistics
    print("\nCumulative Statistics:")
    print(f"├── Total Samples Processed: {len(all_best_results)}")
    print(f"├── Samples with Valid Results: {successful_samples}")
    print(f"├── Total Attempts: {total_attempts}")
    print(f"├── Success by Mechanism Type:")
    for mech_type in sorted(success_by_type.keys()):
        success_count = success_by_type[mech_type]
        fail_count = failure_by_type[mech_type]
        total = success_count + fail_count
        success_rate = (success_count / total) * 100 if total > 0 else 0
        print(f"│   └── {mech_type}: {success_count}/{total} ({success_rate:.1f}%)")
    print(f"└── Failure Reasons by Type:")
    for mech_type in sorted(failure_reasons.keys()):
        if failure_reasons[mech_type]:
            print(f"    ├── {mech_type}:")
            for reason, count in failure_reasons[mech_type].items():
                percentage = (count / failure_by_type[mech_type]) * 100
                print(f"    │   └── {reason}: {count} ({percentage:.1f}%)")

def main():
    print("Starting main function...")
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(mechanism_root, Config.MODEL_SAVE_PATH)
    vocab_path = os.path.join(mechanism_root, Config.VOCAB_SAVE_PATH)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(script_dir, "results_topology")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # pass this to evaluate_single_sample
    print("Loading validation dataset...")
    validation_dataset, vocab, id_to_token, vocab_size = get_validation_dataset()
    
    four_bar_mechanism_types = [
    "RRRR", "RRRP", "RRPR"
    ]

    six_bar_mechanism_types = [
    "Steph1T1", "Steph1T2", "Steph1T3",
    "Steph2T1A1", "Steph2T1A2", "Steph2T2A1", "Steph2T2A2",
    "Steph3T1A1", "Steph3T1A2", "Steph3T2A1", "Steph3T2A2",
    "Watt1T1A1", "Watt1T1A2", "Watt1T2A1", "Watt1T2A2", "Watt1T3A1", "Watt1T3A2",
    "Watt2T1A1", "Watt2T1A2", "Watt2T2A1", "Watt2T2A2"
    ]

    temperature = 0.1
    # Start with a smaller sample size for testing
    n_validation_samples = 1000  # Start with 5 samples first
    batch_size = 50  # Increased to 50 since GPU utilization is very low (2%)
    num_processes = 6  # Keeping at 6 based on RAM usage
    
    print(f"\nSystem info:")
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    print(f"Using {num_processes} processes")
    print(f"Batch size: {batch_size} samples per batch")
    print(f"Total server calls that will be made: {n_validation_samples * len(six_bar_mechanism_types)}")
    print(f"Number of batches: {len(range(0, n_validation_samples, batch_size))}")

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
        mechanism_types=four_bar_mechanism_types,
        temperature=temperature,
        results_dir=results_dir
    )

    print(f"\nProcessing {n_validation_samples} samples in {len(batches)} batches...")
    all_results = {}
    start_time = time.time()
 
    try:
        for batch_num, batch_results in enumerate(tqdm.tqdm(pool.imap(process_batch, batches, chunksize=1), total=len(batches)), 1):
            # Update results
            all_results.update(batch_results)
            
            # Print batch and cumulative statistics
            print_batch_stats(batch_results, batch_num, len(batches), start_time)
            print_cumulative_stats(all_results)
            
            # Save intermediate results
            temp_save_path = os.path.join(results_dir, "all_results_temp.pkl")
            with open(temp_save_path, "wb") as f:
                pickle.dump(all_results, f)
    
    finally:
        pool.close()
        pool.join()
    
    # Save final results
    final_save_path = os.path.join(results_dir, "all_results.pkl")
    with open(final_save_path, "wb") as f:
        pickle.dump(all_results, f)
    
    # Print final statistics
    print("\nProcessing Complete!")
    print(f"Results saved to {final_save_path}")
    print("\nFinal Statistics:")
    print_cumulative_stats(all_results)
    print(f"\nTotal processing time: {datetime.timedelta(seconds=int(time.time() - start_time))}")



if __name__ == "__main__":
    main()
        