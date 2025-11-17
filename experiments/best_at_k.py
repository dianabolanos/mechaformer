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

from config import Config
from wrapper.mechanism_core import MechanismAnimator
from wrapper.mechanism_wrapper import MechanismWrapper
import matplotlib.animation as animation
from inference_mechanism import MechanismInference
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
import datetime 
import psutil
from requests.exceptions import RequestException
import asyncio
import concurrent.futures

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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


async def simulate_mechanism_async(wrapper):
    """Async wrapper for mechanism simulation"""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, wrapper.simulate)

async def evaluate_single_sample(sample_idx, validation_dataset, inference, temperature, run_idx):
    """Evaluate a single sample with a specific temperature."""
    
    try:
        # Set a different random seed for each run to ensure different results
        random_seed = (int(time.time() * 1000) + run_idx) % (2**32 - 1)  # Keep seed in valid range
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        control_points = validation_dataset[sample_idx]["control_points"]
        print(f"\nProcessing sample {sample_idx} (run {run_idx}, seed {random_seed})")  # Debug print

        # Convert to NumPy array if it's a tensor
        if hasattr(control_points, 'numpy'):
            control_points_np = control_points.numpy()
        else:
            control_points_np = control_points   
        
        # Generate original B-spline trajectory for comparison
        degree = 3
        n = len(control_points_np)
        
        if n <= degree:
            print(f"Error: Not enough control points: {n} <= {degree}")  # Debug print
            return None, {"error": "simulation_failure", "message": f"Not enough control points: {n} <= {degree}"}, None
        
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
        
        # Initialize results
        all_results = {
            'dtw_distance': float('inf'),
            'mechanism_type': None,
            'coords': None,
            'coupler_traj': None
        }

        # Single attempt at generating mechanism parameters
        try:
            print(f"Generating mechanism parameters for sample {sample_idx}")  # Debug print
            mechanism_params = inference.generate_mechanism_params(
                control_points, 
                temperature=temperature,
                top_k=5,
                mech_type=None,  # Let model choose mechanism type
                process_curve=False
            )
            print(f"Generated params: {mechanism_params}")  # Debug print
        except Exception as e:
            print(f"Error generating mechanism: {str(e)}")  # Debug print
            return None, {"error": "simulation_failure", "message": f"Failed to generate mechanism: {str(e)}"}, bspline_trajectory

        mech_type = mechanism_params["type"]
        points = mechanism_params.get("params", [])
        
        if not mech_type or not points or len(points) < 3:
            print(f"Invalid mechanism: type={mech_type}, points={len(points) if points else 0}")  # Debug print
            return None, {"error": "simulation_failure", "message": "Invalid mechanism type or insufficient points"}, bspline_trajectory
        
        # Determine bar type
        if mech_type in ["RRRR", "PRPR", "RRRP", "RRPR", "RPPR", "RRPP"]:
            bar_type = "4bar"  
        elif mech_type.startswith("Steph") or mech_type.startswith("Watt"):
            bar_type = "6bar"
        else:
            print(f"Unsupported mechanism type: {mech_type}")  # Debug print
            return None, {"error": "simulation_failure", "message": "Unsupported mechanism type"}, bspline_trajectory

        # Apply condition: only take the first 5 coordinate pairs if bar_type is 4bar
        coords_to_use = points[:5] if bar_type == "4bar" else points
        
        # Validate coordinates are numeric
        try:
            flat_coords = [str(round(float(num), 2)) for pair in coords_to_use for num in pair]
        except (ValueError, TypeError):
            print(f"Invalid coordinate values in: {coords_to_use}")  # Debug print
            return None, {"error": "simulation_failure", "message": "Invalid coordinate values"}, bspline_trajectory
            
        coords = "_" + "_".join(flat_coords) + "_" + mech_type
        print(f"Mechanism coords: {coords}")  # Debug print

        # Initialize mechanism wrapper
        wrapper = MechanismWrapper(coords)
        
        # Async simulate
        print(f"Starting simulation for sample {sample_idx}")  # Debug print
        await simulate_mechanism_async(wrapper)
        print(f"Simulation complete for sample {sample_idx}, poses: {len(wrapper.poses)}")  # Debug print
        
        # Check if mechanism is valid (need more than 1 pose)
        if len(wrapper.poses) <= 1:
            print(f"Invalid mechanism: insufficient poses ({len(wrapper.poses)})")  # Debug print
            return None, {"error": "simulation_failure", "message": "Invalid mechanism: insufficient poses"}, bspline_trajectory

        # Get coupler trajectory
        coupler_traj = wrapper.get_coupler_trajectory()
        if not coupler_traj or len(coupler_traj) == 0:
            print(f"Failed to get coupler trajectory")  # Debug print
            return None, {"error": "simulation_failure", "message": "Failed to get coupler trajectory"}, bspline_trajectory
            
        coupler_traj = np.array(coupler_traj)

        # Calculate DTW distance
        optimal_path, dtw_distance = dtw_path(bspline_trajectory, coupler_traj)
        print(f"DTW distance: {dtw_distance}")  # Debug print
        
        # Validate DTW result
        if np.isnan(dtw_distance) or np.isinf(dtw_distance):
            print(f"Invalid DTW distance: {dtw_distance}")  # Debug print
            return None, {"error": "simulation_failure", "message": "Invalid DTW distance"}, bspline_trajectory
        
        # Update results
        all_results.update({
            'dtw_distance': float(dtw_distance),
            'mechanism_type': mech_type,
            'coords': coords,
            'coupler_traj': coupler_traj.tolist() if hasattr(coupler_traj, 'tolist') else coupler_traj
        })
        print(f"Successfully processed sample {sample_idx}")  # Debug print

        return all_results, None, bspline_trajectory

    except Exception as e:
        print(f"Unexpected error processing sample {sample_idx}: {str(e)}")  # Debug print
        return None, {"error": "simulation_failure", "message": str(e)}, None

async def process_sample_batch_async(sample_indices, validation_dataset, model_path, vocab_path, temperatures, k_values):
    """Process a batch of samples in parallel, testing all k values for each sample.
    Reuses previous runs for higher k values."""
    
    # Initialize model in the worker process
    inference = MechanismInference(model_path, vocab_path)
    
    batch_results = {}
    
    for sample_idx in sample_indices:
        sample_results = {
            'sample_idx': sample_idx,
            'k_values': {},
            'best_overall': {
                'dtw': float('inf'),
                'k': None,
                'k_idx': None,
                'mechanism_type': None,
                'coords': None,
                'coupler_traj': None
            },
            'bspline_trajectory': None  # Store once per sample
        }

        # Store all runs for this sample
        all_runs = []
        current_k = 0
        
        # Process each k value, running only the additional needed samples
        for k in k_values:  # [16, 32]
            # Calculate how many new runs needed for this k
            new_runs_needed = k - current_k
            
            # Generate the additional needed samples
            # Create tasks for all new runs with unique run indices
            tasks = [
                evaluate_single_sample(
                    sample_idx, 
                    validation_dataset, 
                    inference,
                    temperature=temperatures[0],
                    run_idx=current_k + i  # Pass unique run index
                )
                for i in range(new_runs_needed)
            ]
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            for run_idx, (all_results, error, bspline_traj) in enumerate(results):
                # Store bspline_trajectory once per sample if we haven't already
                if sample_results['bspline_trajectory'] is None and bspline_traj is not None:
                    sample_results['bspline_trajectory'] = bspline_traj.tolist() if hasattr(bspline_traj, 'tolist') else bspline_traj
                
                if error or not all_results:
                    all_runs.append({'error': error, 'k_idx': current_k + run_idx})
                else:
                    # Store successful run
                    all_runs.append({
                        'k_idx': current_k + run_idx,
                        'dtw': all_results['dtw_distance'],
                        'mechanism_type': all_results['mechanism_type'],
                        'coords': all_results['coords'],
                        'coupler_traj': all_results['coupler_traj']
                    })
            
            # Update current_k
            current_k = k
            
            # Process results for this k value using all runs so far
            k_results = {
                'dtw_distances': [],
                'mechanism_types': [],
                'coords': [],
                'coupler_trajs': [],
                'all_results': all_runs[:k],  # Use all runs up to k
                'best_k': None,
                'best_dtw': float('inf')
            }
            
            # Process all runs up to this k
            for run in all_runs[:k]:
                if 'error' not in run:
                    dtw = run['dtw']
                    k_results['dtw_distances'].append(dtw)
                    k_results['mechanism_types'].append(run['mechanism_type'])
                    k_results['coords'].append(run['coords'])
                    k_results['coupler_trajs'].append(run['coupler_traj'])
                    
                    # Update best for this k value
                    if dtw < k_results['best_dtw']:
                        k_results['best_dtw'] = dtw
                        k_results['best_k'] = run['k_idx']
                    
                    # Update best overall
                    if dtw < sample_results['best_overall']['dtw']:
                        sample_results['best_overall'].update({
                            'dtw': dtw,
                            'k': k,
                            'k_idx': run['k_idx'],
                            'mechanism_type': run['mechanism_type'],
                            'coords': run['coords'],
                            'coupler_traj': run['coupler_traj']
                        })
            
            sample_results['k_values'][str(k)] = k_results
        
        if any(k_results['dtw_distances'] for k_results in sample_results['k_values'].values()):
            batch_results[str(sample_idx)] = sample_results
            
    return batch_results

def process_batch_wrapper(args):
    """Wrapper function to run async batch processing in a process pool"""
    sample_indices, validation_dataset, model_path, vocab_path, temperatures, k_values = args
    return asyncio.run(process_sample_batch_async(
        sample_indices, validation_dataset, model_path, vocab_path, temperatures, k_values
    ))

def print_batch_stats(batch_results, batch_num, total_batches, start_time):
    """Print statistics for the current batch"""
    if not batch_results:
        print(f"\nBatch {batch_num}/{total_batches}: No successful results")
        return
    
    # Count mechanism types and collect error statistics
    mech_types = defaultdict(int)
    total_attempts = defaultdict(int)  # k_value -> total attempts
    failed_attempts = defaultdict(int)  # k_value -> failed attempts
    successful_attempts = defaultdict(int)  # k_value -> successful attempts
    
    for result in batch_results.values():
        # Count successful mechanism types
        if result['best_overall']['mechanism_type']:
            mech_types[result['best_overall']['mechanism_type']] += 1
            
        # Collect statistics for each k value
        for k, k_data in result['k_values'].items():
            k_int = int(k)
            total_attempts[k_int] += len(k_data['all_results'])
            
            for run_result in k_data['all_results']:
                if 'error' in run_result:
                    failed_attempts[k_int] += 1
                else:
                    successful_attempts[k_int] += 1
    
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
    print(f"├── Statistics by k value:")
    for k in sorted(total_attempts.keys()):
        success_rate = (successful_attempts[k] / total_attempts[k] * 100) if total_attempts[k] > 0 else 0
        fail_rate = (failed_attempts[k] / total_attempts[k] * 100) if total_attempts[k] > 0 else 0
        print(f"│   ├── k={k}:")
        print(f"│   │   ├── Total attempts: {total_attempts[k]}")
        print(f"│   │   ├── Successful: {successful_attempts[k]} ({success_rate:.1f}%)")
        print(f"│   │   └── Failed: {failed_attempts[k]} ({fail_rate:.1f}%)")
    print(f"├── Best Mechanism Types:")
    for mech_type, count in sorted(mech_types.items()):
        print(f"│   └── {mech_type}: {count}")
    print(f"├── Time:")
    print(f"│   ├── Elapsed: {datetime.timedelta(seconds=int(elapsed_time))}")
    print(f"│   └── Estimated Remaining: {datetime.timedelta(seconds=int(estimated_remaining_time))}")
    print(f"└── Memory Usage: {memory_usage:.1f} MB")

def print_cumulative_stats(all_results):
    """Print cumulative statistics for all processed samples"""
    if not all_results['detailed_results']:
        print("\nNo results to report yet")
        return
    
    # Count mechanism types and k values
    mech_types = defaultdict(int)
    best_dtw_by_k = defaultdict(list)  # k -> list of best DTWs for that k value
    
    for result in all_results['detailed_results']:
        # Count overall best mechanism type
        if result['best_overall']['mechanism_type']:
            mech_types[result['best_overall']['mechanism_type']] += 1
        
        # For each k value, get the best DTW achieved
        for k_str, k_data in result['k_values'].items():
            k = int(k_str)
            if k_data['best_dtw'] != float('inf'):
                best_dtw_by_k[k].append(k_data['best_dtw'])
    
    # Print overall statistics
    print("\nCumulative Statistics:")
    print(f"├── Total Samples Processed: {len(all_results['detailed_results'])}")
    print(f"├── K Value Performance:")
    for k in sorted(best_dtw_by_k.keys()):
        dtws = best_dtw_by_k[k]
        count = len(dtws)
        if count > 0:
            percentage = (count / len(all_results['detailed_results'])) * 100
            avg_dtw = np.mean(dtws)
            min_dtw = min(dtws)
            print(f"│   └── k={k}: {count} samples ({percentage:.1f}%) - Avg Best DTW: {avg_dtw:.4f}, Best DTW: {min_dtw:.4f}")
    print(f"└── Overall Mechanism Type Distribution:")
    for mech_type, count in sorted(mech_types.items()):
        percentage = (count / len(all_results['detailed_results'])) * 100
        print(f"    └── {mech_type}: {count} ({percentage:.1f}%)")

def save_results(results, results_dir):
    """Save the results to files."""
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Split results into detailed and summary
    detailed_results = {
        'detailed_results': results['detailed_results']
    }
    
    # Convert defaultdict to regular dict for mechanism_type_counts
    mechanism_type_counts = {}
    for k, counts in results['mechanism_type_counts'].items():
        mechanism_type_counts[k] = dict(counts)
    
    # Collect best DTW values for each sample
    best_dtw_values = []
    for result in results['detailed_results']:
        if result['best_overall']['dtw'] is not None:
            best_dtw_values.append({
                'sample_idx': result['sample_idx'],
                'dtw': result['best_overall']['dtw'],
                'k': result['best_overall']['k'],
                'mechanism_type': result['best_overall']['mechanism_type']
            })
    
    summary_results = {
        'temperatures': results['temperatures'],
        'k_values': results['k_values'],
        'n_samples': results['n_samples'],
        'mechanism_type_counts': mechanism_type_counts,
        'best_dtw_values': best_dtw_values,
        'dtw_statistics': {
            'mean': np.mean([d['dtw'] for d in best_dtw_values]) if best_dtw_values else None,
            'median': np.median([d['dtw'] for d in best_dtw_values]) if best_dtw_values else None,
            'std': np.std([d['dtw'] for d in best_dtw_values]) if best_dtw_values else None,
            'min': min([d['dtw'] for d in best_dtw_values]) if best_dtw_values else None,
            'max': max([d['dtw'] for d in best_dtw_values]) if best_dtw_values else None
        }
    }
    
    # Save detailed results
    detailed_path = os.path.join(results_dir, 'detailed_results.pkl')
    with open(detailed_path, 'wb') as f:
        pickle.dump(detailed_results, f)
    print(f"\nDetailed results saved to '{detailed_path}'")
    
    # Save summary results
    summary_path = os.path.join(results_dir, 'summary_results.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary_results, f)
    print(f"Summary results saved to '{summary_path}'")
    
    # Save intermediate results during processing
    # Convert defaultdict to dict for full results save
    results_copy = dict(results)
    results_copy['mechanism_type_counts'] = mechanism_type_counts
    temp_save_path = os.path.join(results_dir, "all_results_temp.pkl")
    with open(temp_save_path, "wb") as f:
        pickle.dump(results_copy, f)

def main():
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(mechanism_root, Config.MODEL_SAVE_PATH)
    vocab_path = os.path.join(mechanism_root, Config.VOCAB_SAVE_PATH)

    # Create results directory in experiments folder
    results_dir = os.path.join(script_dir, "results-best-at-k")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    validation_dataset, vocab, id_to_token, vocab_size = get_validation_dataset()
    
    # Study parameters
    temperatures = [0.1]  # Keep temperature constant
    k_values = [1]  # Vary k values
    n_validation_samples = 10  # Total samples to process
    
    print(f"\n=== K Value Variation Study (All Results) ===")
    print(f"Testing {n_validation_samples} samples")
    print(f"Temperature: {temperatures[0]}")
    print(f"k values: {k_values}")
    print(f"Total predictions per sample: {max(k_values)}") 

    # Calculate optimal batch size and processes
    num_processes = 7  # Use 7 of 16 available cores
    batch_size = 10  # Process 10 samples at a time, each with all k values
    
    print(f"\nSystem info:")
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    print(f"Using {num_processes} processes")
    print(f"Batch size: {batch_size} samples per batch")
    print(f"Total evaluations per sample: {max(k_values)}")
    
    # Generate random sample indices
    np.random.seed(42)  # Fixed seed for reproducibility
    total_samples = len(validation_dataset)
    random_indices = [int(i) for i in np.random.choice(total_samples, n_validation_samples, replace=False)]
    
    # Split indices into batches
    batches = [random_indices[i:i + batch_size] for i in range(0, len(random_indices), batch_size)]
    print(f"Split {n_validation_samples} samples into {len(batches)} batches")
    
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create batch arguments
    batch_args = [
        (batch, validation_dataset, model_path, vocab_path, temperatures, k_values)
        for batch in batches
    ]
    
    # Process all batches
    results = {
        'temperatures': temperatures,
        'k_values': k_values,
        'n_samples': n_validation_samples,
        'detailed_results': [],
        'mechanism_type_counts': defaultdict(lambda: defaultdict(int))
    }
    
    start_time = time.time()
    print(f"\nProcessing {n_validation_samples} samples in {len(batches)} batches...")
    
    try:
        for batch_num, batch_results in enumerate(tqdm.tqdm(pool.imap(process_batch_wrapper, batch_args, chunksize=1), total=len(batches)), 1):
            if batch_results:  # Only process if we have results
                # Update results
                for sample_result in batch_results.values():
                    results['detailed_results'].append(sample_result)
                    
                    # Update mechanism type counts for each k value
                    for k, k_data in sample_result['k_values'].items():
                        for mech_type in k_data['mechanism_types']:
                            results['mechanism_type_counts'][int(k)][mech_type] += 1
            
            # Print batch and cumulative statistics
            print_batch_stats(batch_results, batch_num, len(batches), start_time)
            print_cumulative_stats(results)
            
            # Save intermediate results
            save_results(results, results_dir)
    
    finally:
        pool.close()
        pool.join()
    
    # Print final comparative statistics
    print("\nProcessing Complete!")
    print(f"Results saved to {results_dir}")
    
    # Print statistics for each k value
    for k in k_values:
        print(f"\nk = {k}:")
        # Count mechanism types and best DTWs for this k
        best_dtws = []
        mechanism_types = defaultdict(int)
        successful_samples = 0
        
        for result in results['detailed_results']:
            if str(k) in result['k_values']:
                k_data = result['k_values'][str(k)]
                if k_data['best_dtw'] != float('inf'):
                    best_dtws.append(k_data['best_dtw'])
                    successful_samples += 1
                    # Get mechanism type from the best run for this k
                    if k_data['mechanism_types']:  # Check if we have any successful mechanisms
                        best_idx = k_data['best_k']
                        if best_idx is not None and best_idx < len(k_data['mechanism_types']):
                            best_mech_type = k_data['mechanism_types'][best_idx]
                            mechanism_types[best_mech_type] += 1
        
        # Print statistics
        if successful_samples > 0:
            percentage = (successful_samples / n_validation_samples) * 100
            avg_best_dtw = np.mean(best_dtws) if best_dtws else 0
            min_dtw = min(best_dtws) if best_dtws else 0
            print(f"├── Successful samples: {successful_samples} ({percentage:.1f}%)")
            print(f"├── Average Best DTW: {avg_best_dtw:.4f}")
            print(f"├── Overall Best DTW: {min_dtw:.4f}")
            print(f"└── Best Mechanism Types:")
            for mech_type, mech_count in sorted(mechanism_types.items()):
                mech_percentage = (mech_count / successful_samples) * 100
                print(f"    └── {mech_type}: {mech_count} ({mech_percentage:.1f}%)")
        else:
            print("└── No successful samples")
    
    # Save final results
    save_results(results, results_dir)

if __name__ == "__main__":
    main() 