import numpy as np
import os
import glob
import sys
import pickle
from tqdm import tqdm
import argparse
from pathlib import Path
from bspline_utils import BSplineCurveProcessor
from config import Config
import multiprocessing as mp
from functools import partial
import concurrent.futures
from collections import defaultdict
import random
from ground_joint_utils import GroundJointNormalizer


class MechanismDataPreprocessor:
    def __init__(self, data_path, max_seq_len=256, max_files=None, random_seed=42):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.max_files = max_files
        self.random_seed = random_seed
        self.data_files = []
        self.mechanism_types = set()
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize B-spline processor
        self.bspline_processor = BSplineCurveProcessor(
            num_control_points=Config.BSPLINE_CONTROL_POINTS,
            degree=Config.BSPLINE_DEGREE,
            smoothing=Config.BSPLINE_SMOOTHING,
        )
        
        # Initialize ground joint normalizer
        self.ground_joint_normalizer = GroundJointNormalizer()
        
        # Initialize vocabulary
        self._build_initial_vocabulary()

    def _collect_data_files(self):
        """Collect all .npy files from the dataset"""
        print("Collecting data files...")

        all_files = []
        
        # Flag to control 8-bar mechanism inclusion
        INCLUDE_8BAR = False  # Set to True to include 8-bar mechanisms
        
        # Look for npy files in different directories
        bar_types = ["4bar-npy", "6bar-npy"]
        if INCLUDE_8BAR:
            bar_types.append("8bar-npy")
            
        for bar_type in bar_types:
            base_path = os.path.join(self.data_path, bar_type)
            if os.path.exists(base_path):
                # Find outputs directory
                outputs_dir = None
                for item in sorted(os.listdir(base_path)):
                    if item.startswith("outputs-"):
                        outputs_dir = os.path.join(base_path, item)
                        break

                if outputs_dir and os.path.exists(outputs_dir):
                    # Find mechanism type directories
                    for mech_type in sorted(os.listdir(outputs_dir)):
                        mech_dir = os.path.join(outputs_dir, mech_type)
                        if os.path.isdir(mech_dir):
                            # Collect .npy files
                            npy_files = sorted(glob.glob(os.path.join(mech_dir, "*.npy")))
                            for npy_file in npy_files:
                                all_files.append((npy_file, mech_type, bar_type))
                                self.mechanism_types.add(mech_type)

        # Limit number of files if specified - ensure diversity
        if self.max_files is not None and len(all_files) > self.max_files:
            print(f"Limiting to {self.max_files} files (out of {len(all_files)} total)")
            
            # Group files by mechanism type for balanced sampling
            files_by_type = {}
            for file_info in all_files:
                mech_type = file_info[1]
                if mech_type not in files_by_type:
                    files_by_type[mech_type] = []
                files_by_type[mech_type].append(file_info)
            
            # Sample proportionally from each mechanism type
            self.data_files = []
            files_per_type = self.max_files // len(files_by_type)
            remaining_files = self.max_files % len(files_by_type)
            
            for i, (mech_type, files) in enumerate(sorted(files_by_type.items())):
                # Add extra file to first few types to account for remainder
                n_files = files_per_type + (1 if i < remaining_files else 0)
                n_files = min(n_files, len(files))  # Don't exceed available files
                sampled_files = random.sample(files, n_files)
                self.data_files.extend(sampled_files)
                print(f"  {mech_type}: {n_files} files")
        else:
            self.data_files = all_files
            
        # Update mechanism types based on sampled data
        self.mechanism_types = set()
        for _, mech_type, _ in self.data_files:
            self.mechanism_types.add(mech_type)

        print(f"Found {len(self.data_files)} data files")
        print(f"Mechanism types: {sorted(self.mechanism_types)}")

    def _build_initial_vocabulary(self):
        """Build initial vocabulary for DSL tokens"""
        # Special tokens
        self.vocab = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }

        # DSL structure tokens
        dsl_tokens = [
            "MECH_TYPE:",
            "POINTS:",
            "X:",
            "Y:",
        ]

        # Add tokens to vocabulary
        for token in dsl_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                
        # Pre-allocate all coordinate bin tokens
        for bin_idx in range(Config.COORD_BINS):
            bin_token = f"BIN_{bin_idx}"
            if bin_token not in self.vocab:
                self.vocab[bin_token] = len(self.vocab)

    def _finalize_vocabulary(self):
        """Finalize vocabulary after processing all data"""
        # Add mechanism types to vocabulary
        for mech_type in sorted(list(self.mechanism_types)):
            if mech_type not in self.vocab:
                self.vocab[mech_type] = len(self.vocab)

        # Create reverse vocabulary
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        print(f"Final vocabulary size: {self.vocab_size}")

    def _parse_filename(self, filename):
        """Parse mechanism parameters from filename"""
        basename = os.path.basename(filename).replace(".npy", "")
        parts = basename.strip("_").split("_")

        # Extract components based on mechanism type
        try:
            # Find mechanism type (non-numeric part)
            mech_type_idx = None
            for i, part in enumerate(parts):
                if not self._is_numeric(part) and len(part) > 2:
                    mech_type_idx = i
                    break

            if mech_type_idx is None:
                return None

            # Extract initial coordinates (before mechanism type)
            coord_parts = parts[:mech_type_idx]
            coords = [float(p) for p in coord_parts if self._is_numeric(p)]

            # Extract mechanism type
            mech_type = parts[mech_type_idx]

            # Extract normalization matrix (after mechanism type)
            norm_parts = parts[mech_type_idx + 1:]
            norm_matrix = [float(p) for p in norm_parts if self._is_numeric(p)]

            return {
                "initial_coords": coords,
                "mechanism_type": mech_type,
                "normalization_matrix": norm_matrix[:6] if len(norm_matrix) >= 6 else norm_matrix,
            }
        except:
            return None

    def _is_numeric(self, s):
        """Check if string represents a number"""
        try:
            float(s)
            return True
        except:
            return False

    def _coords_to_dsl(self, coords, mechanism_type, curve_points, apply_normalization=True):
        """Convert mechanism parameters to DSL representation with ground joint normalization"""
        dsl_tokens = []

        # Start with mechanism type
        dsl_tokens.extend(["<SOS>", "MECH_TYPE:", mechanism_type])

        # Handle mechanism types containing "P" - only use first 5 coordinates
        if 'P' in mechanism_type:
            coords = coords[:10]

        # Apply ground joint normalization if enabled
        if apply_normalization:
            try:
                # Normalize mechanism
                norm_result = self.ground_joint_normalizer.normalize_mechanism(
                    coords, mechanism_type, curve_points
                )
                normalized_coords = norm_result['normalized_coords']
                coords_to_use = normalized_coords
            except Exception as e:
                print(f"Warning: Could not normalize {mechanism_type}: {e}")
                sys.exit(1)

        else:
            coords_to_use = coords

        # Add points section
        dsl_tokens.append("POINTS:")

        # Convert coordinates to point definitions
        if len(coords_to_use) >= 2:
            num_points = len(coords_to_use) // 2
            for i in range(num_points):
                x, y = coords_to_use[i * 2], coords_to_use[i * 2 + 1]
                dsl_tokens.extend([
                    f"P{i}",
                    "X:",
                    self._quantize_coord(x),
                    "Y:",
                    self._quantize_coord(y),
                ])

        dsl_tokens.append("<EOS>")
        return dsl_tokens

    def _quantize_coord(self, coord):
        """Quantize coordinate to fixed bins to control vocabulary size"""
        # Clip coordinate to valid range
        coord_clipped = np.clip(coord, Config.COORD_MIN, Config.COORD_MAX)
        
        # Map to bin index
        bin_size = (Config.COORD_MAX - Config.COORD_MIN) / Config.COORD_BINS
        bin_idx = int((coord_clipped - Config.COORD_MIN) / bin_size)
        
        # Handle edge case where coord equals COORD_MAX
        if bin_idx >= Config.COORD_BINS:
            bin_idx = Config.COORD_BINS - 1
            
        return f"BIN_{bin_idx}"

    def _process_curve_points(self, curve_points):
        """Process curve points using B-spline fitting"""
        # Skip empty or invalid curves
        if len(curve_points) == 0:
            return None

        # Convert to numpy array
        curve_points = np.array(curve_points)

        # Skip if insufficient points
        if len(curve_points) < 4:  # Need at least 4 points for degree 3 B-spline
            return None

        # Fit B-spline to get fixed number of control points
        bspline_data = self.bspline_processor.fit_bspline(curve_points)
        
        # Return None if B-spline fitting failed
        if bspline_data is None:
            return None
            
        control_points = bspline_data["control_points"]
        return control_points.astype(np.float32)

    def _tokens_to_ids(self, tokens):
        """Convert DSL tokens to vocabulary IDs"""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Handle point identifiers dynamically
                if token.startswith("P") or token.startswith("C"):
                    # Add to vocabulary if not exists
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        self.id_to_token = {v: k for k, v in self.vocab.items()}
                        self.vocab_size = len(self.vocab)
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.vocab["<UNK>"])

        # Pad or truncate to max sequence length
        if len(ids) > self.max_seq_len:
            print(f"Warning: Truncating sequence from {len(ids)} to {self.max_seq_len}")
            ids = ids[:self.max_seq_len]
        else:
            ids.extend([self.vocab["<PAD>"]] * (self.max_seq_len - len(ids)))

        return ids

    def _update_vocabulary_batch(self, new_tokens):
        """Update vocabulary with a batch of new tokens"""
        for token in sorted(list(new_tokens)):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def _analyze_sequence_lengths(self, processed_results):
        """Analyze the distribution of sequence lengths and normalized coordinate ranges"""
        print("Analyzing sequence lengths and normalized coordinate ranges...")
        
        sequence_lengths = []
        lengths_by_mechanism = {}
        lengths_by_bar_type = {}
        
        # For analyzing normalized coordinate ranges
        all_normalized_coords = []
        coords_by_mechanism = {}
        coords_by_bar_type = {}
        
        for result in processed_results:
            if result["dsl_tokens"]:
                seq_len = len(result["dsl_tokens"])
                sequence_lengths.append(seq_len)
                
                # Group by mechanism type
                mech_type = result["mechanism_type"]
                if mech_type not in lengths_by_mechanism:
                    lengths_by_mechanism[mech_type] = []
                    coords_by_mechanism[mech_type] = []
                lengths_by_mechanism[mech_type].append(seq_len)
                
                # Group by bar type
                bar_type = result["bar_type"]
                if bar_type not in lengths_by_bar_type:
                    lengths_by_bar_type[bar_type] = []
                    coords_by_bar_type[bar_type] = []
                lengths_by_bar_type[bar_type].append(seq_len)
                
                # Use the actual normalized coordinates if available
                if "normalized_coords" in result and result["normalized_coords"]:
                    coord_values = result["normalized_coords"]
                    all_normalized_coords.extend(coord_values)
                    coords_by_mechanism[mech_type].extend(coord_values)
                    coords_by_bar_type[bar_type].extend(coord_values)
        
        if sequence_lengths:
            seq_lengths_array = np.array(sequence_lengths)
            
            print(f"\nSequence Length Analysis:")
            print(f"  Total sequences: {len(sequence_lengths)}")
            print(f"  Mean length: {np.mean(seq_lengths_array):.2f}")
            print(f"  Median length: {np.median(seq_lengths_array):.2f}")
            
            # Current max_seq_len analysis
            truncated_count = np.sum(seq_lengths_array > self.max_seq_len)
            truncation_percentage = (truncated_count / len(sequence_lengths)) * 100
            print(f"  Sequences exceeding current max_seq_len ({self.max_seq_len}): {truncated_count} ({truncation_percentage:.2f}%)")
            
            # Analysis by mechanism type
            print(f"\nMax Length by Mechanism Type:")
            for mech_type, lengths in lengths_by_mechanism.items():
                lengths_array = np.array(lengths)
                print(f"  {mech_type}: max={np.max(lengths_array)})")
            
            # Analysis by bar type
            print(f"\nMax Length by Bar Type:")
            for bar_type, lengths in lengths_by_bar_type.items():
                lengths_array = np.array(lengths)
                print(f"  {bar_type}: max={np.max(lengths_array)})")
        
        # Analyze normalized coordinate ranges
        if all_normalized_coords:
            coords_array = np.array(all_normalized_coords)
            
            print(f"\n{'='*50}")
            print(f"NORMALIZED COORDINATE RANGE ANALYSIS")
            print(f"{'='*50}")
            
            print(f"\nOverall Normalized Coordinate Statistics:")
            print(f"  Total coordinate values: {len(coords_array)}")
            print(f"  Min value: {np.min(coords_array):.4f}")
            print(f"  Max value: {np.max(coords_array):.4f}")
            print(f"  Mean value: {np.mean(coords_array):.4f}")
            print(f"  Std deviation: {np.std(coords_array):.4f}")
            print(f"  Percentiles:")
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                print(f"    {p}th: {np.percentile(coords_array, p):.4f}")
            
            # Analyze by mechanism type
            print(f"\nNormalized Coordinate Ranges by Mechanism Type:")
            for mech_type, coords in sorted(coords_by_mechanism.items()):
                if coords:
                    coords_arr = np.array(coords)
                    print(f"\n  {mech_type}:")
                    print(f"    Count: {len(coords_arr)}")
                    print(f"    Range: [{np.min(coords_arr):.4f}, {np.max(coords_arr):.4f}]")
                    print(f"    Mean: {np.mean(coords_arr):.4f}, Std: {np.std(coords_arr):.4f}")
            
            # Analyze by bar type
            print(f"\nNormalized Coordinate Ranges by Bar Type:")
            for bar_type, coords in sorted(coords_by_bar_type.items()):
                if coords:
                    coords_arr = np.array(coords)
                    print(f"\n  {bar_type}:")
                    print(f"    Count: {len(coords_arr)}")
                    print(f"    Range: [{np.min(coords_arr):.4f}, {np.max(coords_arr):.4f}]")
                    print(f"    Mean: {np.mean(coords_arr):.4f}, Std: {np.std(coords_arr):.4f}")
            
            # Show the configured coordinate range
            print(f"\nConfigured Coordinate Range (from Config):")
            print(f"  COORD_MIN: {Config.COORD_MIN}")
            print(f"  COORD_MAX: {Config.COORD_MAX}")
            print(f"  COORD_BINS: {Config.COORD_BINS}")
            print(f"  Bin size: {(Config.COORD_MAX - Config.COORD_MIN) / Config.COORD_BINS:.4f}")
        
        return sequence_lengths

    @staticmethod
    def _process_single_file(file_info, data_path=None):
        """Process a single file - static method for multiprocessing"""
        file_path, mech_type, bar_type = file_info
        
        try:
            # Load curve points
            curve_points = np.load(file_path)

            # Parse filename for mechanism parameters
            params = MechanismDataPreprocessor._static_parse_filename(file_path)
            if params is None:
                return None, set()

            # Skip PRPR mechanism types
            if params["mechanism_type"] == "PRPR":
                return None, set()

            # Apply inverse normalization to curve points to match simulation coordinate system
            norm_matrix = MechanismDataPreprocessor._static_parse_normalization_matrix(file_path)
            if norm_matrix is not None:
                curve_points = MechanismDataPreprocessor._static_apply_inverse_normalization(curve_points, norm_matrix)

            # Handle mechanism types containing "P" - only use first 5 coordinates
            coords = params["initial_coords"]
            if 'P' in params["mechanism_type"]:
                coords = coords[:10]

            # Create ground joint normalizer for this worker
            normalizer = GroundJointNormalizer()

            # Apply ground joint normalization to both mechanism and curve
            try:
                norm_result = normalizer.normalize_mechanism(
                    coords, 
                    params["mechanism_type"], 
                    curve_points
                )
                normalized_coords = norm_result['normalized_coords']
                normalized_curve = norm_result['normalized_curve']
            except Exception as e:
                print(f"Warning: Could not normalize {params['mechanism_type']}: {e}")
                return None, set()

            # Process the NORMALIZED curve points using B-spline fitting
            if normalized_curve is None:
                return None, set()
            
            processed_curve = MechanismDataPreprocessor._static_process_curve_points(normalized_curve)
            
            # Skip if B-spline fitting failed
            if processed_curve is None:
                return None, set()

            # Generate DSL representation using already normalized coordinates
            dsl_tokens = MechanismDataPreprocessor._static_coords_to_dsl_normalized(
                normalized_coords, 
                params["mechanism_type"]
            )

            # Collect unique tokens
            unique_tokens = set(dsl_tokens)

            # Compute relative path if data_path is provided
            relative_path = None
            if data_path is not None:
                try:
                    relative_path = os.path.relpath(file_path, data_path)
                except ValueError:
                    # If file_path and data_path are on different drives, use absolute path
                    relative_path = file_path
            else:
                relative_path = file_path
            
            # Return result and tokens
            return {
                "control_points": processed_curve,
                "dsl_tokens": dsl_tokens,
                "mechanism_type": params["mechanism_type"],
                "bar_type": bar_type,
                "normalized_coords": normalized_coords,  # Add the actual normalized coordinates
                "file_path": relative_path,  # Add the relative file path
            }, unique_tokens

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, set()

    def _process_chunk(self, chunk, chunk_id, data_path):
        """Process a chunk of files"""
        results = []
        all_tokens = set()
        
        for file_info in chunk:
            result, tokens = self._process_single_file(file_info, data_path)
            if result is not None:
                results.append(result)
                all_tokens.update(tokens)
        
        return results, all_tokens

    @staticmethod
    def _static_parse_filename(filename):
        """Static version of filename parsing for multiprocessing"""
        basename = os.path.basename(filename).replace(".npy", "")
        parts = basename.strip("_").split("_")

        try:
            # Find mechanism type (non-numeric part)
            mech_type_idx = None
            for i, part in enumerate(parts):
                if not MechanismDataPreprocessor._static_is_numeric(part) and len(part) > 2:
                    mech_type_idx = i
                    break

            if mech_type_idx is None:
                return None

            # Extract initial coordinates (before mechanism type)
            coord_parts = parts[:mech_type_idx]
            coords = [float(p) for p in coord_parts if MechanismDataPreprocessor._static_is_numeric(p)]

            # Extract mechanism type
            mech_type = parts[mech_type_idx]

            # Extract normalization matrix (after mechanism type)
            norm_parts = parts[mech_type_idx + 1:]
            norm_matrix = [float(p) for p in norm_parts if MechanismDataPreprocessor._static_is_numeric(p)]

            return {
                "initial_coords": coords,
                "mechanism_type": mech_type,
                "normalization_matrix": norm_matrix[:6] if len(norm_matrix) >= 6 else norm_matrix,
            }
        except:
            return None

    @staticmethod
    def _static_is_numeric(s):
        """Static version of numeric check for multiprocessing"""
        try:
            float(s)
            return True
        except:
            return False

    @staticmethod
    def _static_process_curve_points(curve_points):
        """Static version of curve processing for multiprocessing"""
        # Skip empty or invalid curves
        if len(curve_points) == 0:
            return None

        # Convert to numpy array if not already
        if not isinstance(curve_points, np.ndarray):
            curve_points = np.array(curve_points, dtype=np.float32)
        else:
            curve_points = curve_points.astype(np.float32)

        # Skip if insufficient points
        if len(curve_points) < 4:  # Need at least 4 points for degree 3 B-spline
            return None

        # Create B-spline processor for this worker
        bspline_processor = BSplineCurveProcessor(
            num_control_points=Config.BSPLINE_CONTROL_POINTS,
            degree=Config.BSPLINE_DEGREE,
            smoothing=Config.BSPLINE_SMOOTHING,
        )

        # Fit B-spline to get fixed number of control points
        bspline_data = bspline_processor.fit_bspline(curve_points)
        
        # Return None if B-spline fitting failed
        if bspline_data is None:
            return None
        
        control_points = bspline_data["control_points"]
        return control_points.astype(np.float32)

    @staticmethod
    def _static_coords_to_dsl(coords, mechanism_type, curve_points, normalizer=None):
        """Static version of DSL generation for multiprocessing with normalization"""
        dsl_tokens = []

        # Start with mechanism type
        dsl_tokens.extend(["<SOS>", "MECH_TYPE:", mechanism_type])

        # Apply ground joint normalization if normalizer provided
        if normalizer is not None:
            try:
                # Normalize mechanism
                norm_result = normalizer.normalize_mechanism(
                    coords, mechanism_type, curve_points
                )
                normalized_coords = norm_result['normalized_coords']
                coords_to_use = normalized_coords
            except Exception as e:
                print(f"Warning: Could not normalize {mechanism_type}: {e}")
                sys.exit(1)
        else:
            coords_to_use = coords

        # Add points section
        dsl_tokens.append("POINTS:")

        # Convert coordinates to point definitions
        if len(coords_to_use) >= 2:
            num_points = len(coords_to_use) // 2
            for i in range(num_points):
                x, y = coords_to_use[i * 2], coords_to_use[i * 2 + 1]
                dsl_tokens.extend([
                    f"P{i}",
                    "X:",
                    MechanismDataPreprocessor._static_quantize_coord(x),
                    "Y:",
                    MechanismDataPreprocessor._static_quantize_coord(y),
                ])

        dsl_tokens.append("<EOS>")
        return dsl_tokens

    @staticmethod
    def _static_quantize_coord(coord):
        """Static version of coordinate quantization for multiprocessing"""
        # Clip coordinate to valid range
        coord_clipped = np.clip(coord, Config.COORD_MIN, Config.COORD_MAX)
        
        # Map to bin index
        bin_size = (Config.COORD_MAX - Config.COORD_MIN) / Config.COORD_BINS
        bin_idx = int((coord_clipped - Config.COORD_MIN) / bin_size)
        
        # Handle edge case where coord equals COORD_MAX
        if bin_idx >= Config.COORD_BINS:
            bin_idx = Config.COORD_BINS - 1
            
        return f"BIN_{bin_idx}"
    
    @staticmethod
    def _static_coords_to_dsl_normalized(normalized_coords, mechanism_type):
        """Static version of DSL generation for already normalized coordinates"""
        dsl_tokens = []

        # Start with mechanism type
        dsl_tokens.extend(["<SOS>", "MECH_TYPE:", mechanism_type])

        # Add points section
        dsl_tokens.append("POINTS:")

        # Convert coordinates to point definitions
        if len(normalized_coords) >= 2:
            num_points = len(normalized_coords) // 2
            for i in range(num_points):
                x, y = normalized_coords[i * 2], normalized_coords[i * 2 + 1]
                dsl_tokens.extend([
                    f"P{i}",
                    "X:",
                    MechanismDataPreprocessor._static_quantize_coord(x),
                    "Y:",
                    MechanismDataPreprocessor._static_quantize_coord(y),
                ])

        dsl_tokens.append("<EOS>")
        return dsl_tokens

    @staticmethod
    def _static_parse_normalization_matrix(filename):
        """Static version of normalization matrix parsing for multiprocessing"""
        basename = os.path.basename(filename).replace(".npy", "")
        parts = basename.strip("_").split("_")
        
        try:
            # Find mechanism type (non-numeric part)
            mech_type_idx = None
            for i, part in enumerate(parts):
                if not MechanismDataPreprocessor._static_is_numeric(part) and len(part) > 2:
                    mech_type_idx = i
                    break
            
            if mech_type_idx is None:
                return None
            
            # Extract normalization matrix (after mechanism type)
            norm_parts = parts[mech_type_idx + 1:]
            norm_matrix = [float(p) for p in norm_parts if MechanismDataPreprocessor._static_is_numeric(p)]
            
            if len(norm_matrix) >= 6:
                # Return as 2x3 affine transformation matrix
                return np.array([
                    [norm_matrix[0], norm_matrix[1], norm_matrix[2]],
                    [norm_matrix[3], norm_matrix[4], norm_matrix[5]]
                ])
            
            return None
        except:
            return None

    @staticmethod
    def _static_apply_inverse_normalization(curve_points, norm_matrix):
        """Static version of inverse normalization for multiprocessing"""
        if norm_matrix is None or len(curve_points) == 0:
            return curve_points
        
        try:
            # Extract the 2x2 linear part and translation vector
            A = norm_matrix[:, :2]  # 2x2 rotation/scale matrix
            b = norm_matrix[:, 2]   # 2x1 translation vector
            
            # Compute inverse: x = A^(-1) @ (x' - b)
            A_inv = np.linalg.inv(A)
            
            # Apply inverse transformation
            translated_points = curve_points - b
            unnormalized_points = (A_inv @ translated_points.T).T
            
            return unnormalized_points
        except np.linalg.LinAlgError:
            # Could not invert normalization matrix, using original points
            return curve_points

    def process_and_save(self, output_path="processed_data.pkl", num_workers=None):
        """Process all data and save to pickle file using multiprocessing"""
        # Collect data files
        self._collect_data_files()
        
        # Determine number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        
        print(f"Processing {len(self.data_files)} files using {num_workers} workers...")
        
        # Create chunks for parallel processing
        chunk_size = max(1, len(self.data_files) // (num_workers * 4))
        
        # Process data in parallel
        processed_results = []
        all_tokens = set()  # Collect all unique tokens for vocabulary
        skipped_count = 0  # Track skipped files
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit chunks to workers
            futures = []
            for i in range(0, len(self.data_files), chunk_size):
                chunk = self.data_files[i:i + chunk_size]
                future = executor.submit(self._process_chunk, chunk, i // chunk_size, self.data_path)
                futures.append(future)
            
            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Processing chunks") as pbar:
                for future in futures:
                    try:
                        chunk_results, chunk_tokens = future.result()
                        processed_results.extend(chunk_results)
                        all_tokens.update(chunk_tokens)
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                    finally:
                        pbar.update(1)
        
        # Update vocabulary with all collected tokens
        self._update_vocabulary_batch(all_tokens)
        
        # Analyze sequence lengths before converting to IDs
        self._analyze_sequence_lengths(processed_results)
        
        # Convert tokens to IDs for all processed data
        print("Converting tokens to IDs...")
        processed_data = []
        for result in tqdm(processed_results, desc="Converting tokens"):
            if result["dsl_tokens"]:
                dsl_ids = self._tokens_to_ids(result["dsl_tokens"])
                processed_data.append({
                    "control_points": result["control_points"],
                    "dsl_ids": np.array(dsl_ids, dtype=np.int64),
                    "mechanism_type": result["mechanism_type"],
                    "bar_type": result["bar_type"],
                    "file_path": result["file_path"],
                })

        # Finalize vocabulary
        self._finalize_vocabulary()
        
        # Calculate actual skipped count
        skipped_count = len(self.data_files) - len(processed_data)

        print(f"Processed {len(processed_data)} samples")
        print(f"Skipped {skipped_count} files due to processing errors")
        
        # Save processed data and vocabulary
        data_to_save = {
            "processed_data": processed_data,
            "vocab": self.vocab,
            "id_to_token": self.id_to_token,
            "vocab_size": self.vocab_size,
            "mechanism_types": list(self.mechanism_types),
            "max_seq_len": self.max_seq_len,
            "num_files_processed": len(self.data_files),
            "num_files_skipped": skipped_count,
            "data_path": self.data_path,  # Store the original data path for reference
            "random_seed": self.random_seed,  # Store the random seed used
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(data_to_save, f)
        
        print(f"Saved processed data to {output_path}")
        print(f"Total samples: {len(processed_data)}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Success rate: {len(processed_data) / len(self.data_files) * 100:.1f}%")
        
        # Analyze sequence lengths
        self._analyze_sequence_lengths(processed_results)
        
        return data_to_save


def main():
    parser = argparse.ArgumentParser(description='Preprocess mechanism data')
    parser.add_argument('--data_path', type=str, 
                       default="",
                       help='Path to raw data directory')
    parser.add_argument('--output_path', type=str, default='processed_data.pkl',
                       help='Path to save processed data')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--max_seq_len', type=int, default=64,
                       help='Maximum sequence length for DSL tokens')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible data sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = MechanismDataPreprocessor(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        max_files=args.max_files,
        random_seed=args.random_seed
    )
    
    # Process and save data
    preprocessor.process_and_save(args.output_path, num_workers=args.num_workers)


if __name__ == "__main__":
    main() 