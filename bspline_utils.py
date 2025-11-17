import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist
import warnings
from sklearn.cluster import DBSCAN

class BSplineCurveProcessor:
    def __init__(self, num_control_points=64, degree=3, smoothing=0.0, closed_threshold=0.05):
        self.num_control_points = num_control_points
        self.degree = degree
        self.smoothing = smoothing
        self.closed_threshold = closed_threshold
        
    def is_curve_closed(self, points, threshold=None):
        """closed curve detection using multiple criteria"""
        if threshold is None:
            threshold = self.closed_threshold
            
        if len(points) < 6:  # Need minimum points for meaningful closed curve
            return False
        
        start_point = points[0]
        end_point = points[-1]
        
        # Calculate various distance metrics
        euclidean_dist = np.linalg.norm(start_point - end_point)
        
        # Calculate curve scale for relative threshold
        curve_bbox = np.max(points, axis=0) - np.min(points, axis=0)
        curve_scale = np.max(curve_bbox)
        
        if curve_scale < 1e-10:
            return False
        
        # Relative distance threshold
        relative_dist = euclidean_dist / curve_scale
        
        # Check if start and end are close
        is_endpoints_close = relative_dist < threshold
        
        # Additional check: look at curvature continuity
        if len(points) >= 10:
            # Check if the curve direction at start and end are consistent
            start_direction = points[5] - points[0]
            end_direction = points[-1] - points[-6]
            
            start_direction = start_direction / np.linalg.norm(start_direction)
            end_direction = end_direction / np.linalg.norm(end_direction)
            
            # For closed curves, end direction should be roughly opposite to start direction
            direction_similarity = np.dot(start_direction, end_direction)
            is_direction_consistent = direction_similarity < -0.5  # Vectors pointing in opposite directions
        else:
            is_direction_consistent = True
        
        return is_endpoints_close and is_direction_consistent
    
    def preprocess_curve(self, points):
        """Preprocess curve points to remove noise and handle problematic points"""
        if len(points) <= 2:
            return points
        
        # Store original extent for coverage checking
        original_bbox = np.max(points, axis=0) - np.min(points, axis=0)
        
        # Remove consecutive duplicates with adaptive tolerance
        points_clean = self._remove_consecutive_duplicates_adaptive(points)
        
        # Check extent loss after duplicate removal
        clean_bbox = np.max(points_clean, axis=0) - np.min(points_clean, axis=0)
        extent_loss = (original_bbox - clean_bbox) / (original_bbox + 1e-10)
        
        # Only remove outliers if we haven't already lost significant coverage
        if np.any(extent_loss > 0.05):  # More than 5% extent loss
            points_final = points_clean
        else:
            # Remove outliers using DBSCAN clustering
            points_after_outliers = self._remove_outliers(points_clean)
            
            # Check extent loss after outlier removal
            outlier_bbox = np.max(points_after_outliers, axis=0) - np.min(points_after_outliers, axis=0)
            outlier_extent_loss = (original_bbox - outlier_bbox) / (original_bbox + 1e-10)
            
            # Only keep outlier removal results if they don't cause excessive extent loss
            if np.any(outlier_extent_loss > 0.10):  # More than 10% extent loss
                points_final = points_clean
            else:
                points_final = points_after_outliers
        
        # Apply light smoothing only
        points_final = self._smooth_curve(points_final, window_size=3)  # Smaller window
        
        return points_final
    
    def _remove_consecutive_duplicates_adaptive(self, points):
        """Remove consecutive duplicates with adaptive tolerance"""
        if len(points) <= 1:
            return points
        
        # Calculate adaptive tolerance based on curve scale
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        median_distance = np.median(distances[distances > 0]) if np.any(distances > 0) else 1e-10
        tolerance = median_distance * 0.001  # 0.1% of median segment length
        
        mask = np.ones(len(points), dtype=bool)
        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[i-1]) < tolerance:
                mask[i] = False
        
        return points[mask]
    
    def _remove_outliers(self, points):
        """Remove outlier points using DBSCAN clustering - less aggressive"""
        if len(points) < 15:  # Increased threshold - don't remove outliers for smaller curves
            return points
        
        try:
            # Use DBSCAN to find the main cluster of points
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            median_distance = np.median(distances) if len(distances) > 0 else 1.0
            eps = median_distance * 5.0
            
            clustering = DBSCAN(eps=eps, min_samples=max(3, len(points) // 20)).fit(points)  # Adaptive min_samples
            labels = clustering.labels_
            
            # Only remove outliers if they represent less than 10% of points
            if len(np.unique(labels)) > 1:
                main_cluster = np.bincount(labels[labels >= 0]).argmax()
                mask = labels == main_cluster
                outliers_ratio = np.sum(~mask) / len(points)
                
                # Only remove outliers if they're less than 10% of the curve
                if outliers_ratio < 0.1:
                    return points[mask]
                else:
                    return points  # Keep all points if too many would be removed
            else:
                return points
        except:
            return points
    
    def _smooth_curve(self, points, window_size=5):
        """Apply light smoothing to reduce noise"""
        if len(points) < window_size:
            return points
        
        # Use moving average for light smoothing
        smoothed = np.copy(points)
        half_window = window_size // 2
        
        for i in range(half_window, len(points) - half_window):
            smoothed[i] = np.mean(points[i-half_window:i+half_window+1], axis=0)
        
        return smoothed
    
    def parameterize_curve(self, points, closed=False):
        """curve parameterization using chord length"""
        if len(points) < 2:
            return np.array([0.0])
        
        if closed:
            # For closed curves, ensure the parameter goes from 0 to 1
            # and the curve is properly closed
            distances = np.linalg.norm(np.diff(np.vstack([points, points[0:1]]), axis=0), axis=1)
            cumulative = np.concatenate([[0], np.cumsum(distances)])
            # Normalize to [0, 1] but don't include the duplicate endpoint
            total_length = cumulative[-1]
            if total_length > 0:
                parameters = cumulative[:-1] / total_length
            else:
                parameters = np.linspace(0, 1, len(points), endpoint=False)
        else:
            # For open curves
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cumulative = np.concatenate([[0], np.cumsum(distances)])
            total_length = cumulative[-1]
            if total_length > 0:
                parameters = cumulative / total_length
            else:
                parameters = np.linspace(0, 1, len(points))
        
        return parameters
    

    
    def _fit_open_bspline(self, points, parameters):
        """open B-spline fitting using parametric representation"""
        # Use parametric B-spline fitting (both x and y together)
        
        # Determine effective degree
        effective_degree = min(self.degree, len(points) - 1)
        
        # Fit parametric B-spline
        tck, u = interpolate.splprep([points[:, 0], points[:, 1]], 
                                     u=parameters,
                                     k=effective_degree,
                                     s=self.smoothing)
        
        # Get the current number of control points
        current_cp = len(tck[1][0])
        
        # If we need to resample to fixed number of control points
        if current_cp != self.num_control_points:
            # Use the fitted parameter range (u) for exact coverage
            # This ensures we sample over the exact same parameter space as the original data
            param_min, param_max = u[0], u[-1]
            
            # Ensure endpoints are exactly preserved
            if self.num_control_points >= 2:
                # Force first and last control points to be at exact endpoints
                new_params = np.linspace(param_min, param_max, self.num_control_points)
                new_params[0] = param_min  # Ensure exact start
                new_params[-1] = param_max  # Ensure exact end
            else:
                new_params = np.array([param_min])
            
            # Evaluate B-spline at new parameter values to get control points
            x_cp, y_cp = interpolate.splev(new_params, tck)
            control_points = np.column_stack([x_cp, y_cp])
        else:
            control_points = np.column_stack([tck[1][0], tck[1][1]])
        
        if len(control_points) != self.num_control_points:
            return None
        
        return {
            'control_points': control_points,
            'is_closed': False,
            'knot_vector': tck[0],
            'original_tck': tck,
            'degree': effective_degree,
            'parameters': parameters
        }
    
    def _fit_closed_bspline(self, points, parameters):
        """closed B-spline fitting"""
        # For closed curves, we need to handle periodicity properly
        effective_degree = min(self.degree, len(points) - 1)
        
        try:
            # Fit periodic B-spline
            tck, u = interpolate.splprep([points[:, 0], points[:, 1]], 
                                         u=parameters,
                                         k=effective_degree,
                                         s=self.smoothing,
                                         per=True)
            
            # Get control points
            current_cp = len(tck[1][0])
            
            if current_cp != self.num_control_points:
                # Use the fitted parameter range (u) for exact coverage
                # For closed curves, ensure proper parameter distribution
                param_min, param_max = u[0], u[-1]
                
                # Create parameter values that span the actual curve extent (closed)
                new_params = np.linspace(param_min, param_max, self.num_control_points, endpoint=False)
                
                # Evaluate B-spline at new parameter values
                x_cp, y_cp = interpolate.splev(new_params, tck)
                control_points = np.column_stack([x_cp, y_cp])
            else:
                control_points = np.column_stack([tck[1][0], tck[1][1]])
            
            if len(control_points) != self.num_control_points:
                return None
            
            return {
                'control_points': control_points,
                'is_closed': True,
                'knot_vector': tck[0],
                'original_tck': tck,
                'degree': effective_degree,
                'parameters': parameters
            }
        except Exception as e:
            # Skip if periodic fitting fails
            warnings.warn(f"Closed B-spline fitting failed: {e}, skipping data point")
            return None
    
    def _handle_degenerate_case(self, points):
        """Handle degenerate cases with very few points"""
        if len(points) == 0:
            control_points = np.zeros((self.num_control_points, 2))
        elif len(points) == 1:
            control_points = np.tile(points[0], (self.num_control_points, 1))
        else:
            # Linear interpolation for few points
            t_old = np.linspace(0, 1, len(points))
            t_new = np.linspace(0, 1, self.num_control_points)
            
            x_new = np.interp(t_new, t_old, points[:, 0])
            y_new = np.interp(t_new, t_old, points[:, 1])
            
            control_points = np.column_stack([x_new, y_new])
        
        if len(control_points) != self.num_control_points:
            return None
        
        return {
            'control_points': control_points,
            'is_closed': False,
            'degree': min(self.degree, len(points) - 1) if len(points) > 0 else 1,
            'degenerate': True
        }
    
    def evaluate_bspline(self, bspline_data, num_points=200):
        """B-spline evaluation"""
        if 'degenerate' in bspline_data and bspline_data['degenerate']:
            return bspline_data['control_points']
        
        if 'original_tck' in bspline_data:
            # Use the original tck for evaluation
            tck = bspline_data['original_tck']
            is_closed = bspline_data['is_closed']
            
            if is_closed:
                t = np.linspace(0, 1, num_points, endpoint=False)
            else:
                t = np.linspace(0, 1, num_points)
            
            try:
                x_eval, y_eval = interpolate.splev(t, tck)
                return np.column_stack([x_eval, y_eval])
            except:
                return bspline_data['control_points']
        else:
            # Fallback to control points
            return bspline_data['control_points']
    
    def calculate_reconstruction_error(self, original_points, bspline_data, num_eval_points=100):
        """Calculate reconstruction error using proper curve alignment"""
        reconstructed = self.evaluate_bspline(bspline_data, num_eval_points)
        
        if len(original_points) == 0 or len(reconstructed) == 0:
            return float('inf')
        
        # Align the curves by finding the best correspondence
        error = self._calculate_hausdorff_distance(original_points, reconstructed)
        return error
    
    def _calculate_hausdorff_distance(self, curve1, curve2):
        """Calculate modified Hausdorff distance between two curves"""
        if len(curve1) == 0 or len(curve2) == 0:
            return float('inf')
        
        # Sample points uniformly from both curves
        n_sample = min(50, len(curve1), len(curve2))
        
        if len(curve1) > n_sample:
            indices1 = np.linspace(0, len(curve1)-1, n_sample, dtype=int)
            sample1 = curve1[indices1]
        else:
            sample1 = curve1
            
        if len(curve2) > n_sample:
            indices2 = np.linspace(0, len(curve2)-1, n_sample, dtype=int)
            sample2 = curve2[indices2]
        else:
            sample2 = curve2
        
        # Calculate distances
        dist_matrix = cdist(sample1, sample2)
        
        # Modified Hausdorff distance (mean of minimum distances)
        min_dists_1to2 = np.min(dist_matrix, axis=1)
        min_dists_2to1 = np.min(dist_matrix, axis=0)
        
        # Use mean instead of max for a more stable metric
        hausdorff_1to2 = np.mean(min_dists_1to2)
        hausdorff_2to1 = np.mean(min_dists_2to1)
        
        return max(hausdorff_1to2, hausdorff_2to1)
    
    def normalize_control_points(self, control_points):
        """Identity normalization - preserves original curve information"""
        if len(control_points) == 0:
            return control_points, (0, 0), 1
        
        # Return original control points without any normalization
        # The curves are already properly positioned from the dataset
        return control_points, (0, 0), 1.0
    
    def denormalize_control_points(self, normalized_points, centroid, scale):
        """Identity denormalization - data is already in original form"""
        return normalized_points
    
    def fit_bspline(self, points, closed=None):
        """Main interface - delegates to fitting with fallback strategies"""
        # Preprocess the curve
        points_clean = self.preprocess_curve(points)
        
        if len(points_clean) < self.degree + 1:
            warnings.warn("Insufficient points for B-spline fitting, skipping data point")
            return None
        
        # Determine if curve is closed
        if closed is None:
            closed = self.is_curve_closed(points_clean)
        
        # Parameterize the curve
        parameters = self.parameterize_curve(points_clean, closed)
        
        try:
            # Try primary strategy
            if closed:
                result = self._fit_closed_bspline(points_clean, parameters)
            else:
                result = self._fit_open_bspline(points_clean, parameters)
            
            # Check if result has good coverage
            if result is not None:
                reconstructed = self.evaluate_bspline(result, num_points=100)
                
                # Calculate coverage ratio
                original_bbox = np.max(points, axis=0) - np.min(points, axis=0)
                recon_bbox = np.max(reconstructed, axis=0) - np.min(reconstructed, axis=0)
                coverage_ratio = recon_bbox / (original_bbox + 1e-10)
                
                # If coverage is poor, try fallback strategy
                if np.any(coverage_ratio < 0.85):  # Less than 85% coverage
                    fallback_result = self._try_fallback_strategy(points, closed)
                    if fallback_result is not None:
                        return fallback_result
                
            return result
            
        except Exception as e:
            # Try fallback strategy before giving up
            fallback_result = self._try_fallback_strategy(points, closed)
            if fallback_result is not None:
                return fallback_result
            
            warnings.warn(f"All B-spline fitting strategies failed: {e}, skipping data point")
            return None

    def _try_fallback_strategy(self, points, closed):
        """Try fallback strategy with different parameters"""
        try:
            # Use less aggressive preprocessing
            points_minimal = self._minimal_preprocess(points)
            
            if len(points_minimal) < 3:  # Need at least 3 points for degree 2
                return None
            
            # Use lower degree and some smoothing for stability
            effective_degree = min(2, len(points_minimal) - 1)
            fallback_smoothing = 0.05
            
            # Parameterize
            if closed is None:
                closed = self.is_curve_closed(points_minimal)
            parameters = self.parameterize_curve(points_minimal, closed)
            
            # Fit with fallback parameters
            if closed:
                tck, u = interpolate.splprep([points_minimal[:, 0], points_minimal[:, 1]], 
                                             u=parameters,
                                             k=effective_degree,
                                             s=fallback_smoothing,
                                             per=True)
            else:
                tck, u = interpolate.splprep([points_minimal[:, 0], points_minimal[:, 1]], 
                                             u=parameters,
                                             k=effective_degree,
                                             s=fallback_smoothing)
            
            # Get true control points from the fitted B-spline
            current_cp = len(tck[1][0])
            if current_cp != self.num_control_points:
                # Step 1: Sample more points from the fitted curve
                param_min, param_max = u[0], u[-1]
                # Use more sample points than needed for better approximation
                n_samples = max(self.num_control_points * 2, 100)
                if closed:
                    sample_params = np.linspace(param_min, param_max, n_samples, endpoint=False)
                else:
                    sample_params = np.linspace(param_min, param_max, n_samples)
                
                x_sampled, y_sampled = interpolate.splev(sample_params, tck)
                sampled_points = np.column_stack([x_sampled, y_sampled])
                
                # Step 2: Fit a NEW B-spline to these sampled points to get true control points
                try:
                    # Use degree 3 if we have enough points, otherwise lower
                    refit_degree = min(3, len(sampled_points) - 1, self.num_control_points - 1)
                    
                    # Parameterize the sampled points
                    sample_parameters = self.parameterize_curve(sampled_points, closed)
                    
                    # Fit B-spline with target number of control points
                    # Use task='interpolation' to force specific number of control points
                    if closed:
                        # For closed curves, we need to be more careful
                        refit_tck, refit_u = interpolate.splprep([sampled_points[:, 0], sampled_points[:, 1]], 
                                                               u=sample_parameters,
                                                               k=refit_degree,
                                                               s=0.0,  # Exact fit
                                                               per=True,
                                                               task=-1,  # Find knots
                                                               t=np.linspace(0, 1, self.num_control_points + refit_degree + 1))
                    else:
                        # For open curves, we can use the approximation approach
                        # Sample exactly the number of points we want as control points
                        target_params = np.linspace(param_min, param_max, self.num_control_points)
                        if self.num_control_points >= 2:
                            target_params[0] = param_min
                            target_params[-1] = param_max
                        
                        x_target, y_target = interpolate.splev(target_params, tck)
                        target_points = np.column_stack([x_target, y_target])
                        
                        # Fit B-spline to these target points
                        target_params_norm = self.parameterize_curve(target_points, closed=False)
                        refit_tck, refit_u = interpolate.splprep([target_points[:, 0], target_points[:, 1]], 
                                                               u=target_params_norm,
                                                               k=min(refit_degree, len(target_points) - 1),
                                                               s=0.0)  # Exact fit
                    
                    # Extract the control points
                    refit_cp_count = len(refit_tck[1][0])
                    if refit_cp_count == self.num_control_points:
                        # Perfect! We got exactly the right number of true control points
                        control_points = np.column_stack([refit_tck[1][0], refit_tck[1][1]])
                        final_tck = refit_tck
                        final_params = refit_u
                    else:
                        # Use the original approach but mark as approximate
                        param_min, param_max = u[0], u[-1]
                        if closed:
                            new_params = np.linspace(param_min, param_max, self.num_control_points, endpoint=False)
                        else:
                            new_params = np.linspace(param_min, param_max, self.num_control_points)
                            if self.num_control_points >= 2:
                                new_params[0] = param_min
                                new_params[-1] = param_max
                        
                        x_cp, y_cp = interpolate.splev(new_params, tck)
                        control_points = np.column_stack([x_cp, y_cp])
                        final_tck = tck
                        final_params = new_params
                        
                except Exception as e:
                    # Fallback to sampled points
                    param_min, param_max = u[0], u[-1]
                    if closed:
                        new_params = np.linspace(param_min, param_max, self.num_control_points, endpoint=False)
                    else:
                        new_params = np.linspace(param_min, param_max, self.num_control_points)
                        if self.num_control_points >= 2:
                            new_params[0] = param_min
                            new_params[-1] = param_max
                    
                    x_cp, y_cp = interpolate.splev(new_params, tck)
                    control_points = np.column_stack([x_cp, y_cp])
                    final_tck = tck
                    final_params = new_params
            else:
                # We got the right number of control points directly - these are true control points
                control_points = np.column_stack([tck[1][0], tck[1][1]])
                final_tck = tck
                final_params = parameters
            
            # STRICT VALIDATION: Ensure exactly the required number of control points
            if len(control_points) != self.num_control_points:
                return None
            
            return {
                'control_points': control_points,
                'is_closed': closed,
                'knot_vector': final_tck[0],
                'original_tck': final_tck,
                'degree': effective_degree,
                'parameters': final_params,
                'fallback': True
            }
            
        except Exception as e:
            return None

    def _minimal_preprocess(self, points):
        """Minimal preprocessing that preserves curve extent"""
        if len(points) <= 2:
            return points
        
        # Only remove exact duplicates
        unique_points = []
        for i, point in enumerate(points):
            if i == 0 or not np.allclose(point, points[i-1], atol=1e-10):
                unique_points.append(point)
        
        return np.array(unique_points)


# Wrapper class for backward compatibility
class BSplineCurveProcessor(BSplineCurveProcessor):
    """Backward compatible wrapper with implementation"""
    
    def __init__(self, num_control_points=64, degree=3, smoothing=0.0):
        # Use defaults
        super().__init__(
            num_control_points=num_control_points,
            degree=degree,
            smoothing=smoothing,
            closed_threshold=0.05
        ) 