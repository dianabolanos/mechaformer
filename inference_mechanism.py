import torch
import numpy as np
import pickle
from train import MechanismTransformer, MechanismDataset
from config import Config
from bspline_utils import BSplineCurveProcessor
import os
from ground_joint_utils import GroundJointNormalizer
from generation_utils import autoregressive_generate

class MechanismInference:
    def __init__(self, model_path, vocab_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize B-spline processor
        self.bspline_processor = BSplineCurveProcessor(
            num_control_points=Config.BSPLINE_CONTROL_POINTS,
            degree=Config.BSPLINE_DEGREE,
            smoothing=Config.BSPLINE_SMOOTHING
        )
        
        # Initialize ground joint normalizer
        self.ground_joint_normalizer = GroundJointNormalizer('wrapper/BSIdict_468.json')
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.vocab = vocab_data['vocab']
            self.id_to_token = vocab_data['id_to_token']
            self.vocab_size = vocab_data['vocab_size']
        
        self.model = MechanismTransformer(
            vocab_size=self.vocab_size,
            d_model=256,                    
            nhead=8,                        
            num_encoder_layers=6,           
            num_decoder_layers=6,           
            num_control_points=Config.BSPLINE_CONTROL_POINTS,
            max_seq_len=Config.MAX_SEQ_LEN,
      
            attn_flash=True,
            use_rmsnorm=True,
            ff_glu=True,
            rotary_pos_emb=True,
            attn_qk_norm=True,
            ff_swish=True,
            ff_no_bias=True,
        ).to(self.device)
        
        # Load model weights (weights-only version)
        try:
            model_weights = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_weights)
            self.model.eval()
            print(f"Model weights loaded successfully from {model_path}")
            print(f"Vocabulary size: {self.vocab_size}")
            print(f"Model architecture: d_model=256, nhead=8, enc_layers=6, dec_layers=6")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")
    
    def preprocess_curve(self, curve_points):
        """Preprocess curve points using B-spline fitting for inference"""
        if len(curve_points) == 0:
            return np.zeros((Config.BSPLINE_CONTROL_POINTS, 2)).astype(np.float32)
        
        # Convert to numpy array
        curve_points = np.array(curve_points)
        
        # Fit B-spline to get fixed number of control points
        bspline_data = self.bspline_processor.fit_bspline(curve_points)
        control_points = bspline_data['control_points']
        
        # Normalize control points (same as training)
        normalized_cp, centroid, scale = self.bspline_processor.normalize_control_points(control_points)
        
        return normalized_cp.astype(np.float32)
    
    def generate_dsl(self, curve_points, max_length=Config.MAX_SEQ_LEN, process_curve=True, temperature=0.01, mech_type=None, top_k=5):
        """Generate DSL sequence from curve points using autoregressive generation"""
        with torch.no_grad():
            # Preprocess curve to B-spline control points
            if process_curve:
                control_points = self.preprocess_curve(curve_points)
            else:
                control_points = curve_points
            control_tensor = torch.FloatTensor(control_points).unsqueeze(0).to(self.device)
            
            # Prepare prefix tokens if mechanism type is specified
            prefix_tokens = None
            if mech_type is not None:
                if 'MECH_TYPE:' in self.vocab and mech_type in self.vocab:
                    prefix_tokens = [self.vocab['MECH_TYPE:'], self.vocab[mech_type]]
            
            # Use autoregressive_generate from generation_utils
            generated_tensor = autoregressive_generate(
                self.model,
                control_tensor,
                self.vocab,
                self.device,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                prefix_tokens=prefix_tokens
            )
            
            # Convert tensor to list, removing batch dimension
            generated_sequence = generated_tensor[0].cpu().tolist()
            
            return generated_sequence
    
    def sequence_to_dsl(self, sequence):
        """Convert token sequence to readable DSL string"""
        tokens = []
        for token_id in sequence:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['<PAD>', '<SOS>', '<EOS>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def parse_dsl_to_mechanism(self, dsl_string):
        """Parse DSL string to extract mechanism parameters with ground joint reconstruction"""
        tokens = dsl_string.split()
        mechanism_info = {
            'mechanism_type': None,
            'points': []
        }
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == 'MECH_TYPE:' and i + 1 < len(tokens):
                mechanism_info['mechanism_type'] = tokens[i + 1]
                i += 2
            elif token.startswith('P') and i + 4 < len(tokens):
                # Parse point: P0 X: BIN_123 Y: BIN_456
                if tokens[i + 1] == 'X:' and tokens[i + 3] == 'Y:':
                    try:
                        x_token = tokens[i + 2]
                        y_token = tokens[i + 4]
                        
                        # Convert BIN tokens back to coordinates
                        x_val = self.parse_num_token(x_token)
                        y_val = self.parse_num_token(y_token)
                        
                        mechanism_info['points'].append((x_val, y_val))
                        i += 5
                    except:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        # Reconstruct ground joints using BSI dictionary
        if mechanism_info['mechanism_type'] is not None:
            try:
                # Get ground joint info from BSI dictionary
                ground_indices = self.ground_joint_normalizer.get_ground_joint_indices(mechanism_info['mechanism_type'])
                is_ternary = len(ground_indices) >= 3
                
                # Determine which indices were removed during preprocessing
                removed_indices = ground_indices[:2] if is_ternary else ground_indices
                
                # Create full point list with ground joints
                total_points = len(mechanism_info['points']) + len(removed_indices)
                full_points = [None] * total_points
                
                # Insert ground joints at their fixed positions
                for i, idx in enumerate(removed_indices):
                    if i == 0:
                        full_points[idx] = (0.0, 0.0)  # First ground joint
                    elif i == 1:
                        full_points[idx] = (1.0, 0.0)  # Second ground joint
                
                # Insert other points
                other_idx = 0
                for i in range(total_points):
                    if i not in removed_indices:
                        if other_idx < len(mechanism_info['points']):
                            full_points[i] = mechanism_info['points'][other_idx]
                            other_idx += 1
                
                mechanism_info['points'] = full_points
            except Exception as e:
                # If we can't get ground joint info, just use the points as-is
                print(f"Warning: Could not reconstruct ground joints for {mechanism_info['mechanism_type']}: {e}")
        
        return mechanism_info
    
    def parse_num_token(self, num_token):
        """Parse BIN token back to float coordinate"""
        if num_token.startswith('BIN_'):
            try:
                bin_idx = int(num_token[4:])  # Remove 'BIN_' prefix
                # Convert bin index back to coordinate value (use bin center)
                bin_size = (Config.COORD_MAX - Config.COORD_MIN) / Config.COORD_BINS
                coord_value = Config.COORD_MIN + (bin_idx + 0.5) * bin_size
                return coord_value
            except:
                return 0.0
        return 0.0

    def generate_mechanism_params(self, curve_points, temperature=0.01, mech_type=None, top_k=50, process_curve=True):
        """Generate mechanism parameters directly from curve points for backend API"""
        try:
            # Generate DSL sequence
            generated_sequence = self.generate_dsl(curve_points, temperature=temperature, mech_type=mech_type, top_k=top_k, process_curve=process_curve)
            
            # Convert to readable DSL
            dsl_string = self.sequence_to_dsl(generated_sequence)
            
            # Parse to mechanism info
            mechanism_info = self.parse_dsl_to_mechanism(dsl_string)

            # Convert points to the expected format
            coord_params = []
            for point in mechanism_info['points']:
                if point is not None:  # Handle None values from ground joint reconstruction
                    coord_params.append([float(point[0]), float(point[1])])

            return {
                'params': coord_params,
                'type': mechanism_info.get('mechanism_type'),
                'dsl': dsl_string,
                'success': True,
                'generated_sequence': generated_sequence,
                'vocab_size': self.vocab_size
            }
            
        except Exception as e:
            print(f"Error generating mechanism parameters: {e}")
            return {
                'params': [],
                'type': 'RRRR',
                'dsl': '',
                'success': False,
                'error': str(e)
            }

def demo_inference():
    """Demonstrate the inference process with the new model parameters"""
    
    # Check if trained model exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Trained model not found at {Config.MODEL_SAVE_PATH}")
        print("Please train a model first or download one from S3.")
        return
    
    if not os.path.exists(Config.VOCAB_SAVE_PATH):
        print(f"Vocabulary file not found at {Config.VOCAB_SAVE_PATH}")
        print("Please train a model first to create the vocabulary.")
        return
    
    # Initialize inference
    try:
        inference = MechanismInference(Config.MODEL_SAVE_PATH, Config.VOCAB_SAVE_PATH)
    except Exception as e:
        print(f"Failed to initialize inference: {e}")
        return
    
    # Load a sample curve from the dataset for demonstration
    print("Loading sample data for demonstration...")
    
    # Try to load a sample .npy file
    sample_file = None
    data_path = Config.DATA_PATH
    
    for bar_type in ['4bar-npy', '6bar-npy', '8bar-npy']:
        base_path = os.path.join(data_path, bar_type)
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                if item.startswith('outputs-'):
                    outputs_dir = os.path.join(base_path, item)
                    for mech_type in os.listdir(outputs_dir):
                        mech_dir = os.path.join(outputs_dir, mech_type)
                        if os.path.isdir(mech_dir):
                            npy_files = [f for f in os.listdir(mech_dir) if f.endswith('.npy')]
                            if npy_files:
                                sample_file = os.path.join(mech_dir, npy_files[0])
                                break
                    if sample_file:
                        break
            if sample_file:
                break
    
    if sample_file:
        print(f"Using sample file: {sample_file}")
        
        # Load sample curve
        curve_points = np.load(sample_file)
        print(f"Loaded curve with {len(curve_points)} points")
        
        # Generate mechanism parameters directly
        print("Generating mechanism parameters...")
        try:
            mechanism_params = inference.generate_mechanism_params(curve_points)
            
            if mechanism_params['success']:
                print("✅ Mechanism generation successful!")
                print(f"📋 DSL: {mechanism_params['dsl']}")
                print(f"🔧 Type: {mechanism_params['type']}")
                print(f"📍 Parameters: {mechanism_params['params']}")
                print(f"🎯 Generated sequence length: {len(mechanism_params['generated_sequence'])}")
                print(f"📚 Vocabulary size: {mechanism_params['vocab_size']}")
            else:
                print("⚠️ Mechanism generation failed")
                print(f"❌ Error: {mechanism_params.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return
        
    else:
        print("No sample data found. Creating synthetic curve for demonstration...")
        
        # Create a simple synthetic curve (circle)
        t = np.linspace(0, 2*np.pi, 100)
        curve_points = np.column_stack([np.cos(t), np.sin(t)])
        
        print("Generating mechanism parameters from synthetic circle curve...")
        mechanism_params = inference.generate_mechanism_params(curve_points)
        
        if mechanism_params['success']:
            print("✅ Mechanism generation successful!")
            print(f"DSL: {mechanism_params['dsl']}")
            print(f"Type: {mechanism_params['type']}")
            print(f"Parameters: {mechanism_params['params']}")
        else:
            print("⚠️ Mechanism generation failed")
            print(f"❌ Error: {mechanism_params.get('error', 'Unknown error')}")

if __name__ == "__main__":
    demo_inference() 