# MechaFormer (AAAI 2026): Supplementary Materials

This repository contains the source code, evaluation scripts, and supplementary materials for the paper "MechaFormer: Sequence Learning for Kinematic Mechanism Design Automation".

## Overview

Designing mechanical mechanisms to trace specific paths is a classic yet notoriously difficult engineering problem, characterized by a vast and complex search space of discrete topologies and continuous parameters. We introduce MechaFormer, a Transformer-based model that tackles this challenge by treating mechanism design as a conditional sequence generation task.

## Key Features

- **Transformer-based Architecture**: Utilizes X-Transformers with advanced features including flash attention, RMSNorm, GLU feed-forward networks, and rotary positional embeddings

- **B-spline Processing**: Automatic conversion of input curves to standardized B-spline control points
- **Ground Joint Reconstruction**: Reconstruction of ground joints using BSI dictionary (note that the BSI dictionary is not included in this repository, and must be downloaded separately from https://github.com/purwarlab/vae_dataset_project.git)
- **Comprehensive Evaluation**: Multiple evaluation strategies including best-of-K sampling and topology constraints

**Note**: The simulator API must be run from the https://github.com/purwarlab/vae_dataset_project.git repository. Please follow the instructions provided in that repository for proper setup and usage.

## Quick Start

### Download Model Weights

Before running any code, you must download the model weights:

1. Download `mechanism_transformer_weights.pth` from:
   ```
   https://drive.google.com/file/d/1dw_PhEFzT51fU3SjGYwSPJQvJnRDfOQp/view?usp=sharing
   ```

2. Place the downloaded file in the root directory of this repository.

Note: The vocabulary file (`vocab.pkl`) is already included in the repository.

### Prerequisites

Before running the code, ensure you have Python 3.8+ and a virtual environment set up:

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation

Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn x-transformers einops loguru tqdm requests h5py
```

### Data Preparation

**Important**: You must first prepare the training data before running the demonstration:

1. **Download the Dataset**: Download the "Four, Six, and Eight Bar Mechanisms with Curves" dataset from Kaggle:
   ```
   https://www.kaggle.com/datasets/purwarlab/four-six-and-eight-bar-mechanisms-with-curves
   ```

2. **Extract and Set Path**: Extract the downloaded dataset and note the full path to the data directory.

3. **Update Configuration**: Edit `config.py` and set the `DATA_PATH` variable to point to your extracted dataset:
   ```python
   DATA_PATH = "/path/to/your/downloaded/dataset"
   ```

4. **Run Preprocessing**: Execute the preprocessing script to create the required `processed_data.pkl` file:
   ```bash
   python preprocessing_data.py --data_path /path/to/your/dataset --output_path processed_data.pkl
   ```
   
   **Optional parameters:**
   ```bash
   python preprocessing_data.py \
     --data_path /path/to/dataset \
     --output_path processed_data.pkl \
     --max_files 10000 \ # remove this parameter to process the full dataset
     --num_workers 4 \
     --random_seed 42
   ```
   
   This process will:
   - Process up to 846,480 mechanism-curve pairs (or subset if `--max_files` is not specified)
   - Apply B-spline curve fitting to standardize curves
   - Generate DSL (Domain Specific Language) representations
   - Create vocabulary mappings for tokenization
   - Save everything to `processed_data.pkl`
   
   **Processing time**: Full dataset processing takes minutes. For testing, use `--max_files 1000` to process a smaller subset.


### Running the Demonstration

Execute the main demonstration script:

```bash
python demo_curve_to_mechanism.py
```

This will:
1. Load a target curve from the validation dataset
2. Generate corresponding mechanism parameters using the pre-trained MechaFormer model
3. Simulate the resulting mechanism's motion
4. Save an animation (`.mp4`) showing the mechanism tracing the target curve

### Configuration

Modify the demonstration by editing variables in the `main()` function of `demo_curve_to_mechanism.py`:

- `SAMPLE_INDEX`: Integer to select different target curves from validation set (0 to validation_size-1)
- `OUTPUT_DIR`: Directory for saving output animations

## Repository Structure

```
mechaformer-supp/
├── demo_curve_to_mechanism.py           # Main demonstration script
├── inference_mechanism.py               # Model inference engine
├── train.py                            # Training script for the transformer model
├── config.py                           # System configuration and hyperparameters
├── bspline_utils.py                    # B-spline curve processing utilities
├── ground_joint_utils.py               # Ground joint reconstruction utilities
├── generation_utils.py                 # Text generation utilities for autoregressive decoding
├── mechanism_transformer_weights.pth   # Trained model weights (73MB)
├── vocab.pkl                           # Vocabulary file for sequence tokenization
├── processed_data.pkl                  # Preprocessed dataset (training and validation)
├── wrapper/                            # Mechanism simulation environment
│   ├── mechanism_core.py               # Core kinematic solver and animation
│   ├── mechanism_wrapper.py            # High-level simulation wrapper
└── experiments/                        # Evaluation and analysis scripts
    ├── best_at_k.py                   # Best-of-K sampling evaluation
    ├── best_at_rotation.py            # Rotational sampling evaluation
    └── best_at_topology.py            # Topology-constrained evaluation
```

## Model Architecture

The MechaFormer model features:

- **Encoder-Decoder Transformer**: 6 encoder layers, 6 decoder layers
- **Model Dimensions**: 256-dimensional model, 8 attention heads
- **Advanced Features**: 
  - Flash attention for efficient memory usage
  - RMSNorm for improved training stability
  - GLU (Gated Linear Units) in feed-forward layers
  - Rotary positional embeddings
  - Query-key normalization in attention
  - SwiGLU activation functions

## Data Processing Pipeline

### Input Processing

1. **Curve Input**: Accept target trajectories as sequences of 2D points
2. **B-spline Fitting**: Convert to standardized 64 B-spline control points
3. **Normalization**: Apply coordinate normalization for model stability
4. **Tokenization**: Convert mechanism parameters to discrete tokens using coordinate binning

### Output Processing

1. **Sequence Generation**: Autoregressive generation of mechanism specification tokens
2. **Parsing**: Convert token sequences back to mechanism parameters
3. **Ground Joint Reconstruction**: Restore ground joints using topology-specific BSI data
4. **Validation**: Kinematic validation through simulation


### Weights File (mechanism_transformer_weights.pth)
- **Size**: 73MB
- **Contents**: Model weights only
- **Use Case**: Inference, deployment, faster loading

## Evaluation Scripts

### Best-of-K Sampling (`experiments/best_at_k.py`)

Evaluates model performance by generating K candidate mechanisms and selecting the best based on curve similarity metrics.

```bash
python experiments/best_at_k.py
```

### Rotational Sampling (`experiments/best_at_rotation.py`)

Tests model robustness by evaluating performance across different curve orientations.

```bash
python experiments/best_at_rotation.py
```

### Topology Evaluation (`experiments/best_at_topology.py`)

Evaluates model performance when constrained to specific mechanism topologies.

```bash
python experiments/best_at_topology.py
```

## Dataset

This work utilizes the "Four, Six, and Eight Bar Mechanisms with Curves" dataset:
- **Source**: https://www.kaggle.com/datasets/purwarlab/four-six-and-eight-bar-mechanisms-with-curves
- **Size**: 846,480 mechanism-curve pairs
- **Coverage**: 24 different mechanism topologies
- **Applications**: Training, validation, and evaluation

## Acknowledgments

- Kaggle dataset contributors for mechanism data:
 https://www.kaggle.com/datasets/purwarlab/four-six-and-eight-bar-mechanisms-with-curves
- Simulating the mechanism motion is done using the simulator API from the https://github.com/purwarlab/vae_dataset_project.git repository.
