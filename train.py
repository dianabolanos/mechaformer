import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import json
from tqdm import tqdm
import math
import pickle
import argparse
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config import Config

# Import x-transformers
from x_transformers import TransformerWrapper, Decoder, Encoder, ContinuousTransformerWrapper
import torch.nn.functional as F


class BinDistanceLoss(nn.Module):
    """
    Custom loss function that applies distance-based loss for bin tokens (0-200)
    and standard cross-entropy for other tokens.
    """
    def __init__(self, vocab_size, bin_start_id=Config.BIN_START_ID, bin_end_id=Config.BIN_END_ID, 
                 ignore_index=0, temperature=2.0, bin_weight=1.0):
        """
        Args:
            vocab_size: Total vocabulary size
            bin_start_id: Start ID for bin tokens (inclusive)
            bin_end_id: End ID for bin tokens (inclusive)
            ignore_index: Token ID to ignore (usually padding)
            temperature: Temperature for soft target generation
            bin_weight: Weight for bin loss relative to regular CE loss
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.bin_start_id = bin_start_id
        self.bin_end_id = bin_end_id
        self.num_bins = bin_end_id - bin_start_id + 1
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.bin_weight = bin_weight
        
        # Standard cross-entropy for non-bin tokens
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        
    def create_soft_targets(self, true_bins, num_classes):
        """Create soft targets with Gaussian-like distribution around true bin."""
        batch_size = true_bins.size(0)
        soft_targets = torch.zeros(batch_size, num_classes, device=true_bins.device)
        
        # Create indices for all classes
        indices = torch.arange(num_classes, device=true_bins.device).unsqueeze(0)
        
        # Calculate distances from true bin for each sample
        true_bins_expanded = true_bins.unsqueeze(1)
        distances = torch.abs(indices - true_bins_expanded).float()
        
        # Apply Gaussian-like distribution
        # exp(-distance^2 / (2 * temperature^2))
        soft_targets = torch.exp(-distances.pow(2) / (2 * self.temperature ** 2))
        
        # Normalize to sum to 1
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        
        return soft_targets
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model outputs of shape (batch_size * seq_len, vocab_size)
            targets: True token IDs of shape (batch_size * seq_len)
        """
        # Identify which targets are bins
        is_bin = (targets >= self.bin_start_id) & (targets <= self.bin_end_id) & (targets != self.ignore_index)
        is_not_bin = ~is_bin & (targets != self.ignore_index)
        
        # Initialize loss
        loss = torch.zeros_like(targets, dtype=torch.float)
        
        # Process bin tokens with distance-based loss
        if is_bin.any():
            bin_indices = torch.where(is_bin)[0]
            bin_predictions = predictions[bin_indices]
            bin_targets = targets[bin_indices]
            
            # Extract predictions for bin tokens only
            bin_logits = bin_predictions[:, self.bin_start_id:self.bin_end_id+1]
            
            # Convert target IDs to bin indices (0-based)
            bin_target_indices = bin_targets - self.bin_start_id
            
            # Create soft targets
            soft_targets = self.create_soft_targets(bin_target_indices, self.num_bins)
            
            # Compute KL divergence loss
            log_probs = F.log_softmax(bin_logits, dim=1)
            bin_loss = -(soft_targets * log_probs).sum(dim=1)
            
            # Apply bin weight
            loss[bin_indices] = bin_loss * self.bin_weight
        
        # Process non-bin tokens with standard cross-entropy
        if is_not_bin.any():
            ce_losses = self.ce_loss(predictions, targets)
            loss[is_not_bin] = ce_losses[is_not_bin]
        
        # Return mean loss over non-ignored tokens
        valid_tokens = targets != self.ignore_index
        if valid_tokens.any():
            return loss[valid_tokens].mean()
        else:
            return loss.mean()


class MechanismDataset(Dataset):
    def __init__(self, processed_data_path="processed_data.pkl", max_samples=None, min_mechanism_instances=20000, allowed_mechanism_types=None):
        self.processed_data_path = processed_data_path
        self.max_samples = max_samples
        self.min_mechanism_instances = min_mechanism_instances
        self.allowed_mechanism_types = allowed_mechanism_types

        # Load pre-processed data
        self._load_processed_data()

    def _load_processed_data(self):
        """Load pre-processed data from pickle file"""
        print(f"Loading pre-processed data from {self.processed_data_path}...")
        
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
        
        print(f"Loaded {len(self.processed_data)} processed samples before filtering")
        print(f"Original mechanism types: {sorted(self.mechanism_types)}")
        
        # Apply mechanism type filtering first
        if self.allowed_mechanism_types is not None:
            self._filter_by_mechanism_type()
        
        # Then apply minimum instance filtering
        if self.min_mechanism_instances > 0:
            self._filter_by_mechanism_count()
        
        # Limit samples if specified (after filtering)
        if self.max_samples is not None and len(self.processed_data) > self.max_samples:
            print(f"Limiting to {self.max_samples} samples (out of {len(self.processed_data)} total)")
            self.processed_data = self.processed_data[:self.max_samples]
        
        print(f"Final dataset: {len(self.processed_data)} processed samples")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Final mechanism types: {sorted(self.mechanism_types)}")

    def _filter_by_mechanism_type(self):
        """Filter by allowed mechanism types using bar type keywords"""
        print(f"\nFiltering by mechanism type keywords: {self.allowed_mechanism_types}")
        
        # Get current mechanism counts before filtering
        mechanism_counts = self._count_mechanism_instances()
        available_types = set(mechanism_counts.keys())
        
        print(f"Available mechanism types in dataset: {sorted(available_types)}")
        
        # Map bar type keywords to actual mechanism type names
        selected_types = set()
        
        for keyword in self.allowed_mechanism_types:
            keyword_lower = keyword.lower()
            
            if keyword_lower == "4-bar":
                # 4-bar mechanisms: RRRP, RRRR, RPRR, RRPR
                four_bar_types = {"RRRP", "RRRR", "RPRR", "RRPR"}
                found_types = four_bar_types.intersection(available_types)
                selected_types.update(found_types)
                print(f"  4-bar keyword mapped to: {sorted(found_types)}")
                if four_bar_types - available_types:
                    print(f"    Expected but not found: {sorted(four_bar_types - available_types)}")
                    
            elif keyword_lower == "6-bar":
                # 6-bar mechanisms: start with "Watt" or "Steph"
                six_bar_types = {t for t in available_types if t.startswith("Watt") or t.startswith("Steph")}
                selected_types.update(six_bar_types)
                print(f"  6-bar keyword mapped to: {sorted(six_bar_types)}")
                
            elif keyword_lower == "8-bar":
                # 8-bar mechanisms: start with "Type"
                eight_bar_types = {t for t in available_types if t.startswith("Type")}
                selected_types.update(eight_bar_types)
                print(f"  8-bar keyword mapped to: {sorted(eight_bar_types)}")
                
            else:
                # Direct mechanism type name (fallback for exact matches)
                if keyword in available_types:
                    selected_types.add(keyword)
                    print(f"  Direct match: {keyword}")
                else:
                    print(f"  Unknown keyword/type: {keyword}")
        
        # Identify removed types
        removed_types = available_types - selected_types
        
        print(f"\nMechanism type filtering analysis:")
        print(f"  Requested keywords: {sorted(self.allowed_mechanism_types)}")
        print(f"  Selected mechanism types: {sorted(selected_types)}")
        print(f"  Total selected types: {len(selected_types)}")
        
        if removed_types:
            print(f"  Removed mechanism types: {sorted(removed_types)}")
            for mech_type in sorted(removed_types):
                count = mechanism_counts[mech_type]
                print(f"    {mech_type}: {count} instances")
        
        # Filter the processed data
        original_count = len(self.processed_data)
        self.processed_data = [
            sample for sample in self.processed_data 
            if sample["mechanism_type"] in selected_types
        ]
        filtered_count = len(self.processed_data)
        
        # Update mechanism types set
        self.mechanism_types = selected_types
        
        print(f"\nMechanism type filtering summary:")
        print(f"  Original samples: {original_count}")
        print(f"  Filtered samples: {filtered_count}")
        print(f"  Removed samples: {original_count - filtered_count}")
        print(f"  Retention rate: {filtered_count / original_count * 100:.1f}%")
        
        if filtered_count == 0:
            print("WARNING: No samples remaining after mechanism type filtering!")
            print("Available mechanism types were:", sorted(available_types))
            print("Check if your keywords match the expected patterns.")

    def _filter_by_mechanism_count(self):
        """Filter out mechanism types with less than min_mechanism_instances samples"""
        print(f"\nFiltering mechanisms with less than {self.min_mechanism_instances} instances...")
        
        # Count instances of each mechanism type
        mechanism_counts = self._count_mechanism_instances()
        
        # Identify mechanism types to keep
        valid_mechanisms = set()
        removed_mechanisms = set()
        
        for mechanism_type, count in mechanism_counts.items():
            if count >= self.min_mechanism_instances:
                valid_mechanisms.add(mechanism_type)
            else:
                removed_mechanisms.add(mechanism_type)
        
        # Report filtering results
        print(f"Mechanism type analysis:")
        print(f"  Total mechanism types: {len(mechanism_counts)}")
        print(f"  Kept (>= {self.min_mechanism_instances} instances): {len(valid_mechanisms)}")
        print(f"  Removed (< {self.min_mechanism_instances} instances): {len(removed_mechanisms)}")
        
        if removed_mechanisms:
            print(f"\nRemoved mechanism types:")
            for mechanism_type in sorted(removed_mechanisms):
                count = mechanism_counts[mechanism_type]
                print(f"  {mechanism_type}: {count} instances")
        
        if valid_mechanisms:
            print(f"\nKept mechanism types:")
            for mechanism_type in sorted(valid_mechanisms):
                count = mechanism_counts[mechanism_type]
                print(f"  {mechanism_type}: {count} instances")
        
        # Filter the processed data
        original_count = len(self.processed_data)
        self.processed_data = [
            sample for sample in self.processed_data 
            if sample["mechanism_type"] in valid_mechanisms
        ]
        filtered_count = len(self.processed_data)
        
        # Update mechanism types set
        self.mechanism_types = valid_mechanisms
        
        print(f"\nFiltering summary:")
        print(f"  Original samples: {original_count}")
        print(f"  Filtered samples: {filtered_count}")
        print(f"  Removed samples: {original_count - filtered_count}")
        print(f"  Retention rate: {filtered_count / original_count * 100:.1f}%")

    def _count_mechanism_instances(self):
        """Count the number of instances for each mechanism type"""
        mechanism_counts = {}
        
        for sample in self.processed_data:
            mechanism_type = sample["mechanism_type"]
            mechanism_counts[mechanism_type] = mechanism_counts.get(mechanism_type, 0) + 1
        
        return mechanism_counts

    def get_mechanism_statistics(self):
        """Get detailed statistics about mechanism types in the dataset"""
        mechanism_counts = self._count_mechanism_instances()
        
        stats = {
            'total_samples': len(self.processed_data),
            'total_mechanism_types': len(mechanism_counts),
            'mechanism_counts': mechanism_counts,
            'min_instances': min(mechanism_counts.values()) if mechanism_counts else 0,
            'max_instances': max(mechanism_counts.values()) if mechanism_counts else 0,
            'avg_instances': sum(mechanism_counts.values()) / len(mechanism_counts) if mechanism_counts else 0
        }
        
        return stats

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


class MechanismTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=16,
        num_encoder_layers=8,
        num_decoder_layers=8,
        num_control_points=32,
        max_seq_len=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
        attn_flash=True,
        use_rmsnorm=True,
        ff_glu=True,
        rotary_pos_emb=True,
        attn_qk_norm=True,
        ff_swish=True,
        ff_no_bias=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_control_points = num_control_points
        self.vocab_size = vocab_size

        self.curve_encoder = ContinuousTransformerWrapper(
            dim_in=2,  # 2D control points (x, y)
            max_seq_len=num_control_points,
            use_abs_pos_emb=not rotary_pos_emb,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_encoder_layers,
                heads=nhead,
                attn_flash=attn_flash,
                use_rmsnorm=use_rmsnorm,
                ff_glu=ff_glu,
                rotary_pos_emb=rotary_pos_emb,
                attn_qk_norm=attn_qk_norm,
                ff_swish=ff_swish,
                ff_no_bias=ff_no_bias,
            )
        )

        self.dsl_decoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=not rotary_pos_emb,
            attn_layers=Decoder(
                dim=d_model,
                depth=num_decoder_layers,
                heads=nhead,
                attn_flash=attn_flash,
                use_rmsnorm=use_rmsnorm,
                ff_glu=ff_glu,
                rotary_pos_emb=rotary_pos_emb,
                attn_qk_norm=attn_qk_norm,
                ff_swish=ff_swish,
                ff_no_bias=ff_no_bias,
                cross_attend=True,
            )
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, control_points, tgt_sequence, tgt_mask=None):
        curve_encoded = self.curve_encoder(
            control_points,
            return_embeddings=True
        )

        output = self.dsl_decoder(
            tgt_sequence,
            context=curve_encoded,
            mask=tgt_mask,
            return_embeddings=False
        )
        
        return output

    def generate_square_subsequent_mask(self, sz):
        return torch.ones(sz, sz, dtype=torch.bool).tril()


def setup(rank, world_size):
    """Initialize the process group for distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # NCCL environment variables for better stability
    os.environ['NCCL_TIMEOUT_MS'] = '1800000'  # 30 minutes timeout
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback interface for localhost
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P for localhost training
    
    # Choose backend based on availability and world size
    if torch.cuda.is_available() and world_size <= 2:
        backend = "gloo"
        print(f"Using GLOO backend for rank {rank}")
    elif torch.cuda.is_available():
        backend = "nccl" 
        print(f"Using NCCL backend for rank {rank}")
    else:
        backend = "gloo"
        print(f"Using GLOO backend (CPU) for rank {rank}")
    
    # Initialize the process group with timeout
    dist.init_process_group(
        backend=backend, 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )
    
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        print(f"Set CUDA device to {rank}")
    else:
        print("CUDA not available, using CPU")

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, best_loss, vocab_size, checkpoint_path):
    """Save comprehensive checkpoint for resume training"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_loss': best_loss,
        'vocab_size': vocab_size,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint for resume training"""
    if not os.path.exists(checkpoint_path):
        return None, 0, [], [], float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (
        checkpoint,
        checkpoint['epoch'],
        checkpoint.get('train_losses', []),
        checkpoint.get('val_losses', []),
        checkpoint.get('best_loss', float('inf'))
    )

def save_loss_history(train_losses, val_losses, train_ppls, val_ppls, json_path):
    """Save loss history to JSON file"""
    loss_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
        'num_epochs': len(train_losses),
        'last_updated': datetime.now().isoformat()
    }
    
    with open(json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)

def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training & perplexity curves"""
    plt.figure(figsize=(12, 5))

    epochs = range(0, len(train_losses))
    train_ppls = [math.exp(l) for l in train_losses]
    val_ppls = [math.exp(l) for l in val_losses] if val_losses else []

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Perplexity subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_ppls, 'g-', label='Training PPL', linewidth=2)
    if val_ppls:
        plt.plot(epochs, val_ppls, 'orange', label='Validation PPL', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def analyze_split_distribution(dataset, train_indices, val_indices):
    """Analyze and report mechanism type distribution in train/validation splits"""
    train_mechanisms = {}
    val_mechanisms = {}
    
    # Count mechanism types in training set
    for idx in train_indices:
        sample = dataset.processed_data[idx]
        mech_type = sample["mechanism_type"]
        train_mechanisms[mech_type] = train_mechanisms.get(mech_type, 0) + 1
    
    # Count mechanism types in validation set
    for idx in val_indices:
        sample = dataset.processed_data[idx]
        mech_type = sample["mechanism_type"]
        val_mechanisms[mech_type] = val_mechanisms.get(mech_type, 0) + 1
    
    print("\nMechanism Type Distribution Analysis:")
    print("=" * 60)
    print(f"{'Type':<20} {'Train':<10} {'Val':<10} {'Val %':<10} {'Total':<10}")
    print("-" * 60)
    
    all_types = set(train_mechanisms.keys()) | set(val_mechanisms.keys())
    total_train = sum(train_mechanisms.values())
    total_val = sum(val_mechanisms.values())
    
    for mech_type in sorted(all_types):
        train_count = train_mechanisms.get(mech_type, 0)
        val_count = val_mechanisms.get(mech_type, 0)
        total_count = train_count + val_count
        val_percentage = (val_count / total_count * 100) if total_count > 0 else 0
        
        print(f"{mech_type:<20} {train_count:<10} {val_count:<10} {val_percentage:<10.1f} {total_count:<10}")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_train:<10} {total_val:<10} {total_val/(total_train+total_val)*100:<10.1f} {total_train+total_val:<10}")
    print("=" * 60)

def train_model(rank, world_size, args):
    """Main training function for distributed training"""
    print(f"Running DDP on rank {rank} of {world_size}")
    
    # Setup distributed training
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    
    # Load dataset
    if rank == 0:
        print("Loading dataset...")
    
    dataset = MechanismDataset(
        processed_data_path=args.processed_data_path,
        max_samples=args.max_samples,
        min_mechanism_instances=args.min_mechanism_instances,
        allowed_mechanism_types=args.allowed_mechanism_types # Pass allowed types to dataset
    )

    # Save vocabulary for later use (only on rank 0)
    if rank == 0:
        with open("vocab.pkl", "wb") as f:
            pickle.dump(
                {
                    "vocab": dataset.vocab,
                    "id_to_token": dataset.id_to_token,
                    "vocab_size": dataset.vocab_size,
                },
                f,
            )

    # Create train/validation split
    train_indices, val_indices = create_validation_split(dataset, val_split=args.val_split)
    
    # Analyze split distribution (only on rank 0)
    if rank == 0:
        analyze_split_distribution(dataset, train_indices, val_indices)
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create distributed samplers and data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Using max_seq_len: {dataset.max_seq_len} (from processed data)")

    # Initialize model with x-transformers features
    model = MechanismTransformer(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_control_points=Config.BSPLINE_CONTROL_POINTS,
        max_seq_len=dataset.max_seq_len,
        # x-transformers features
        attn_flash=getattr(args, 'attn_flash', True),
        use_rmsnorm=getattr(args, 'use_rmsnorm', True),
        ff_glu=getattr(args, 'ff_glu', True),
        rotary_pos_emb=getattr(args, 'rotary_pos_emb', True),
        attn_qk_norm=getattr(args, 'attn_qk_norm', True),
        ff_swish=getattr(args, 'ff_swish', True),
        ff_no_bias=getattr(args, 'ff_no_bias', True),
        # regularisation
        attn_dropout=args.dropout_rate,
        ff_dropout=args.dropout_rate,
    ).to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[rank] if torch.cuda.is_available() else None,
        )
        print(f"Wrapped model with DDP on rank {rank}")
    else:
        print(f"Single GPU training on rank {rank}, no DDP wrapping needed")

    # Loss and optimizer
    if args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<PAD>"])
        if rank == 0:
            print("Using standard cross-entropy loss")
    elif args.loss_type == 'bin_distance':
        # Determine bin token IDs based on vocabulary structure
        # Special tokens: 0-3, DSL tokens: 4-7, Bin tokens: 8 onwards
        bin_start_id = Config.BIN_START_ID
        bin_end_id = Config.BIN_END_ID
        
        criterion = BinDistanceLoss(
            vocab_size=dataset.vocab_size,
            bin_start_id=bin_start_id,
            bin_end_id=bin_end_id,
            ignore_index=dataset.vocab["<PAD>"],
            temperature=args.bin_temperature,
            bin_weight=args.bin_weight
        )
        if rank == 0:
            print(f"Using bin distance loss with temperature={args.bin_temperature}, bin_weight={args.bin_weight}")
            print(f"Bin token range: {bin_start_id}-{bin_end_id} (total {Config.COORD_BINS} bins)")
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Initialize loss tracking
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    start_epoch = 0
    best_loss = float("inf")
    
    # Setup file paths
    checkpoint_path = args.checkpoint_path or "checkpoint.pth"
    loss_json_path = args.loss_json_path or "loss_history.json"
    plot_path = args.plot_path or "training_curves.png"
    
    # Try to resume from checkpoint
    if args.resume and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Resuming from checkpoint: {checkpoint_path}")
        
        _, start_epoch, train_losses, val_losses, best_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
        
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")

    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Starting training from epoch {start_epoch + 1}")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        # Training phase
        model.train()
        train_sampler.set_epoch(epoch)  # Important for proper shuffling in distributed training
        
        train_loss = 0
        num_train_batches = 0

        # Only show progress bar on rank 0
        if rank == 0:
            train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Train]", mininterval=2.0, miniters=500)
        else:
            train_progress_bar = train_dataloader

        for batch in train_progress_bar:
            control_points = batch["control_points"].to(device)
            dsl_sequence = batch["dsl_sequence"].to(device)

            # Prepare input and target
            tgt_input = dsl_sequence[:, :-1]  # All tokens except last
            tgt_output = dsl_sequence[:, 1:]  # All tokens except first

            # Forward pass (x-transformers handles causal masking automatically)
            optimizer.zero_grad()
            output = model(control_points, tgt_input)

            # Calculate loss
            loss = criterion(
                output.reshape(-1, dataset.vocab_size), tgt_output.reshape(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

            if rank == 0:
                train_progress_bar.set_postfix({"loss": loss.item()})

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            if rank == 0:
                val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Val]", mininterval=2.0, miniters=500)
            else:
                val_progress_bar = val_dataloader
                
            for batch in val_progress_bar:
                control_points = batch["control_points"].to(device)
                dsl_sequence = batch["dsl_sequence"].to(device)

                # Prepare input and target
                tgt_input = dsl_sequence[:, :-1]
                tgt_output = dsl_sequence[:, 1:]

                # Forward pass
                output = model(control_points, tgt_input)

                # Calculate loss
                loss = criterion(
                    output.reshape(-1, dataset.vocab_size), tgt_output.reshape(-1)
                )

                val_loss += loss.item()
                num_val_batches += 1

                if rank == 0:
                    val_progress_bar.set_postfix({"val_loss": loss.item()})

        # Average losses across all GPUs
        avg_train_loss = train_loss / max(num_train_batches, 1)
        avg_val_loss = val_loss / max(num_val_batches, 1)
        
        # Synchronize losses across all processes
        train_loss_tensor = torch.tensor(avg_train_loss, device=device)
        val_loss_tensor = torch.tensor(avg_val_loss, device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        
        avg_train_loss = (train_loss_tensor / world_size).item()
        avg_val_loss = (val_loss_tensor / world_size).item()
        
        # Update learning rate scheduler with validation loss
        scheduler.step(avg_val_loss)

        # Store losses and perplexities for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_ppls.append(math.exp(avg_train_loss))
        val_ppls.append(math.exp(avg_val_loss))

        if rank == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train PPL: {math.exp(avg_train_loss):.2f}, Val PPL: {math.exp(avg_val_loss):.2f}")

        # Save checkpoint and best model (only on rank 0)
        if rank == 0:
            # Save regular checkpoint
            save_checkpoint(
                epoch, model, optimizer, scheduler, 
                train_losses, val_losses, best_loss, 
                dataset.vocab_size, checkpoint_path
            )
            
            # Save loss history to JSON
            save_loss_history(train_losses, val_losses, train_ppls, val_ppls, loss_json_path)
            
            # Save best model if improved
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "vocab_size": dataset.vocab_size,
                    },
                    "best_mechanism_transformer.pth",
                )
                print(f"Saved best model with val loss: {avg_val_loss:.4f}")
            
            # Plot and save training curves each epoch
            plot_training_curves(train_losses, val_losses, plot_path)
    
    # Cleanup distributed training
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Distributed Training for Mechanism Transformer')
    
    # Data parameters
    parser.add_argument('--processed_data_path', type=str, default='processed_data.pkl',
                       help='Path to pre-processed data file')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use for training (for testing)')
    parser.add_argument('--min_mechanism_instances', type=int, default=25000,
                       help='Minimum number of instances required for a mechanism type to be included (set to 0 to disable filtering)')
    parser.add_argument('--allowed_mechanism_types', type=str, nargs='+', default=None,
                       help='List of mechanism types to allow using keywords: "4-bar" (RRRP,RRRR,RPRR,RRPR), "6-bar" (Watt*,Steph*), "8-bar" (Type*), or exact mechanism names')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers')
    
    # x-transformers features
    parser.add_argument('--attn_flash', action='store_true', default=True, help='Use flash attention (default: True)')
    parser.add_argument('--no_attn_flash', action='store_false', dest='attn_flash', help='Disable flash attention')
    parser.add_argument('--use_rmsnorm', action='store_true', default=True, help='Use RMSNorm instead of LayerNorm (default: True)')
    parser.add_argument('--no_rmsnorm', action='store_false', dest='use_rmsnorm', help='Disable RMSNorm')
    parser.add_argument('--ff_glu', action='store_true', default=True, help='Use GLU in feedforward (default: True)')
    parser.add_argument('--no_ff_glu', action='store_false', dest='ff_glu', help='Disable GLU in feedforward')
    parser.add_argument('--rotary_pos_emb', action='store_true', default=True, help='Use rotary positional embeddings (default: True)')
    parser.add_argument('--no_rotary_pos_emb', action='store_false', dest='rotary_pos_emb', help='Disable rotary positional embeddings')
    parser.add_argument('--attn_qk_norm', action='store_true', default=True, help='Use QK normalization in attention (default: True)')
    parser.add_argument('--no_attn_qk_norm', action='store_false', dest='attn_qk_norm', help='Disable QK normalization')
    parser.add_argument('--ff_swish', action='store_true', default=True, help='Use Swish activation in feedforward (default: True)')
    parser.add_argument('--no_ff_swish', action='store_false', dest='ff_swish', help='Disable Swish activation')
    parser.add_argument('--ff_no_bias', action='store_true', default=True, help='Remove bias from feedforward layers (default: True)')
    parser.add_argument('--ff_with_bias', action='store_false', dest='ff_no_bias', help='Keep bias in feedforward layers')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warm-up steps for LR scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for attention and feed-forward layers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing value for CE loss')
    
    # Loss function parameters
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'bin_distance'], 
                       default='cross_entropy', help='Type of loss function to use')
    parser.add_argument('--bin_temperature', type=float, default=1.0, 
                       help='Temperature for soft targets in bin distance loss')
    parser.add_argument('--bin_weight', type=float, default=2.0, 
                       help='Weight for bin loss relative to regular CE loss')
    
    # Resume training and checkpointing
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='Path to checkpoint file')
    parser.add_argument('--loss_json_path', type=str, default='loss_history.json', help='Path to loss history JSON file')
    parser.add_argument('--plot_path', type=str, default='training_curves.png', help='Path to training curves plot')
    parser.add_argument('--plot_interval', type=int, default=5, help='Plot training curves every N epochs')
    
    # Distributed training
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    if args.world_size > 1:
        # Multi-GPU training
        import torch.multiprocessing as mp
        mp.spawn(train_model, args=(args.world_size, args), nprocs=args.world_size)
    else:
        # Single GPU training
        train_model(0, 1, args)

if __name__ == "__main__":
    main()
