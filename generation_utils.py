"""
Utility functions for autoregressive generation with various sampling strategies.
"""

import torch
import torch.nn.functional as F


def top_k_sampling(logits, k, temperature=1.0):
    """
    Apply top-k sampling to logits.
    
    Args:
        logits: Logits from the model (vocab_size,)
        k: Number of top tokens to consider
        temperature: Temperature for sampling
    
    Returns:
        Sampled token index
    """
    # Apply temperature
    logits = logits / temperature
    
    # Get top k tokens
    top_k_values, top_k_indices = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)
    
    # Create a new tensor filled with -inf
    filtered_logits = torch.full_like(logits, float('-inf'))
    
    # Set the top k values
    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
    
    # Apply softmax and sample
    probs = F.softmax(filtered_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    return sampled_idx.item()


def autoregressive_generate(model, control_points, vocab, device, max_length=512, temperature=1.0, top_k=50, prefix_tokens=None):
    """
    Generate DSL sequence autoregressively starting from <SOS> token.
    
    Args:
        model: The transformer model
        control_points: Control points tensor (batch_size, num_points, 2)
        vocab: Vocabulary dictionary
        device: Device to run on
        max_length: Maximum sequence length
        temperature: Temperature for sampling
        top_k: Number of top tokens to consider for sampling
        prefix_tokens: Optional list of token IDs to start with after SOS (e.g., mechanism type)
    
    Returns:
        Generated sequence tensor
    """
    model.eval()
    batch_size = control_points.size(0)
    
    # Start with SOS token
    generated = torch.full((batch_size, 1), vocab["<SOS>"], dtype=torch.long, device=device)
    
    # Add prefix tokens if provided
    if prefix_tokens is not None:
        prefix_tensor = torch.tensor([prefix_tokens] * batch_size, dtype=torch.long, device=device)
        generated = torch.cat([generated, prefix_tensor], dim=1)
    
    with torch.no_grad():
        for _ in range(max_length - 1):
            # Get model predictions
            logits = model(control_points, generated)
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :]
            
            # Sample next tokens for each sequence in batch
            next_tokens = []
            for i in range(batch_size):
                if generated[i, -1] == vocab["<EOS>"] or generated[i, -1] == vocab["<PAD>"]:
                    # If already ended, add padding
                    next_tokens.append(vocab["<PAD>"])
                else:
                    # Sample next token
                    next_token = top_k_sampling(next_token_logits[i], k=top_k, temperature=temperature)
                    next_tokens.append(next_token)
            
            # Append next tokens
            next_tokens = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Check if all sequences have ended
            if all(generated[i, -1] == vocab["<EOS>"] or generated[i, -1] == vocab["<PAD>"] 
                   for i in range(batch_size)):
                break
    
    return generated