import numpy as np
from collections import Counter

def compute_codebook_stats(token_sequences, vocab_size=None, eps=1e-12):
    """
    token_sequences: list of lists (or arrays) of token IDs
                     e.g., [[1, 5, 3], [2, 2, 7, 1], ...]
    vocab_size: optional, total size of codebook (K)

    Returns:
        entropy (H)
        perplexity (exp(H))
        usage_fraction (#tokens used / K)
    """

    # Flatten all tokens
    all_tokens = np.concatenate(token_sequences)

    total = len(all_tokens)

    # Count frequencies
    counts = Counter(all_tokens)

    # Convert to probabilities
    probs = np.array([c / total for c in counts.values()])

    # Entropy
    entropy = -np.sum(probs * np.log(probs + eps))

    # Perplexity
    perplexity = np.exp(entropy)

    # Codebook usage (if vocab size known)
    if vocab_size is not None:
        usage_fraction = len(counts) / vocab_size
    else:
        usage_fraction = None

    return {
        "entropy": entropy,
        "perplexity": perplexity,
        "usage_fraction": usage_fraction,
        "num_active_tokens": len(counts),
        "total_tokens": total,
    }