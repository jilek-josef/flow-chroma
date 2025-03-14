import torch

def cosine_optimal_transport(X, Y):
    """
    Approximate optimal transport between two sets using cosine similarity,
    enforcing one-to-one assignment like the Hungarian method.

    Args:
        X (torch.Tensor): (N, D) Feature vectors on CUDA.
        Y (torch.Tensor): (M, D) Feature vectors on CUDA.

    Returns:
        transport_cost (torch.Tensor): Optimal transport cost (scalar).
        (row_indices, col_indices): Hard assignment indices.
    """
    # Normalize vectors for cosine similarity
    X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (torch.norm(Y, dim=1, keepdim=True) + 1e-8)

    # Compute cosine similarity cost matrix
    cost_matrix = -torch.mm(X_norm, Y_norm.T)  # Negative since we minimize

    # Get one-to-one assignment
    row_indices = torch.arange(cost_matrix.shape[0], device=cost_matrix.device)
    col_indices = torch.zeros_like(row_indices)  # Placeholder for assignments

    assigned_cols = set()
    for i in range(cost_matrix.shape[0]):
        # Find the best column match for each row
        sorted_cols = torch.argsort(cost_matrix[i])
        for col in sorted_cols:
            if col.item() not in assigned_cols:
                col_indices[i] = col
                assigned_cols.add(col.item())
                break

    # Compute transport cost
    transport_cost = cost_matrix.gather(1, col_indices.unsqueeze(1)).sum()

    return transport_cost, (row_indices, col_indices)
