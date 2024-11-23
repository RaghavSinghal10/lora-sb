import torch


def get_svd_grad(input_matrix, rank, n_iter=10):
    """Use PyTorch's SVD which can utilize GPU acceleration"""
    
    # Handle meta tensors
    if hasattr(input_matrix, 'is_meta') and input_matrix.is_meta:
        input_matrix = input_matrix.to('cpu')
        input_matrix = input_matrix.clone().detach()
    
    # Convert to torch tensor if not already
    if not torch.is_tensor(input_matrix):
        input_matrix = torch.from_numpy(input_matrix)
    
    # Move to GPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_matrix = input_matrix.to(device)
    
    # Temporarily convert to float32 for SVD
    input_matrix_float = input_matrix.to(torch.float32)
    
    # Create random tensor in same dtype
    torch.manual_seed(42)  # for reproducibility
    size = input_matrix.size(1)
    R = torch.randn(size, rank, device=device, dtype=torch.float32)
    
    # Compute SVD with controlled dtypes
    U, s, V = torch.svd_lowrank(input_matrix_float, q=rank, niter=n_iter)
    
    # Convert results back to bfloat16
    U = U.to(torch.bfloat16)
    s = s.to(torch.bfloat16)
    V = V.to(torch.bfloat16)
    diag_s = torch.diag(s)
    
    # remove intermediate tensors from memory
    del input_matrix, input_matrix_float, R
    
    return U, diag_s, V.T


    
