import torch

__all__ = ['cosine_similarity', 'angle_differences', 'l2_differences']

def cosine_similarity(A, B):
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)

    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=torch.float32)
    
    magnitude_A = torch.norm(A)
    magnitude_B = torch.norm(B)
    
    if magnitude_A == 0 or magnitude_B == 0:
        cosine_sim = 0.
    else:
        dot_product = torch.dot(A, B)
        cosine_sim = dot_product / (magnitude_A * magnitude_B)
        cosine_sim = cosine_sim.item()
    
    return cosine_sim

def angle_differences(A, B):
    cosine_sim = cosine_similarity(A, B)
    cosine_sim = max(min(cosine_sim, 1.0), -1.0)

    angle_radians = torch.acos(torch.tensor(cosine_sim))

    return angle_radians.item()

def l2_differences(A, B):
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)

    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=torch.float32)
 
    l2_diff = torch.norm(A - B)

    return l2_diff.item()


