import torch


def gradient_penalty(critic, real, fake):
    """Penalize deviation of interpolated-input gradient norms from one."""
    alpha = torch.rand((real.shape[0], 1, 1, 1), device=real.device, dtype=real.dtype)
    mixed = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    scores = critic(mixed)
    gradients = torch.autograd.grad(scores, mixed, torch.ones_like(scores), create_graph=True)[0]
    norms = gradients.reshape(real.shape[0], -1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()
