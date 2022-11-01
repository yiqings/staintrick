from torch import nn
import torch
from kornia.color import lab_to_rgb, rgb_to_lab
from einops import repeat

class LabPreNorm(nn.Module):
    def __init__(
        self, 
        model=None,
        epsilon=1e-7, 
        requires_grad=True,
    ):
        super(LabPreNorm, self).__init__()
        self.epsilon = epsilon
        self.mu = nn.Parameter(
            torch.tensor([64, 20, -10], dtype=torch.float32, requires_grad=requires_grad)
        )
        self.sigma = nn.Parameter(
            torch.tensor([12, 7, 5], dtype=torch.float32, requires_grad=requires_grad)
        )
        self.model = model if model is not None else nn.Identity()

    def forward(self,x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), f"image should be scaled to [0,1] rather than [0,256], current scale {x.min()}-{x.max()}"
        
        x = rgb_to_lab(x)
        
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W)

        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x = lab_to_rgb(x)

        return self.model(x)

if __name__ == "__main__":
    import torch

    lab_norm = LabPreNorm()
    x = torch.randn((1, 3, 224, 224)).clip(min=0,max=1)
    y = lab_norm(x)
    print((y-x).max())
