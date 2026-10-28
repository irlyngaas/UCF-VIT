import torch
import math
import torch.nn as nn

class SinusoidalEmbeddings(nn.Module):
    """Fixed sinusoidal timestep embeddings, e.g. for conditioning a diffusion model on `t`.

    Precomputes a `(time_steps, embed_dim)` table of sine/cosine embeddings and looks
    up rows by timestep index.
    """

    def __init__(self, time_steps:int, embed_dim: int):
        """Precomputes the sinusoidal embedding table.

        Args:
            time_steps: Number of distinct timesteps to precompute embeddings for.
            embed_dim: Dimensionality of each embedding vector.
        """
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        """Looks up the embedding for each timestep in `t`.

        Args:
            x: Reference tensor used only to determine the target device.
            t: Tensor of timestep indices to look up.

        Returns:
            Tensor of embeddings for `t`, moved to `x`'s device.
        """
        embeds = self.embeddings[t].to(x.device)
        return embeds
