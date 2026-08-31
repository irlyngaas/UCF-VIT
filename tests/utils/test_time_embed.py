import torch

from UCF_VIT.utils.time_embed import SinusoidalEmbeddings


def test_sinusoidal_embeddings_lookup():
    embed_dim = 6
    time_steps = 10
    module = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=embed_dim)

    x = torch.zeros(2, 1)
    t = torch.tensor([0, 5])
    out = module(x, t)

    assert out.shape == (2, embed_dim)
    torch.testing.assert_close(out[0], module.embeddings[0])
    torch.testing.assert_close(out[1], module.embeddings[5])
    # timestep 0 -> sin(0)=0, cos(0)=1
    torch.testing.assert_close(out[0, 0::2], torch.zeros(embed_dim // 2))
    torch.testing.assert_close(out[0, 1::2], torch.ones(embed_dim // 2))
