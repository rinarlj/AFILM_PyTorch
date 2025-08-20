import torch
import torch.nn as nn

def SubPixel1D(I, r):
    # x shape: [batch, width, channels] où channels doit être divisible par r
    batch, width, channels = I.size()
    assert channels % r == 0, f"channels ({channels}) must be divisible by r ({r})"

    # Équivalent aux transpositions et batch_to_space_nd de TensorFlow
    # 1. Transpose [batch, width, channels] -> [channels, width, batch]
    I = I.permute(2, 1, 0)  # [channels, width, batch]

    # 2. Reshape pour simuler batch_to_space_nd
    # batch_to_space_nd avec block_shape=[r] réorganise les données
    channels_out = channels // r
    I = I.view(channels_out, r, width, batch)  # [channels//r, r, width, batch]
    I = I.permute(0, 2, 3, 1)  # [channels//r, width, batch, r]
    I = I.reshape(channels_out, width * r, batch)  # [channels//r, width*r, batch]

    # 3. Transpose final [channels//r, width*r, batch] -> [batch, width*r, channels//r]
    I = I.permute(2, 1, 0)  # [batch, width*r, channels//r]

    return I


class SubPixel1DLayer(nn.Module):
    """PyTorch nn.Module wrapper for SubPixel1D function"""
    def __init__(self, upscale_factor):
        super(SubPixel1DLayer, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return SubPixel1D(x, self.upscale_factor)
