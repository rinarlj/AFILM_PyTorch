import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.subpixel import SubPixel1D
from .layers.transformer import TransformerBlock

class AFiLM(nn.Module):
    def __init__(self, n_step, block_size, n_filters):
        super(AFiLM, self).__init__()
        self.block_size = block_size
        self.n_filters = n_filters
        self.n_step = n_step

        # Initialize transformer
        max_len = int(self.n_step / self.block_size)
        self.transformer = TransformerBlock(num_layers=4, embed_dim=n_filters,
                                          num_heads=8, ff_dim=2048, 
                                          maximum_position_encoding=max_len)
        self.max_pool = nn.MaxPool1d(kernel_size=self.block_size, 
                                   stride=self.block_size, padding=0)

    def make_normalizer(self, x_in):
        """Pools to downsample along temporal dimension and runs transformer"""
        # Input shape: (batch, length, channels)
        batch_size, length, channels = x_in.shape
        
        # Ensure length is divisible by block_size
        if length % self.block_size != 0:
            pad_length = self.block_size - (length % self.block_size)
            x_in = F.pad(x_in, (0, 0, 0, pad_length))
            length = length + pad_length
        
        # Transpose for pooling: (batch, channels, length)
        x_in_transposed = x_in.transpose(1, 2)
        x_in_down = self.max_pool(x_in_transposed)
        
        # Transpose back: (batch, length_down, channels)
        x_in_down = x_in_down.transpose(1, 2)
        
        x_transformer = self.transformer(x_in_down, training=self.training)
        return x_transformer

    def apply_normalizer(self, x_in, x_norm):
        """Applies normalization weights by multiplying into respective blocks"""
        batch_size, length, channels = x_in.shape
        
        # Ensure divisible by block_size
        if length % self.block_size != 0:
            pad_length = self.block_size - (length % self.block_size)
            x_in = F.pad(x_in, (0, 0, 0, pad_length))
            length = length + pad_length
        
        n_blocks = length // self.block_size
        
        # Reshape into blocks
        x_norm = x_norm.reshape(batch_size, n_blocks, 1, self.n_filters)
        x_in_reshaped = x_in.reshape(batch_size, n_blocks, self.block_size, self.n_filters)
        
        # Apply normalization
        x_out = x_norm * x_in_reshaped
        
        # Return to original shape
        x_out = x_out.reshape(batch_size, n_blocks * self.block_size, self.n_filters)
        
        # Remove padding if needed
        if x_out.shape[1] > length:
            x_out = x_out[:, :length, :]
            
        return x_out

    def forward(self, x):
        assert len(x.shape) == 3, 'Input should be (batch_size, steps, num_features)'
        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x


class AFiLMModel(nn.Module):
    def __init__(self, n_layers=4, scale=4):
        super(AFiLMModel, self).__init__()
        
        self.n_layers = n_layers
        self.scale = scale

        # Model parameters
        n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
        n_step = [4096, 2048, 1024, 512, 256, 512, 1024, 2048, 4096]

        # DOWN SAMPLING LAYERS
        self.down_blocks = nn.ModuleList()
        
        in_channels = 1
        for l in range(n_layers):
            nf = n_filters[l]
            fs = n_filtersizes[l]
            ns = n_step[l]
            nb = 128 // (2 ** l)  # block size for this layer

            down_block = nn.ModuleDict({
                'conv': nn.Conv1d(in_channels, nf, fs, dilation=2, padding='same'),
                'pool': nn.MaxPool1d(2, stride=2),
                'afilm': AFiLM(ns, nb, nf)
            })
            nn.init.orthogonal_(down_block['conv'].weight)
            self.down_blocks.append(down_block)
            in_channels = nf

        # BOTTLENECK
        self.bottleneck_conv = nn.Conv1d(in_channels, n_filters[n_layers], 
                                       n_filtersizes[n_layers], dilation=2, padding='same')
        self.bottleneck_pool = nn.MaxPool1d(2, stride=2)
        self.bottleneck_dropout = nn.Dropout(0.5)
        nb_bottleneck = 128 // (2 ** n_layers)
        self.bottleneck_afilm = AFiLM(n_step[n_layers], nb_bottleneck, n_filters[n_layers])
        nn.init.orthogonal_(self.bottleneck_conv.weight)

        # UP SAMPLING LAYERS
        self.up_blocks = nn.ModuleList()
        
        for l in range(n_layers-1, -1, -1):
            nf = n_filters[l]
            fs = n_filtersizes[l]
            ns = n_step[l]
            nb = 128 // (2 ** l)

            # Input channels: from previous layer + skip connection
            up_in_channels = n_filters[l+1] if l == n_layers-1 else n_filters[l+1] * 2
            
            up_block = nn.ModuleDict({
                'conv': nn.Conv1d(up_in_channels, 2 * nf, fs, dilation=2, padding='same'),
                'dropout': nn.Dropout(0.5),
                'afilm': AFiLM(ns, nb, nf)
            })
            nn.init.orthogonal_(up_block['conv'].weight)
            self.up_blocks.append(up_block)

        # OUTPUT LAYER
        self.output_conv = nn.Conv1d(n_filters[0] * 2, 2, 9, padding='same')
        nn.init.normal_(self.output_conv.weight)

    def forward(self, x):
        # Input: (batch, length, channels) -> (batch, channels, length)
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.transpose(1, 2)
        
        inputs = x
        skip_connections = []

        # DOWN SAMPLING
        for i, block in enumerate(self.down_blocks):
            x = block['conv'](x)
            x = block['pool'](x)
            x = F.leaky_relu(x, 0.2)
            
            # AFiLM processing
            x_afilm = x.transpose(1, 2)  # (batch, length, channels)
            x_afilm = block['afilm'](x_afilm)
            x = x_afilm.transpose(1, 2)  # back to (batch, channels, length)
            
            skip_connections.append(x)

        # BOTTLENECK
        x = self.bottleneck_conv(x)
        x = self.bottleneck_pool(x)
        x = self.bottleneck_dropout(x)
        x = F.leaky_relu(x, 0.2)
        
        x_afilm = x.transpose(1, 2)
        x_afilm = self.bottleneck_afilm(x_afilm)
        x = x_afilm.transpose(1, 2)

        # UP SAMPLING
        for i, block in enumerate(self.up_blocks):
            x = block['conv'](x)
            x = block['dropout'](x)
            x = F.relu(x)
            
            # SubPixel upsampling
            x = x.transpose(1, 2)
            x = SubPixel1D(x, r=2)
            
            # AFiLM
            x = block['afilm'](x)
            
            # Skip connection
            skip = skip_connections[-(i+1)]
            skip = skip.transpose(1, 2)  # (batch, length, channels)
            x = torch.cat([x, skip], dim=-1)
            
            x = x.transpose(1, 2)  # back to (batch, channels, length)

        # OUTPUT
        x = self.output_conv(x)
        x = x.transpose(1, 2)
        x = SubPixel1D(x, r=2)
        
        # Residual connection
        outputs = x + inputs.transpose(1, 2)  # inputs is (batch, channels, length)

        return outputs  # (batch, length, channels)


def get_afilm(n_layers=4, scale=4):
    return AFiLMModel(n_layers=n_layers, scale=scale)
