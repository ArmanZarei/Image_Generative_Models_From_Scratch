import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, dim_embeddings):
        super().__init__()

        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(num_embeddings, dim_embeddings)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x, return_indices=False):
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        x_flatted = x_permuted.view(-1, x.shape[1])

        # Using Broadcasting: (N, 1, dim_embeddings) - (num_embeddings, dim_embeddings) = (N, num_embeddings, dim_embeddings)
        indices = torch.norm(x_flatted.unsqueeze(1) - self.embedding.weight, dim=2).argmin(dim=1)

        if return_indices:
            return indices.view(x.shape[0], x.shape[2], x.shape[3])

        quantized = self.embedding(indices) # (-1, dim_embeddings)

        return quantized.view(x_permuted.shape).permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self.net(x)
    

class Encoder(nn.Module):
    def __init__(self, img_channels=3, dim_hidden=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(img_channels, dim_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_hidden, dim_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(dim_hidden, dim_hidden),
            ResidualBlock(dim_hidden, dim_hidden),
            nn.Conv2d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, dim_hidden=256, img_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            ResidualBlock(dim_hidden, dim_hidden),
            ResidualBlock(dim_hidden, dim_hidden),
            nn.ConvTranspose2d(dim_hidden, dim_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim_hidden, img_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        return self.net(x)

