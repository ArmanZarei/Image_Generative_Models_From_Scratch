import torch
from torch import nn
import math
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, max_pos, d_model, out_dim):
        super().__init__()

        assert d_model % 2 == 0

        numerator = torch.arange(0, max_pos, dtype=torch.float)
        denominator = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        frac = numerator.unsqueeze(1) @ denominator.unsqueeze(0) # (pos_max, d_model//2)

        self.embd = torch.stack((torch.sin(frac), torch.cos(frac)), dim=-1).view(max_pos, d_model)

        self.net = nn.Sequential(
            nn.Embedding.from_pretrained(self.embd),
            nn.Linear(d_model, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.net(t)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gp_norm_groups=16):
        super().__init__()

        self.gp_norm = nn.GroupNorm(num_groups=gp_norm_groups, num_channels=in_channels, eps=1e-6)

        self.proj_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape

        h = self.gp_norm(x)

        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        alpha = q.permute(0, 2, 3, 1).view(B, H*W, C) @ k.view(B, C, H*W)
        alpha = alpha / np.sqrt(C)
        alpha = torch.softmax(alpha, dim=-1)

        out = v.view(B, C, H*W) @ alpha.permute(0, 2, 1)
        out = out.view(B, C, H, W)
        # out = alpha @ v.permute(0, 2, 3, 1).view(B, H*W, C)
        # out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        out = self.proj_out(out)

        return out + x


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.net(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_embd_channels, dropout_prob, use_attention=False, gp_norm_groups=16):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=gp_norm_groups, num_channels=in_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        if t_embd_channels > 0:
            self.proj_t = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_embd_channels, out_channels)
            )

        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=gp_norm_groups, num_channels=out_channels, eps=1e-6),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.shortcut_fn = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.attention = None if not use_attention else AttentionBlock(out_channels, gp_norm_groups=gp_norm_groups)

    def forward(self, x, t_embd):
        out = self.net1(x)

        if t_embd is not None:
            out = out + self.proj_t(t_embd)[:, :, None, None]
            
        out = self.net2(out)
        out = out + self.shortcut_fn(x)
        if self.attention is not None:
            out = self.attention(out)

        return out


class UNet(nn.Module):
    def __init__(self, 
            img_channels,
            start_channels,
            channels_multiplier_arr: list,
            levels_to_apply_attention: list,
            dropout_prob=0.1,
            num_resnet_blocks=2,
            max_time=1000,
            gp_norm_groups=16
        ):
        super().__init__()

        assert len(channels_multiplier_arr) == len(levels_to_apply_attention)

        time_embedding_dim = start_channels * 4

        self.time_embedding = SinusoidalTimeEmbedding(max_pos=max_time, d_model=start_channels, out_dim=time_embedding_dim)

        self.from_rgb_conv = nn.Conv2d(img_channels, start_channels, kernel_size=3, padding=1)
        
        ################ Building Down Blocks ################
        self.down_blocks = nn.ModuleList()
        curr_in_channels = start_channels
        skips_channels = [curr_in_channels]
        for level, ch_multiplier in enumerate(channels_multiplier_arr):
            out_channels = start_channels * ch_multiplier
            for _ in range(num_resnet_blocks):
                self.down_blocks.append(
                    ResnetBlock(
                        curr_in_channels,
                        out_channels,
                        time_embedding_dim,
                        dropout_prob,
                        use_attention=levels_to_apply_attention[level], 
                        gp_norm_groups=gp_norm_groups
                    )
                )
                curr_in_channels = out_channels
                skips_channels.append(curr_in_channels)
            if level != len(channels_multiplier_arr)-1:
                self.down_blocks.append(DownSample(curr_in_channels))
                skips_channels.append(curr_in_channels)
        
        ################ Building Middle Blocks ################
        self.middle_blocks = nn.ModuleList([
            ResnetBlock(curr_in_channels, curr_in_channels, time_embedding_dim, dropout_prob, use_attention=True, gp_norm_groups=gp_norm_groups),
            ResnetBlock(curr_in_channels, curr_in_channels, time_embedding_dim, dropout_prob, use_attention=False, gp_norm_groups=gp_norm_groups),
        ])

        ################ Building Up Blocks ################
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channels_multiplier_arr))):
            ch_multiplier = channels_multiplier_arr[level]
            out_channels = start_channels * ch_multiplier
            for _ in range(num_resnet_blocks + 1):
                self.up_blocks.append(
                    ResnetBlock(
                        curr_in_channels + skips_channels.pop(),
                        out_channels,
                        time_embedding_dim,
                        dropout_prob,
                        use_attention=levels_to_apply_attention[level],
                        gp_norm_groups=gp_norm_groups
                    )
                )
                curr_in_channels = out_channels
            if level != 0:
                self.up_blocks.append(UpSample(curr_in_channels))
        
        assert len(skips_channels) == 0

        self.to_noise = nn.Sequential(
            nn.GroupNorm(num_groups=gp_norm_groups, num_channels=curr_in_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(curr_in_channels, img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_embd = self.time_embedding(t)

        ######## Down Blocks ########
        out = self.from_rgb_conv(x)
        skips = [out]
        for block in self.down_blocks:
            if isinstance(block, ResnetBlock):
                out = block(out, t_embd)
            elif isinstance(block, DownSample):
                out = block(out)
            else:
                raise Exception("Unexpected block in Down Blocks")
            skips.append(out)
        
        ######## Middle Blocks ########
        for block in self.middle_blocks:
            out = block(out, t_embd)
        
        ######## Up Blocks ########
        for block in self.up_blocks:
            if isinstance(block, ResnetBlock):
                out = block(torch.cat((skips.pop(), out), dim=1), t_embd)
            elif isinstance(block, UpSample):
                out = block(out)
            else:
                raise Exception("Unexpected block in Up Blocks")
        assert len(skips) == 0
        
        out = self.to_noise(out)

        return out

