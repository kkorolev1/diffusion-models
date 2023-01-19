import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1)
        self.init()

    def init(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.init()

    def init(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        return self.conv(F.interpolate(x, scale_factor=2))


class TimeEmbedding(nn.Module):
    def __init__(self, T, embedding_dim, time_dim):
        assert embedding_dim % 2 == 0
        super().__init__()
        emb = torch.arange(0, embedding_dim, step=2) / embedding_dim * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, embedding_dim // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, embedding_dim // 2, 2]
        emb = emb.view(T, embedding_dim)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=embedding_dim, out_features=time_dim),
            Swish(),
            nn.Linear(in_features=time_dim, out_features=time_dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        num_groups = 32
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.proj_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, dropout, enable_attention=False, num_groups=32):
        super().__init__()    

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            Swish(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.time_embedding_proj = nn.Sequential(
            Swish(),
            nn.Linear(in_features=time_embedding_dim, out_features=out_channels)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.residual = nn.Identity()
        
        if enable_attention:
            self.attention = AttentionBlock(in_channels=out_channels)
        else:
            self.attention = nn.Identity()

        self.init()
    
    def init(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, time_embedding):
        proj = self.block1(x) + self.time_embedding_proj(time_embedding)[:, :, None, None]
        proj = self.block2(proj) + self.residual(x)
        return self.attention(proj)


class Unet(nn.Module):
    def __init__(self, T, n_channels, channels_mul, n_residual_blocks, dropout):
        super().__init__()

        num_groups = 32

        time_dim = 4 * n_channels
        self.time_embedding = TimeEmbedding(T=T, embedding_dim=n_channels, time_dim=time_dim)

        self.head = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1)

        n_channels_list = [n_channels]
        now_ch = n_channels

        self.down_blocks = nn.ModuleList()

        for i, mul in enumerate(channels_mul):
            out_channels = n_channels * mul

            for _ in range(n_residual_blocks):
                self.down_blocks.append(ResidualBlock(
                    in_channels=now_ch, out_channels=out_channels, time_embedding_dim=time_dim,
                    dropout=dropout, enable_attention=True))
                now_ch = out_channels
                n_channels_list.append(now_ch)
            
            if i != len(channels_mul) - 1:
                self.down_blocks.append(Downsample(now_ch))
                n_channels_list.append(now_ch)

        self.middle_blocks = nn.ModuleList([
            ResidualBlock(in_channels=now_ch, out_channels=now_ch, time_embedding_dim=time_dim, dropout=dropout, enable_attention=True),
            ResidualBlock(in_channels=now_ch, out_channels=now_ch, time_embedding_dim=time_dim, dropout=dropout, enable_attention=False),
        ])

        self.up_blocks = nn.ModuleList()

        for i, mul in reversed(list(enumerate(channels_mul))):
            out_channels = n_channels * mul
            for _ in range(n_residual_blocks + 1):
                self.up_blocks.append(ResidualBlock(
                    in_channels=n_channels_list.pop() + now_ch, out_channels=out_channels, time_embedding_dim=time_dim,
                    dropout=dropout, enable_attention=True))
                now_ch = out_channels
            if i != 0:
                self.up_blocks.append(Upsample(now_ch))
        assert len(n_channels_list) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=now_ch),
            Swish(),
            nn.Conv2d(in_channels=now_ch, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        self.init()
    
    def init(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        temb = self.time_embedding(t)
        
        h = self.head(x)
        hs = [h]
        for layer in self.down_blocks:
            h = layer(h, temb)
            hs.append(h)
        
        for layer in self.middle_blocks:
            h = layer(h, temb)
        
        for layer in self.up_blocks:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
        
if __name__ == '__main__':
    batch_size = 32
    model = Unet(T=1000, n_channels=128, channels_mul=[1, 2, 2, 2], n_residual_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)