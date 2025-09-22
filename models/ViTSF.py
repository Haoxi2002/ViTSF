import einops
import torch
from torch import nn

from layers.Embed import PositionalEmbedding
from layers.Mlp_Mixer import MixerBlock


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.token_dim = (args.h // args.patch_size[0]) * (args.seq_len // args.patch_size[1])  # token <==> patch
        self.conv_embedding = nn.Conv2d(1, args.hidden_dim, stride=args.patch_size, kernel_size=args.patch_size, padding=0)
        self.position_embedding = PositionalEmbedding(args.hidden_dim)
        self.static_embedding = nn.Linear(7, args.pred_len)
        self.blocks = nn.ModuleList([
            MixerBlock(args.hidden_dim, self.token_dim, args.token_mlp_dim, args.channel_mlp_dim, args.dropout) for _ in range(args.n_blocks)
        ])
        self.out = nn.Linear(self.token_dim * args.hidden_dim, args.pred_len)
        self.linear = nn.Linear(args.pred_len, args.pred_len)

    """
    input:
        x: (batch_size, features, h, seq_len)
        static: (batch_size, features, 7)
    output:
        y: (batch_size, pred_len, features)
    """
    def forward(self, x, static):
        bs, f_c, h, w = x.size()
        # CI
        x = torch.reshape(x, (-1, 1, h, w))
        static = torch.reshape(static, (-1, 1, 7))

        # encoder
        x = self.conv_embedding(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.position_embedding(x)
        static = self.static_embedding(static)

        # backbone
        for l in self.blocks:
            x = l(x)

        # decode
        x = einops.rearrange(x, 'b p c -> b 1 (p c)')
        x = self.out(x)
        x = x * static
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)

        # de CI
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (bs, -1, self.args.pred_len))
        x = torch.transpose(x, 1, 2)
        return x
