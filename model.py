import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_basis=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_basis = num_basis
        self.basis_weights = nn.Parameter(torch.randn(num_basis, out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.basis_weights, mode='fan_out', nonlinearity='relu')

    def forward(self, x, mixing_coeffs):
        batch_size = x.size(0)
        coeffs = mixing_coeffs.reshape(batch_size, self.num_basis, 1, 1, 1)
        basis = self.basis_weights.unsqueeze(0)
        dynamic_weights = (coeffs * basis).sum(dim=1)
        x_reshaped = x.reshape(1, batch_size * self.in_channels, -1)
        w_reshaped = dynamic_weights.reshape(batch_size * self.out_channels, self.in_channels, self.kernel_size)
        out = F.conv1d(x_reshaped, w_reshaped, bias=None, stride=self.stride, groups=batch_size)
        out = out.reshape(batch_size, self.out_channels, -1)
        return out + self.bias.reshape(1, -1, 1)

class HAC_Net(nn.Module):
    def __init__(self, num_train_tissues, embed_dim=32, num_basis=4):
        super().__init__()
        self.tissue_embed = nn.Embedding(num_train_tissues, embed_dim)
        self.router = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, num_basis), nn.Softmax(dim=1))
        self.seq_embed = nn.Embedding(5, 16)
        self.u5_conv = DynamicConv1d(16, 32, kernel_size=7, stride=1, num_basis=num_basis)
        self.cds_conv = DynamicConv1d(16, 32, kernel_size=3, stride=3, num_basis=num_basis)
        self.u3_conv = DynamicConv1d(16, 32, kernel_size=7, stride=1, num_basis=num_basis)
        self.regressor = nn.Sequential(nn.Linear(32*3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, u5, cds, u3, tissue_id=None, external_ctx_emb=None):
        if external_ctx_emb is not None: ctx = external_ctx_emb
        else: ctx = self.tissue_embed(tissue_id)
        coeffs = self.router(ctx)
        u5_x = self.seq_embed(u5).permute(0, 2, 1)
        cds_x = self.seq_embed(cds).permute(0, 2, 1)
        u3_x = self.seq_embed(u3).permute(0, 2, 1)
        u5_feat = F.adaptive_max_pool1d(self.u5_conv(u5_x, coeffs), 1).squeeze(-1)
        cds_feat = F.adaptive_avg_pool1d(self.cds_conv(cds_x, coeffs), 1).squeeze(-1)
        u3_feat = F.adaptive_max_pool1d(self.u3_conv(u3_x, coeffs), 1).squeeze(-1)
        combined = torch.cat([u5_feat, cds_feat, u3_feat], dim=1)
        return self.regressor(combined).squeeze(-1)