import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    try:
        from selective_scan import selective_scan_fn as selective_scan_fn_v1
        selective_scan_fn = selective_scan_fn_v1
    except ImportError:
        print("Warning: selective_scan not found. Using fallback implementation.")
        selective_scan_fn = None


# Inception Block 
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        branch_channels = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.out_conv(out)
        return out



# SS2D (Mamba-style)

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        A_log = torch.log(A)

        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)

        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, merge=True):
        D = torch.ones(d_inner)

        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)

        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is required for SS2D")

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
            ],
            dim=1
        ).view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(
            out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3
        ).contiguous().view(B, -1, L)

        invwh_y = torch.transpose(
            inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3
        ).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out



# VSS Block

class ProposedVSSBlock(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=3,
        expand=2,
        dropout=0.0,
        bias=True
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        # Left gating branch
        self.gate_proj = nn.Linear(dim, dim, bias=bias)

        # Main feature branch
        self.in_proj = nn.Linear(dim, dim, bias=bias)

        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,   # depthwise conv
            bias=True
        )

        self.ss2d = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bias=bias
        )

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, H, W, C]
        residual = x

        x_ln = self.norm(x)

        # Branch 1: gating branch
        gate = self.gate_proj(x_ln)

        # Branch 2: main branch
        feat = self.in_proj(x_ln)                    # [B, H, W, C]
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        feat = self.dwconv(feat)
        feat = feat.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        feat = self.ss2d(feat)                      # [B, H, W, C]
        feat = self.out_norm(feat)

        # Element-wise multiplication
        out = gate * feat

        # Final linear
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection
        out = out + residual
        return out


# InMamba Block
class InceptionMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0.0, d_state=16):
        super().__init__()
        self.inception = InceptionBlock(in_channels, out_channels)
        self.mamba = ProposedVSSBlock(
            dim=out_channels,
            d_state=d_state,
            dropout=drop_path
        )

    def forward(self, x):
        x = self.inception(x)              # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)         # [B, H, W, C]
        x = self.mamba(x)                 # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)         # [B, C, H, W]
        return x


# Attention Gate
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = x * psi
        return out


#Decoder UpBlock
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(UpBlock, self).__init__()

        self.att = Attention_block(
            F_g=in_ch,
            F_l=skip_ch,
            F_int=out_ch
        )

        self.reduce = nn.Conv2d(in_ch, skip_ch, kernel_size=1)
        self.conv = InceptionBlock(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        skip_att = self.att(x, skip)
        x = self.reduce(x)
        x = F.interpolate(x, size=skip_att.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_att], dim=1)
        x = self.conv(x)
        return x

# Encoder-Decoder 
class InMambaAttentionUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_ch=64, drop_path=0.1, d_state=16):
        super(InceptionMambaAttentionUNet, self).__init__()

        # Encoder
        self.enc1 = InceptionMambaBlock(in_channels, base_ch, drop_path=drop_path, d_state=d_state)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = InceptionMambaBlock(base_ch, base_ch * 2, drop_path=drop_path, d_state=d_state)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = InceptionMambaBlock(base_ch * 2, base_ch * 4, drop_path=drop_path, d_state=d_state)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = InceptionMambaBlock(base_ch * 4, base_ch * 8, drop_path=drop_path, d_state=d_state)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = InceptionMambaBlock(base_ch * 8, base_ch * 16, drop_path=drop_path, d_state=d_state)

        # Decoder
        self.up4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        # Final output
        self.final_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Final prediction and sigmoid activation for binary segmentation
        out = self.final_conv(d1)
        out = torch.sigmoid(out)  # Sigmoid activation
        return out
