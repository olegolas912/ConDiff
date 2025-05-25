# architectures/IAFNO_pt_modified.py

import math
import torch
import torch.nn as nn


# ---------- spectral conv (unchanged) ----------
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1 / math.sqrt(in_ch * out_ch)
        self.wr = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes))
        self.wi = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes))

    # def forward(self, x):  # (B,C,H,W)
    #     B, C, H, W = x.shape
    #     x_ft = torch.fft.rfftn(x, dim=(-2, -1))  # (B,C,H,W//2+1) complex
    #     out_ft = torch.zeros(
    #         B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
    #     )
    #     x_low = x_ft[:, :, : self.modes, : self.modes]
    #     weight = torch.complex(self.wr, self.wi)  # (Cin,Cout,m,m)
    #     out_ft[:, :, : self.modes, : self.modes] = torch.einsum(
    #         "b i x y , i o x y -> b o x y", x_low, weight
    #     )
    #     return torch.fft.irfftn(out_ft, s=(H, W), dim=(-2, -1))

    def forward(self, x):
        # --- Adaptive spectral modes: увеличиваем/уменьшаем число мод в зависимости от контраста ---
        B, C, H, W = x.shape
        contrast = x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(
            dim=(1, 2, 3), keepdim=True
        )
        alpha = torch.sigmoid(contrast.mean())  # в (0,1)
        raw = int(self.modes * (1 + alpha.item()))
        eff = max(1, min(self.modes, raw))

        x_ft = torch.fft.rfftn(x, dim=(-2, -1))  # (B,C,H,W//2+1)
        out_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )

        # подрезаем как спектр, так и веса
        x_low = x_ft[:, :, :eff, :eff]
        w_complex = torch.complex(self.wr, self.wi)
        w_low = w_complex[:, :, :eff, :eff]

        out_ft[:, :, :eff, :eff] = torch.einsum(
            "b i x y, i o x y -> b o x y", x_low, w_low
        )
        return torch.fft.irfftn(out_ft, s=(H, W), dim=(-2, -1))


# ---------- IAFNO block with modifications ----------
class IAFNOBlock(nn.Module):
    def __init__(self, channels, modes, n_imp=1):
        super().__init__()
        self.n_imp = n_imp
        self.spec = SpectralConv2d(channels, channels, modes)
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()
        self.theta = nn.Parameter(torch.tensor(0.5))
        # Modified: use InstanceNorm2d instead of GroupNorm(1, channels)
        self.norm = nn.GroupNorm(1, channels)
        # self.norm = nn.InstanceNorm2d(channels, affine=True, eps=1e-02)

    def forward(self, x):
        for _ in range(self.n_imp):
            g = self.spec(x)
            local_feat = self.local(x)  # ----
            g = g + local_feat  # ---
            g = self.act(self.pw1(g))
            g = self.pw2(g)
            # Modified: direct residual instead of weighted
            x = self.theta * x + (1 - self.theta) * g
        # apply normalization at end of block
        x = self.norm(x)
        return x


# ---------- full IAFNO model with modified input channels ----------
class IAFNO(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, width=64, depth=8, modes=16, n_imp=2):
        super().__init__()
        # Modified: in_ch defaults to 3 (forcing term + X coord + Y coord)
        self.inp = nn.Conv2d(in_ch, width, 1)
        self.body = nn.ModuleList(
            [IAFNOBlock(width, modes, n_imp) for _ in range(depth)]
        )
        self.out = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        # x shape: (B, 3, H, W)
        x = self.inp(x)
        res = x
        for blk in self.body:
            x = blk(x)
        # Modified: pass residual through final conv
        return self.out(x + res)


# ---------- factory for your training pipeline ----------
def get_IAFNO_pt(
    grid,
    device,
    width=64,
    depth=8,
    modes=None,
    n_imp=4,
    lr=1e-4,
    wd=1e-6,  # Modified: reduced weight decay from 1e-3 → 1e-6
    batch=8,  # Modified: reduced batch size from 32 → 16
):
    if modes is None:
        # Modified: use fewer spectral modes (grid//4 instead of grid//3)
        modes = grid // 4

    model = IAFNO(
        in_ch=3,  # Modified: 3 input channels (f, x-coordinate, y-coordinate)
        out_ch=1,
        width=width,
        depth=depth,
        modes=modes,
        n_imp=n_imp,
    ).to(device)

    model_data = {"model": model}
    optim_spec = {
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": batch,
        "N_epochs": 100,
    }
    return model_data, optim_spec
