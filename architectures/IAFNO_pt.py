# architectures/IAFNO_pt.py
import math, torch, torch.nn as nn


# ---------- spectral conv ----------
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1 / math.sqrt(in_ch * out_ch)
        self.wr = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes))
        self.wi = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes))

    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))  # (B,C,H,W//2+1) complex
        out_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )

        x_low = x_ft[:, :, : self.modes, : self.modes]
        weight = torch.complex(self.wr, self.wi)  # (Cin,Cout,m,m)
        out_ft[:, :, : self.modes, : self.modes] = torch.einsum(
            "bcih,cjoh->bjoh", x_low, weight
        )
        return torch.fft.irfftn(out_ft, s=(H, W), dim=(-2, -1))  # (B,Cout,H,W)


# ---------- IAFNO-block ----------
class IAFNOBlock(nn.Module):
    def __init__(self, channels, modes, n_imp=1):
        super().__init__()
        self.n_imp = n_imp
        self.spec = SpectralConv2d(channels, channels, modes)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()
        self.theta = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        for _ in range(self.n_imp):
            g = self.spec(x)
            g = self.act(self.pw1(g))
            g = self.pw2(g)
            x = x + self.theta * (g - x)
            x = self.norm(x)
        return x


# ---------- вся сеть ----------
class IAFNO(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, width=64, depth=8, modes=16, n_imp=2):
        super().__init__()
        self.inp = nn.Conv2d(in_ch, width, 1)
        self.body = nn.ModuleList(
            [IAFNOBlock(width, modes, n_imp) for _ in range(depth)]
        )
        self.out = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        x = self.inp(x)
        res = x
        for blk in self.body:
            x = blk(x)
        return self.out(x + res)


# ---------- фабрика под твой пайплайн ----------
def get_IAFNO_pt(
    grid, device, width=64, depth=8, modes=None, n_imp=2, lr=3e-4, wd=1e-3, batch=32
):
    if modes is None:
        modes = grid // 3

    model = IAFNO(1, 1, width, depth, modes, n_imp).to(device)
    model_data = {"model": model}
    optim_spec = {
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": batch,
        "N_epochs": 300,
    }
    return model_data, optim_spec
