"""Architecture and input features of the stronger magnitude-to-face predictor (trained in phase_net2.py)."""
import torch, torch.nn as nn
def feat2(y, FU):
    yc = y.clamp(min=0); a = FU.ifft2_m((yc ** 2).to(torch.complex64)).real; a = a / a[..., 192:193, 192:193].clamp(min=1e-8)
    return torch.cat([torch.nn.functional.avg_pool2d(a[..., 64:320, 64:320], 2), torch.log1p(yc)[..., 128:256, 128:256]], 1)
class Res(nn.Module):
    def __init__(s, c): super().__init__(); s.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1))
    def forward(s, x): return x + s.b(x)
class Net2(nn.Module):
    def __init__(s, K=8):
        super().__init__(); s.K = K; ch = [64, 96, 192, 384, 512]; L = [nn.Conv2d(6, 64, 3, 1, 1)]
        for i in range(4): L += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), Res(ch[i + 1]), Res(ch[i + 1])]
        s.enc = nn.Sequential(*L, nn.GroupNorm(8, 512), nn.SiLU(), nn.Conv2d(512, 128, 1)); s.fc = nn.Sequential(nn.Flatten(), nn.Linear(8192, 3072), nn.SiLU(), nn.Linear(3072, 8192), nn.SiLU())
        D = [nn.Conv2d(128, 512, 3, 1, 1)]; dc = [512, 384, 192, 96]
        for i in range(3): D += [Res(dc[i]), nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dc[i], dc[i + 1], 3, 1, 1)]
        s.dec = nn.Sequential(*D, Res(96), nn.GroupNorm(8, 96), nn.SiLU(), nn.Conv2d(96, K * 3, 3, 1, 1))
    def forward(s, f): return torch.sigmoid(s.dec(s.fc(s.enc(f)).view(-1, 128, 8, 8))).view(-1, s.K, 3, 64, 64)
