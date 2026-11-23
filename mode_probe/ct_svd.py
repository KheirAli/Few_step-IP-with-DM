"""One-time decomposition of the sparse-view CT operator for a fixed geometry, so that the data-fit becomes a closed form (no iterations).
A (rays x pixels) is built column by column from the projector of unified.py, then A A' = U S^2 U' (rays x rays) and V = A' U / S.
Saved: V [pixels, r] and s [r] (singular values above 1e-3 of the largest).   usage: python ct_svd.py <views> [gpu]"""
import sys, math, time, torch
K = int(sys.argv[1]); dev = torch.device(f"cuda:{sys.argv[2] if len(sys.argv) > 2 else 0}"); n = 256; N = n * n; m = K * 364; t0 = time.time()
def radon(x):                                    # identical to unified._radon, any number of channels
    xp = torch.nn.functional.pad(x, (54, 54, 54, 54)).expand(K, -1, -1, -1); th = torch.arange(K, device=x.device).float() * math.pi / K
    M = torch.stack([torch.stack([th.cos(), -th.sin(), torch.zeros_like(th)], 1), torch.stack([th.sin(), th.cos(), torch.zeros_like(th)], 1)], 1)
    rot = torch.nn.functional.grid_sample(xp, torch.nn.functional.affine_grid(M, list(xp.shape), align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    return rot.sum(dim=2).permute(1, 0, 2)[None] / 16.0
A = torch.empty(m, N, device=dev); B = 128; eye = torch.eye(N, device="cpu")
for s in range(0, N, B):
    x = torch.zeros(1, B, n, n, device=dev); x.view(B, N)[torch.arange(B), torch.arange(s, s + B)] = 1.0
    A[:, s:s + B] = radon(x)[0].reshape(B, m).T
print(f"built A {tuple(A.shape)} in {time.time() - t0:.0f}s", flush=True)
G = (A @ A.T).double(); lam, U = torch.linalg.eigh(G); del G
keep = lam > (1e-3 ** 2) * lam.max(); s = lam[keep].sqrt().float(); U = U[:, keep].float(); print(f"rank kept {int(keep.sum())} of {m}; s max {float(s.max()):.3f} min {float(s.min()):.5f}; {time.time() - t0:.0f}s", flush=True)
V = torch.empty(N, U.shape[1], device=dev)
for c in range(0, U.shape[1], 2048): V[:, c:c + 2048] = (A.T @ U[:, c:c + 2048]) / s[c:c + 2048]
x = torch.rand(N, device=dev); print("check |A x|^2 vs |S V'x|^2:", float((A @ x).pow(2).sum()), float((s * (V.T @ x)).pow(2).sum()), "| V orthonormal error", float((V[:, :64].T @ V[:, :64] - torch.eye(64, device=dev)).abs().max()))
torch.save({"V": V.cpu(), "s": s.cpu(), "K": K}, f"ct_svd_{K}.pt"); print(f"saved ct_svd_{K}.pt in {time.time() - t0:.0f}s")
