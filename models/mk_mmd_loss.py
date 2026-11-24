import torch


def _rbf_sum(X, Y, gamma, block=2048, exclude_diag=False, self_kernel=False):
    s = torch.zeros((), device=X.device, dtype=X.dtype)
    n, m = X.shape[0], Y.shape[0]
    for i in range(0, n, block):
        Xi = X[i : i + block]
        xi2 = (Xi * Xi).sum(dim=1, keepdim=True)
        for j in range(0, m, block):
            Yj = Y[j : j + block]
            yj2 = (Yj * Yj).sum(dim=1, keepdim=True)
            dist2 = xi2 + yj2.T - 2.0 * (Xi @ Yj.T)
            K = torch.exp(-gamma * torch.clamp(dist2, min=0.0))
            if self_kernel and exclude_diag and i == j:
                s = s + (K.sum() - K.diagonal().sum())
            else:
                s = s + K.sum()
    return s


def mmd_mk_nd_unbiased(X, Y, bws):
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return torch.zeros((), device=X.device, dtype=X.dtype)
    out = torch.zeros((), device=X.device, dtype=X.dtype)
    for bw in bws:
        gamma = 1.0 / (2.0 * (bw**2) + 1e-12)
        kxx = _rbf_sum(X, X, gamma, exclude_diag=True, self_kernel=True)
        kyy = _rbf_sum(Y, Y, gamma, exclude_diag=True, self_kernel=True)
        kxy = _rbf_sum(X, Y, gamma, exclude_diag=False, self_kernel=False)
        out = out + (
            kxx / (n * (n - 1) + 1e-12) + kyy / (m * (m - 1) + 1e-12) - 2.0 * kxy / (n * m)
        )
    return out / float(len(bws))


def mmd_mk_1d_unbiased(x, y, bws):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return mmd_mk_nd_unbiased(x, y, bws)
