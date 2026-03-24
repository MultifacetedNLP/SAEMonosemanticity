import torch
import math
import torch.nn.functional as F
from collections.abc import Sequence


class TorchKDEHistogram1DVectorized:
    def __init__(self, bins: int = 100, bandwidth: float | str = "scott"):
        self.bins = bins
        self.bandwidth = bandwidth

    def fit(self, X: torch.Tensor):
        """
        X: [D, N]  — D datasets each with N samples
        """
        assert X.ndim == 2, "Expect a 2D tensor of shape [D, N]"
        D, N = X.shape
        device, dtype = X.device, X.dtype

        # per‐dataset statistics
        self.n = torch.full((D,), N, device=device, dtype=dtype)
        x_min = X.min(dim=1).values
        x_max = X.max(dim=1).values

        counts = []
        centers = []
        h       = []
        log_counts = []
        log_norm   = []

        # we do have to loop over D here
        for d in range(D):
            x_d = X[d]
            # histogram
            c = torch.histc(x_d, bins=self.bins,
                            min=x_min[d].item(), max=x_max[d].item(),
                            out=None).to(device=device, dtype=dtype)
            # bin centers
            edges = torch.linspace(x_min[d], x_max[d], self.bins + 1,
                                   device=device, dtype=dtype)
            cen = (edges[:-1] + edges[1:]) / 2

            # bandwidth
            if isinstance(self.bandwidth, str) and self.bandwidth.lower() == "scott":
                hd = (N ** (-1.0 / 5))
            else:
                hd = float(self.bandwidth)
            hd = torch.as_tensor(hd, device=device, dtype=dtype)

            # log‐counts (for zero counts → −∞)
            lc = torch.where(c > 0, torch.log(c), torch.full_like(c, -float("inf")))
            # normalization constant per dataset
            ln = math.log(N * hd.item() * math.sqrt(2 * math.pi))

            counts.append(c)
            centers.append(cen)
            h.append(hd)
            log_counts.append(lc)
            log_norm.append(ln)

        # stack into batched tensors
        self.counts     = torch.stack(counts,     dim=0)  # [D, bins]
        self.centers    = torch.stack(centers,    dim=0)  # [D, bins]
        self.h          = torch.stack(h,          dim=0)  # [D]
        self.log_counts = torch.stack(log_counts, dim=0)  # [D, bins]
        self.log_norm   = torch.tensor(log_norm,  device=device, dtype=dtype)  # [D]
        
    def fitSAE(self, X_list: Sequence[torch.Tensor]):
        device, dtype = X_list[0].device, X_list[0].dtype

        counts = []
        centers = []
        h       = []
        log_counts = []
        log_norm   = []

        # we do have to loop over D here — but D=10 is tiny, and this is only at fit time
        for x_d in X_list:
            N = x_d.numel()
            
            if N == 0:
                # build a “degenerate” empty histogram
                c  = torch.zeros(self.bins, device=device, dtype=dtype)
                cen = torch.zeros(self.bins, device=device, dtype=dtype)
                hd = torch.tensor(1.0, device=device, dtype=dtype)
                lc = torch.full((self.bins,), -float("inf"), device=device, dtype=dtype)
                ln = 0.0
            else:
                x_min, x_max = x_d.min(), x_d.max()
                
                c = torch.histc(x_d, bins=self.bins, min=x_min.item(), max=x_max.item())
                edges = torch.linspace(x_min, x_max, self.bins + 1, device=device, dtype=dtype)
                cen = (edges[:-1] + edges[1:]) / 2
                hd = (N ** (-1.0 / 5)) if isinstance(self.bandwidth, str) else float(self.bandwidth)
                hd = torch.as_tensor(hd, device=device, dtype=dtype)
                lc = torch.where(c > 0, torch.log(c), torch.full_like(c, -float("inf")))
                ln = math.log(N * hd.item() * math.sqrt(2 * math.pi))

            counts.append(c)
            centers.append(cen)
            h.append(hd)
            log_counts.append(lc)
            log_norm.append(ln)

        # stack into batched tensors
        self.counts     = torch.stack(counts,     dim=0)  # [D, bins]
        self.centers    = torch.stack(centers,    dim=0)  # [D, bins]
        self.h          = torch.stack(h,          dim=0)  # [D]
        self.log_counts = torch.stack(log_counts, dim=0)  # [D, bins]
        self.log_norm   = torch.tensor(log_norm,  device=device, dtype=dtype)  # [D]

    def score_samples(self, X_query: torch.Tensor) -> torch.Tensor:
        """
        Vectorized scoring.
        If X_query.ndim == 1 (shape [M]), you get back [D, M].
        If X_query.ndim == 2 (shape [D, M]), you get back [D, M].
        """
        D = self.counts.size(0)
        # make Xq into shape [D, M]
        if X_query.ndim == 1:
            xq = X_query.to(device=self.centers.device, dtype=self.centers.dtype)
            xq = xq.unsqueeze(0).expand(D, -1)  # [D, M]
        else:
            assert X_query.size(0) == D, "When 2D, first dim must match number of datasets"
            xq = X_query.to(device=self.centers.device, dtype=self.centers.dtype)

        # [D, M, bins]
        diff = (xq.unsqueeze(2) - self.centers.unsqueeze(1)) / self.h[:, None, None]
        # [D, M, bins]
        base = -0.5 * diff.pow(2) + self.log_counts.unsqueeze(1) - self.log_norm[:, None, None]
        # [D, M]
        log_pdf = torch.logsumexp(base, dim=2)
        return log_pdf