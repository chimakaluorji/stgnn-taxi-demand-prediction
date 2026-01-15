from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Compute D^{-1/2} (A+I) D^{-1/2}."""
    n = A.shape[0]
    A_hat = A + np.eye(n, dtype=float)
    D = np.sum(A_hat, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    Dm = np.diag(D_inv_sqrt)
    return Dm @ A_hat @ Dm

class GraphConv(nn.Module):
    def __init__(self, A_norm: torch.Tensor, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("A_norm", A_norm)  # [N,N]
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, N, Fin] -> [B, N, Fout]"""
        # aggregate neighbors then linear
        X_agg = torch.matmul(self.A_norm, X)  # [B,N,Fin]
        return self.lin(X_agg)

class STGNN(nn.Module):
    """Simple ST-GNN: GraphConv per time step + GRU across time (per node) + linear head."""
    def __init__(self, A_norm: torch.Tensor, in_features: int = 1, gcn_hidden: int = 32, gru_hidden: int = 64):
        super().__init__()
        self.gcn = GraphConv(A_norm, in_features, gcn_hidden)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, T, N, Fin] -> yhat: [B, N]"""
        B, T, N, Fin = X.shape
        # apply GCN at each time step
        X2 = X.reshape(B*T, N, Fin)
        H = torch.relu(self.gcn(X2))  # [B*T, N, gcn_hidden]
        H = H.reshape(B, T, N, -1)    # [B, T, N, gcn_hidden]

        # GRU per node: treat nodes as batch dimension
        Hn = H.permute(0, 2, 1, 3).contiguous()  # [B, N, T, gcn_hidden]
        Hn = Hn.reshape(B*N, T, -1)              # [B*N, T, gcn_hidden]
        out, _ = self.gru(Hn)                    # [B*N, T, gru_hidden]
        last = out[:, -1, :]                     # [B*N, gru_hidden]
        y = self.head(last).squeeze(-1)          # [B*N]
        return y.reshape(B, N)

@dataclass
class TrainArtifacts:
    state_dict: Dict[str, Any]
    A_norm: np.ndarray
    zone_ids: list[int]
    mean: np.ndarray
    std: np.ndarray
    time_bin_minutes: int
    seq_len: int

def train_stgnn(
    X_in: np.ndarray,   # [M, seq_len, N, 1]
    y: np.ndarray,      # [M, N]
    A: np.ndarray,      # [N, N]
    zone_ids: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    time_bin_minutes: int,
    seq_len: int,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[TrainArtifacts, float]:
    device_t = torch.device(device)
    A_norm_np = normalize_adjacency(A)
    A_norm = torch.tensor(A_norm_np, dtype=torch.float32, device=device_t)

    model = STGNN(A_norm=A_norm, in_features=1).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(X_in, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    last_loss = None
    for _ in range(epochs):
        epoch_losses = []
        for xb, yb in dl:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad()
            pred = model(xb)  # [B,N]
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        last_loss = float(np.mean(epoch_losses)) if epoch_losses else None

    artifacts = TrainArtifacts(
        state_dict=model.state_dict(),
        A_norm=A_norm_np,
        zone_ids=zone_ids,
        mean=mean,
        std=std,
        time_bin_minutes=time_bin_minutes,
        seq_len=seq_len,
    )
    return artifacts, float(last_loss if last_loss is not None else 0.0)

@torch.no_grad()
def predict_next(
    model: STGNN,
    X_seq: np.ndarray,   # [1, seq_len, N, 1] normalized
    mean: np.ndarray,
    std: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    device_t = torch.device(device)
    model.eval()
    xb = torch.tensor(X_seq, dtype=torch.float32, device=device_t)
    pred_norm = model(xb).detach().cpu().numpy()[0]  # [N]
    pred = pred_norm * std + mean
    pred = np.maximum(pred, 0.0)
    return pred
