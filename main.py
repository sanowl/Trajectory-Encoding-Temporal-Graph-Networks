
from __future__ import annotations
import math, itertools
from collections import defaultdict
from typing import Tuple, List, Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import secrets

class LearnableID(nn.Module):
    def __init__(self, d: int, init_nodes: int = 0):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(init_nodes if init_nodes > 0 else 1, d)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, idx: Tensor) -> Tensor:
        max_idx = int(idx.max().item())
        if max_idx >= self.emb.num_embeddings:
            self._expand(max_idx + 1)
        return self.emb(idx)

    def _expand(self, new_size: int):
        old_weight = self.emb.weight.data
        self.emb = nn.Embedding(new_size, self.d, device=old_weight.device)
        nn.init.xavier_uniform_(self.emb.weight)
        self.emb.weight.data[: old_weight.size(0)] = old_weight

class TrajectoryMemory(NamedTuple):
    tp: Tensor      # temporal positional feature  (TP_i(t))
    last_t: float   # last update time

class TrajectoryEncoder(nn.Module):
    def __init__(self, d: int, alpha: float, beta: float):
        super().__init__()
        self.d, self.alpha, self.beta = d, alpha, beta

    def encode(self, x: Tensor, dt: Tensor) -> Tensor:
        """TE_exp(x, dt) = α * x * exp(-β * dt)   (Eq. 11)."""
        return self.alpha * x * torch.exp(-self.beta * dt.unsqueeze(-1))

    def aggregate(self, self_tp: Tensor, nbr_msgs: List[Tensor]) -> Tensor:
        if not nbr_msgs:
            return self_tp
        return self_tp + torch.stack(nbr_msgs, dim=0).sum(dim=0)  # Eq. 13

    def update(self, id_vec: Tensor, agg_msg: Tensor) -> Tensor:
        return id_vec + agg_msg  # Eq. 14


class TGNMemory(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.state: Dict[int, Tensor] = {}
        self.last_t: Dict[int, float] = {}
        self.gru = nn.GRUCell(d, d)

    def get(self, idx: Tensor) -> Tensor:
        out = []
        zero = torch.zeros(self.d, device=idx.device)
        for i in idx.tolist():
            if i not in self.state:
                self.state[i] = zero.clone()
                self.last_t[i] = 0.0
            out.append(self.state[i])
        return torch.stack(out, dim=0)

    def update(self, idx: Tensor, msg: Tensor, t: float):
        h0 = self.get(idx)
        h1 = self.gru(msg, h0)
        for i, h in zip(idx.tolist(), h1):
            self.state[i] = h.detach()
            self.last_t[i] = t

class TGNLayer(nn.Module):
    def __init__(self, d: int, e_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(2 * d + 1 + e_dim, d)
        self.act = nn.ReLU()

    def forward(self,
                src_h: Tensor,
                dst_h: Tensor,
                dt: Tensor,
                e_feat: Tensor) -> Tensor:
        x = torch.cat([src_h, dst_h, dt.unsqueeze(-1), e_feat], dim=-1)
        return self.act(self.lin_msg(x))

# ------------------------------------------------------------------------------
# 4.  Fusion & Heads
# ------------------------------------------------------------------------------
class FuseAttn(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=heads, batch_first=True)
        self.proj = nn.Linear(d, d)

    def forward(self, mem: Tensor, traj: Tensor) -> Tensor:
        # concatenate as a 2‑token sequence: [state, traj]
        seq = torch.stack([mem, traj], dim=1)
        fused, _ = self.attn(seq, seq, seq)        # (B,2,d)
        return self.proj(fused.mean(dim=1))        # average the two tokens

class NodeCLSHead(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out_dim))

    def forward(self, z): return self.mlp(z)

class TETGN(nn.Module):
    def __init__(self, d_id:int, d_mem:int, e_dim:int,
                 layers:int, heads:int,
                 alpha:float, beta:float):
        super().__init__()
        self.id_emb = LearnableID(d_id)
        self.traj = TrajectoryEncoder(d_id, alpha, beta)
        self.traj_mem: Dict[int, TrajectoryMemory] = {}
        self.mem = TGNMemory(d_mem)
        self.tgn_layers = nn.ModuleList([TGNLayer(d_mem, e_dim) for _ in range(layers)])
        self.fuse = FuseAttn(d_mem + d_id, heads)  # after concat

    def _get_traj_tp(self, idx: Tensor, device) -> Tensor:
        out = []
        for i in idx.tolist():
            if i not in self.traj_mem:
                id_vec = self.id_emb(torch.tensor([i], device=device)).squeeze(0)
                self.traj_mem[i] = TrajectoryMemory(tp=id_vec, last_t=0.0)
            out.append(self.traj_mem[i].tp)
        return torch.stack(out, dim=0)

    def _set_traj_tp(self, idx: Tensor, tp_new: Tensor, t: float):
        for i, tp in zip(idx.tolist(), tp_new):
            self.traj_mem[i] = TrajectoryMemory(tp=tp.detach(), last_t=t)

    def forward(self,
                events: List[Tuple[int,int,float,Tensor]],
                neg_k: int = 5,
                task: str = "link") -> Tuple[Tensor, dict]:
        device = next(self.parameters()).device
        # Accumulators
        pos_pred, neg_pred = [], []

        for (src, dst, t, e_feat) in events:
            src_t = torch.tensor([src], device=device)
            dst_t = torch.tensor([dst], device=device)
            e_feat = e_feat.to(device)

            # === 1. trajectory message passing stream ========================
            #   obtain TP_i(t‑) for src & dst
            tp_src_prev = self._get_traj_tp(src_t, device)
            tp_dst_prev = self._get_traj_tp(dst_t, device)
            #   encode message (each node sends its own TP)
            dt_src = torch.tensor([t - self.traj_mem[src].last_t], device=device) \
                     if src in self.traj_mem else torch.tensor([t], device=device)
            dt_dst = torch.tensor([t - self.traj_mem[dst].last_t], device=device) \
                     if dst in self.traj_mem else torch.tensor([t], device=device)

            msg_src = self.traj.encode(tp_src_prev, dt_src)  # α x exp(‑βΔt)
            msg_dst = self.traj.encode(tp_dst_prev, dt_dst)

            #   aggregate (only one neighbour in an event; extendable)
            agg_src = self.traj.aggregate(tp_src_prev, [msg_dst])
            agg_dst = self.traj.aggregate(tp_dst_prev, [msg_src])

            #   update TP_i(t)
            id_src = self.id_emb(src_t).squeeze(0)
            id_dst = self.id_emb(dst_t).squeeze(0)
            tp_src_new = self.traj.update(id_src, agg_src)
            tp_dst_new = self.traj.update(id_dst, agg_dst)
            self._set_traj_tp(src_t, tp_src_new.unsqueeze(0), t)
            self._set_traj_tp(dst_t, tp_dst_new.unsqueeze(0), t)

            # === 2. MP‑TGN message passing stream ===========================
            h_src_prev = self.mem.get(src_t)
            h_dst_prev = self.mem.get(dst_t)
            delta_t_src = torch.tensor([t - self.mem.last_t[src]], device=device) \
                          if src in self.mem.last_t else torch.tensor([t], device=device)
            delta_t_dst = torch.tensor([t - self.mem.last_t[dst]], device=device) \
                          if dst in self.mem.last_t else torch.tensor([t], device=device)

            msg = []
            for layer in self.tgn_layers:
                msg_layer = layer(h_src_prev, h_dst_prev, delta_t_src, e_feat.unsqueeze(0))
                msg.append(msg_layer)
            msg = sum(msg) / len(msg)
            # update memory for src & dst
            self.mem.update(src_t, msg, t)
            self.mem.update(dst_t, msg, t)

            # === 3. Fuse ======================================================
            h_src = torch.cat([self.mem.get(src_t), tp_src_new.unsqueeze(0)], dim=-1)
            h_dst = torch.cat([self.mem.get(dst_t), tp_dst_new.unsqueeze(0)], dim=-1)
            z_src = self.fuse(h_src, tp_src_new.unsqueeze(0))
            z_dst = self.fuse(h_dst, tp_dst_new.unsqueeze(0))

            # === 4. Link prediction ==========================================
            score_pos = torch.sigmoid((z_src * z_dst).sum(-1))  # dot product
            pos_pred.append(score_pos)

            # negative samples
            neg_nodes = torch.randint(0, self.id_emb.emb.num_embeddings,
                                      (neg_k,), device=device)
            z_neg = self.fuse(
                torch.cat([self.mem.get(neg_nodes),
                           self._get_traj_tp(neg_nodes, device)], dim=-1),
                self._get_traj_tp(neg_nodes, device))
            score_neg = torch.sigmoid((z_src * z_neg).sum(-1))  # (neg_k,)
            neg_pred.append(score_neg)

        # BCE loss
        pos_pred = torch.cat(pos_pred)
        neg_pred = torch.cat(neg_pred)
        y_pos = torch.ones_like(pos_pred)
        y_neg = torch.zeros_like(neg_pred)
        loss = F.binary_cross_entropy(
            torch.cat([pos_pred, neg_pred]),
            torch.cat([y_pos, y_neg])
        )
        diag = dict(pos=pos_pred.mean().item(),
                    neg=neg_pred.mean().item())
        return loss, diag

@torch.no_grad()
def evaluate(model: TETGN, loader):
    model.eval()
    ap_numer, ap_denom = 0.0, 0.0
    for batch in loader:
        loss, diag = model(batch, neg_k=5)
        ap_numer += diag["pos"]  # crude proxy; compute real AP if needed
        ap_denom += 1
    return ap_numer / ap_denom

def train_one_epoch(model: TETGN, loader, optim, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optim.zero_grad()
        loss, _ = model(batch, neg_k=5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    cfg = dict(
        d_id = 20,
        d_mem = 128,
        e_dim = 172,          # Wikipedia / Reddit; set 0 for -fm / LastFM
        layers = 2,
        heads = 4,
        alpha = 2.0,
        beta  = 0.1,
        lr    = 1e-4,
        epochs= 5,
        batch_size = 200,
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )

    model = TETGN(cfg["d_id"], cfg["d_mem"], cfg["e_dim"],
                  cfg["layers"], cfg["heads"],
                  cfg["alpha"], cfg["beta"]).to(cfg["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    def dummy_loader(num_batches=10, bs=cfg["batch_size"]):
        for _ in range(num_batches):
            batch=[]
            for _ in range(bs):
                s,d = secrets.SystemRandom().randint(0,1999), secrets.SystemRandom().randint(0,1999)
                t   = secrets.SystemRandom().random()*1000
                e   = torch.randn(cfg["e_dim"])
                batch.append((s,d,t,e))
            # sort by time like real pre‑processing
            batch.sort(key=lambda x: x[2])
            yield batch

    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, dummy_loader(), opt, cfg["device"])
        val  = evaluate(model, dummy_loader(2))
        print(f"Epoch {epoch:02d} | loss {loss:.4f} | val_ap~ {val:.4f}")
