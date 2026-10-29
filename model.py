"""
HA-Net: Hardware-aware Attention Network for cross-subject MI decoding.

Input: (B, 1, C, T) EEG after online EA, C=22, T=500 (4 s @ 125 Hz)

 Block 1  temporal conv   Conv2d(1->F1, (1,K1), same)              + BN     (BN folded at export)
 Block 2  spatial dw conv Conv2d(F1->F1*D, (C,1), groups=F1)       + BN + ReLU + AvgPool(1,P1)
 Block 3  separable conv  dw Conv2d(F2, (1,K2), groups=F2) -> pw 1x1 + BN + ReLU + AvgPool(1,P2)
          -> L = T/(P1*P2) tokens of dimension F2              (500/32 -> 15 tokens x 16)
 Block 4  HW-attention    x + MHSA(BN(x))      softmax -> base-2 PWL softmax (approx.py)
                          x + FFN(BN(x))       FFN: Linear(F2->2F2)-ReLU-Linear(2F2->F2)
 Head     flatten (L*F2) -> Linear(n_cls)

Hardware choices (vs. EEGNet / ATCNet):
 * ReLU instead of ELU (no exp), power-of-two pooling (shift), BatchNorm instead of LayerNorm
   (folded into the following linear layer -> no sqrt/div at inference)
 * attention scale log2(e)/sqrt(d_h) folded into W_q at export
 * ~6k parameters -> all weights fit in on-chip BRAM of the XC7Z020
"""
import math
import torch
import torch.nn as nn
from approx import softmax_pow2, pow2_pwl, LOG2E


def hw_elu(x, approx=True):
    """ELU with the SAME base-2 PWL exponential unit used by the softmax:
       x>0 -> x ;  x<=0 -> 2^(x*log2 e) - 1."""
    neg = (pow2_pwl(torch.clamp(x, max=0.0) * LOG2E) - 1.0) if approx else torch.expm1(torch.clamp(x, max=0.0))
    return torch.where(x > 0, x, neg)


class ConvAct(nn.Module):
    def __init__(self, kind="relu", approx=True):
        super().__init__()
        self.kind, self.approx = kind, approx

    def forward(self, x):
        return torch.relu(x) if self.kind == "relu" else hw_elu(x, self.approx)


class HWAttention(nn.Module):
    def __init__(self, d, heads, approx=True):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.approx = heads, d // heads, approx
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.scale = (LOG2E if approx else 1.0) / math.sqrt(self.dh)
        self.last_attn = None

    def forward(self, x):                          # x: (B, L, d)
        B, L, d = x.shape
        sh = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)   # (B,h,L,dh)
        q, k, v = sh(self.q(x)), sh(self.k(x)), sh(self.v(x))
        s = (q @ k.transpose(-1, -2)) * self.scale
        p = softmax_pow2(s) if self.approx else s.softmax(-1)
        self.last_attn = p.detach()
        out = (p @ v).transpose(1, 2).reshape(B, L, d)
        return self.o(out)


class HANet(nn.Module):
    def __init__(self, C=22, T=500, n_cls=4, F1=8, D=2, K1=32, P1=8, K2=16, P2=4,
                 heads=2, ffn_mult=2, attention=True, approx=True, dropout=0.25, act="relu"):
        super().__init__()
        F2 = F1 * D
        self.cfg = dict(C=C, T=T, n_cls=n_cls, F1=F1, D=D, K1=K1, P1=P1, K2=K2, P2=P2,
                        heads=heads, ffn_mult=ffn_mult, attention=attention, approx=approx, act=act)
        self.temporal = nn.Conv2d(1, F1, (1, K1), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F2, (C, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d((1, P1))
        self.sep_dw = nn.Conv2d(F2, F2, (1, K2), padding="same", groups=F2, bias=False)
        self.sep_pw = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, P2))
        self.drop = nn.Dropout(dropout)
        self.act = ConvAct(act, approx)     # conv blocks: ReLU or HW-ELU
        self.ffn_act = nn.ReLU()            # FFN keeps ReLU
        self.L = T // P1 // P2
        self.attention = attention
        if attention:
            self.bn_a = nn.BatchNorm1d(F2)
            self.attn = HWAttention(F2, heads, approx)
            self.bn_f = nn.BatchNorm1d(F2)
            self.ffn1 = nn.Linear(F2, ffn_mult * F2)
            self.ffn2 = nn.Linear(ffn_mult * F2, F2)
        self.head = nn.Linear(self.L * F2, n_cls)

    def features(self, x):                          # x: (B,1,C,T)
        x = self.bn1(self.temporal(x))
        x = self.drop(self.pool1(self.act(self.bn2(self.spatial(x)))))
        x = self.bn3(self.sep_pw(self.sep_dw(x)))
        x = self.drop(self.pool2(self.act(x)))      # (B,F2,1,L)
        x = x.squeeze(2).transpose(1, 2)            # (B,L,F2) tokens
        if self.attention:
            bn = lambda m, t: m(t.transpose(1, 2)).transpose(1, 2)
            x = x + self.drop(self.attn(bn(self.bn_a, x)))
            x = x + self.drop(self.ffn2(self.ffn_act(self.ffn1(bn(self.bn_f, x)))))
        return x.flatten(1)                          # (B, L*F2)

    def forward(self, x):
        return self.head(self.features(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_macs(cfg):
    C, T, F1, D, K1, P1, K2, P2 = (cfg[k] for k in ("C", "T", "F1", "D", "K1", "P1", "K2", "P2"))
    F2, L = F1 * D, T // P1 // P2
    macs = {
        "temporal": F1 * C * T * K1,
        "spatial": F2 * C * T,
        "sep_dw": F2 * (T // P1) * K2,
        "sep_pw": F2 * F2 * (T // P1),
    }
    if cfg["attention"]:
        dff = cfg["ffn_mult"] * F2
        macs["qkvo"] = 4 * L * F2 * F2
        macs["attn_scores+pv"] = 2 * L * L * F2
        macs["ffn"] = 2 * L * F2 * dff
    macs["head"] = L * F2 * cfg["n_cls"]
    macs["total"] = sum(macs.values())
    return macs


if __name__ == "__main__":
    m = HANet()
    x = torch.randn(4, 1, 22, 500)
    print(m(x).shape, "params:", count_params(m))
    print(count_macs(m.cfg))


# ------------------------------------------------------------------ baselines (same protocol)
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., J. Neural Eng. 2018), adapted to 125 Hz input."""
    def __init__(self, C=22, T=500, n_cls=4, F1=8, D=2, K1=64, dropout=0.25, **_):
        super().__init__()
        F2 = F1 * D
        self.cfg = dict(model="eegnet", C=C, T=T, n_cls=n_cls, F1=F1, D=D, K1=K1)
        self.block = nn.Sequential(
            nn.Conv2d(1, F1, (1, K1), padding="same", bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.head = nn.Linear(F2 * (T // 4 // 8), n_cls)

    def features(self, x):
        return self.block(x).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al., Hum. Brain Mapp. 2017), adapted to 125 Hz input."""
    def __init__(self, C=22, T=500, n_cls=4, F=40, K=13, pool=38, stride=8, dropout=0.5, **_):
        super().__init__()
        self.cfg = dict(model="shallow", C=C, T=T, n_cls=n_cls)
        self.temporal = nn.Conv2d(1, F, (1, K))
        self.spatial = nn.Conv2d(F, F, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(F)
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(dropout)
        L = ((T - K + 1) - pool) // stride + 1
        self.head = nn.Linear(F * L, n_cls)

    def features(self, x):
        x = self.bn(self.spatial(self.temporal(x)))
        x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        return self.drop(x).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))
