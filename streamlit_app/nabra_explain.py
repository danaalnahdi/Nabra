# AASIST layers adapted from DeepFense (Apache License 2.0):
#   https://github.com/Yaselley/deepfense-framework  deepfense/models/backends/aasist.py
import math, os, random, types
from threading import Lock
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from scipy.signal import resample_poly
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINT = os.environ.get("NABRA_CHECKPOINT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt"))
# Main model: XLS-R 300M + WPT + AASIST (run xlsr300m__wpt__ftrecipe__lr0.0001__seed42, epoch 18 chosen on dev).
# Threshold = EER threshold on the Track 1 development-test set (whole clip); final-test EER 5.93%.
THRESHOLD = 10.4609375
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = dict(device_type=DEVICE.type, dtype=torch.bfloat16)
CFG = dict(hf_model="facebook/wav2vec2-xls-r-300m", sample_rate=16000, max_len=64600, trim_silence=True, rms_dbfs=-23.0,
           n_prompts=6, n_wavelet_prompts=4, prompt_dropout=0.1, spec_augment=False, gradient_checkpointing=False,
           aasist=dict(filts=[70, [1, 32], [32, 32], [32, 64], [64, 64]], gat_dims=[64, 32],
                       pool_ratios=[0.5, 0.7, 0.5, 0.5], temperatures=[2.0, 2.0, 100.0, 100.0]))
SR, N, HOP = CFG["sample_rate"], CFG["max_len"], 320
THR = THRESHOLD


def load_wave(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(1)
    if sr != SR:
        g = math.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def trim_silence(x, top_db=40.0, frame=400, hop=160):
    if len(x) < frame: return x
    nf = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(nf)[:, None]
    e = 10 * np.log10((x[idx] ** 2).mean(1) + 1e-10)
    v = np.flatnonzero(e > e.max() - top_db)
    return x[v[0] * hop: v[-1] * hop + frame] if len(v) else x


def rms_normalize(x, target_dbfs=-23.0):
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    return np.clip(x * (10 ** (target_dbfs / 20) / rms), -1, 1).astype(np.float32)


def clean_wave(x):
    if CFG["trim_silence"]: x = trim_silence(x)
    return rms_normalize(x, CFG["rms_dbfs"])


def crop_or_pad(x, n, mode, rng=None):
    if len(x) == 0: return np.zeros(n, np.float32)
    if len(x) < n: x = np.tile(x, int(np.ceil(n / len(x))))
    if len(x) == n: return x
    if mode == "random": start = rng.randint(0, len(x) - n + 1)
    elif mode == "center": start = (len(x) - n) // 2
    else: start = 0
    return x[start:start + n]


def make_windows(x, n):
    if len(x) <= n: return crop_or_pad(x, n, "start")[None, :]
    starts = list(range(0, len(x) - n + 1, n))
    if starts[-1] + n < len(x): starts.append(len(x) - n)
    return np.stack([x[s:s + n] for s in starts])


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.0
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        x = self.input_drop(x)

        att_map = self._derive_att_map(x)

        x = self._project(x, att_map)

        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)

        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.0
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)

        att_map = self._derive_att_map(x, num_type1, num_type2)

        master = self._update_master(x, master)

        x = self._project(x, att_map)

        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )

        att_map = att_board

        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, pool=(1, 3), first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d(pool)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out


class AASIST(nn.Module):
    def __init__(self, input_dim=1024, filts=(70, [1, 32], [32, 32], [32, 64], [64, 64]), gat_dims=(64, 32),
                 pool_ratios=(0.5, 0.7, 0.5, 0.5), temperatures=(2.0, 2.0, 100.0, 100.0)):
        super().__init__()
        self.LL = nn.Linear(input_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

    def forward(self, x):
        x = self.LL(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)

        return last_hidden


def haar_dwt(q):
    even, odd = q[..., 0::2], q[..., 1::2]
    return (even + odd) / math.sqrt(2), (even - odd) / math.sqrt(2)


class WPTEncoder(nn.Module):
    def __init__(self, hf_name):
        super().__init__()
        conf = AutoConfig.from_pretrained(hf_name)
        conf.apply_spec_augment = CFG["spec_augment"]
        self.ssl = AutoModel.from_pretrained(hf_name, config=conf)
        self.ssl.requires_grad_(False)
        self.is_wavlm = conf.model_type == "wavlm"
        self.pre_ln = bool(conf.do_stable_layer_norm)
        L, d = conf.num_hidden_layers, conf.hidden_size
        assert d % 2 == 0
        self.n_p, self.n_w = CFG["n_prompts"], CFG["n_wavelet_prompts"]
        bound = math.sqrt(6.0 / (2 * d))
        self.P = nn.Parameter(torch.empty(L, self.n_p, d).uniform_(-bound, bound))
        self.Q = nn.Parameter(torch.empty(L, self.n_w, d).uniform_(-bound, bound))
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(CFG["prompt_dropout"])

    def train(self, mode=True):
        super().train(mode)
        self.ssl.eval()
        return self

    def forward(self, wav, use_prompts=True):
        B, enc = wav.shape[0], self.ssl.encoder
        with torch.no_grad():
            f = self.ssl.feature_extractor(wav).transpose(1, 2)
            h, _ = self.ssl.feature_projection(f)
            h = h + enc.pos_conv_embed(h)
            if not self.pre_ln:
                h = enc.layer_norm(h)
            h = enc.dropout(h)
        n = self.n_w + self.n_p if use_prompts else 0
        bias = None
        for i, layer in enumerate(enc.layers):
            if use_prompts:
                A, D = haar_dwt(self.Q[i])
                wp = self.drop(self.proj(torch.cat([A, D], -1)).expand(B, -1, -1))
                p = self.drop(self.P[i].expand(B, -1, -1))
                x = torch.cat([wp.to(h.dtype), p.to(h.dtype), h], 1)
            else:
                x = h
            if self.is_wavlm:
                run = lambda x, bias, layer=layer, i=i: layer(x, position_bias=bias, index=i)
            else:
                run = lambda x, bias, layer=layer: (layer(x), None)
            if CFG["gradient_checkpointing"] and self.training:
                x, bias = checkpoint(run, x, bias, use_reentrant=False)
            else:
                x, bias = run(x, bias)
            h = x[:, n:]
        if self.pre_ln:
            h = enc.layer_norm(h)
        return h


class Detector(nn.Module):
    def __init__(self, hf_name):
        super().__init__()
        self.enc = WPTEncoder(hf_name)
        self.do_normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(hf_name), "do_normalize", False))
        self.backend = AASIST(input_dim=self.enc.ssl.config.hidden_size, **CFG["aasist"])
        self.fc = nn.Linear(5 * CFG["aasist"]["gat_dims"][1], 2)

    def forward(self, wav):
        if self.do_normalize:
            with torch.autocast("cuda", enabled=False):
                wav = wav.float()
                wav = (wav - wav.mean(1, keepdim=True)) / torch.sqrt(wav.var(1, keepdim=True, unbiased=False) + 1e-7)
        return self.fc(self.backend(self.enc(wav)))


def score_of(logits):
    return (logits[:, 1] - logits[:, 0]).float()


def attn_keep(self, hidden_states, key_value_states=None, attention_mask=None, output_attentions=False, **kw):
    # XLS-R (wav2vec2) attention that keeps the per-head attention map A and its gradient; same output as the original
    if not ATT["keep"]: return HF_ATTN(self, hidden_states, key_value_states, attention_mask, output_attentions, **kw)
    B, T, _ = hidden_states.shape; H, hd = self.num_heads, self.head_dim
    q, k, v = [p(hidden_states).view(B, T, H, hd).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj)]
    A = torch.softmax((q @ k.transpose(-1, -2)) * self.scaling, dim=-1)
    A.retain_grad(); ATT["A"].append(A)
    return self.out_proj((A @ v).transpose(1, 2).reshape(B, T, H * hd)), A


def gatr_from(As, n):
    T = As[0].shape[-1] - n; I = torch.eye(T, device=As[0].device); R = I.clone()
    for A in As:
        R = R + (A.grad * A).clamp(min=0).mean(1)[0, n:, n:] @ R
    W = As[-1].grad.mean(1)[0, n:].norm(dim=1)
    return ((W[:, None] * (R - I)).sum(0) / W.sum()).detach().float().cpu().numpy()


def gatr_window(x, target):
    _MODEL.zero_grad(set_to_none=True); ATT.update(keep=True, A=[])
    try:
        with torch.enable_grad():
            logits = _MODEL(torch.as_tensor(np.asarray(x, np.float32)).reshape(1, -1).to(DEVICE))
            logits[0, int(target)].backward()
        return gatr_from(ATT["A"], _MODEL.enc.n_w + _MODEL.enc.n_p), logits.detach().float().cpu().numpy()[0]
    finally:
        ATT.update(keep=False, A=[])


@torch.no_grad()
def window_scores(W):
    with torch.autocast(**AMP):
        return np.concatenate([score_of(_MODEL(torch.from_numpy(np.ascontiguousarray(b)).to(DEVICE))).float().cpu().numpy()
                               for b in np.array_split(W, int(np.ceil(len(W) / 16)))])


def explain_clip(x, target=None):
    x = clean_wave(np.asarray(x, np.float32))
    W = make_windows(x, N)
    starts = [0] if len(x) <= N else sorted(set(range(0, len(x) - N + 1, N)) | {len(x) - N})
    score = float(window_scores(W).mean())
    c = (1 if score >= THR else 0) if target is None else int(target)
    nf = int(np.ceil(len(x) / HOP)); acc, cnt = np.zeros(nf), np.zeros(nf)
    for st, w in zip(starts, W):
        r, _ = gatr_window(w, c); o = int(round(st / HOP)); k = min(len(r), nf - o)
        acc[o:o + k] += r[:k]; cnt[o:o + k] += 1
    rel = np.clip(acc / np.maximum(cnt, 1), 0, None)
    return dict(score=score, cls=c, pred="bona fide" if c == 1 else "spoof", frames=rel, flat=bool(rel.max() <= 0), wave=x)


def top_segments(rel, k=3, width=25):
    if rel.max() <= 0: return []
    s, out = np.convolve(rel, np.ones(width), "same"), []
    for _ in range(k):
        i = int(s.argmax())
        if not np.isfinite(s[i]) or s[i] <= 0: break
        a, b = max(0, i - width // 2), min(len(rel), i + width // 2 + 1)
        out.append((a * HOP / SR, b * HOP / SR)); s[max(0, a - width // 2): b + width // 2] = -np.inf
    return sorted(out)


def plot_explanation(e, title=""):
    x, rel = e["wave"], e["frames"]; fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(np.arange(len(x)) / SR, x, lw=.4, color="0.6")
    if not e["flat"]:
        ax2 = ax.twinx(); tc = (np.arange(len(rel)) + .5) * HOP / SR
        ax2.fill_between(tc, rel / rel.max(), color="tab:red", alpha=.35, step="mid"); ax2.set_ylim(0, 1.05); ax2.set_yticks([])
        for a, b in top_segments(rel): ax.axvspan(a, b, color="tab:orange", alpha=.2)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{title} prediction: {e['pred']} (score {e['score']:.2f}, threshold {THR:.2f})"
                 + (" | no localized evidence" if e["flat"] else ""))
    plt.tight_layout(); return fig


_MODEL, _LOCK, ATT, HF_ATTN = None, Lock(), {"keep": False, "A": []}, None


def load_model():
    global _MODEL, HF_ATTN
    with _LOCK:
        if _MODEL is None:
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            m = Detector(CFG["hf_model"])
            ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            m.load_state_dict(ck["model"]); m.to(DEVICE).eval()
            layers = m.enc.ssl.encoder.layers
            HF_ATTN = type(layers[0].attention).forward
            for layer in layers:
                layer.attention.forward = types.MethodType(attn_keep, layer.attention)
            _MODEL = m
    return _MODEL


def _result(score):
    p_real = 1 / (1 + math.exp(-(score - THRESHOLD)))
    real = score >= THRESHOLD
    return {"label": "real" if real else "fake", "confidence": p_real if real else 1 - p_real, "score": score}


def predict(audio_path: str) -> dict:
    load_model()
    x = clean_wave(load_wave(audio_path))
    with _LOCK:
        return _result(float(window_scores(make_windows(x, N)).mean()))


def explain(audio_path: str) -> dict:
    load_model()
    with _LOCK:
        e = explain_clip(load_wave(audio_path))
    out = _result(e["score"])
    out.update(segments=top_segments(e["frames"]), frames=e["frames"], figure=plot_explanation(e))
    return out
