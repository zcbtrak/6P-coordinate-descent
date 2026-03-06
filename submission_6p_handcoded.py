"""
AdderBoard submission: 6-parameter adder transformer (hand-coded, pure numpy).

Architecture: 1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2.
Based on Lokimorty's accepted 10-parameter hand-coded submission.

Reduces the 10-parameter architecture to 6 by hardcoding:
  - Q_proj as architectural constant: [cos(PHI), -sin(PHI)] where
    PHI = (2*pi/19)*(10+0.3). This is the attention routing angle,
    derived from RoPE period 19 — analogous to choosing a RoPE base theta.
  - Final RMSNorm weights folded into the output head as constants:
      logits = rms_norm(h, w) @ emb.T   (10-param model)
             ≡ unit_rms_norm(h) @ (w ⊙ emb).T   (6-param model)
    The two formulations are algebraically identical, but the second
    absorbs the norm weights [50√2, -10√2] into the tied output table,
    requiring zero extra parameters.

The 6 learnable parameters:
  embed_w0  = 1000.0    — embedding bias (parabolic vertex)
  embed_w1  = 0.001     — embedding quadratic coefficient
  v_proj_w  ≈ -15556.3  — V projection scalar
  gate_w0   = -12032.0  — tied gate weight (digit-sum threshold)
  gate_w1   = 128000.0  — tied gate weight (carry-mix slope)
  carry_w   ≈ 0.3906    — shared carry projection scalar
"""

import math
import numpy as np

# ── Architecture constants ──────────────────────────────────────────────────

VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
PROMPT_LEN = 31

MODEL_DIM = 2
HEAD_DIM = 2

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)  # hardcoded Q angle

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
ATTN_SCALE = (HEAD_DIM ** -0.5) * (QK_NORM_SCALE ** 2)

EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
CARRY_ALPHA = 256.0 / CONST_NORM

# Hardcoded norm weights (folded into output head)
NORM_W0 = 50.0 * math.sqrt(2.0)    # ≈ 70.711
NORM_W1 = -10.0 * math.sqrt(2.0)   # ≈ -14.142

RMS_EPS = 1e-6

# ── The 6 hand-coded parameter values ───────────────────────────────────────

PARAMS = np.array([
    EMBED_CONST,                                   # embed_w0 = 1000.0
    1e-3,                                          # embed_w1 = 0.001
    -22.0 * DIGIT_SCALE,                           # v_proj_w ≈ -15556.349
    CARRY_ALPHA * (-94.0) / CONST_NORM,            # gate_w0  = -12032.0
    CARRY_ALPHA * DIGIT_SCALE,                     # gate_w1  = 128000.0
    (100.0 / CARRY_ALPHA) * (1.0 / CONST_NORM),   # carry_w  ≈ 0.390625
], dtype=np.float64)

PARAM_NAMES = ['embed_w0', 'embed_w1', 'v_proj_w', 'gate_w0', 'gate_w1', 'carry_w']

# ── Forward pass helpers ────────────────────────────────────────────────────

def _unit_rms_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _apply_rope(x, positions):
    a = positions * OMEGA
    cos_a = np.cos(a)[..., np.newaxis]
    sin_a = np.sin(a)[..., np.newaxis]
    return np.concatenate([
        x[..., 0:1] * cos_a - x[..., 1:2] * sin_a,
        x[..., 0:1] * sin_a + x[..., 1:2] * cos_a,
    ], axis=-1)


def _build_embed_table(w0, w1):
    d = np.arange(VOCAB_SIZE, dtype=np.float64)
    return np.stack([w0 - w1 * d * d, -d], axis=-1)


def _forward(params, token_ids):
    """Full forward pass → logits (B, T, V)."""
    embed_w0, embed_w1, v_proj_w, gate_a, gate_c, carry_w = params

    B, T = token_ids.shape
    embed_table = _build_embed_table(embed_w0, embed_w1)
    h = embed_table[token_ids]

    # ── Attention ──
    hn = _unit_rms_norm(h)

    # Hardcoded Q: [x0*cos(PHI), -x0*sin(PHI)]
    COS_PHI = math.cos(PHI)
    SIN_PHI = math.sin(PHI)
    q = np.stack([hn[..., 0] * COS_PHI, hn[..., 0] * (-SIN_PHI)], axis=-1)
    k = np.stack([hn[..., 0], np.zeros_like(hn[..., 0])], axis=-1)
    v = np.stack([hn[..., 1] * v_proj_w, np.zeros_like(hn[..., 1])], axis=-1)

    q = _unit_rms_norm(q)
    k = _unit_rms_norm(k)

    positions = np.arange(T, dtype=np.float64)[np.newaxis, :]
    q = _apply_rope(q, positions)
    k = _apply_rope(k, positions)

    scores = np.einsum('btd,bsd->bts', q, k) * ATTN_SCALE
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal[np.newaxis], -np.inf, scores)
    attn_w = _softmax(scores, axis=-1)
    attn_out = np.einsum('bts,bsd->btd', attn_w, v)

    h = h + np.stack([np.zeros_like(attn_out[..., 0]), attn_out[..., 0]], axis=-1)

    # ── MLP ──
    hn = _unit_rms_norm(h)
    g0 = hn[..., 0] * gate_a + hn[..., 1] * gate_c
    g1 = hn[..., 0] * (gate_a - gate_c / EMBED_CONST) + hn[..., 1] * gate_c
    gate = np.stack([g0, g1], axis=-1)

    base = hn[..., 0]
    up = np.stack([base, base], axis=-1)
    mix = _silu(gate) * up

    h = h + np.stack([np.zeros_like(base), carry_w * (mix[..., 1] - mix[..., 0])], axis=-1)

    # ── Output (UnitRMSNorm + folded output head) ──
    h = _unit_rms_norm(h)
    norm_w = np.array([NORM_W0, NORM_W1])
    folded_table = embed_table * norm_w[np.newaxis, :]
    return np.einsum('btd,vd->btv', h, folded_table)


# ── Encoding ────────────────────────────────────────────────────────────────

def _encode_prompt(a, b):
    a_digits = [int(c) for c in f"{a:010d}"][::-1]
    b_digits = [int(c) for c in f"{b:010d}"][::-1]
    return [0] + a_digits + [0] * 9 + b_digits + [0]


# ── Public API (required by verify.py) ──────────────────────────────────────

class _Model:
    """Thin wrapper holding the 6 parameters as a numpy array."""
    def __init__(self, params):
        self.params = params


def build_model():
    model = _Model(PARAMS)
    metadata = {
        "name": "qwen6_fixedq_foldednorm",
        "author": "zcbtrak",
        "params": 6,
        "architecture": "1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2 (fixed Q, folded norm)",
        "tricks": [
            "RoPE period-19 geometry",
            "hardcoded Q_proj (routing as architectural constant, 0 params)",
            "norm weights folded into tied output head (0 extra params)",
            "tied carry hinge gate",
            "shared carry-scale scalar",
            "2-parameter embedding e(d)=[c0-c1*d^2,-d]",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")
    seq = _encode_prompt(a, b)
    for _ in range(OUTPUT_DIGITS):
        logits = _forward(model.params, np.array([seq], dtype=np.int64))
        seq.append(int(np.argmax(logits[0, -1, :])))
    digits = seq[-OUTPUT_DIGITS:]
    return int(''.join(str(d) for d in digits)[::-1])
