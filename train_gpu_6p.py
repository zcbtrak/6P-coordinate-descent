%%writefile train_gpu_6p.py
#!/usr/bin/env python3
"""
train_gpu_6p.py — GPU-accelerated coordinate descent for the 6-parameter adder.

This script uses PyTorch to parallelize the evaluation of grid search candidates
across training problems.

Usage:
    python train_gpu_6p.py --starts 10 --cycles 30 --n-points 100 --n-train 128
"""

import argparse
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

# =============================================================================
# Architectural Constants
# =============================================================================
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
PROMPT_LEN = 31

MODEL_DIM = 2
HEAD_DIM = 2

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD

PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)
COS_PHI = math.cos(PHI)
SIN_PHI = math.sin(PHI)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
# Normalized scale for attention scores
ATTN_SCALE = (HEAD_DIM ** -0.5) * (QK_NORM_SCALE ** 2)

EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
CARRY_ALPHA = 256.0 / CONST_NORM

# Folded norm weights
NORM_W0 = 50.0 * math.sqrt(2.0)
NORM_W1 = -10.0 * math.sqrt(2.0)

RMS_EPS = 1e-6
N_PARAMS = 6
PARAM_NAMES = ['embed_w0', 'embed_w1', 'v_proj_w', 'gate_w0', 'gate_w1', 'carry_w']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# PyTorch Forward Pass (Batched)
# =============================================================================

def unit_rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_EPS)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def apply_rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    # pos: (T,) or (B, T) or (1, T)
    angles = pos * OMEGA
    cos_a = torch.cos(angles).unsqueeze(-1)
    sin_a = torch.sin(angles).unsqueeze(-1)
    
    # x: (..., 2)
    x0 = x[..., 0:1]
    x1 = x[..., 1:2]
    return torch.cat([
        x0 * cos_a - x1 * sin_a,
        x0 * sin_a + x1 * cos_a
    ], dim=-1)

def forward_pass_batched(params: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    params: (BC, 6) where BC is batch_size * num_candidates
    token_ids: (BC, T) 
    """
    BC, T = token_ids.shape
    w0, w1, vw, ga, gc, cw = params.unbind(dim=-1) # (BC,) each
    
    # Build embed table per candidate
    # d: (10,)
    d = torch.arange(VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    # embed_table: (BC, 10, 2)
    et = torch.stack([
        w0.unsqueeze(-1) - w1.unsqueeze(-1) * (d * d),
        -d.expand(BC, 10)
    ], dim=-1)
    
    # h: (BC, T, 2)
    # We need to use gather to pick embeddings based on token_ids
    # Actually, token_ids are (BC, T), and we want to index into et which is (BC, 10, 2)
    # We can use torch.gather or just index if we flatten et
    h = et[torch.arange(BC, device=DEVICE).unsqueeze(-1), token_ids]

    # Attention
    hn = unit_rms_norm(h)
    
    # q, k, v (BC, T, 2)
    q = torch.stack([hn[..., 0] * COS_PHI, hn[..., 0] * (-SIN_PHI)], dim=-1)
    k = torch.stack([hn[..., 0], torch.zeros_like(hn[..., 0])], dim=-1)
    v = torch.stack([hn[..., 1] * vw.view(BC, 1), torch.zeros_like(hn[..., 1])], dim=-1)
    
    q = unit_rms_norm(q)
    k = unit_rms_norm(k)
    
    pos = torch.arange(T, dtype=torch.float32, device=DEVICE).view(1, T)
    q = apply_rope(q, pos)
    k = apply_rope(k, pos)
    
    # scores: (BC, T, T)
    scores = torch.matmul(q, k.transpose(-1, -2)) * ATTN_SCALE
    mask = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    
    h = h + torch.stack([torch.zeros_like(attn_out[..., 0]), attn_out[..., 0]], dim=-1)
    
    # MLP
    hn = unit_rms_norm(h)
    # ga, gc: (BC,)
    ga = ga.view(BC, 1, 1)
    gc = gc.view(BC, 1, 1)
    
    g0 = hn[..., 0:1] * ga + hn[..., 1:2] * gc
    g1 = hn[..., 0:1] * (ga - gc / EMBED_CONST) + hn[..., 1:2] * gc
    gate = torch.cat([g0, g1], dim=-1)
    
    base = hn[..., 0:1]
    up = torch.cat([base, base], dim=-1)
    mix = silu(gate) * up
    
    h = h + torch.stack([
        torch.zeros_like(base.squeeze(-1)),
        cw.view(BC, 1) * (mix[..., 1] - mix[..., 0])
    ], dim=-1)
    
    # Output
    h = unit_rms_norm(h)
    nw = torch.tensor([NORM_W0, NORM_W1], device=DEVICE).view(1, 1, 2)
    # folded_et: (BC, 10, 2)
    folded_et = et * nw
    
    # logits: (BC, T, 10)
    # Using matmul: (BC, T, 1, 2) * (BC, 1, 10, 2).transpose(-1, -2) -> (BC, T, 1, 10)
    logits = torch.matmul(h.unsqueeze(-2), folded_et.unsqueeze(1).transpose(-1, -2)).squeeze(-2)
    return logits

# =============================================================================
# Batched Fitness Evaluation
# =============================================================================

def encode_problems(problems: List[Tuple[int, int]]) -> torch.Tensor:
    B = len(problems)
    prompts = torch.zeros((B, PROMPT_LEN), dtype=torch.long, device=DEVICE)
    for i, (a, b) in enumerate(problems):
        a_digits = [int(c) for c in f"{a:010d}"][::-1]
        b_digits = [int(c) for c in f"{b:010d}"][::-1]
        prompts[i] = torch.tensor([0] + a_digits + [0] * 9 + b_digits + [0], device=DEVICE)
    return prompts

def get_problem_targets(problems: List[Tuple[int, int]]) -> torch.Tensor:
    B = len(problems)
    targets = torch.zeros((B, OUTPUT_DIGITS), dtype=torch.long, device=DEVICE)
    for i, (a, b) in enumerate(problems):
        sum_val = a + b
        t_digits = [int(c) for c in f"{sum_val:011d}"][::-1]
        targets[i] = torch.tensor(t_digits, device=DEVICE)
    return targets

def evaluate_fitness_batched(
    params_candidates: torch.Tensor, # (NC, 6)
    prompts: torch.Tensor,          # (B, PROMPT_LEN)
    targets: torch.Tensor,          # (B, OUTPUT_DIGITS)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        accuracy: (NC,) mean digit accuracy per candidate
        margin: (NC,) mean margin per candidate
        prob_acc: (NC,) mean problem accuracy per candidate (all digits correct)
    """
    NC = params_candidates.shape[0]
    B = prompts.shape[0]
    BC = NC * B
    
    # Flatten candidates and problems into one batch
    # params_flat: (BC, 6)
    params_flat = params_candidates.repeat_interleave(B, dim=0)
    # seq: (BC, PROMPT_LEN)
    seq = prompts.repeat(NC, 1)
    # targets_flat: (BC, OUTPUT_DIGITS)
    targets_flat = targets.repeat(NC, 1)
    
    total_correct = torch.zeros(BC, device=DEVICE)
    total_margin = torch.zeros(BC, device=DEVICE)
    
    for i in range(OUTPUT_DIGITS):
        logits = forward_pass_batched(params_flat, seq)
        # We only care about the last token's prediction
        logit_vec = logits[:, -1, :] # (BC, 10)
        
        preds = torch.argmax(logit_vec, dim=-1)
        seq = torch.cat([seq, preds.unsqueeze(-1)], dim=-1)
        
        target_digit = targets_flat[:, i]
        correct_mask = (preds == target_digit)
        total_correct += correct_mask.float()
        
        # Margin calculation
        correct_logits = logit_vec.gather(1, target_digit.unsqueeze(-1)).squeeze(-1)
        top2_vals, _ = torch.topk(logit_vec, 2, dim=-1)
        best_vals = top2_vals[:, 0]
        second_best_vals = top2_vals[:, 1]
        
        alt_logit = torch.where(correct_mask, second_best_vals, best_vals)
        margin = correct_logits - alt_logit
        total_margin += margin
        
    # Reshape back to (NC, B) and average over problems
    accuracy = total_correct.view(NC, B).mean(dim=1) / OUTPUT_DIGITS
    margin = total_margin.view(NC, B).mean(dim=1) / OUTPUT_DIGITS
    prob_acc = (total_correct == OUTPUT_DIGITS).float().view(NC, B).mean(dim=1)
    
    return accuracy, margin, prob_acc

# =============================================================================
# Random restart search (domain-knowledge-free)
# =============================================================================

def random_restart_search(
    param_ranges: list,
    smart_random_fn,
    n_candidates: int,
    prompts: torch.Tensor,
    targets: torch.Tensor,
    max_bc: int = 100_000,
) -> Tuple[torch.Tensor, float]:
    """
    Sample n_candidates random parameter vectors from smart_random ranges,
    evaluate all in batched chunks, return the best.
    Zero domain knowledge — just brute-force GPU parallelism.
    """
    candidates = torch.stack([smart_random_fn() for _ in range(n_candidates)])

    B = prompts.shape[0]
    chunk_size = max(1, max_bc // B)

    best_prob_acc = -1.0
    best_params = candidates[0].clone()

    for i in range(0, n_candidates, chunk_size):
        chunk = candidates[i : i + chunk_size]
        _, _, prob_accs = evaluate_fitness_batched(chunk, prompts, targets)
        idx = torch.argmax(prob_accs)
        if prob_accs[idx].item() > best_prob_acc:
            best_prob_acc = prob_accs[idx].item()
            best_params = chunk[idx].clone()

    return best_params, best_prob_acc

def perturb_search(
    current_params: torch.Tensor,
    param_ranges: list,
    n_candidates: int,
    prompts: torch.Tensor,
    targets: torch.Tensor,
    dims_to_perturb: list = None,
    max_bc: int = 100_000,
) -> Tuple[torch.Tensor, float]:
    """
    Try progressively larger perturbations around current_params.
    Only perturb specified dims (all if None).
    Start small (1% of range) and grow. Stop as soon as improvement found.
    """
    n_dims = len(param_ranges)
    if dims_to_perturb is None:
        dims_to_perturb = list(range(n_dims))

    ranges_t = torch.tensor(param_ranges, dtype=torch.float32, device=DEVICE)
    B = prompts.shape[0]
    chunk_size = max(1, max_bc // B)

    # Mask: only perturb selected dims
    dim_mask = torch.zeros(n_dims, device=DEVICE)
    dim_mask[dims_to_perturb] = 1.0

    best_prob_acc = -1.0
    best_params = current_params.clone()

    for scale in [0.01, 0.03, 0.1, 0.3]:
        widths = (ranges_t[:, 1] - ranges_t[:, 0]) * scale
        noise = torch.randn(n_candidates, n_dims, device=DEVICE) * widths.unsqueeze(0) * dim_mask.unsqueeze(0)
        candidates = current_params.unsqueeze(0) + noise
        candidates = torch.clamp(candidates, ranges_t[:, 0], ranges_t[:, 1])

        for i in range(0, n_candidates, chunk_size):
            chunk = candidates[i : i + chunk_size]
            _, _, prob_accs = evaluate_fitness_batched(chunk, prompts, targets)
            idx = torch.argmax(prob_accs)
            if prob_accs[idx].item() > best_prob_acc:
                best_prob_acc = prob_accs[idx].item()
                best_params = chunk[idx].clone()

        if best_prob_acc > 0:
            break

    return best_params, best_prob_acc

# =============================================================================
# Coordinate Descent (Batched)
# =============================================================================

def coordinate_descent_gpu(
    n_starts: int,
    n_cycles: int,
    n_points: int,
    n_train: int,
    seed: int,
    verbose: bool = True
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parameter ranges
    param_ranges = [
        (50, 5000),          # embed_w0
        (1e-5, 0.1),         # embed_w1
        (-100000, -100),     # v_proj_w
        (-100000, -100),     # gate_w0
        (1000, 500000),      # gate_w1
        (0.01, 10.0),        # carry_w
    ]
    
    def smart_random():
        return torch.tensor([
            np.random.uniform(100, 2000),
            np.random.uniform(1e-4, 0.01),
            np.random.uniform(-30000, -1000),
            np.random.uniform(-20000, -1000),
            np.random.uniform(10000, 200000),
            np.random.uniform(0.05, 2.0),
        ], dtype=torch.float32, device=DEVICE)

    best_overall_params = None
    best_overall_acc = -1.0
    
    # Pre-generate validation set (large enough to avoid overfitting illusion)
    val_raw = []
    rng_val = np.random.default_rng(2025)
    for _ in range(500):
        val_raw.append((int(rng_val.integers(0, MAX_ADDEND+1)), int(rng_val.integers(0, MAX_ADDEND+1))))
    val_prompts = encode_problems(val_raw)
    val_targets = get_problem_targets(val_raw)

    for start_idx in range(n_starts):
        # Training problems for this start
        train_raw = []
        rng_train = np.random.default_rng(seed + start_idx)
        for _ in range(n_train):
            train_raw.append((int(rng_train.integers(0, MAX_ADDEND+1)), int(rng_train.integers(0, MAX_ADDEND+1))))
        train_prompts = encode_problems(train_raw)
        train_targets = get_problem_targets(train_raw)

        # Phase 1: Random search for a good init (5000 candidates)
        if verbose:
            print(f"\nStart {start_idx+1}/{n_starts} | Random search (5000 candidates)...")
        params, init_prob_acc = random_restart_search(
            param_ranges, smart_random, 5000,
            train_prompts, train_targets
        )
        if verbose:
            print(f"  Best random init ProbAcc: {init_prob_acc:.2%}")
        
        # Phase 2: Coordinate descent to refine
        ranges = [list(r) for r in param_ranges]
        shrink_factor = 0.6
        best_train_prob_acc = init_prob_acc
        stall_count = 0

        for cycle in range(n_cycles):
            dim_diag = []  # per-dim landscape diagnostics
            for dim in range(N_PARAMS):
                lo, hi = ranges[dim]
                center = params[dim].item()
                width = hi - lo
                lo_new = max(param_ranges[dim][0], center - width / 2)
                hi_new = min(param_ranges[dim][1], center + width / 2)
                
                grid = torch.linspace(lo_new, hi_new, n_points, device=DEVICE)
                candidates = params.repeat(n_points, 1)
                candidates[:, dim] = grid
                
                accs, _, _ = evaluate_fitness_batched(candidates, train_prompts, train_targets)
                best_idx = torch.argmax(accs)

                # Landscape diagnostics (zero extra compute)
                dim_diag.append({
                    'edge': best_idx.item() == 0 or best_idx.item() == n_points - 1,
                    'sens': accs.std().item(),
                    'gain': (accs[best_idx] - accs[n_points // 2]).item(),
                })

                params = candidates[best_idx].clone()

            # Evaluate prob_acc for stall/convergence decisions
            _, _, train_prob = evaluate_fitness_batched(params.unsqueeze(0), train_prompts, train_targets)
            train_prob_acc = train_prob.item()

            if verbose:
                v_acc, _, v_prob = evaluate_fitness_batched(params.unsqueeze(0), val_prompts, val_targets)
                print(
                    f"  Cycle {cycle+1:3d} | TrainProbAcc: {train_prob_acc:.2%} | "
                    f"ValAcc: {v_acc.item():.4f} | ValProbAcc: {v_prob.item():.2%}"
                )

            # Early stop if converged
            if train_prob_acc >= 0.99:
                if verbose:
                    print(f"  Converged at ProbAcc {train_prob_acc:.2%}")
                break

            # Stall detection
            if train_prob_acc > best_train_prob_acc + 1e-6:
                best_train_prob_acc = train_prob_acc
                stall_count = 0
            else:
                stall_count += 1

            if stall_count >= 3:
                if best_train_prob_acc >= 0.99:
                    break

                # --- Landscape-aware escape ---
                edge_dims = [d for d in range(N_PARAMS) if dim_diag[d]['edge']]
                active_dims = [d for d in range(N_PARAMS) if dim_diag[d]['sens'] > 1e-4]

                if edge_dims:
                    # Strategy 1: some dims hitting range boundaries — widen just those
                    for d in edge_dims:
                        c = params[d].item()
                        cur_w = ranges[d][1] - ranges[d][0]
                        new_w = min(param_ranges[d][1] - param_ranges[d][0], cur_w * 3.0)
                        ranges[d] = [
                            max(param_ranges[d][0], c - new_w / 2),
                            min(param_ranges[d][1], c + new_w / 2)
                        ]
                    if verbose:
                        names = [PARAM_NAMES[d] for d in edge_dims]
                        print(f"  >> Edge-bound: {', '.join(names)} — widening")
                    stall_count = 0
                    continue

                elif active_dims and best_train_prob_acc >= 0.5:
                    # Strategy 2: near-ish convergence, some dims still sensitive
                    # Targeted perturb of only the active dims
                    if verbose:
                        names = [PARAM_NAMES[d] for d in active_dims]
                        sens_str = ', '.join(f"{PARAM_NAMES[d]}={dim_diag[d]['sens']:.4f}" for d in active_dims)
                        print(f"  >> Stuck at {best_train_prob_acc:.2%}, perturbing active: {sens_str}")
                    new_params, new_prob_acc = perturb_search(
                        params, param_ranges, 5000,
                        train_prompts, train_targets,
                        dims_to_perturb=active_dims
                    )
                    if new_prob_acc > best_train_prob_acc:
                        params = new_params
                        best_train_prob_acc = new_prob_acc
                        for d in range(N_PARAMS):
                            c = params[d].item()
                            orig_w = param_ranges[d][1] - param_ranges[d][0]
                            ranges[d] = [
                                max(param_ranges[d][0], c - orig_w * 0.3),
                                min(param_ranges[d][1], c + orig_w * 0.3)
                            ]
                        if verbose:
                            print(f"  >> Perturb found ProbAcc {new_prob_acc:.2%}")
                        stall_count = 0
                        continue
                    elif verbose:
                        print(f"  >> Perturb didn't improve ({new_prob_acc:.2%})")
                    # Fall through to fresh restart

                # Strategy 3: flat landscape or perturb failed — fresh random restart
                if verbose:
                    print(f"  >> Fresh random restart (5000 candidates)...")
                new_params, new_prob_acc = random_restart_search(
                    param_ranges, smart_random, 5000,
                    train_prompts, train_targets
                )
                if new_prob_acc > best_train_prob_acc:
                    params = new_params
                    best_train_prob_acc = new_prob_acc
                    for d in range(N_PARAMS):
                        c = params[d].item()
                        orig_w = param_ranges[d][1] - param_ranges[d][0]
                        ranges[d] = [
                            max(param_ranges[d][0], c - orig_w * 0.3),
                            min(param_ranges[d][1], c + orig_w * 0.3)
                        ]
                    if verbose:
                        print(f"  >> Restart found ProbAcc {new_prob_acc:.2%}")
                elif verbose:
                    print(f"  >> Restart didn't improve ({new_prob_acc:.2%})")
                stall_count = 0
                continue

            # Narrow ranges
            for dim in range(N_PARAMS):
                center = params[dim].item()
                cur_width = ranges[dim][1] - ranges[dim][0]
                new_width = cur_width * shrink_factor
                ranges[dim] = [
                    max(param_ranges[dim][0], center - new_width / 2),
                    min(param_ranges[dim][1], center + new_width / 2)
                ]

        # Final start evaluation
        _, _, final_v_prob = evaluate_fitness_batched(params.unsqueeze(0), val_prompts, val_targets)
        print(f"Start {start_idx+1} Result: ProbAcc {final_v_prob.item():.2%}")
        
        if final_v_prob > best_overall_acc:
            best_overall_acc = final_v_prob.item()
            best_overall_params = params.clone()
            print(f"*** New Overall Best: ProbAcc {best_overall_acc:.2%} ***")

    return best_overall_params, best_overall_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', type=int, default=10)
    parser.add_argument('--cycles', type=int, default=25)
    parser.add_argument('--n-points', type=int, default=80)
    parser.add_argument('--n-train', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args, _ = parser.parse_known_args()

    print(f"Accelerating on {DEVICE}")
    t0 = time.time()
    best_params, best_acc = coordinate_descent_gpu(
        args.starts, args.cycles, args.n_points, args.n_train, args.seed
    )
    elapsed = time.time() - t0
    
    print("\n" + "="*50)
    print("Final Result")
    print("="*50)
    print(f"Best Accuracy: {best_acc:.4%}")
    print("Best Parameters:")
    for name, val in zip(PARAM_NAMES, best_params.tolist()):
        print(f"  {name:12s} = {val:>15.6f}")
    print(f"Total time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
