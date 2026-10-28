"""Standalone utility to search for good run_merge_batch tuning values.

`GPUPatchify2D`/`GPUPatchify3D.run_merge_batch` has three performance-only
tuning knobs (`merge_batch_ratio`, `merge_min_batch`, `merge_max_batch` --
see their own `__init__` docstring entries) that control the per-iteration
candidate-batch size of its level-parallel merge loop. They don't affect
correctness (a separate `max_needed` floor guarantees the loop always
terminates at exactly `fixed_length` regardless of these three), only wall-
clock time -- and their defaults were carried over unchanged from a 2D-only
reference implementation, never re-tuned for this class's own (potentially
much larger, especially in 3D) candidate pools.

This script builds one real `GPUPatchify2D`/`GPUPatchify3D` instance for a
given `img_size`/`batch_size` (2 sizes -> 2D, 3 sizes -> 3D), computes its
level structure and merge costs on random synthetic input *once*, then
times `run_merge_batch` (which only reads the three tuning knobs as plain
instance attributes, so they can be swapped between calls with no need to
reconstruct the instance or recompute the level structure) across a grid of
candidate values, and reports the fastest.

Usage:
    python utils/tune_gpu_adaptive_patching.py --img-size 256,256 --batch-size 8
    python utils/tune_gpu_adaptive_patching.py --img-size 128,256,256 --batch-size 2 --fixed-length 512
    python utils/tune_gpu_adaptive_patching.py --img-size 256,256 --batch-size 16 \\
        --batch-ratios 0.05,0.15,0.3 --min-batches 1024,4096 --max-batches 200000,500000,1000000
"""

import argparse
import itertools
import time

import torch

from UCF_VIT.model.gpu_adaptive_patching import GPUPatchify2D, GPUPatchify3D


def _build_patchifier(cls, img_size, min_size, fixed_length, modulus, **kwargs):
    """Constructs `cls`, nudging `fixed_length` up to the nearest value satisfying its congruence assertion.

    `GPUPatchify2D`/`GPUPatchify3D.__init__` requires `(initial_leaves -
    fixed_length) % modulus == 0` (see either class's own docstring) --
    rather than duplicating that arithmetic here, just tries `fixed_length`,
    `fixed_length + 1`, ... up to `modulus - 1` more, and takes whichever
    first constructs successfully. `initial_leaves` doesn't depend on
    `fixed_length`, so this never changes the level structure being tuned
    against, only which exact leaf count it targets.

    Returns:
        `(instance, fixed_length_used)`.
    """
    last_error = None
    for delta in range(modulus):
        candidate = fixed_length + delta
        try:
            return cls(img_size=img_size, min_size=min_size, fixed_length=candidate, **kwargs), candidate
        except AssertionError as e:
            last_error = e
    raise RuntimeError(
        f"Could not find a valid fixed_length within [{fixed_length}, {fixed_length + modulus - 1}] "
        f"for img_size={img_size}, min_size={min_size}. Last error: {last_error}"
    )


def _default_max_batches(patchifier, ndims, batch_size):
    """A handful of candidate `merge_max_batch` values scaled to this problem's real level-1 candidate pool."""
    level1_pool = batch_size * max(1, patchifier.initial_leaves // (2 ** ndims))
    candidates = {250_000, 500_000, 1_000_000, 2_000_000, level1_pool}
    return sorted(v for v in candidates if v >= 1000)


def _time_run_merge_batch(patchifier, padded_img, merge_costs, level_shapes, trials, warmup, device):
    """Times `trials` calls to `run_merge_batch` against the same precomputed level structure.

    Reuses `merge_costs`/`level_shapes`/`padded_img` across every call (all
    read-only, unaffected by the tuning knobs under test) -- only `run_
    merge_batch`'s own internal `alive`/`leaves_remaining` state is rebuilt
    fresh each call, so repeated timing runs are safe and directly
    comparable.

    Returns:
        List of per-trial wall-clock seconds (`trials` entries, after
        `warmup` untimed calls).
    """
    is_cuda = device.type == "cuda"

    def _run():
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        patchifier.run_merge_batch(merge_costs, level_shapes, padded_img)
        if is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - start

    for _ in range(warmup):
        _run()
    return [_run() for _ in range(trials)]


def tune(img_size, batch_size, fixed_length=None, min_size=2, channels=1, score_fn="variance",
         device=None, trials=5, warmup=2, seed=0, batch_ratios=None, min_batches=None, max_batches=None):
    """Searches `merge_batch_ratio`/`merge_min_batch`/`merge_max_batch` for the fastest `run_merge_batch`.

    Args:
        img_size: `(H, W)` for 2D or `(D, H, W)` for 3D.
        batch_size: Number of images/volumes per batch.
        fixed_length: Target leaf count; `None` uses the class's own
            default (196 for 2D, 344 for 3D), auto-nudged up to the
            nearest congruence-satisfying value either way.
        min_size: Finest block side length -- see either class's own
            `min_size` docstring entry.
        channels: Synthetic input channel count.
        score_fn: `"variance"` (default, no threshold tuning needed) or
            `"canny"`.
        device: `torch.device` or device string; `None` auto-selects CUDA
            if available. CPU timings are directly comparable to each
            other but not representative of real GPU wall-clock behavior.
        trials: Timed repeats per candidate configuration.
        warmup: Untimed repeats before timing starts (lets CUDA kernels/
            caches warm up).
        seed: Random seed for the synthetic input -- fixed across every
            candidate so the same merge decisions are being compared.
        batch_ratios: Candidate `merge_batch_ratio` values; `None` uses
            `[0.05, 0.15, 0.3, 0.5]`.
        min_batches: Candidate `merge_min_batch` values; `None` uses
            `[1024, 4096, 16384]`.
        max_batches: Candidate `merge_max_batch` values; `None` derives a
            handful of values scaled to this problem's own level-1
            candidate pool size (see `_default_max_batches`).

    Returns:
        `(results, cls_name)`: `results` is a list of `(mean_seconds,
        std_seconds, batch_ratio, min_batch, max_batch)` tuples, sorted
        fastest first; `cls_name` is `"GPUPatchify2D"`/`"GPUPatchify3D"`.
    """
    ndims = len(img_size)
    assert ndims in (2, 3), f"img_size must have 2 (H,W) or 3 (D,H,W) entries, got {img_size}"
    cls = GPUPatchify2D if ndims == 2 else GPUPatchify3D
    modulus = 3 if ndims == 2 else 7
    default_fixed_length = 196 if ndims == 2 else 344

    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(f"WARNING: running on {device} -- timings are only meaningful relative to each other, "
              "not representative of real GPU wall-clock behavior.")

    patchifier, fixed_length_used = _build_patchifier(
        cls, tuple(img_size), min_size, fixed_length or default_fixed_length, modulus, score_fn=score_fn,
    )
    patchifier = patchifier.to(device)
    if fixed_length_used != (fixed_length or default_fixed_length):
        print(f"fixed_length nudged from {fixed_length or default_fixed_length} to {fixed_length_used} "
              f"to satisfy the congruence requirement.")
    print(f"{cls.__name__}: img_size={tuple(img_size)}, batch_size={batch_size}, min_size={min_size}, "
          f"fixed_length={fixed_length_used}, initial_leaves={patchifier.initial_leaves}, "
          f"max_level={patchifier.max_level}, device={device}")

    torch.manual_seed(seed)
    img = torch.randn(batch_size, channels, *img_size, device=device)
    padded_img, _ = patchifier._pad_tensor(img)
    merge_costs, level_shapes = patchifier._compute_all_levels_batch(padded_img.float())
    padded_spatial_last = padded_img.permute(0, *range(2, ndims + 2), 1)  # -> [..., C], matches run_merge_batch's own convention

    batch_ratios = batch_ratios or [0.05, 0.15, 0.3, 0.5]
    min_batches = min_batches or [1024, 4096, 16384]
    max_batches = max_batches or _default_max_batches(patchifier, ndims, batch_size)

    results = []
    for ratio, min_b, max_b in itertools.product(batch_ratios, min_batches, max_batches):
        if min_b > max_b:
            continue
        patchifier.merge_batch_ratio = ratio
        patchifier.merge_min_batch = min_b
        patchifier.merge_max_batch = max_b
        times = _time_run_merge_batch(patchifier, padded_spatial_last, merge_costs, level_shapes, trials, warmup, device)
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        results.append((mean, std, ratio, min_b, max_b))

    results.sort(key=lambda r: r[0])
    return results, cls.__name__


def _print_results(results, cls_name, top_n=10):
    print(f"\n{'rank':>4}  {'mean_ms':>10}  {'std_ms':>8}  {'batch_ratio':>11}  {'min_batch':>10}  {'max_batch':>10}")
    for rank, (mean, std, ratio, min_b, max_b) in enumerate(results[:top_n], 1):
        print(f"{rank:>4}  {mean * 1000:>10.3f}  {std * 1000:>8.3f}  {ratio:>11}  {min_b:>10}  {max_b:>10}")

    best = results[0]
    print(f"\nBest: merge_batch_ratio={best[2]}, merge_min_batch={best[3]}, merge_max_batch={best[4]} "
          f"({best[0] * 1000:.3f} ms mean over the timed trials)")
    print(f"Pass these to {cls_name}'s constructor directly, e.g.:\n"
          f"    {cls_name}(img_size=..., merge_batch_ratio={best[2]}, "
          f"merge_min_batch={best[3]}, merge_max_batch={best[4]})")


def _parse_int_list(s):
    return [int(x) for x in s.split(",")] if s else None


def _parse_float_list(s):
    return [float(x) for x in s.split(",")] if s else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-size", required=True, help="Comma-separated: 'H,W' (2D) or 'D,H,W' (3D)")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fixed-length", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--score-fn", choices=["variance", "canny"], default="variance")
    parser.add_argument("--device", default=None, help="e.g. 'cuda', 'cuda:0', 'cpu'; default: cuda if available")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-ratios", type=_parse_float_list, default=None, help="e.g. '0.05,0.15,0.3'")
    parser.add_argument("--min-batches", type=_parse_int_list, default=None, help="e.g. '1024,4096,16384'")
    parser.add_argument("--max-batches", type=_parse_int_list, default=None, help="e.g. '250000,500000,1000000'")
    parser.add_argument("--top-n", type=int, default=10, help="How many ranked results to print")
    args = parser.parse_args()

    img_size = tuple(int(x) for x in args.img_size.split(","))
    results, cls_name = tune(
        img_size=img_size, batch_size=args.batch_size, fixed_length=args.fixed_length, min_size=args.min_size,
        channels=args.channels, score_fn=args.score_fn, device=args.device, trials=args.trials, warmup=args.warmup,
        seed=args.seed, batch_ratios=args.batch_ratios, min_batches=args.min_batches, max_batches=args.max_batches,
    )
    _print_results(results, cls_name, top_n=args.top_n)


if __name__ == "__main__":
    main()
