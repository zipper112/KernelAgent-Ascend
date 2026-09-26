# k1-allreduce-micro candidate: cube-path reduce_sum via x @ ones (revise of c003)
# w01 (many rows): single MM  x2 @ ones(K,1)  -> Cube streams x once from HBM,
#   fp32 accumulation, *1.0 exact (replaces the vector-reduce pass)
# w02 (one long row): reshape row to (nchunk, CHUNK) and reduce two-level so
#   rows*nchunk chunk-rows spread the 2**20-element stream over all ~40 AI
#   cores, killing the single-core tail; tiny second-level mm over partials.
# pure torch_npu; no .item()/.cpu(); bench untouched.

import torch

CHUNK = 4096      # chunk length for long-row split (divides 2**20 exactly)
MIN_ROWS = 160    # ~4 rows/core on ~40 cores -> enough M-parallelism for direct MM

_ONES = {}


def _ones(k, dtype, device):
    # ones vector is read-only in mm; cache to avoid re-alloc across bench iters
    key = (k, dtype, device)
    o = _ONES.get(key)
    if o is None:
        o = torch.ones((k, 1), dtype=dtype, device=device)
        _ONES[key] = o
    return o


def _mm_reduce(x2, k, dtype, device):
    # (rows, k) @ (k, 1) -> (rows,); Cube fp32 accumulation, *1.0 is exact
    return torch.mm(x2, _ones(k, dtype, device)).squeeze(-1)


def kernel(inputs):
    x = inputs[0]
    out_shape = x.shape[:-1]
    k = x.shape[-1]
    dtype, device = x.dtype, x.device

    if x.numel() == 0:
        return torch.zeros(out_shape, dtype=dtype, device=device)

    rows = x.numel() // k
    x2 = x.reshape(rows, k)  # view for contiguous bench inputs

    if rows >= MIN_ROWS or k <= CHUNK:
        # enough independent rows (or short K): one cube-path MM, one HBM pass
        return _mm_reduce(x2, k, dtype, device).reshape(out_shape)

    # long-row tail case (e.g. w02 1x1x1048576): split each row into chunk
    # rows so rows*nchunk saturate all cores, then second-level mm reduce
    nchunk = (k + CHUNK - 1) // CHUNK
    pad = nchunk * CHUNK - k
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, pad))  # zero-pad is exact for sums
    x3 = x2.reshape(rows * nchunk, CHUNK)
    part = _mm_reduce(x3, CHUNK, dtype, device).reshape(rows, nchunk)
    return _mm_reduce(part, nchunk, dtype, device).reshape(out_shape)
