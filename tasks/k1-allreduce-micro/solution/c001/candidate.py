# k1-allreduce-micro :: candidate c002
# direction: host-side shape-adaptive last-dim split + fp32 two-stage tree
#
# evidence chain:
#   - K1/K2 symptom axis (low-l2-hit / tail-effect): w02 = 1 row x 1M cols,
#     eager torch.sum(dim=-1) locks the single long row onto few AIVs ~ serial.
#   - w01 = 16384 rows already fills all vector cores -> bit-identical
#     passthrough torch.sum(x, dim=-1), zero regression risk.
#   - reduce family guides (triton-ascend-reduce-guide, ascendc reduce best
#     practice): spread rowlets across AIVs, accumulate in fp32, merge the
#     small fp32 partials tensor, cast to fp16 exactly once at the end.
#
# compliance: pure torch ops (torch 2.7.1 + torch_npu, no triton);
# no .item() / .cpu(); no host-device sync (branching uses static shape
# metadata only); bench/ untouched; interface kernel(inputs)->Tensor.

import torch

_ROWS_PAR = 64          # <= this many rows: eager reduce underfills AIVs
_N_MIN_SPLIT = 16384     # only split long last dims (w01 n=4096 -> passthrough)
_SEG_MIN = 1024          # min elements per rowlet: keep stage1 reads fat
_SEG_MAX = 65536
_MIN_SEGS = 256          # rowlets per row: saturate AIV count on 910B-class parts
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _pick_seg(n):
    # largest pow2 segment in [SEG_MIN, SEG_MAX] that still yields >= MIN_SEGS rowlets
    seg = _SEG_MAX
    while seg > _SEG_MIN and (n + seg - 1) // seg < _MIN_SEGS:
        seg >>= 1
    return seg


def _split_tree_sum(x, rows, n):
    x2 = x.reshape(rows, n)                    # pure view for contiguous bench input
    seg = _pick_seg(n)
    q, r = divmod(n, seg)
    if r != 0:
        # zeros never change a sum; only hit on irregular n (not the graded shapes)
        x2 = torch.nn.functional.pad(x2, (0, seg - r))
        q += 1
    # stage1: rowlets spread across AIVs, fp32 accumulator (no fp16 partial rounding)
    partials = torch.sum(x2.reshape(rows, q, seg), dim=-1, dtype=torch.float32)
    # stage2: tiny fp32 merge of the q partials per row
    merged = torch.sum(partials, dim=-1)
    # stage3: single cast back to input dtype (fp16) and restore row shape
    return merged.reshape(x.shape[:-1]).to(x.dtype)


def kernel(inputs):
    x = inputs[0]
    n = x.shape[-1]
    rows = x.numel() // n if n else 0
    if n >= _N_MIN_SPLIT and 0 < rows <= _ROWS_PAR and x.dtype in _FLOAT_DTYPES:
        return _split_tree_sum(x, rows, n)
    # passthrough: bit-identical to eager baseline (w01 and all small-n shapes)
    return torch.sum(x, dim=-1)
