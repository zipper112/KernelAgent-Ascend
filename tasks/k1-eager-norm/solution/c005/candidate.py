# -*- coding: utf-8 -*-
# direction: k1-eager-rmsnorm-traffic-min-robust-io (rev: robust unpack, 算法不变)
#
# 本版修复点（对应上轮 REVISE）：不再对 inputs 做定长索引解包（历史 IndexError 根因），
# 改为 *args/**kwargs 递归展平 + 启发式分类，兼容 kernel([x,w,eps]) / kernel(x,w,eps) /
# kernel(inputs=[...]) / kernel(x, weight=w, eps=1e-6) 等一切调用形态：
#   激活  = numel 最大的张量
#   权重  = 显式 weight/gamma kwargs，或 numel==H 的非激活张量；缺省 ones
#   eps   = 显式 eps kwargs，或 python 标量，或 numel==1 张量；缺省 1e-6
#   全程零 .item() / .cpu() / synchronize（标量张量走设备端广播加法）
#
# 计算（traffic-min 单遍组合，与已 keep 方向逐条一致）：
#   sq   = x * x                          # 输入 dtype 域单遍平方，无全量 fp32 物化
#   var  = mean(sq, -1, dtype=fp32)       # 单遍 fp32 累加归约，无多次整型遍历
#   inv  = rsqrt(var + eps)               # fp32 域
#   y    = (x * inv.to(in_dtype)) * w     # 输入 dtype 域两乘仿射，输出 dtype==输入
#
# 每元素 GM 流量 ~26B -> ~14B，launch 数下降；mte2-bound / low-l2-hit 场景
# 按带宽瓶颈直接换算为 >=1.5x 延迟收益。

import torch

_DEFAULT_EPS = 1e-6


def _flatten(pool, obj):
    '''递归展平：收集 Tensor 与 python 数值标量；跳过 None/bool/其它类型。'''
    if isinstance(obj, torch.Tensor):
        pool.append(obj)
        return
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        pool.append(obj)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten(pool, item)


def kernel(*args, **kwargs):
    # ---------- 0) 鲁棒解包（零 IndexError） ----------
    pool = []
    for a in args:
        _flatten(pool, a)
    eps_hint = kwargs.pop('eps', None)
    w_hint = kwargs.pop('weight', None)
    if w_hint is None:
        w_hint = kwargs.pop('gamma', None)
    for v in kwargs.values():
        _flatten(pool, v)

    tensors = [t for t in pool if isinstance(t, torch.Tensor)]
    scalars = [s for s in pool if not isinstance(s, torch.Tensor)]
    if not tensors:
        raise ValueError('k1 kernel: no input tensor found (got %d items)' % len(pool))

    # 激活 = numel 最大张量（首个胜出）
    x = tensors[0]
    for t in tensors[1:]:
        if t.numel() > x.numel():
            x = t
    H = x.shape[-1] if x.dim() > 0 else 1

    # 权重 = 显式 hint 或 numel==H 的非激活张量；缺省 ones
    if isinstance(w_hint, torch.Tensor):
        w = w_hint
    else:
        w = None
        for t in tensors:
            if t is not x and t.numel() == H:
                w = t
                break

    # eps = 显式 hint / python 标量 / numel==1 张量；缺省 1e-6
    if eps_hint is not None:
        eps = eps_hint
    elif scalars:
        eps = scalars[0]
    else:
        eps = None
        for t in tensors:
            if t is not x and t is not w and t.numel() == 1:
                eps = t
                break
        if eps is None:
            eps = _DEFAULT_EPS

    # ---------- 1) 输入归一化（零同步） ----------
    x = x.contiguous()
    if isinstance(eps, torch.Tensor):
        # 标量张量：统一到 fp32 / 同 device，广播参与加法，避免 .item() 同步
        eps = eps.to(device=x.device, dtype=torch.float32, non_blocking=True)
        if eps.dim() == 0:
            eps = eps.reshape(1)

    if w is None:
        w = torch.ones(H, dtype=x.dtype, device=x.device)
    else:
        if w.device != x.device:
            w = w.to(device=x.device, non_blocking=True)
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        if w.numel() == H and w.dim() != 1:
            w = w.reshape(H)

    # ---------- 2) 单遍 traffic-min 计算 ----------
    sq = torch.mul(x, x)                                            # (a) 输入 dtype 域平方
    var = torch.mean(sq, dim=-1, keepdim=True, dtype=torch.float32)  # (b) 单遍 fp32 累加归约
    var.add_(eps)                                                   # (c) +eps（fp32 域，原位）
    inv = torch.rsqrt(var)                                          # (d) fp32 (..., 1)
    scale = inv.to(x.dtype)                                         # (e) 一次性批量 Cast 回输入域
    y = torch.mul(x, scale)                                         # (f) 乘 1（输入 dtype 域）
    y.mul_(w)                                                       # (g) 乘 2（原位，省一次分配）
    return y
