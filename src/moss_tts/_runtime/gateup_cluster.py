"""Experimental exact G32 gate/up projection sharded across SM90 CTAs.

Projection rows are sharded; only the 32 BF16 output values are gathered for
the unchanged output quantizer. Compile with isolated Triton 3.8, then load
the resulting cubins in the selected 3.7 host. No production dispatch changes.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .dp4a_norm_pdl import _round, _dot
from .qkv_cluster_prepare import _mul, _norm_sum_legacy
from .pdl_control import wait, trigger
from .bulk_address import _bulk


@gluon.jit
def _four(x):
    even, odd = gl.split(gl.reshape(x, (32, 32, 2, 2)))
    x0, x2 = gl.split(even)
    x1, x3 = gl.split(odd)
    return x0, x1, x2, x3


@gluon.jit
def _projection(a, b, xs, W, S, base, I: gl.constexpr, F: gl.constexpr):
    ri = base + gl.arange(0, 32, layout=gl.SliceLayout(1, gl.SliceLayout(2, I)))
    gi = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(2, I)))
    c = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(0, I)))
    pos = gi[None, :, None] * 4 + gl.expand_dims(gl.expand_dims(c, 0), 0)
    w = gl.load(gl.cast(W, gl.pointer_type(gl.uint32)) + ri[:, None, None]*512 + pos)
    dot = _dot(w, a[None, :, :], b[None, :, :])
    sums = gl.convert_layout((gl.sum(dot, 2) >> 4).to(gl.float32), F)
    rf = base + gl.arange(0, 32, layout=gl.SliceLayout(1, F))
    gf = gl.arange(0, 128, layout=gl.SliceLayout(0, F))
    scale = gl.load(S + rf[:, None]*128 + gf[None, :]).to(gl.float32)
    d0, d1, d2, d3 = _four(sums)
    s0, s1, s2, s3 = _four(_mul(scale, xs[None, :]))
    partial = gl.fma(d0, s0, _mul(d1, s1))
    partial = gl.fma(d2, s2, partial)
    partial = gl.fma(d3, s3, partial)
    return gl.sum(partial, 1)


@gluon.jit
def _kernel(X, RES, NW, W, S, SUM, Y, OQ, OS, NY, NQ, NS,
            EPS: gl.constexpr, ADD: gl.constexpr, CTAS: gl.constexpr,
            DIV: gl.constexpr, TRIGGER: gl.constexpr, DEBUG: gl.constexpr, DISTRIBUTED: gl.constexpr,
            V: gl.constexpr, I: gl.constexpr, F: gl.constexpr, O: gl.constexpr):
    pid = gl.program_id(0)
    base = pid*32
    if DIV:
        rank = gl.inline_asm_elementwise('mov.u32 $0,%cluster_ctarank;', '=r', [],
                                        dtype=gl.int32, is_pure=False, pack=1)
        pointer = W + (base + rank*(32//CTAS))*2048
        size = pid*0 + (32//CTAS)*2048//DIV
        _bulk(pointer, size, 0)
        _bulk(pointer + 12288*2048, size, 0)
    wait()
    if TRIGGER == 1:
        trigger()
    i = gl.arange(0, 4096, layout=V)
    x = gl.load(X+i).to(gl.float32)
    if ADD:
        x = (x + gl.load(RES+i).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
        gl.store(SUM+i, x, pid == 0)
    inv = gl.rsqrt(_norm_sum_legacy(x)/4096 + EPS)
    normalized = ((x*inv).to(gl.bfloat16).to(gl.float32)*gl.load(NW+i).to(gl.float32)).to(gl.bfloat16)
    grouped = gl.reshape(normalized.to(gl.float32), (128, 32))
    scale = gl.maximum(gl.div_rn(gl.max(gl.abs(grouped), 1), 127.0), 1e-8)
    quant = _round(grouped*gl.div_rn(1.0, scale)[:, None])
    if DEBUG:
        gl.store(NY+i, normalized, pid == 0)
        quant_flat = gl.convert_layout(gl.reshape(quant, (4096,)).to(gl.int8), V)
        gl.store(NQ+i, quant_flat, pid == 0)
        gi = gl.arange(0, 128, layout=scale.type.layout)
        gl.store(NS+gi, scale, pid == 0)
    q4 = gl.reshape(quant, (1024, 4))
    c = gl.arange(0, 4, layout=gl.SliceLayout(0, q4.type.layout))
    words = gl.sum((q4 & 255) << (c[None, :]*8), 1)
    a, b = gl.split(gl.reshape(words, (512, 2)))
    a = gl.convert_layout(gl.reshape(a, (128, 4)), gl.SliceLayout(0, I))
    b = gl.convert_layout(gl.reshape(b, (128, 4)), gl.SliceLayout(0, I))
    scale = gl.convert_layout(scale, gl.SliceLayout(0, F))
    if TRIGGER == 2:
        trigger()
    gate = _projection(a, b, scale, W, S, base, I, F).to(gl.bfloat16).to(gl.float32)
    up = _projection(a, b, scale, W, S, base+12288, I, F).to(gl.bfloat16).to(gl.float32)
    silu = (gate/(1 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    value = (silu*up).to(gl.bfloat16)
    if TRIGGER == 3:
        trigger()
    if DISTRIBUTED:
        rows = base + gl.arange(0, 32, layout=value.type.layout)
        gl.store(Y+rows, value)
        output = value.to(gl.float32)
        output_scale = gl.maximum(gl.div_rn(gl.max(gl.abs(output), 0), 127.0), 1e-8)
        output_quant = _round(output*gl.div_rn(1.0, output_scale)).to(gl.int8)
        gl.store(OQ+rows, output_quant)
        gl.store(OS+pid, output_scale)
    else:
        value = gl.convert_layout(value, O)
        rows = base + gl.arange(0, 32, layout=O)
        gl.store(Y+rows, value)
        output = gl.reshape(value.to(gl.float32), (1, 32))
        output_scale = gl.maximum(gl.div_rn(gl.max(gl.abs(output), 1), 127.0), 1e-8)
        output_quant = _round(output*gl.div_rn(1.0, output_scale)[:, None])
        packed_output = gl.convert_layout(gl.reshape(output_quant, (32,)).to(gl.int8), O)
        gl.store(OQ+rows, packed_output)
        group = pid + gl.arange(0, 1, layout=output_scale.type.layout)
        gl.store(OS+group, output_scale)


def linear(x, residual, norm_weight, eps, w, s, *, ctas=2,
           integer_groups=2, integer_rows=4, divisor=16, trigger_mode=1,
           distributed_quant=False, debug=False, return_kernel=False, compiled=None):
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError('SM90 only')
    if ctas not in (1, 2, 4, 8) or integer_groups not in (1, 2, 4) or integer_rows not in (1, 2, 4):
        raise ValueError('Unsupported cluster layout')
    if divisor not in (0, 8, 16, 32) or trigger_mode not in (0, 1, 2, 3):
        raise ValueError('Unsupported launch schedule')
    tensors = (x, norm_weight, w, s) + (() if residual is None else (residual,))
    if not x.is_cuda or any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError('Contiguous same-device CUDA tensors required')
    if x.numel() != 4096 or x.dtype != torch.bfloat16 or norm_weight.numel() != 4096 or norm_weight.dtype != x.dtype:
        raise ValueError('One BF16 K4096 state required')
    if w.shape != (24576, 2048) or w.dtype != torch.uint8 or s.shape != (24576, 128) or s.dtype != torch.bfloat16:
        raise ValueError('Selected G32 gate/up weights required')
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype):
        raise ValueError('Matching residual required')
    if any(t.data_ptr() % 16 for t in tensors):
        raise ValueError('Aligned inputs required')
    summed = torch.empty_like(x) if residual is not None else x
    y = torch.empty((*x.shape[:-1], 12288), device=x.device, dtype=x.dtype)
    oq = torch.empty(12288, device=x.device, dtype=torch.int8)
    os = torch.empty(384, device=x.device, dtype=torch.float32)
    ny = torch.empty_like(x) if debug else torch.empty(0, device=x.device, dtype=x.dtype)
    nq = torch.empty(4096 if debug else 0, device=x.device, dtype=torch.int8)
    ns = torch.empty(128 if debug else 0, device=x.device, dtype=torch.float32)
    bits = ctas.bit_length()-1
    replicated = [[0] for _ in range(bits)]
    vlayout = gl.BlockedLayout([8], [32], [4], [0], cga_layout=replicated)
    ilayout = gl.BlockedLayout([1, integer_groups, 4], [1, 32, 1],
        [integer_rows, 4//integer_rows, 1], [2, 1, 0], cga_layout=[[1 << i, 0, 0] for i in range(bits)])
    flayout = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0], cga_layout=[[1 << i, 0] for i in range(bits)])
    olayout = gl.BlockedLayout([1], [32], [4], [0], cga_layout=replicated)
    if compiled is None:
        kernel = _kernel[(384,)](x, residual, norm_weight, w, s, summed, y, oq, os, ny, nq, ns,
            eps, residual is not None, ctas, divisor, trigger_mode, debug, distributed_quant,
            vlayout, ilayout, flayout, olayout, num_warps=4, num_ctas=ctas, launch_pdl=True)
    else:
        compiled(dict(X=x, RES=residual, NW=norm_weight, W=w, S=s, SUM=summed,
                      Y=y, OQ=oq, OS=os, NY=ny, NQ=nq, NS=ns), 384, eps)
        kernel = compiled
    result = (summed, y, (oq, os))
    value = (result, (ny, nq, ns)) if debug else result
    return (value, kernel) if return_kernel else value


def configs():
    choices = {}
    for ctas in (1, 2, 4, 8):
        choices[f'c{ctas}'] = dict(ctas=ctas, integer_groups=2, integer_rows=4, divisor=16, trigger_mode=1)
        choices[f'c{ctas}_dist'] = {**choices[f'c{ctas}'], 'distributed_quant': True}
    for ctas in (1, 2, 4):
        for ig, ir in ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4)):
            choices[f'c{ctas}_ig{ig}ir{ir}'] = {**choices[f'c{ctas}'], 'integer_groups': ig, 'integer_rows': ir}
        for div in (0, 8, 32):
            choices[f'c{ctas}_d{div}'] = {**choices[f'c{ctas}'], 'divisor': div}
        for mode in (0, 2, 3):
            choices[f'c{ctas}_t{mode}'] = {**choices[f'c{ctas}'], 'trigger_mode': mode}
    return choices
