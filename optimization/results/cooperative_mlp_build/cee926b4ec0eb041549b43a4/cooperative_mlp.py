"""Benchmark-only cooperative fusion of the selected exact G32 MLP PTX.

The selected up and down arithmetic is copied verbatim into separate PTX
scopes. Only dispatch, dependency release and program-id mapping change.
The grid is launched cooperatively, after checking the driver occupancy
limit. A reusable sense barrier joins the two stages. One instance belongs
to one sequential CUDA stream; this is not installed in serving.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import torch

from .common import RESULTS
from .ptx_resources import Attribute, LaunchConfig, check


NAMES = ('X', 'RES', 'NW', 'UW', 'US', 'SUM', 'Y', 'OQ', 'OS', 'DW', 'DS', 'DY', 'BARRIER')


def extract(ptx, signature, mapping, prefix):
    """Discard debug metadata, retain every original arithmetic instruction."""
    text = re.sub(r'//[^\n]*', '', ptx)
    match = re.search(r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)\s*\.reqntid\s+128\s*\{', text, re.S)
    if not match:
        raise ValueError('Expected one 128-thread selected PTX entry')
    params = re.findall(r'\.param\s+[^,]*?\b(\w+)\s*(?:,|$)', match[2])
    if len(params) != len(signature) + 2:
        raise ValueError(('PTX parameter ABI changed', len(params), signature))
    start = match.end(); depth = 1; end = start
    while depth:
        if text[end] == '{': depth += 1
        elif text[end] == '}': depth -= 1
        end += 1
    body = text[start:end-1]
    body = re.sub(r'^\s*\.loc[^\n]*', '', body, flags=re.M)
    body = re.sub(r'\$L__[\w$]+', lambda m: prefix + m[0], body)
    for source, (name, _) in zip(params, signature):
        target = mapping.get(name)
        if target is None:
            # Debug-only and scratch pointers must be unused in production.
            if re.search(r'\b' + re.escape(source) + r'\b', body):
                raise ValueError('Unexpected used pointer: ' + name)
        else:
            body = re.sub(r'\b' + re.escape(source) + r'\b', 'arg_' + target, body)
    if any(re.search(r'\b' + re.escape(p) + r'\b', body) for p in params):
        raise ValueError('Unmapped PTX parameter')
    if body.count('ret;') != 1:
        raise ValueError('Unexpected early returns')
    body = body.replace('ret;', '')
    if body.count('griddepcontrol.wait;') != 1 or body.count('griddepcontrol.launch_dependents;') != 1:
        raise ValueError('Expected selected PDL schedule')
    body = body.replace('griddepcontrol.wait;', '').replace('griddepcontrol.launch_dependents;', '')
    body = body.replace('%ctaid.x', '%virtual')
    # These two mutable producer/consumer stages need coherent global loads.
    if 'ld.global.nc' in body or 'exit;' in body:
        raise ValueError('Unsupported cache policy or early exit')
    return body


def source(up_ptx, up_signature, down_ptx, down_signature, blocks, early, cluster=1):
    up = extract(up_ptx, up_signature, {
        'X': 'X', 'RES': 'RES', 'NW': 'NW', 'W': 'UW', 'S': 'US',
        'SUM': 'SUM', 'Y': 'Y', 'OQ': 'OQ', 'OS': 'OS'}, 'up_')
    down = extract(down_ptx, down_signature, {
        'Q': 'OQ', 'XS': 'OS', 'W': 'DW', 'S': 'DS', 'Y': 'DY'}, 'down_')
    if cluster > 1:
        # Both copied stages were compiled for one CTA per program. Only
        # the new join uses clusters; preserve the original program IDs.
        up = up.replace('%cluster_ctarank', '0')
        down = down.replace('%cluster_ctarank', '0')
    cluster_sync = 'barrier.cluster.arrive; barrier.cluster.wait;' if cluster > 1 else ''
    params = ',\n'.join('.param .u64 arg_' + n for n in NAMES)
    return f'''// Selected PTX stage composition; cooperative launch required.
.version 9.0
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry cooperative_mlp({params})
.reqntid 128
{{
 .reg .u32 %virtual, %block, %bar_tid, %step, %old, %seen, %flip, %rank;
 .reg .u64 %counter;
 .reg .pred %again, %leader, %blockzero, %pending, %clusterleader;
 mov.u32 %block, %ctaid.x;
 mov.u32 %virtual, %block;
 griddepcontrol.wait;
 {'griddepcontrol.launch_dependents;' if early else ''}
 setp.lt.u32 %again, %virtual, 384;
 @!%again bra UP_DONE;
UP_LOOP:
 {{ {up} }}
 bar.sync 0;
 add.u32 %virtual, %virtual, {blocks};
 setp.lt.u32 %again, %virtual, 384;
 @%again bra UP_LOOP;
UP_DONE:

 // All stage stores happen before each CTA leader's release. The last
 // arrival flips the sense bit. The acquire plus CTA barrier publishes
 // all gate/up quantized values to every down thread. No inter-kernel spin.
 bar.sync 0;
 {cluster_sync}
 mov.u32 %bar_tid, %tid.x;
 setp.eq.u32 %leader, %bar_tid, 0;
 rem.u32 %rank, %block, {cluster};
 setp.eq.u32 %clusterleader, %rank, 0;
 and.pred %leader, %leader, %clusterleader;
 @!%leader bra BARRIER_EXIT;
 ld.param.u64 %counter, [arg_BARRIER];
 setp.eq.u32 %blockzero, %block, 0;
 selp.u32 %step, {0x80000000-(blocks//cluster-1)}, 1, %blockzero;
 atom.add.release.gpu.u32 %old, [%counter], %step;
BARRIER_POLL:
 ld.acquire.gpu.u32 %seen, [%counter];
 xor.b32 %flip, %seen, %old;
 and.b32 %flip, %flip, 2147483648;
 setp.eq.u32 %pending, %flip, 0;
 @%pending bra BARRIER_POLL;
BARRIER_EXIT:
 bar.sync 0;
 {cluster_sync}

 mov.u32 %virtual, %block;
DOWN_LOOP:
 {{ {down} }}
 bar.sync 0;
 add.u32 %virtual, %virtual, {blocks};
 setp.lt.u32 %again, %virtual, 512;
 @%again bra DOWN_LOOP;
 {'' if early else 'griddepcontrol.launch_dependents;'}
 ret;
}}
'''


class Fused:
    def __init__(self, entry, down_kernel, *, blocks=384, early=False, cap=None, cluster=1):
        if blocks not in (128, 256, 384, 512) or cap not in (None, 128, 144, 160, 168):
            raise ValueError('Unsupported cooperative schedule')
        if torch.cuda.get_device_capability() != (9, 0):
            raise ValueError('Qualified SM90 only')
        if cluster not in (1, 2, 4, 8) or blocks % cluster:
            raise ValueError('Portable cluster size and aligned grid required')
        self.device = torch.cuda.current_device()
        self.blocks = blocks; self.cluster = cluster
        self.add = entry['raw']['has_residual']
        self.eps = entry['raw']['eps']
        folder = RESULTS/'gateup_exact_bundle_v1'
        manifest = json.loads((folder/'manifest.json').read_text())
        record = manifest['binaries'][f'c1_ig1ir2_add{int(self.add)}_debug0']
        if record['eps'] != self.eps or record['metadata']['num_ctas'] != 1:
            raise ValueError('Selected up specialization required')
        up_ptx = (folder/(record['prefix'] + '.ptx')).read_text()
        up_cubin = (folder/(record['prefix'] + '.cubin')).read_bytes()
        if hashlib.sha256(up_cubin).hexdigest() != record['cubin_sha256']:
            raise ValueError('Selected producer bundle hash mismatch')
        down_signature = [(n, t) for n, t in down_kernel.src.signature.items() if t.startswith('*')]
        ptx = source(up_ptx, record['pointer_signature'], down_kernel.asm['ptx'], down_signature, blocks, early, cluster)
        if cap is not None:
            ptx = ptx.replace('.reqntid 128', f'.reqntid 128\n.maxnreg {cap}')
        compiler = Path('/workspace/cuda-13.0-ptxas/ptxas')
        compiler_hash = hashlib.sha256(compiler.read_bytes()).hexdigest()
        key = hashlib.sha256((ptx + compiler_hash).encode()).hexdigest()[:24]
        build = RESULTS/'cooperative_mlp_build'/key
        build.mkdir(parents=True, exist_ok=True)
        cubin = build/'kernel.cubin'
        if not cubin.exists():
            (build/'kernel.ptx').write_text(ptx)
            (build/'up.original.ptx').write_text(up_ptx)
            (build/'down.original.ptx').write_text(down_kernel.asm['ptx'])
            (build/'cooperative_mlp.py').write_bytes(Path(__file__).read_bytes())
            run = subprocess.run([str(compiler), '-arch=sm_90a', '-O3', '-v', str(build/'kernel.ptx'), '-o', str(cubin)], capture_output=True, text=True)
            (build/'build.log').write_text(run.stdout + run.stderr)
            run.check_returncode()
            (build/'kernel.sass').write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(cubin)], text=True))
        self.lib = C.CDLL('libcuda.so.1')
        self.lib.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.lib.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
        self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
        self.lib.cuLaunchKernelEx.argtypes = [C.POINTER(LaunchConfig), C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        self.lib.cuDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
        self.lib.cuOccupancyMaxActiveClusters.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.POINTER(LaunchConfig)]
        cooperative = C.c_int()
        check(self.lib.cuDeviceGetAttribute(C.byref(cooperative), 95, self.device), 'cooperative capability')
        if not cooperative.value:
            raise ValueError('Cooperative launch unsupported')
        self.module = C.c_void_p(); self.function = C.c_void_p()
        check(self.lib.cuModuleLoad(C.byref(self.module), str(cubin).encode()), 'cooperative module load')
        check(self.lib.cuModuleGetFunction(C.byref(self.function), self.module, b'cooperative_mlp'), 'cooperative function')
        self.shared = max(record['metadata']['shared'], down_kernel.metadata.shared)
        resources = {}
        for n, code in [('registers', 4), ('local_bytes_per_thread', 3), ('static_shared_bytes', 1)]:
            v = C.c_int(); check(self.lib.cuFuncGetAttribute(C.byref(v), code, self.function), n)
            resources[n] = v.value
        active = C.c_int()
        check(self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(active), self.function, 128, self.shared), 'cooperative residency')
        sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        max_clusters = None
        if cluster > 1:
            cfg, attributes = self.launch_config()
            value = C.c_int()
            check(self.lib.cuOccupancyMaxActiveClusters(C.byref(value), self.function, C.byref(cfg)), 'cooperative cluster capacity')
            max_clusters = value.value
        self.resources = {**resources, 'max_active_blocks_per_sm': active.value,
            'sm_count': sm, 'grid_blocks': blocks, 'cooperative_capacity': active.value*sm,
            'cluster_size': cluster, 'max_active_clusters': max_clusters,
            'dynamic_shared_bytes': self.shared, 'early_release': early, 'register_cap': cap,
            'build': str(build), 'cubin_sha256': hashlib.sha256(cubin.read_bytes()).hexdigest(),
            'up_cubin_sha256': record['cubin_sha256'],
            'down_cubin_sha256': hashlib.sha256(down_kernel.asm['cubin']).hexdigest(),
            'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(), 'compiler_sha256': compiler_hash}
        (build/'resources.json').write_text(json.dumps(self.resources, indent=2)+'\n')
        if blocks > active.value*sm:
            raise ValueError(('Unsafe cooperative grid refused before launch', self.resources))
        if max_clusters is not None and blocks//cluster > max_clusters:
            raise ValueError(('Unsafe cooperative cluster grid refused before launch', self.resources))
        self.counter = torch.zeros(1, device='cuda', dtype=torch.int32)
        self.metadata = SimpleNamespace(shared=self.shared)
        self.n_regs = resources['registers']; self.n_spills = None
        self.asm = {'cubin': cubin.read_bytes()}

    def launch_config(self):
        attributes = (Attribute*3)()
        attributes[0].id = 2; attributes[0].value.serialization = 1
        attributes[1].id = 6; attributes[1].value.serialization = 1
        attributes[2].id = 4
        for i, value in enumerate((self.cluster, 1, 1)):
            C.cast(C.byref(attributes[2].value), C.POINTER(C.c_uint))[i] = value
        cfg = LaunchConfig(self.blocks, 1, 1, 128, 1, 1, self.shared,
            torch.cuda.current_stream(self.device).cuda_stream, attributes, 3 if self.cluster > 1 else 2)
        return cfg, attributes

    def __call__(self, entry, index=0):
        d = entry['raw']; w = entry['weights']
        if d['has_residual'] != self.add or d['eps'] != self.eps:
            raise ValueError('Different residual/norm specialization')
        x = d['x'][index:index+1]
        res = d['residual'][index:index+1] if self.add else x
        summed = torch.empty_like(x) if self.add else x
        y = torch.empty((1, 12288), dtype=x.dtype, device=x.device)
        oq = torch.empty(12288, dtype=torch.int8, device=x.device)
        os = torch.empty(384, dtype=torch.float32, device=x.device)
        dy = torch.empty_like(x)
        tensors = (x, res, d['weight'], *w['up'], summed, y, oq, os, *w['down'], dy, self.counter)
        expected = (torch.bfloat16,)*3 + (torch.uint8, torch.bfloat16) + (torch.bfloat16,)*2 + (torch.int8, torch.float32, torch.uint8, torch.bfloat16, torch.bfloat16, torch.int32)
        if any(t.device != x.device or t.device.index != self.device or not t.is_contiguous() or t.dtype != dt for t, dt in zip(tensors, expected, strict=True)):
            raise ValueError('Invalid cooperative buffers')
        if x.numel() != 4096 or w['up'][0].shape != (24576, 2048) or w['down'][0].shape != (4096, 6144):
            raise ValueError('Selected G32 MLP shape required')
        values = [C.c_uint64(t.data_ptr()) for t in tensors]
        pointers = (C.c_void_p*len(values))(*[C.cast(C.pointer(v), C.c_void_p) for v in values])
        cfg, attributes = self.launch_config()
        check(self.lib.cuLaunchKernelEx(C.byref(cfg), self.function, pointers, None), 'cooperative MLP launch')
        return (summed, y, (oq, os)), dy
