"""Save the actual SM90 kernels for the exact short-scale projection preset.

This is a static machine-code audit, not a bandwidth or latency measurement.
"""
import argparse
import hashlib
import json
import re
import subprocess

import torch
import triton

from .common import RESULTS
from .benchmark_group128 import quantize
from .dp4a_packing import pack_interleaved
from .dp4a_scaled import _gemv, SELECTED
from .dp4a_layout import _kernel
from .short_scales import PLAN


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe nonempty tag required')
    folder = RESULTS / ('short_scales_codegen_' + args.tag)
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    rows = []
    for name, k in (('out', 4096), ('down', 12288)):
        saved = torch.load(RESULTS / f'gptq_v1_g32_d10/17_{name}.pt',
                           map_location='cuda', weights_only=True)
        w = pack_interleaved(saved['packed'])
        sf = saved['scales'].float()
        sb = saved['scales'].bfloat16()
        assert torch.equal(sf, sb.float())
        states = torch.load(RESULTS / f'calibration_v1/17_{name}.pt', weights_only=True)
        x = states[231:232].cuda()
        q, sx = quantize(x, 32)
        n = w.shape[0]
        outputs = []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for variant in ('fp32', 'short'):
                y = torch.empty((1, n), device='cuda', dtype=torch.bfloat16)
                oq = torch.empty(0, device='cuda', dtype=torch.int8)
                os = torch.empty(0, device='cuda', dtype=torch.float32)
                if variant == 'fp32':
                    p = SELECTED[name]
                    kernel = _gemv[(triton.cdiv(n, p['rows']),)](
                        q, sx, w, sf, y, oq, os, n, k, triton.next_power_of_2(k // 32),
                        p['rows'], p['mode'], False, False, 0, num_warps=p['warps'])
                else:
                    p = PLAN[name]
                    kernel = _kernel[(triton.cdiv(n, p['rows']),)](
                        q, sx, w, sb, y, oq, os, n, k, triton.next_power_of_2(k // 32),
                        p['rows'], False, p['warps'], p['integer_groups'],
                        p['integer_rows'], num_warps=p['warps'])
                outputs.append(y)
                stem = folder / (name + '_' + variant)
                for key in ('ptx', 'ttgir'):
                    stem.with_suffix('.' + key).write_text(kernel.asm[key])
                cubin = kernel.asm['cubin']
                stem.with_suffix('.cubin').write_bytes(cubin)
                sass = subprocess.check_output([
                    '/usr/local/cuda/bin/cuobjdump', '--dump-sass',
                    str(stem.with_suffix('.cubin'))], text=True)
                stem.with_suffix('.sass').write_text(sass)
                instructions = []
                for line in sass.splitlines():
                    match = re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)', line)
                    if match:
                        instructions.append(match.group(1))
                counts = {op: instructions.count(op) for op in sorted(set(instructions))}
                rows.append({'projection': name, 'variant': variant, 'plan': p,
                             'registers': kernel.n_regs, 'spills': kernel.n_spills,
                             'shared_bytes': kernel.metadata.shared,
                             'cubin_sha256': hashlib.sha256(cubin).hexdigest(),
                             'static_instruction_count': len(instructions),
                             'static_opcodes': counts})
            stream.synchronize()
            assert torch.equal(*outputs), name
        torch.cuda.current_stream().wait_stream(stream)
    result = {'codebooks': 32, 'torch': torch.__version__, 'triton': triton.__version__,
              'gpu': torch.cuda.get_device_name(), 'cases': rows, 'all_outputs_exact': True,
              'scope': 'Two real layer-17 inputs; static code for the selected full projection shapes. '
                       'Counts are not dynamic execution or hardware-counter measurements. '
                       'All-layer numerical and latency validation are separate artifacts.'}
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
