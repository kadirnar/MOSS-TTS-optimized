"""Measure exact active audio-head prefixes without dropping any codec channel.

The delay schedule masks not-yet-active heads. This isolated experiment skips
their matrix rows, pads logits back to 32 heads, and preserves sampling shape.
It is not a serving option or an end-to-end TTFA measurement.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .common import RESULTS
from .llm import sample_topk


def prefix(h, weight, count):
    if count == 32:
        return F.linear(h, weight).view(32, 1024)
    out = torch.zeros((32, 1024), dtype=h.dtype, device=h.device)
    out[:count].copy_(F.linear(h, weight[:count * 1024]).view(count, 1024))
    return out


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        raise ValueError('Safe tag required')
    target = RESULTS / f'audio_head_prefix_{args.tag}.json'
    if target.exists():
        raise FileExistsError('Preserve previous results')
    torch.set_num_threads(4)
    checkpoint = Path('/workspace/models/moss-tts-v15')
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    heads = []
    for i in range(1, 33):
        key = f'lm_heads.{i}.weight'
        with safe_open(checkpoint / index[key], framework='pt', device='cpu') as f:
            heads.append(f.get_tensor(key)[:1024].clone())
    weight = torch.cat(heads).cuda().contiguous()
    del heads
    inputs = []
    for file in sorted((RESULTS / 'compiler_capture_v1').glob('*add_rmsnorm.pt')):
        item = torch.load(file, weights_only=True, map_location='cuda')
        inputs.append((file.name, item['expected'][1].reshape(1, 4096)))
    torch.manual_seed(9826)
    for i in range(8):
        inputs.append((f'random_{i}', torch.randn(1, 4096, device='cuda', dtype=torch.bfloat16)))
    zero = torch.zeros_like(inputs[0][1])
    spike = zero.clone()
    spike[0, -1] = 100
    inputs.extend([('zero', zero), ('spike', spike)])
    records = []
    for label, h in inputs:
        expected = prefix(h, weight, 32)
        for count in range(1, 33):
            actual = prefix(h, weight, count)
            mismatches = int((expected[:count] != actual[:count]).sum())
            torch.cuda.manual_seed(7721)
            a = sample_topk(expected)
            torch.cuda.manual_seed(7721)
            b = sample_topk(actual)
            records.append({'input': label, 'heads': count, 'active_logit_mismatches': mismatches,
                            'active_token_mismatches': int((a[:count] != b[:count]).sum())})
    print('Exactness', len(records), 'cases;', sum(r['active_logit_mismatches'] for r in records),
          'logit differences;', sum(r['active_token_mismatches'] for r in records), 'token differences', flush=True)
    # Even one-head slices cover 128 MiB across this ring, exceeding H200 L2.
    ring = [weight] + [weight.clone() for _ in range(15)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    timings = []
    graphs = {}
    for count in (32, 1, 2, 4, 8, 16, 24):
        with torch.cuda.stream(stream):
            for i in range(3):
                prefix(inputs[0][1], ring[i], count)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = [prefix(inputs[0][1], ring[i % 16], count) for i in range(32)]
        graphs[count] = (graph, outputs)
    for repeat in range(9):
        order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for count in order:
            graph, _ = graphs[count]
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            with torch.cuda.stream(stream):
                graph.replay()
                start.record()
                graph.replay()
                end.record()
            end.synchronize()
            timings.append({'repeat': repeat, 'heads': count, 'us': start.elapsed_time(end) * 1000 / 32})
    medians = {str(count): statistics.median(r['us'] for r in timings if r['heads'] == count) for count in graphs}
    result = {'method': 'Isolated BF16 cuBLAS prefix projection with zero padding to 32x1024 logits; private-stream graphs, 16-weight ring, reversed timing order, nine repeats.',
              'codebooks': 32, 'torch': torch.__version__, 'checks': records, 'timings': timings,
              'median_us_by_head_count': medians,
              'all_active_logits_exact': all(r['active_logit_mismatches'] == 0 for r in records),
              'all_sampled_active_tokens_exact': all(r['active_token_mismatches'] == 0 for r in records),
              'scope': 'Three captured real final hidden states plus eight random/zero/spike inputs. No full-model integration, quality qualification, or TTFA claim.'}
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'timings')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
