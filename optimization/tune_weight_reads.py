"""HBM GEMV tile trials: rotate eight matrices to avoid a misleading L2-only test."""
import json
import torch
import triton
from .common import RESULTS
from .kernels import _tiled_weight_gemv as tiled_gemv


def measure(fn, weights):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for w in weights:
            fn(w)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            for w in weights:
                fn(w)
    times = []
    for _ in range(9):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); graph.replay(); end.record(); end.synchronize()
        times.append(start.elapsed_time(end)*1000/(3*len(weights)))
    return sorted(times)[4]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    results = {'method': 'CUDA graph, eight distinct weight matrices per ring, 24 kernels per replay, median of nine replays; synthetic values, actual backbone shapes', 'cases': {}}
    for dtype in (torch.bfloat16, torch.int8, torch.float8_e4m3fn):
        for n, k in ((24576,4096), (4096,12288), (6144,4096), (4096,4096)):
            weights = [(torch.randn(n,k,device='cuda')*20).to(dtype) for _ in range(8)]
            x = torch.randn(k,device='cuda',dtype=torch.bfloat16)
            y = torch.empty(n,device='cuda',dtype=torch.bfloat16)
            scale = torch.rand(n,device='cuda')*0.003
            quantized = dtype != torch.bfloat16
            ref = torch.nn.functional.linear(x.float(),weights[0].float())
            if quantized:
                ref *= scale
            ref = ref.bfloat16()
            case = {}
            for rows in (1,2,4):
                for warps in (2,4,8):
                    def fn(w):
                        tiled_gemv[(triton.cdiv(n,rows),)](x,w,scale,y,n,k,rows,triton.next_power_of_2(k),quantized,num_warps=warps)
                    fn(weights[0])
                    err = ((y.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt()).item()
                    assert err < 0.001, (dtype,n,k,rows,warps,err)
                    case[f'r{rows}_w{warps}'] = {'us':measure(fn, weights),'relative_rms_vs_fp32_reference':err}
            key = f'{str(dtype).split(".")[-1]}_{n}x{k}'
            results['cases'][key] = case
            best = min(case, key=lambda c:case[c]['us'])
            print(key, best, case[best], flush=True)
            (RESULTS/'weight_read_tiles.json').write_text(json.dumps(results,indent=2)+'\n')
            del weights, x, y, ref


if __name__ == '__main__':
    main()
