"""CPU-only stage and resident-interval accounting for saved profiler traces."""
import argparse
import collections
import json
from pathlib import Path

RESULTS=Path(__file__).resolve().parent/'results'


def analyze(path):
    events=json.loads(Path(path).read_text())['traceEvents']
    kernels=[event for event in events if event.get('cat')=='kernel']
    groups=collections.defaultdict(list)
    for event in kernels:groups[event.get('args',{}).get('correlation')].append(event)
    prefill=[group for group in groups.values() if sum('cudnn_generated_fort_native_sdpa' in e['name'] for e in group)==36]
    if len(prefill)!=1:raise ValueError('Expected one 36-layer prefill graph')
    group=prefill[0];by_name=collections.defaultdict(list)
    for event in group:by_name[event['name']].append(event['dur'])
    edges=[];counts=collections.Counter()
    for event in kernels:
        name=event['name'];kind='other'
        if name in ('_norm','_project','_kernel'):kind='projection'
        elif name in ('_qk_rope_cache','_reduce_quant') or 'attention_ordered_pdl<' in name:kind='llm_attention'
        counts[kind]+=1
        edges.extend(((event['ts'],kind,1),(event['ts']+event['dur'],kind,-1)))
    active=collections.Counter();durations=collections.Counter();previous=min(t for t,_,_ in edges)
    for current,kind,delta in sorted(edges):
        label='+'.join(sorted(k for k,n in active.items() if n)) or 'no_kernel'
        durations[label]+=current-previous;active[kind]+=delta;previous=current
    return {'trace':str(path),'kernel_events':len(kernels),'counts':dict(counts),
        'interval_classes_us':dict(durations),'prefill':{'kernel_events':len(group),
        'span_us':max(e['ts']+e['dur'] for e in group)-min(e['ts'] for e in group),
        'kernels':[{'name':name,'calls':len(values),'sum_us':sum(values)}
            for name,values in sorted(by_name.items(),key=lambda item:-sum(item[1]))]},
        'scope':'Post-timing 34-step request, 33 LLM steps and two codec chunks. Resident intervals include PDL waits, not utilization or pure arithmetic. Separate traces are not paired speedup evidence.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag or not all(c.isalnum() or c=='_' for c in args.tag):raise ValueError('Safe tag required')
    path=RESULTS/f'prefill_qkv_profile_summary_{args.tag}.json'
    assert not path.exists(),'Preserve evidence'
    result={name:analyze(RESULTS/file) for name,file in (
        ('prior_bulk','bulk_prefetch_profile_v2_trace.json'),
        ('prefill','prefill_qkv_profile_'+args.tag+'_trace.json'))}
    path.write_text(json.dumps(result,indent=2)+'\n')
    print({name:{'kernels':row['kernel_events'],'prefill_kernels':row['prefill']['kernel_events'],
        'prefill_span_us':row['prefill']['span_us']} for name,row in result.items()})


if __name__=='__main__':main()
