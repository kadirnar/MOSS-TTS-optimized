"""CPU-only resident-interval accounting for clustered-QKV profiler traces."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def analyze(path):
    events=[e for e in json.loads(Path(path).read_text())['traceEvents'] if e.get('cat')=='kernel']
    groups=collections.defaultdict(list);edges=[]
    for e in events:
        name=e['name'];grid=e.get('args',{}).get('grid');kind='other'
        if name=='_project':kind='mlp_down'
        elif name=='_norm':kind='mlp_gate_up'
        elif name=='_kernel' and grid==[384,1,1]:kind='fused_qkv_preparation'
        elif name=='_kernel' and grid==[256,1,1]:kind='attention_output'
        elif name=='_reduce_quant' or 'attention_ordered_pdl<' in name:kind='attention'
        groups[kind].append(e['dur'])
        edges.extend(((e['ts'],kind,1),(e['ts']+e['dur'],kind,-1)))
    active=collections.Counter();durations=collections.Counter();previous=min(t for t,_,_ in edges)
    for current,kind,delta in sorted(edges):
        label='+'.join(sorted(k for k,n in active.items() if n)) or 'no_kernel'
        durations[label]+=current-previous;active[kind]+=delta;previous=current
    return {'trace':str(path),'kernel_events':len(events),
        'groups':{n:{'calls':len(v),'sum_us':sum(v),'median_us':statistics.median(v)} for n,v in groups.items()},
        'intervals_us':dict(durations),
        'scope':'Post-timing 34-step request. Kernel resident intervals include dependency waits and overlap; not utilization or an arithmetic lower bound. Fusion changes which stage contains head normalization/RoPE/cache writes.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag or not all(c.isalnum() or c=='_' for c in a.tag):raise ValueError('Safe tag required')
    folder=Path(__file__).resolve().parent/'results';path=folder/f'qkv_cluster_profile_summary_{a.tag}.json'
    assert not path.exists(),'Preserve evidence'
    result=analyze(folder/f'qkv_cluster_profile_{a.tag}_trace.json')
    path.write_text(json.dumps(result,indent=2)+'\n');print(result['groups'])


if __name__=='__main__':main()
