"""Delay draining, leading padding and duplicate snapshot validation."""
import json
from pathlib import Path
import torch
from .vllm_delay_stream import extract_complete_frames

torch.manual_seed(32)
cases=[]
for length in (1,2,31,32,73,170):
    for leading in (0,1,5):
        original = torch.randint(0,1024,(length,32))
        delayed = torch.full((length+31+leading+7,32),1024,dtype=torch.long)
        for q in range(32):
            delayed[leading+q:leading+q+length,q]=original[:,q]
        emitted=[]
        consumed=0
        first=None
        for n in range(1,len(delayed)+1):
            frames,consumed=extract_complete_frames(delayed[:n],consumed)
            if frames.numel():
                emitted.append(frames)
                first=n if first is None else first
            duplicate,again=extract_complete_frames(delayed[:n],consumed)
            assert duplicate.numel()==0 and again==consumed
        assert torch.equal(torch.cat(emitted),original)
        assert first==leading+32
        cases.append({'frames':length,'leading_padding':leading,'exact':True,'first_available_step':first})
result={'codebooks':32,'cases':cases,'exact_all_cases':True}
Path('optimization/results/vllm_delay_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('Validated 18 streaming cases, including full drain and repeated snapshots')
