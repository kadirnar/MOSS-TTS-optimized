import torch
from .common import RESULTS,load_models
from .streaming import StreamingTTS

def main():
    model,codec,processor=load_models()
    fixture=torch.load(RESULTS/'fixture.pt',weights_only=True)
    engine=StreamingTTS(model,codec,processor)
    engine.warmup()
    list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=40))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        list(engine.stream(fixture['text'],fixture['reference'],max_new_tokens=34))
    prof.export_chrome_trace(str(RESULTS/'final_trace.json'))
    (RESULTS/'final_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))

if __name__=='__main__':main()
