"""Incrementally de-delay complete 32-codebook frames for vLLM-Omni."""
import torch


def extract_complete_frames(snapshot, consumed):
    if snapshot.ndim != 2 or snapshot.shape[1] != 32:
        raise ValueError('Expected a full snapshot of 32 delayed codebooks')
    complete = max(0,snapshot.shape[0]-31)
    if complete < consumed:
        raise ValueError('The accumulated delay snapshot shrank')
    if complete == consumed:
        return snapshot.new_empty((0,32)), complete
    channels = torch.arange(32,device=snapshot.device)
    rows = torch.arange(consumed,complete,device=snapshot.device)
    frames = snapshot[rows[:,None]+channels,channels]
    # Leading text rows and the drained tail can be partially padded. Only a
    # fully populated frame can be sent to the 32-codebook decoder.
    valid = ((frames >= 0)&(frames < 1024)).all(-1)
    return frames[valid], complete


def talker2codec_delay_async_chunk(transfer_manager, multimodal_output, request, is_finished=False):
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
    req_id = str(getattr(request,'request_id',id(request)))
    if not hasattr(transfer_manager,'_moss_optimized_delay_state'):
        transfer_manager._moss_optimized_delay_state = {}
    state = transfer_manager._moss_optimized_delay_state
    consumed = state.get(req_id,0)
    snapshot = ((multimodal_output or {}).get('codes',{}) or {}).get('audio')
    frames = None
    if isinstance(snapshot,torch.Tensor) and snapshot.numel():
        frames,consumed = extract_complete_frames(snapshot,consumed)
    state[req_id] = consumed
    if is_finished:
        state.pop(req_id,None)
    if frames is None or not frames.numel():
        if not is_finished:
            return None
        flat = []
    else:
        flat = frames.transpose(0,1).contiguous().reshape(-1).cpu().tolist()
    return OmniPayloadStruct(codes=CodesStruct(audio=flat),meta=MetaStruct(
        req_id=[req_id],left_context_size=0,
        codec_chunk_frames=len(flat)//32,codec_left_context_frames=0,
        stream_finished=torch.tensor(bool(is_finished),dtype=torch.bool),
        finished=torch.tensor(bool(is_finished),dtype=torch.bool)),request_id=req_id)
