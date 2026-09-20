import imageio, os, torch, warnings, torchvision, torchaudio, argparse, json, random, math
import time
import numpy as np
from typing import Optional
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import get_wsd_schedule
from torchcodec.decoders import AudioDecoder
from .cache_shards import ShardedBinIdxWriter, ShardedBinIdxReader, DEFAULT_META_FILENAME
from safetensors.torch import load_file as safe_load_file

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        sample_rate=None,  # target sample rate; if None, keep original
        num_samples=None,  # target number of samples; if None, keep original length
        min_num_samples=2048,  # minimum number of samples; pad to this if shorter
        max_num_samples=30*44100,  # maximum number of samples; center-crop if longer
        mono=False,
        data_file_keys=("audio", "audio_latent"),
        audio_file_extension=("wav", "mp3", "flac", "ogg", "m4a", "aac", "wma", "", "mp4", "aiff", "wv"),
        repeat=1,
        drop_prompt_prob=0.1,
        cache_folder=None,
        append_duration_suffix=False,
        append_duration_suffix_prob=0.5,
        duration_precision=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            if metadata_path is None:
                metadata_path = args.dataset_metadata_path
            # Optional arguments; use getattr to avoid hard dependency in parsers
            sample_rate = getattr(args, "sample_rate", sample_rate)
            num_samples = getattr(args, "num_audio_samples", num_samples)
            min_num_samples = getattr(args, "min_num_audio_samples", min_num_samples)
            max_num_samples = getattr(args, "max_num_audio_samples", max_num_samples)
            mono = getattr(args, "mono", mono)
            data_file_keys = getattr(args, "data_file_keys", ",".join(data_file_keys)).split(",")
            repeat = args.dataset_repeat
            drop_prompt_prob = getattr(args, "drop_prompt_prob", drop_prompt_prob)  # 0.1
            cache_folder = getattr(args, "cache_folder", cache_folder)
            append_duration_suffix = getattr(args, "append_duration_suffix", append_duration_suffix)
            append_duration_suffix_prob = getattr(args, "append_duration_suffix_prob", append_duration_suffix_prob)
            duration_precision = getattr(args, "duration_precision", duration_precision)
        
        self.base_path = base_path
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.min_num_samples = min_num_samples
        self.max_num_samples = max_num_samples
        self.mono = mono
        self.data_file_keys = data_file_keys
        self.audio_file_extension = audio_file_extension
        self.repeat = repeat
        self.drop_prompt_prob = drop_prompt_prob
        self.cache_folder = cache_folder
        self.append_duration_suffix = append_duration_suffix
        if append_duration_suffix_prob is None:
            append_duration_suffix_prob = 1.0
        self.append_duration_suffix_prob = float(append_duration_suffix_prob)
        # Clamp to [0, 1] to avoid surprises
        self.append_duration_suffix_prob = max(0.0, min(1.0, self.append_duration_suffix_prob))
        self.duration_precision = int(duration_precision) if duration_precision is not None else 1
        self._sharded_reader = None

        self.max_duration_s = self.max_num_samples / self.sample_rate
        
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = metadata.to_dict("records")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path, dtype="string")
            self.data = metadata.to_dict("records")
        
    
    def generate_metadata(self, folder):
        audio_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.audio_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            audio_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["audio"] = audio_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def is_audio(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.audio_file_extension
    
    
    def _ensure_mono(self, waveform):
        # waveform: Tensor [C, T] or [T]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if self.mono:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        else:
            # If mono=False but input is single-channel, duplicate to stereo
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                # Some files have 6+ channels; keep only the first two.
                waveform = waveform[0:2]
        return waveform
    
    
    def _apply_num_samples(self, waveform):
        if self.num_samples is None:
            return waveform
        target = int(self.num_samples)
        num_current = waveform.shape[-1]
        if num_current == target:
            return waveform
        if num_current > target:
            return waveform[..., :target]
        # pad time dimension to target length
        pad_T = target - num_current
        pad = torch.zeros((waveform.shape[0], pad_T), dtype=waveform.dtype, device=waveform.device)
        return torch.cat([waveform, pad], dim=-1)

    def _ensure_max_num_samples(self, waveform):
        if self.max_num_samples is None:
            return waveform
        max_target = int(self.max_num_samples)
        num_current = waveform.shape[-1]
        if num_current <= max_target:
            return waveform
        start = (num_current - max_target) // 2
        end = start + max_target
        return waveform[..., start:end]

    
    def _ensure_min_num_samples(self, waveform):
        if self.min_num_samples is None:
            return waveform
        min_target = int(self.min_num_samples)
        num_current = waveform.shape[-1]
        if num_current >= min_target:
            return waveform
        pad_T = min_target - num_current
        pad = torch.zeros((waveform.shape[0], pad_T), dtype=waveform.dtype, device=waveform.device)
        return torch.cat([waveform, pad], dim=-1)

    def load_audio(self, file_path, start_time=None, end_time=None):
        if not self.is_audio(file_path):
            raise ValueError(f"File {file_path} is not an audio file.")
        try:
            if not os.path.exists(file_path):
                warnings.warn(f"audio file not found: {file_path}.")
                return None

            # Decode with soundfile first; fall back to ffmpeg -> wav for
            # formats it cannot read (e.g. some mp3). Avoids torchcodec entirely
            # to dodge segfaults seen on certain builds.
            import soundfile as sf
            try:
                data, sr = sf.read(file_path, dtype='float32')
                waveform = torch.from_numpy(data)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.t()
                waveform = waveform.float()
            except Exception:
                # soundfile cannot decode this file (e.g. some mp3 variants);
                # transcode to wav with ffmpeg, then re-read.
                import subprocess
                target_sr = int(self.sample_rate) if self.sample_rate is not None else 48000
                target_ac = 1 if self.mono else 2
                wav_tmp = file_path + ".sf_fallback.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", file_path,
                     "-ar", str(target_sr), "-ac", str(target_ac),
                     "-f", "wav", "-acodec", "pcm_f32le", wav_tmp],
                    capture_output=True, check=True,
                )
                data, sr = sf.read(wav_tmp, dtype='float32')
                os.unlink(wav_tmp)
                waveform = torch.from_numpy(data)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.t()
                waveform = waveform.float()
            duration_seconds = waveform.shape[-1] / sr

            # When start/end times are given and valid, read that segment.
            use_segment = (start_time is not None and end_time is not None)
            if use_segment:
                s = max(0, int(start_time * sr))
                e = min(waveform.shape[-1], int(end_time * sr))
                waveform = waveform[:, s:e]

            elif (self.max_num_samples is not None and self.sample_rate is not None and duration_seconds > self.max_duration_s):
                # Center-crop to max length at decode time for long samples.
                max_dur = float(self.max_duration_s)
                start_c = max(0.0, float(duration_seconds - max_dur) / 2.0)
                # Cap the start offset to avoid pathological decode time on
                # extremely long audio (observed on some freesound files).
                start_c = min(start_c, 60.0)
                end_c = start_c + max_dur
                s = int(start_c * sr)
                e = int(end_c * sr)
                waveform = waveform[:, s:e]

            elif duration_seconds > 60*60:
                warnings.warn(f"Duration of {file_path} is {duration_seconds} seconds, which is longer than 60 minutes.")
                waveform = waveform[:, :int(60*60*sr)]

            if self.sample_rate is not None and sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            waveform = self._ensure_mono(waveform)  # ensure [C, T]
            # Apply max-length center crop first.
            waveform = self._ensure_max_num_samples(waveform)  # center-crop if longer than max
            # Compute true non-pad duration (seconds) before any padding.
            effective_sr = self.sample_rate if self.sample_rate is not None else sr
            num_current = waveform.shape[-1]
            if self.num_samples is None:
                valid_T = num_current
            else:
                target = int(self.num_samples)
                # If it would be cropped, take target; if it would be padded, keep original.
                valid_T = min(num_current, target)
            # Then apply exact-length and min-length pad/crop.
            waveform = self._apply_num_samples(waveform)  # crop/pad on T to exact num_samples if provided
            waveform = self._ensure_min_num_samples(waveform)  # ensure at least min_num_samples
            nonpad_duration_s = float(valid_T) / float(effective_sr)
            return waveform, nonpad_duration_s
        except Exception:
            warnings.warn(f"cannot load audio file {file_path}.")
            return None
    
    
    def load_data(self, file_path, start_time=None, end_time=None):
        return self.load_audio(file_path, start_time=start_time, end_time=end_time)
    
    
    def __getitem__(self, data_id):
        import warnings
        # Prefer the cache, if any.
        if self.cache_folder is not None:
            # New multi-shard bin+idx layout (enabled when the meta file exists).
            meta_path = os.path.join(self.cache_folder, DEFAULT_META_FILENAME)
            if os.path.exists(meta_path):
                if self._sharded_reader is None:
                    self._sharded_reader = ShardedBinIdxReader(self.cache_folder)
                loaded = self._sharded_reader.get(int(data_id))
                if not isinstance(loaded, dict):
                    warnings.warn(f"cache miss for data_id={data_id}, skipping.")
                    return None

                # context has shape [1, T, D]; right-pad T to 512 with zeros.
                context = loaded["context"]
                pad_T = 512 - context.shape[1]
                pad = torch.zeros((1, pad_T, context.shape[2]), dtype=context.dtype, device=context.device)
                context = torch.cat([context, pad], dim=1)
                loaded["context"] = context

                return {"cached": torch.tensor(True), "data_id": data_id, **loaded}
            # Legacy single-file cache layout.
            cache_path_npz = os.path.join(self.cache_folder, f"{data_id}.npz")
            cache_path_pth = os.path.join(self.cache_folder, f"{data_id}.pth")
            if os.path.exists(cache_path_npz) or os.path.exists(cache_path_pth):
                try:
                    if os.path.exists(cache_path_npz):
                        npz_file = np.load(cache_path_npz)
                        loaded = {k: torch.from_numpy(np.array(v)) if hasattr(v, "dtype") else v for k, v in npz_file.items()}
                    else:
                        loaded = torch.load(cache_path_pth, map_location="cpu")
                    if isinstance(loaded, dict):
                        return {"cached": torch.tensor(True), "data_id": data_id, **loaded}
                except Exception:
                    warnings.warn(f"cannot load cache file {cache_path_npz if os.path.exists(cache_path_npz) else cache_path_pth}.")
                    pass
            warnings.warn(f"cannot load cache file {cache_path_npz if os.path.exists(cache_path_npz) else cache_path_pth}.")

        data = self.data[data_id % len(self.data)].copy()
        computed_duration_s = None
        data['data_id'] = data_id

        # Read start_time / end_time from the row (if present). Use to_numeric
        # so non-numeric strings yield NaN instead of raising.
        start_time_s = None
        end_time_s = None
        if "start_time" in data:
            val = pd.to_numeric(data.get("start_time"), errors="coerce")
            if pd.notna(val):
                start_time_s = float(val)
        if "end_time" in data:
            val = pd.to_numeric(data.get("end_time"), errors="coerce")
            if pd.notna(val):
                end_time_s = float(val)

        # If audio_latent is provided, load it from safetensors and build a
        # dummy audio tensor of matching length.
        latent_loaded = False
        latent_T = None
        if "audio_latent" in data and pd.notna(data["audio_latent"]):
            try:
                latent_path = os.path.join(self.base_path, data["audio_latent"])
                latent_obj = safe_load_file(latent_path)
                if "latents" not in latent_obj:
                    raise ValueError("key 'latents' not found in safetensors file")
                latents = latent_obj["latents"]
                if isinstance(latents, np.ndarray):
                    latents = torch.from_numpy(latents)
                latents = latents.float()
                if latents.ndim != 2:
                    raise ValueError(f"audio_latent expected shape [64|128, T], got {tuple(latents.shape)}")
                channels = int(latents.shape[0])
                if channels not in (64, 128):
                    raise ValueError(f"audio_latent expected shape [64|128, T], got {tuple(latents.shape)}")
                data["audio_latent"] = latents
                latent_T = int(latents.shape[1])
                # Dummy audio: 64-ch latent -> [2, T*2048]; 128-ch -> [1, T*960].
                if channels == 64:
                    data["audio"] = torch.zeros((2, latent_T * 2048), dtype=torch.float32)
                else:  # channels == 128
                    data["audio"] = torch.zeros((1, latent_T * 960), dtype=torch.float32)
                latent_loaded = True
            except Exception as e:
                warnings.warn(f"cannot load audio_latent file {data.get('audio_latent')}: {e}.")
                return None
        else:
            if "audio_latent" in data:
                del data["audio_latent"]

        for key in self.data_file_keys:
            if key in data:
                # Skip keys already populated above.
                if key == "audio_latent" and latent_loaded:
                    continue
                if key == "audio" and latent_loaded:
                    # audio already replaced by a dummy tensor from audio_latent.
                    continue
                path = data[key]
                if key == "audio":
                    loaded = self.load_data(path, start_time=start_time_s, end_time=end_time_s)
                    if isinstance(loaded, tuple) and len(loaded) == 2:
                        data[key], computed_duration_s = loaded
                    else:
                        data[key] = loaded
                else:
                    loaded = self.load_data(path)
                    data[key] = loaded[0] if isinstance(loaded, tuple) else loaded
                if data[key] is None:
                    warnings.warn(f"cannot load file for key={key}.")
                    return None
        # Duration is now sourced from load_audio's return value only.
        # Randomly drop prompt with given probability
        if "prompt" in data and isinstance(data["prompt"], str):
            if self.drop_prompt_prob is not None and self.drop_prompt_prob > 0:
                if random.random() < float(self.drop_prompt_prob):
                    data["prompt"] = ""
        else:
            warnings.warn(f"prompt is not a string: {data['prompt']}.")
        # Optionally append a duration suffix to the prompt.
        if self.append_duration_suffix:
            # Probability of appending; default 0.5. 1.0 means always append.
            prob = getattr(self, "append_duration_suffix_prob", 1.0)
            if prob is None:
                prob = 1.0
            prob = max(0.0, min(1.0, float(prob)))
            if computed_duration_s is not None:
                if prob >= 1.0 or (prob > 0 and random.random() < prob):
                    computed_duration_s = min(computed_duration_s, self.max_duration_s)
                    fmt = f"{{:.{max(0, int(self.duration_precision))}f}}"
                    duration_text = fmt.format(computed_duration_s)
                    suffix = f" duration: {duration_text}s"
                    if "prompt" in data and isinstance(data["prompt"], str):
                        data["prompt"] = data["prompt"] + suffix
            elif not latent_loaded and prob > 0:
                warnings.warn("duration info unavailable; skip duration suffix.")
        return data
    
    
    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None, loss=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

    def save_training_state(self, accelerator, epoch_id, global_step, micro_step):
        """Save the full training state to output_path/training_state/ for resume."""
        accelerator.wait_for_everyone()
        state_dir = os.path.join(self.output_path, "training_state")
        # Save the full accelerator state (model, optimizer, scheduler, RNG).
        accelerator.save_state(state_dir)
        # Then save extra metadata (epoch, step, num_steps).
        if accelerator.is_main_process:
            metadata = {
                "epoch_id": epoch_id,
                # optimizer step; only incremented when accelerator.sync_gradients is True.
                "global_step": global_step,
                "micro_step": micro_step,
                # micro step; incremented on every on_step_end call.
                # global_step = num_steps / gradient_accumulation_steps
                "num_steps": self.num_steps,
            }
            metadata_path = os.path.join(state_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

    @staticmethod
    def load_training_metadata(state_dir):
        metadata_path = os.path.join(state_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            return json.load(f)

def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    batch_size: int = 1,
    clip_grad_norm: float = 1.0,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    log_dir: Optional[str] = None,
    prefetch_factor: int = 2,
    resume_from=None,     # Resume directory (points to a previous output_path).
):
    def collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        in_order=True, collate_fn=collate_skip_none
    )
    # Enable logging with Accelerator if log_dir is provided
    log_with = None
    if log_dir is not None:
        log_with = "tensorboard"
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        log_with=log_with,
        project_dir=log_dir,
    )
    from accelerate.utils import set_seed
    set_seed(42, device_specific=True)
    # Build the scheduler before accelerator.prepare so world_size is known.
    if scheduler is None:
        steps_per_epoch = math.ceil(len(dataset) / max(1, batch_size))
        optim_steps_per_epoch = math.ceil(steps_per_epoch / max(1, gradient_accumulation_steps))
        total_steps = max(1, optim_steps_per_epoch * max(1, num_epochs))
        warmup_steps = min(100, max(0, total_steps - 1))
        decay_steps = max(1, int(round(total_steps * 0.10)))
        stable_steps = max(0, total_steps - warmup_steps - decay_steps)
        scheduler = get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_stable_steps=stable_steps,
            num_decay_steps=decay_steps,
        )
    # accelerator.prepare wraps the model in DDP (for multi-GPU), shards the
    # dataloader across ranks, and handles mixed-precision casting.
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    
    global_step = 0
    micro_step = 0
    # --- Resume Logic ---
    start_epoch = 0
    if resume_from is not None:
        state_dir = os.path.join(resume_from, "training_state")
        accelerator.load_state(state_dir)
        metadata = ModelLogger.load_training_metadata(state_dir)
        # Resume from the next epoch after the one we last saved (save runs at epoch end).
        start_epoch = metadata["epoch_id"] + 1
        global_step = metadata["global_step"]
        micro_step = metadata["micro_step"]
        model_logger.num_steps = metadata["num_steps"]
        if accelerator.is_main_process:
            print(f"Resumed from epoch {metadata['epoch_id']}, global_step {global_step}, "
                f"micro_step {micro_step}, continuing from epoch {start_epoch}")
    
    if accelerator.log_with is not None:
        accelerator.init_trackers(
            "training",
            config={
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
        )
    
    prev_iter_end_time = time.time()
    prev_log_time = time.time()
    for epoch_id in range(start_epoch, num_epochs):
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            if data is None:
                prev_iter_end_time = time.time()
                continue
            # Measure data loading time
            # accelerator.wait_for_everyone()
            data_loading_end = time.time()
            data_loading_time = data_loading_end - prev_iter_end_time
            with accelerator.accumulate(model):
                compute_start_time = time.time()
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss)
                scheduler.step()
                compute_end_time = time.time()
                compute_time = compute_end_time - compute_start_time
                # Log time metrics for every micro step
                if accelerator.log_with is not None:
                    micro_step += 1
                    cur_log_time = time.time()
                    accelerator.log(
                        {
                            "time/data_loading": float(data_loading_time),
                            "time/compute": float(compute_time),
                            "time/step": float(cur_log_time - prev_log_time),
                            "train/epoch": epoch_id,
                        },
                        step=micro_step,
                    )
                    prev_log_time = cur_log_time
                # Log loss/lr only when optimizer actually steps
                if accelerator.log_with is not None and accelerator.sync_gradients:
                    global_step += 1
                    # Reduce loss across processes for a global mean
                    loss_to_log = loss.detach()
                    reduced_loss = accelerator.reduce(loss_to_log, reduction="mean")
                    accelerator.log(
                        {
                            "train/loss": float(reduced_loss.float().item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/grad_norm": float(grad_norm),
                        },
                        step=global_step,
                    )
                # if micro_step == 1000:
                #     torch.cuda.cudart().cudaProfilerStart()
                #     torch.autograd.profiler.emit_nvtx(record_shapes=False).__enter__()
                # elif micro_step == 1020:
                #     torch.cuda.cudart().cudaProfilerStop()
            prev_iter_end_time = time.time()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        model_logger.save_training_state(accelerator, epoch_id, global_step, micro_step)
    model_logger.on_training_end(accelerator, model, save_steps)
    accelerator.end_training()


def launch_data_process_task(model: DiffusionTrainingModule, dataset, cache_folder, num_shards: int = 16, skip_first_batches: int = 0, num_workers: int = 0, prefetch_factor: int = 4):
    accelerator = Accelerator()
    def collate_first_valid(batch):
        for b in batch:
            if b is not None:
                return b
        return {"__skip__": torch.tensor(True)}
    dataloader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sampler,
        shuffle=False,
        # collate_fn=lambda x: x[0],
        collate_fn=collate_first_valid,
        # num_workers=8,
        num_workers=num_workers,   
        pin_memory=True,
        # prefetch_factor=4,
        prefetch_factor=prefetch_factor if num_workers > 0 else None, 
    )
    model, dataloader = accelerator.prepare(model, dataloader)
    if skip_first_batches and skip_first_batches > 0:
        dataloader = accelerator.skip_first_batches(dataloader, int(skip_first_batches))
    os.makedirs(cache_folder, exist_ok=True)
    writer = ShardedBinIdxWriter(cache_folder, num_shards=num_shards)
    prev_step_end_time = time.time()
    for i, data in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
        if isinstance(data, dict) and data.get("__skip__", None) is not None:
            accelerator.wait_for_everyone()
            prev_step_end_time = time.time()
            continue
        step_start_time = time.time()
        data_id = data['data_id']
        with torch.no_grad():
            unwrapped_model = model.module if hasattr(model, "module") else model
            preprocess_start_time = time.time()
            # forward_preprocess encodes audio with VAE and prompt with the
            # text encoder, yielding input_latents and context.
            inputs = unwrapped_model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in unwrapped_model.model_input_keys if key in inputs}
            context = inputs["context"]
            # context has shape [1, T, D]; the tail is padded with zeros from
            # some position onward. Trim those trailing all-zero rows.
            # [1, T]
            zero_mask = (context == 0).all(dim=-1)

            if zero_mask.any():
                # Index of the first all-zero row.
                t_end = int(torch.where(zero_mask[0])[0][0])
            else:
                t_end = context.size(1)

            context = context[:, :t_end]
            inputs["context"] = context
            preprocess_end_time = time.time()
            try:
                # Write {input_latents, context} for this sample to disk.
                writer.write_sample(int(data_id), inputs, compress=False)
            except ValueError as e:
                print(f"Skipping sample {data_id} due to error: {e}")
            write_end_time = time.time()
            # print(f"write time: {write_end_time - preprocess_end_time}")
        accelerator.wait_for_everyone()
        prev_step_end_time = time.time()
