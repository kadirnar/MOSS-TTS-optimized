# MOSS-TTS Model Card

**MOSS-TTS** is a next-generation, production-grade TTS foundation model focused on **voice cloning**, **ultra-long stable speech generation**, **token-level duration control**, **multilingual & code-switched synthesis**, and **fine-grained Pinyin/phoneme-level pronunciation control**. It is built on a clean autoregressive discrete-token recipe that emphasizes high-quality audio tokenization, large-scale diverse pre-training data, and efficient discrete token modeling.

**MOSS-TTS-v1.5** continues from MOSS-TTS 1.0, keeps the same generation API, expands MOSS-TTS multilingual coverage to 31 languages, improves voice-cloning stability, improves long-reference short-text cloning, follows punctuation-driven pauses more reliably, and supports explicit inline pause markers such as `[pause 3.2s]`.



## 1. Overview

### 1.1 TTS Family Positioning
MOSS-TTS is the **flagship base model** in our open-source **TTS Family**. It is designed as a production-ready synthesis backbone that can serve as the primary high-quality engine for scalable voice applications, and as a strong research baseline for controllable TTS and discrete audio token modeling.

**Design goals**
- **Production readiness**: robust voice cloning with stable, on-brand speaker identity at scale
- **Controllability**: duration and pronunciation controls that integrate into real workflows
- **Long-form stability**: consistent identity and delivery for extended narration
- **Multilingual coverage**: multilingual and code-switched synthesis as first-class capabilities



### 1.2 Key Capabilities

MOSS-TTS delivers state-of-the-art quality while providing the fine-grained controllability and long-form stability required for production-grade voice applications, from zero-shot cloning and hour-long narration to token- and phoneme-level control across multilingual and code-switched speech.

* **State-of-the-art evaluation performance** — top-tier objective and subjective results across standard TTS benchmarks and in-house human preference testing, validating both fidelity and naturalness.
* **Zero-shot Voice Cloning (Voice Clone)** — clone a target speaker’s timbre (and part of speaking style) from short reference audio, without speaker-specific fine-tuning.
* **Ultra-long Speech Generation (up to 1 hour)** — support continuous long-form speech generation for up to one hour in a single run, designed for extended narration and long-session content creation.
* **Token-level Duration Control** — control pacing, rhythm, pauses, and speaking rate at token resolution for precise alignment and expressive delivery.
* **Phoneme-level Pronunciation Control** — supports:

  * pure **Pinyin** input
  * pure **IPA** phoneme input
  * mixed **Chinese / English / Pinyin / IPA** input in any combination
* **Multilingual support** — high-quality multilingual synthesis with robust generalization across languages and accents.
* **Code-switching** — natural mixed-language generation within a single utterance (e.g., Chinese–English), with smooth transitions, consistent speaker identity, and pronunciation-aware rendering on both sides of the switch.



### 1.3 Model Architecture

MOSS-TTS includes **two complementary architectures**, both trained and released to explore different performance/latency tradeoffs and to support downstream research.

**Architecture A: Delay Pattern (MossTTSDelay)**
- Single Transformer backbone with **(n_vq + 1) heads**.
- Uses **delay scheduling** for multi-codebook audio tokens.
- Strong long-context stability, efficient inference, and production-friendly behavior.

**Architecture B: Global Latent + Local Transformer (MossTTSLocal)**
- Backbone produces a **global latent** per time step.
- A lightweight **Local Transformer** emits a token block per step.
- **Streaming-friendly** with simpler alignment (no delay scheduling).

**Why train both?**
- **Exploration of architectural potential** and validation across multiple generation paradigms.
- **Different tradeoffs**: Delay pattern tends to be faster and more stable for long-form synthesis; Local is smaller and excels on objective benchmarks.
- **Open-source value**: two strong baselines for research, ablation, and downstream innovation.

For full details, see:
- **`moss_tts_delay/README.md`**
- **`moss_tts_local/README.md`**



### 1.4 Released Models

| Model | Description |
|---|---|
| **MossTTSDelay-8B v1.5** | **Recommended for production**. Latest MOSS-TTS checkpoint with stronger multilingual synthesis when language tags are provided, more stable voice cloning, and explicit pause control. |
| **MossTTSDelay-8B 1.0** | Original 8B delay-pattern release with strong long-context stability and robust voice cloning quality. |
| **MossTTSLocal-4B v1.5** | **Recommended for evaluation and research**. Upgraded backbone (Qwen3-4B), uses **MOSS-Audio-Tokenizer-v2** for **48 kHz stereo** output, with 12-codebook RVQ. |
| **MossTTSLocal-1.7B** | Original 1.7B local-transformer release (Qwen3-1.7B backbone) with SOTA objective metrics and 24 kHz mono output via MOSS-Audio-Tokenizer (v1), 32-codebook RVQ. |

**Recommended decoding hyperparameters (per model)**

| Model | audio_temperature | audio_top_p | audio_top_k | audio_repetition_penalty |
|---|---:|---:|---:|---:|
| **MossTTSDelay-8B v1.5 / 1.0** | 1.7 | 0.8 | 25 | 1.0 |
| **MossTTSLocal-4B v1.5** | 1.7 | 0.8 | 25 | 1.0 |
| **MossTTSLocal-1.7B** | 1.0 | 0.95 | 50 | 1.1 |

### 1.5 Supported Languages

MOSS-TTS-v1.5 currently supports **31 languages**. It keeps the 20 languages supported by MOSS-TTS 1.0 and extends multilingual continued training to Cantonese, Dutch, Finnish, Hindi, Macedonian, Malay, Romanian, Swahili, Tagalog, Thai, and Vietnamese.

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|---|---|---|---|---|---|---|---|---|
| Chinese | zh | 🇨🇳 | Cantonese | yue | 🇭🇰 | English | en | 🇺🇸 |
| Arabic | ar | 🇸🇦 | Czech | cs | 🇨🇿 | Danish | da | 🇩🇰 |
| Dutch | nl | 🇳🇱 | Finnish | fi | 🇫🇮 | French | fr | 🇫🇷 |
| German | de | 🇩🇪 | Greek | el | 🇬🇷 | Hebrew | he | 🇮🇱 |
| Hindi | hi | 🇮🇳 | Hungarian | hu | 🇭🇺 | Italian | it | 🇮🇹 |
| Japanese | ja | 🇯🇵 | Korean | ko | 🇰🇷 | Macedonian | mk | 🇲🇰 |
| Malay | ms | 🇲🇾 | Persian (Farsi) | fa | 🇮🇷 | Polish | pl | 🇵🇱 |
| Portuguese | pt | 🇵🇹 | Romanian | ro | 🇷🇴 | Russian | ru | 🇷🇺 |
| Spanish | es | 🇪🇸 | Swahili | sw | 🇹🇿 | Swedish | sv | 🇸🇪 |
| Tagalog | tl | 🇵🇭 | Thai | th | 🇹🇭 | Turkish | tr | 🇹🇷 |
| Vietnamese | vi | 🇻🇳 | | | | | | |


## 2. Quick Start

> Tip: For production usage, prioritize **MossTTSDelay-8B v1.5**. The examples below use this model; **MossTTSLocal** (both 4B v1.5 and 1.7B 1.0) supports the same API, and a practical walkthrough is available in [moss_tts_local/README.md](../moss_tts_local/README.md).

> Tip: MOSS-TTS-v1.5 uses the same generation API as the 1.0 `MossTTSDelay-8B` checkpoint. For multilingual inputs, set `language` whenever the language is known.

MOSS-TTS provides a convenient `generate` interface for rapid usage. The examples below cover:
1. Direct generation (Chinese / English / multilingual text with language tags / Pinyin / IPA)
2. Voice cloning
3. Duration control
4. Explicit pause control with `[pause X.Ys]`

```python
from pathlib import Path
import importlib.util
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = "Dear friend,\nHello.\n\nToday, I want to share some important words in a sincere and gentle voice.\nLike a small star, I hope these words will slowly shine in your heart.\n\nFirst, I wish you peace and happiness every day.\n\nWhen you wake in the morning,\nmay there be light outside your window and quiet in your room,\nand may your heart feel light, without hurry or fear.\n\nMay you enjoy your meals, walk with steady steps,\nand have sweet dreams each night.\n\nI hope you always stay curious.\nAsk questions about the world and take an interest in the sky,\nthe stars, flowers, books, and stories.\nWhenever you ask why, may someone listen with care.\n\nI also hope you learn to be gentle:\nwith your friends, with animals, and with yourself.\nIf you make a mistake, do not be too quick to blame yourself.\nEveryone who grows learns better ways along the journey.\n\nMay you have courage in unfamiliar places,\nwhen you raise your hand for the first time,\nand when you face something difficult or frightening.\nMay you quietly tell yourself, \"I can try.\"\n\nIt is all right if you do not succeed at once.\nFailure simply tells you that you are making an effort.\n\nI hope you learn to share happiness.\nTell others about the things that make you smile.\nShared joy becomes brighter and greater.\n\nIf you feel sad one day, remember that sadness is nothing to be ashamed of,\nand crying does not make you weak.\nMay you find a safe place to say what is in your heart,\nthen look up again and see hope.\n\nI also hope you have dreams, whether large or small,\neven if you cannot describe them clearly yet.\nYour dreams will grow with you and become clearer over time.\n\nFinally, here is my most important wish:\nmay the world treat you gently, and may you be a gentle person.\nMay every day be worth remembering and cherishing.\n\nDear friend, remember that you are unique.\nYou are already doing well, and your future can slowly grow brighter.\nI wish you health, courage, and happiness.\nMay you always move forward with a smile.\n"
text_2 = "We stand on the threshold of the AI era.\nArtificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."
text_3 = "nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？"
text_4 = "nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？"
text_5 = "您好，请问您来自哪 zuo4 cheng2 shi4？"
text_6 = "/həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/"
text_7 = "Bonjour, je voudrais essayer une voix française naturelle et stable."
text_8 = "Today I learned a poem called [pause 3.2s] Quiet Night Thoughts."

# Use audio from ./assets/audio to avoid downloading from the cloud.
ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Direct TTS (no reference). Language tags are recommended in v1.5.
    [processor.build_user_message(text=text_1)],
    [processor.build_user_message(text=text_2)],
    # Direct TTS (no reference). For languages other than Chinese and English,
    # set the language tag whenever it is known.
    [processor.build_user_message(text=text_7, language="French")],
    # Pinyin or IPA input
    [processor.build_user_message(text=text_3)],
    [processor.build_user_message(text=text_4)],
    [processor.build_user_message(text=text_5)],
    [processor.build_user_message(text=text_6)],
    # Explicit pause control. Use [pause X.Ys], such as [pause 3.2s].
    [processor.build_user_message(text=text_8)],
    # Voice cloning (with reference)
    [processor.build_user_message(text=text_1, reference=[ref_audio_1])],
    [processor.build_user_message(text=text_2, reference=[ref_audio_2])],
    # Duration control
    [processor.build_user_message(text=text_2, tokens=325)],
    [processor.build_user_message(text=text_2, tokens=600)],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

batch_size = 1

save_dir = Path("inference_root")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)

```

### Continuation + Voice Cloning (Prefix Audio + Text)

MOSS-TTS supports continuation-based cloning: provide a prefix audio clip in the assistant message, and make sure the **prefix transcript** is included in the text. The model continues in the same speaker identity and style.

```python
from pathlib import Path
import importlib.util
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = "Dear friend,\nHello.\n\nToday, I want to share some important words in a sincere and gentle voice.\nLike a small star, I hope these words will slowly shine in your heart.\n\nFirst, I wish you peace and happiness every day.\n\nWhen you wake in the morning,\nmay there be light outside your window and quiet in your room,\nand may your heart feel light, without hurry or fear.\n\nMay you enjoy your meals, walk with steady steps,\nand have sweet dreams each night.\n\nI hope you always stay curious.\nAsk questions about the world and take an interest in the sky,\nthe stars, flowers, books, and stories.\nWhenever you ask why, may someone listen with care.\n\nI also hope you learn to be gentle:\nwith your friends, with animals, and with yourself.\nIf you make a mistake, do not be too quick to blame yourself.\nEveryone who grows learns better ways along the journey.\n\nMay you have courage in unfamiliar places,\nwhen you raise your hand for the first time,\nand when you face something difficult or frightening.\nMay you quietly tell yourself, \"I can try.\"\n\nIt is all right if you do not succeed at once.\nFailure simply tells you that you are making an effort.\n\nI hope you learn to share happiness.\nTell others about the things that make you smile.\nShared joy becomes brighter and greater.\n\nIf you feel sad one day, remember that sadness is nothing to be ashamed of,\nand crying does not make you weak.\nMay you find a safe place to say what is in your heart,\nthen look up again and see hope.\n\nI also hope you have dreams, whether large or small,\neven if you cannot describe them clearly yet.\nYour dreams will grow with you and become clearer over time.\n\nFinally, here is my most important wish:\nmay the world treat you gently, and may you be a gentle person.\nMay every day be worth remembering and cherishing.\n\nDear friend, remember that you are unique.\nYou are already doing well, and your future can slowly grow brighter.\nI wish you health, courage, and happiness.\nMay you always move forward with a smile.\n"
text_2 = "We stand on the threshold of the AI era.\nArtificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."
ref_text_1 = "太阳系八大行星之一。"
ref_text_2 = "But I really can't complain about not having a normal college experience to you."
# Use audio from ./assets/audio to avoid downloading from the cloud.
ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Continuatoin only
    [
        processor.build_user_message(text=ref_text_1 + text_1),
        processor.build_assistant_message(audio_codes_list=[ref_audio_1])
    ],
    # Continuation with voice cloning
    [
        processor.build_user_message(text=ref_text_2 + text_2, reference=[ref_audio_2]),
        processor.build_assistant_message(audio_codes_list=[ref_audio_2])
    ],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

batch_size = 1

save_dir = Path("inference_root")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="continuation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)

```



### Input Types

**UserMessage**

| Field | Type | Required | Description |
|---|---|---:|---|
| `text` | `str` | Yes | Text to synthesize. MOSS-TTS-v1.5 supports 31 languages. Text can mix raw text with Pinyin or IPA for pronunciation control and can include explicit pause markers such as `[pause 3.2s]`. |
| `language` | `str` | No | Language tag for multilingual synthesis, for example `"French"`. Recommended in v1.5 whenever the language is known. |
| `reference` | `List[str]` | No | Reference audio for voice cloning. For current MOSS-TTS, **one audio** is expected in the list. |
| `tokens` | `int` | No | Expected number of audio tokens. **1s ≈ 12.5 tokens**. |

**AssistantMessage**

| Field | Type | Required | Description |
|---|---|---:|---|
| `audio_codes_list` | `List[str]` | Only for continuation | Prefix audio for continuation-based cloning. Use audio file paths or URLs. |



### Generation Hyperparameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `max_new_tokens` | `int` | — | Controls total generated audio tokens. Use duration rule: **1s ≈ 12.5 tokens**. |
| `audio_temperature` | `float` | 1.7 | Higher values increase variation; lower values stabilize prosody. |
| `audio_top_p` | `float` | 0.8 | Nucleus sampling cutoff. Lower values are more conservative. |
| `audio_top_k` | `int` | 25 | Top-K sampling. Lower values tighten sampling space. |
| `audio_repetition_penalty` | `float` | 1.0 | >1.0 discourages repeating patterns. |

> Note: MOSS-TTS is a pretrained base model and is **sensitive to decoding hyperparameters**. See **Released Models** for recommended defaults.



### Pinyin Input

Use tone-numbered Pinyin such as `ni3 hao3 wo3 men1`. You can convert Chinese text with [pypinyin](https://github.com/mozillazg/python-pinyin), then adjust tones for pronunciation control.

```python
import re
from pypinyin import pinyin, Style

CN_PUNCT = r"，。！？；：、（）“”‘’"


def fix_punctuation_spacing(s: str) -> str:
    s = re.sub(rf"\s+([{CN_PUNCT}])", r"\1", s)
    s = re.sub(rf"([{CN_PUNCT}])\s+", r"\1", s)
    return s


def zh_to_pinyin_tone3(text: str, strict: bool = True) -> str:
    result = pinyin(
        text,
        style=Style.TONE3,
        heteronym=False,
        strict=strict,
        errors="default",
    )

    s = " ".join(item[0] for item in result)
    return fix_punctuation_spacing(s)

text = zh_to_pinyin_tone3("您好，请问您来自哪座城市？")
print(text)

# Expected: nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？
# Try: nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？
```



### IPA Input

Use `/.../` to wrap IPA sequences so they are distinct from normal text. You can use [DeepPhonemizer](https://github.com/spring-media/DeepPhonemizer) to convert English paragraphs or words into IPA sequences.

```python
from dp.phonemizer import Phonemizer

# Download a phonemizer checkpoint from https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
model_path = "<path-to-phonemizer-checkpoint>"
phonemizer = Phonemizer.from_checkpoint(model_path)

english_texts = "Hello, may I ask which city you are from?"
phoneme_outputs = phonemizer(
    english_texts,
    lang="en_us",
    batch_size=8
)
model_input_text = f"/{phoneme_outputs}/"
print(model_input_text)

# Expected: /həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/
```



## 3. Evaluation
MOSS-TTS achieved state-of-the-art results on the open-source zero-shot TTS benchmark Seed-TTS-eval, not only surpassing all open-source models but also rivaling the most powerful closed-source models.

| Model | Params | Open‑source | EN WER (%) ↓ | EN SIM (%) ↑ | ZH CER (%) ↓ | ZH SIM (%) ↑ |
|---|---:|:---:|---:|---:|---:|---:|
| DiTAR | 0.6B | ❌ | 1.69 | 73.5 | 1.02 | 75.3 |
| FishAudio‑S1 | 4B | ❌ | 1.72 | 62.57 | 1.22 | 72.1 |
| CosyVoice3 | 1.5B | ❌ | 2.22 | 72 | 1.12 | 78.1 |
| Seed‑TTS |  | ❌ | 2.25 | 76.2 | 1.12 | 79.6 |
| MiniMax‑Speech |  | ❌ | 1.65 | 69.2 | 0.83 | 78.3 |
|  |  |  |  |  |  |  |
| CosyVoice | 0.3B | ✅ | 4.29 | 60.9 | 3.63 | 72.3 |
| CosyVoice2 | 0.5B | ✅ | 3.09 | 65.9 | 1.38 | 75.7 |
| CosyVoice3 | 0.5B | ✅ | 2.02 | 71.8 | 1.16 | 78 |
| F5‑TTS | 0.3B | ✅ | 2 | 67 | 1.53 | 76 |
| SparkTTS | 0.5B | ✅ | 3.14 | 57.3 | 1.54 | 66 |
| FireRedTTS | 0.5B | ✅ | 3.82 | 46 | 1.51 | 63.5 |
| FireRedTTS‑2 | 1.5B | ✅ | 1.95 | 66.5 | 1.14 | 73.6 |
| Qwen2.5‑Omni | 7B | ✅ | 2.72 | 63.2 | 1.7 | 75.2 |
| FishAudio‑S1‑mini | 0.5B | ✅ | 1.94 | 55 | 1.18 | 68.5 |
| IndexTTS2 | 1.5B | ✅ | 2.23 | 70.6 | 1.03 | 76.5 |
| VibeVoice | 1.5B | ✅ | 3.04 | 68.9 | 1.16 | 74.4 |
| HiggsAudio‑v2 | 3B | ✅ | 2.44 | 67.7 | 1.5 | 74 |
| GLM-TTS | 1.5B | ✅ | 2.23 | 67.2 | 1.03 | 76.1 |
| GLM-TTS-RL | 1.5B | ✅ | 1.91 | 68.1 | **0.89** | 76.4 |
| VoxCPM | 0.5B | ✅ | 1.85 | 72.9 | 0.93 | 77.2 |
| Qwen3‑TTS | 0.6B | ✅ | 1.68 | 70.39 | 1.23 | 76.4 |
| Qwen3‑TTS | 1.7B | ✅ | **1.5** | 71.45 | 1.33 | 76.72 |
|  |  |  |  |  |  |  |
| **MossTTSDelay** | **8B** | ✅ | 1.84 | 70.86 | 1.37 | 76.98 |
| **MossTTSLocal** | **1.7B** | ✅ | 1.93 | **73.28** | 1.44 | **79.62** |

