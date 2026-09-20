"""Materialize small metadata/code files; keep large pinned weight files linked."""
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from .common import TTS_REVISION,CODEC_REVISION,RESULTS


def main():
    roots={}
    for model,revision,name in [
        ('OpenMOSS-Team/MOSS-TTS-v1.5',TTS_REVISION,'moss-tts-v15'),
        ('OpenMOSS-Team/MOSS-Audio-Tokenizer',CODEC_REVISION,'moss-codec')]:
        source=Path(snapshot_download(model,revision=revision,local_files_only=True))
        target=Path('/workspace/models')/name
        target.mkdir(parents=True,exist_ok=True)
        for p in source.rglob('*'):
            if not p.is_file():continue
            dest=target/p.relative_to(source)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.safetensors':
                if not dest.exists():dest.symlink_to(p.resolve())
            else:
                shutil.copyfile(p,dest)
        roots[name]={'model':model,'revision':revision,'source':str(source),'view':str(target)}
    (RESULTS/'checkpoint_views.json').write_text(json.dumps(roots,indent=2)+'\n')
    print(json.dumps(roots,indent=2))


if __name__=='__main__':main()
