"""Extract a pinned CUDA 13.0 assembler without changing any Python environment."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


URL='https://files.pythonhosted.org/packages/71/8b/a546c12881fffeba927d810598987df25d74b8b241788c7db8dfc93b0173/nvidia_cuda_nvcc-13.0.88-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'
WHEEL_SHA256='56fe502eb77625a12f25172caa3cdddb4e4c8ba2c8c17dba44b164761b380f03'
PTXAS_SHA256='daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2'


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('/workspace/cuda-13.0-ptxas'));a=p.parse_args()
    a.directory.mkdir(parents=True,exist_ok=True);target=a.directory/'ptxas'
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=PTXAS_SHA256:raise ValueError('Existing assembler does not match; preserve it and choose another directory')
        print('Verified existing pinned assembler:',target);return
    with urllib.request.urlopen(URL,timeout=120) as response:wheel=response.read()
    if hashlib.sha256(wheel).hexdigest()!=WHEEL_SHA256:raise ValueError('Wheel checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:data=archive.read('nvidia/cu13/bin/ptxas')
    if hashlib.sha256(data).hexdigest()!=PTXAS_SHA256:raise ValueError('Assembler checksum mismatch')
    with target.open('xb') as output:output.write(data)
    target.chmod(0o755)
    (a.directory/'provenance.json').write_text(json.dumps({'source':URL,'wheel_sha256':WHEEL_SHA256,'ptxas_sha256':PTXAS_SHA256,'extracted_files':['nvidia/cu13/bin/ptxas']},indent=2)+'\n')
    print('Extracted pinned assembler:',target)


if __name__=='__main__':main()
