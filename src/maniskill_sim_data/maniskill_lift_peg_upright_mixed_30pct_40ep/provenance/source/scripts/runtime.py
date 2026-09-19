"""Project-local runtime configuration and a bounded NumPy socket protocol."""
import io
import json
import os
from pathlib import Path
import socket
import struct

ROOT = Path(__file__).resolve().parents[1]

def configure():
    paths = {'HF_HOME':'.cache/huggingface','HF_HUB_CACHE':'.cache/huggingface/hub',
             'XDG_CACHE_HOME':'.cache/xdg','XDG_CONFIG_HOME':'.cache/config',
             'XDG_DATA_HOME':'.cache/data','TORCH_HOME':'.cache/torch',
             'TORCH_EXTENSIONS_DIR':'.cache/torch_extensions','CUDA_CACHE_PATH':'.cache/cuda',
             'TMPDIR':'tmp','HOME':'runtime/home','MS_ASSET_DIR':'.cache/maniskill',
             'PIP_CACHE_DIR':'.cache/pip'}
    for key, path in paths.items():
        target=ROOT/path; target.mkdir(parents=True,exist_ok=True); os.environ[key]=str(target)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')

def receive_exact(sock, n):
    parts=[]
    while n:
        part=sock.recv(min(n,1048576))
        if not part: raise EOFError('peer disconnected')
        parts.append(part); n-=len(part)
    return b''.join(parts)

def send(sock, meta, **arrays):
    import numpy as np
    stream=io.BytesIO()
    np.savez(stream, metadata=np.array(json.dumps(meta)), **arrays)
    data=stream.getvalue()
    if len(data)>10_000_000: raise ValueError('message too large')
    sock.sendall(struct.pack('!I',len(data))+data)

def receive(sock):
    import numpy as np
    n=struct.unpack('!I',receive_exact(sock,4))[0]
    if n>10_000_000: raise ValueError('message too large')
    with np.load(io.BytesIO(receive_exact(sock,n)),allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata']))
        arrays={k:archive[k] for k in archive.files if k!='metadata'}
    if meta.get('error'): raise RuntimeError(meta['error'])
    return meta,arrays

def save_json(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(path)
