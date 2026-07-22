"""Frozen, candidate-first T3 evaluation contract (no training)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_contract(path: Path, split: str) -> dict:
    if split not in ('valid','test'): raise ValueError('evaluation split must be valid or test')
    c=json.loads(path.read_text())
    d=c.get('dataset',{})
    if not d.get('canonical_sha256') or not d.get('r2_key'): raise ValueError('dataset contract incomplete')
    return {'contract':c,'split':split}

def validate_checkpoint(path: Path) -> str:
    if not path.is_file(): raise FileNotFoundError(path)
    digest=sha256_file(path)
    import torch
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(obj,dict) or 'model' not in obj: raise ValueError('checkpoint lacks model weights')
    return digest

def smoke(path: Path, contract: Path, split: str, output: Path) -> dict:
    c=load_contract(contract,split); digest=validate_checkpoint(path); output.mkdir(parents=True,exist_ok=True)
    result={'evaluation_only':True,'checkpoint_valid':True,'dataset_contract_valid':True,'split':split,'evaluator_ready':True,'training_called':False,'checkpoint_sha256':digest,'yolo':{'status':'NOT_REQUESTED'}}
    (output/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
