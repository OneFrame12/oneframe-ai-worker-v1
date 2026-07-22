#!/usr/bin/env python3
import argparse, hashlib, json, os, subprocess, tempfile, tarfile
from pathlib import Path
CANON='cc8d2b5dd07891928a0f83dab8af4899b75ba7ef6aec12de351c015da5a83410'; CKSHA='d8f70210e425a4a4234d547737f57500bcc4ac24a333b99e33d9d5a371e0b80f'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def load_env(path):
 for line in Path(path).read_text().splitlines():
  if line.strip() and not line.lstrip().startswith('#') and '=' in line:
   k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip().strip('"\''))
def canonical_hash(dataset):
 import sys
 candidates=[dataset.parent.parent/'src']+[p/'ai_worker_v1/src' for p in dataset.parents]
 for p in candidates:
  if (p/'dataset_hashing.py').exists():
   sys.path.insert(0,str(p)); from dataset_hashing import compute_training_payload_hash
   return compute_training_payload_hash(dataset)['training_payload_hash']
 raise RuntimeError('canonical hash implementation not found')
def make_archive(dataset,out):
 files=[]
 for p in sorted(dataset.rglob('*')):
  if p.is_file() and not any(x in p.parts for x in ('__pycache__','runs','outputs','model_cache')) and p.name not in ('.DS_Store',): files.append(p)
 with tarfile.open(out,'w') as t:
  for p in files:
   info=t.gettarinfo(str(p),arcname=str(p.relative_to(dataset.parent))); info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
   with open(p,'rb') as f: t.addfile(info,f)
 subprocess.run(['zstd','-q','-f',str(out)],check=True)
 return files,Path(str(out)+'.zst')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dataset-dir',type=Path,required=True); ap.add_argument('--checkpoint-path',type=Path,required=True); ap.add_argument('--contract',type=Path,required=True); ap.add_argument('--env-file',type=Path); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 if a.dry_run==a.execute: ap.error('choose exactly one mode')
 if a.env_file: load_env(a.env_file)
 if not a.dataset_dir.is_dir() or not a.checkpoint_path.is_file(): raise SystemExit('dataset/checkpoint missing')
 print('[1/8] validating dataset',flush=True); ch=canonical_hash(a.dataset_dir)
 print('[2/8] validating checkpoint',flush=True); ck=sha(a.checkpoint_path)
 if ch!=CANON or ck!=CKSHA: raise SystemExit(json.dumps({'canonical_ok':ch==CANON,'checkpoint_ok':ck==CKSHA}))
 report={'canonical_sha256':ch,'checkpoint_sha256':ck,'dataset_files':sum(p.is_file() for p in a.dataset_dir.rglob('*')),'execute':a.execute}
 if a.dry_run: print(json.dumps(report)); return
 try: import boto3
 except ImportError as e: raise SystemExit('boto3 is required for execute') from e
 bucket=os.getenv('R2_INPUT_BUCKET') or os.getenv('AI_WORKER_V1_R2_INPUT_BUCKET'); endpoint=os.getenv('R2_ENDPOINT') or os.getenv('AI_WORKER_V1_R2_ENDPOINT')
 if not bucket or not endpoint: raise SystemExit('R2 bucket/endpoint missing')
 print('[3/8] building deterministic tar',flush=True)
 with tempfile.TemporaryDirectory() as td:
  raw=Path(td)/'dataset.tar'; files,arc=make_archive(a.dataset_dir,raw); print('[4/8] compressing dataset',flush=True); arcsha=sha(arc)
  client=boto3.client('s3',endpoint_url=endpoint,aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID') or os.getenv('AI_WORKER_V1_R2_WRITE_ACCESS_KEY_ID'),aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY') or os.getenv('AI_WORKER_V1_R2_WRITE_SECRET_ACCESS_KEY'))
  print('[5/8] checking immutable R2 objects',flush=True)
  keys=[(f'datasets/ball/v0/{CANON}/dataset.tar.zst',str(arc),arcsha),(f'models/rfdetr/base/{CKSHA}/rf-detr-base.pth',str(a.checkpoint_path),ck)]
  for key,path,digest in keys:
   try:
    head=client.head_object(Bucket=bucket,Key=key); old=(head.get('Metadata') or {}).get('sha256')
    if old!=digest: raise SystemExit('immutable object collision: '+key)
    print('already_verified '+key,flush=True); continue
   except client.exceptions.ClientError as e:
    if e.response.get('ResponseMetadata',{}).get('HTTPStatusCode')!=404: raise
   print('[6/8] uploading '+key,flush=True); client.upload_file(path,bucket,key,ExtraArgs={'Metadata':{'sha256':digest}})
  print('[7/8] uploading manifest',flush=True)
  manifest={'dataset_canonical_sha256':CANON,'dataset_archive_sha256':arcsha,'archive_bytes':arc.stat().st_size,'file_count':len(files),'checkpoint_sha256':ck,'checkpoint_bytes':a.checkpoint_path.stat().st_size,'bucket':bucket,'created_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
  client.put_object(Bucket=bucket,Key=f'datasets/ball/v0/{CANON}/manifest.json',Body=json.dumps(manifest,sort_keys=True).encode(),Metadata={'sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()})
  print('[8/8] verifying HEAD metadata',flush=True); report.update({'archive_sha256':arcsha,'archive_bytes':arc.stat().st_size,'file_count':len(files),'bucket':bucket,'manifest':manifest}); print(json.dumps(report))
if __name__=='__main__': main()
