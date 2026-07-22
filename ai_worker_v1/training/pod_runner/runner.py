#!/usr/bin/env python3
"""Single-shot Ball T3 runner. All I/O uses time-limited presigned URLs."""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

JOB = Path('/workspace/oneframe_job')
ROOT = JOB / 't3_package' / 'oneframe'
OUT = JOB / 'output'

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def url_get(url: str, dest: Path) -> None:
    if not url: raise RuntimeError('missing presigned GET URL')
    dest.parent.mkdir(parents=True, exist_ok=True)
    req=urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(req, timeout=180) as r, dest.open('wb') as f:
        shutil.copyfileobj(r,f,1024*1024)

def url_put(url: str, path: Path) -> None:
    if not url: return
    req=urllib.request.Request(url, data=path.read_bytes(), method='PUT', headers={'Content-Type':'application/octet-stream'})
    with urllib.request.urlopen(req, timeout=180): pass

def heartbeat(state: dict) -> None:
    url=os.getenv('HEARTBEAT_PUT_URL','')
    while not state['stop'].wait(60):
        payload={'run_id':state['run_id'],'status':'RUNNING','timestamp_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':int(time.time()-state['started']),'last_log_line':state['last'],'epoch':state.get('epoch')}
        p=OUT/'heartbeat.json'; p.write_text(json.dumps(payload),encoding='utf-8')
        try: url_put(url,p)
        except Exception: pass

def main() -> int:
    run_id=os.environ['ONEFRAME_RUN_ID']; started=time.time(); OUT.mkdir(parents=True,exist_ok=True); ROOT.mkdir(parents=True,exist_ok=True)
    state={'run_id':run_id,'started':started,'last':'','stop':threading.Event()}; log=OUT/'train.log'; rc=1; first_error=None
    try:
        ds=JOB/'input'/'dataset.tar.zst'; ck=JOB/'input'/'rf-detr-base.pth'; url_get(os.environ['DATASET_GET_URL'],ds); url_get(os.environ['CHECKPOINT_GET_URL'],ck)
        if sha256(ds)!=os.environ['DATASET_ARCHIVE_SHA256']: raise RuntimeError('dataset archive SHA256 mismatch')
        if sha256(ck)!=os.environ['CHECKPOINT_SHA256']: raise RuntimeError('checkpoint SHA256 mismatch')
        (ROOT/'checkpoints').mkdir(parents=True,exist_ok=True); shutil.copy2(ck,ROOT/'checkpoints'/'rf-detr-base.pth')
        # T3's canonical script resolves /workspace/oneframe; expose the packaged
        # tree there without changing the training code or its import contract.
        canonical=Path('/workspace/oneframe')
        if not canonical.exists(): canonical.symlink_to(ROOT, target_is_directory=True)
        zstd=subprocess.run(['zstd','-d','-f',str(ds),'-o',str(JOB/'input'/'dataset.tar')],capture_output=True,text=True)
        if zstd.returncode: raise RuntimeError(zstd.stderr[-500:])
        tar=subprocess.run(['tar','--no-same-owner','-xf',str(JOB/'input'/'dataset.tar'),'-C',str(ROOT)],capture_output=True,text=True)
        if tar.returncode: raise RuntimeError(tar.stderr[-500:])
        hb=threading.Thread(target=heartbeat,args=(state,),daemon=True); hb.start()
        cmd=[sys.executable,'scripts/run_t3_remote_training.py']; proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        with log.open('w',encoding='utf-8') as lf:
            for line in proc.stdout:
                state['last']=line.rstrip(); lf.write(line); lf.flush(); print(line,end='',flush=True)
        rc=proc.wait()
        if rc: first_error=state['last']
    except Exception as exc:
        first_error=str(exc); rc=1
    finally:
        state['stop'].set();
        if not log.exists(): log.write_text((first_error or '')+'\n',encoding='utf-8')
        artifacts={}
        candidates=list(ROOT.rglob('best.pth'))+list(ROOT.rglob('last.pth'))+list(ROOT.rglob('*metrics*.json'))+list(ROOT.rglob('*config*.json'))
        urls={'best.pth':'BEST_PUT_URL','last.pth':'LAST_PUT_URL','metrics.json':'METRICS_PUT_URL','resolved_config.json':'CONFIG_PUT_URL','train.log':'LOG_PUT_URL'}
        for p in candidates+[log]:
            key=p.name
            if key in urls:
                try: url_put(os.getenv(urls[key],''),p); artifacts[key]=sha256(p)
                except Exception as exc: first_error=first_error or str(exc); rc=rc or 1
        result={'status':'SUCCESS' if rc==0 else 'FAILED','exit_code':rc,'run_id':run_id,'started_at':datetime.fromtimestamp(started,timezone.utc).isoformat(),'ended_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':int(time.time()-started),'gpu':os.getenv('GPU_NAME'),'artifact_sha256':artifacts,'first_error':first_error}
        rp=OUT/'result.json'; rp.write_text(json.dumps(result,indent=2),encoding='utf-8')
        try: url_put(os.getenv('RESULT_PUT_URL',''),rp)
        except Exception: pass
    return rc
if __name__=='__main__': raise SystemExit(main())
