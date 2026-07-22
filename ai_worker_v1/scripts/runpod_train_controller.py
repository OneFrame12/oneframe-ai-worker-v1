#!/usr/bin/env python3
"""Single-Pod Ball T3 controller with verified R2 outputs and cleanup."""
from __future__ import annotations
import argparse,json,os,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; CONTRACT=ROOT/'ai_worker_v1/training/jobs/ball_v0_t3_pod.json'; ACTIVE=ROOT/'ai_worker_v1/runtime/active_training_pod.json'
GPU_PRIORITY=['NVIDIA RTX A6000','NVIDIA A40','NVIDIA GeForce RTX 4090','NVIDIA L40S']

def load_env():
    p=Path(os.getenv('ENV_FILE','.env.local'))
    if p.exists():
        for line in p.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip().strip('"\''))

def api(method,path,payload=None):
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request('https://rest.runpod.io'+path,data=data,method=method,headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            body=r.read()
            if r.status==204 or not body:return None
            return json.loads(body)
    except urllib.error.HTTPError as e:
        if e.code==404:return {'_http_status':404}
        raise RuntimeError(f'RunPod HTTP {e.code}: {e.read(300).decode(errors="replace")}')

def r2():
    import boto3
    from botocore.config import Config
    return boto3.client('s3',endpoint_url=os.environ['R2_ENDPOINT'],aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],region_name='auto',config=Config(signature_version='s3v4',connect_timeout=10,read_timeout=60))

def inputs(c,run_id):
    s=r2(); bucket=os.environ.get('R2_INPUT_BUCKET','one-frame'); ds=c['dataset']['r2_key']; ck=c['checkpoint']['r2_key']; manifest=ds.rsplit('/',1)[0]+'/manifest.json'
    for key in (ds,ck,manifest): s.head_object(Bucket=bucket,Key=key)
    outbucket=os.environ.get('R2_OUTPUT_BUCKET',bucket); prefix=c['output_prefix'].replace('{run_id}',run_id)
    urls={'DATASET_GET_URL':s.generate_presigned_url('get_object',Params={'Bucket':bucket,'Key':ds},ExpiresIn=14400),'CHECKPOINT_GET_URL':s.generate_presigned_url('get_object',Params={'Bucket':bucket,'Key':ck},ExpiresIn=14400),'OUTPUT_PREFIX':prefix}
    names={'heartbeat.json':'HEARTBEAT_PUT_URL','result.json':'RESULT_PUT_URL','best.pth':'BEST_PUT_URL','last.pth':'LAST_PUT_URL','metrics.json':'METRICS_PUT_URL','resolved_config.json':'CONFIG_PUT_URL','training_summary.json':'TRAINING_SUMMARY_PUT_URL','train.log':'LOG_PUT_URL'}
    for name,var in names.items(): urls[var]=s.generate_presigned_url('put_object',Params={'Bucket':outbucket,'Key':prefix+name},ExpiresIn=14400)
    return urls,prefix

def verify_delete(pid):
    for delay in (1,2,4):
        res=api('GET','/v1/pods/'+pid)
        if isinstance(res,dict) and res.get('_http_status')==404:return True
        time.sleep(delay)
    return False

def delete_verified(pid):
    for _ in range(3):
        api('DELETE','/v1/pods/'+pid)
        if verify_delete(pid): return True
        time.sleep(2)
    raise RuntimeError('Pod deletion could not be verified with 404')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true'); p.add_argument('--execute',action='store_true'); p.add_argument('--cleanup-only',action='store_true'); p.add_argument('--job-mode',choices=('train','evaluate'),default='train'); p.add_argument('--source-run-id'); p.add_argument('--eval-checkpoint-key'); p.add_argument('--eval-checkpoint-sha256'); p.add_argument('--eval-split',choices=('valid','test'),default='valid'); a=p.parse_args()
    if sum(bool(x) for x in (a.dry_run,a.execute,a.cleanup_only))!=1:p.error('choose exactly one mode')
    load_env()
    if a.cleanup_only:
        if ACTIVE.exists(): delete_verified(json.loads(ACTIVE.read_text())['pod_id']); ACTIVE.unlink()
        return 0
    c=json.loads(CONTRACT.read_text()); run=('ball-v0-t3-' if a.job_mode=='train' else 'ball-eval-')+time.strftime('%Y%m%dT%H%M%SZ')+'-'+os.getenv('GITHUB_SHA','local')[:7]; urls,prefix=inputs(c,run)
    if a.job_mode=='evaluate':
        s=r2(); b=os.environ.get('R2_INPUT_BUCKET','one-frame'); key=a.eval_checkpoint_key or ''; s.head_object(Bucket=b,Key=key); urls['EVAL_CHECKPOINT_GET_URL']=s.generate_presigned_url('get_object',Params={'Bucket':b,'Key':key},ExpiresIn=14400); urls['EVAL_CHECKPOINT_SHA256']=a.eval_checkpoint_sha256; prefix=c['output_prefix'].replace('{run_id}',a.source_run_id or run)+'validation-recovery-'+time.strftime('%Y%m%dT%H%M%SZ')+'/'
    if a.dry_run: print(json.dumps({'dry_run':True,'run_id':run,'output_prefix':prefix,'pods_created':0,'presigned_urls_generated':True,'max_hourly_cost':float(os.getenv('MAX_HOURLY_COST','0.80'))})); return 0
    pods=api('GET','/v1/pods');
    if not isinstance(pods,list): raise RuntimeError('GET /v1/pods did not return list')
    if any(str(x.get('name','')).startswith('oneframe-train-') and x.get('desiredStatus') not in ('EXITED','TERMINATED') for x in pods): raise RuntimeError('active oneframe-train Pod exists')
    image=os.environ.get('TRAINING_IMAGE','oneframecontent/oneframe-ai-worker-v1:train-'+os.getenv('GITHUB_SHA','local')); env=dict(urls,ONEFRAME_RUN_ID=run,ONEFRAME_JOB_MODE=a.job_mode,EVAL_SPLIT=a.eval_split,DATASET_ARCHIVE_SHA256=c['dataset']['archive_sha256'],CHECKPOINT_SHA256=c['checkpoint']['sha256'])
    payload={'name':('oneframe-train-' if a.job_mode=='train' else 'oneframe-eval-')+run,'imageName':image,'computeType':'GPU','cloudType':'SECURE','interruptible':False,'gpuCount':1,'gpuTypeIds':GPU_PRIORITY,'gpuTypePriority':'availability','containerDiskInGb':80,'volumeInGb':0,'ports':[],'dockerStartCmd':['python','-u','/workspace/oneframe_job/runner.py'],'env':env}
    pod=api('POST','/v1/pods',payload); pid=pod.get('id') or pod.get('podId'); cost=float(pod.get('costPerHr',pod.get('adjustedCostPerHr',0)) or 0)
    ACTIVE.parent.mkdir(parents=True,exist_ok=True); ACTIVE.write_text(json.dumps({'pod_id':pid,'run_id':run}))
    try:
        if cost>float(os.getenv('MAX_HOURLY_COST','0.80')): raise RuntimeError(f'costPerHr {cost} exceeds ceiling')
        deadline=time.time()+int(os.getenv('MAX_RUNTIME_MINUTES','120'))*60; s=r2(); outbucket=os.environ.get('R2_OUTPUT_BUCKET',os.environ.get('R2_INPUT_BUCKET','one-frame'))
        while time.time()<deadline:
            st=api('GET','/v1/pods/'+pid); print(json.dumps({'pod_id':pid,'status':st.get('desiredStatus'),'cost_per_hr':cost}),flush=True)
            result_key=prefix+'result.json';
            try: result=json.loads(s.get_object(Bucket=outbucket,Key=result_key)['Body'].read())
            except Exception: result=None
            if result is not None:
                if result.get('status')!='SUCCESS' or result.get('exit_code')!=0: raise RuntimeError('training FAILED: '+str(result.get('first_error')))
                for key in ('best.pth','train.log','training_summary.json','resolved_config.json'): s.head_object(Bucket=outbucket,Key=prefix+key)
                print(json.dumps({'training':'SUCCESS','output_prefix':prefix})); return 0
            if st.get('desiredStatus') in ('EXITED','TERMINATED'): raise RuntimeError('Pod terminated without successful result.json')
            time.sleep(30)
        raise TimeoutError('training exceeded max runtime')
    finally:
        delete_verified(pid); ACTIVE.unlink(missing_ok=True)
if __name__=='__main__': raise SystemExit(main())
