#!/usr/bin/env python3
import json, os, time, urllib.request
def api(method,path):
    req=urllib.request.Request('https://rest.runpod.io'+path,method=method,headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']})
    with urllib.request.urlopen(req,timeout=30) as r:return json.loads(r.read()) if r.headers.get('content-type','').startswith('application/json') else {}
def main():
    pods=api('GET','/v1/pods').get('pods',[]); deleted=[]; now=time.time()
    for p in pods:
        if not str(p.get('name','')).startswith('oneframe-train-'): continue
        created=p.get('createdAt') or p.get('created_at') or 0
        try: age=now-float(created)/1000 if float(created)>1e10 else now-float(created)
        except Exception: age=0
        if age>9000 or p.get('desiredStatus') in ('EXITED','TERMINATED'):
            api('DELETE','/v1/pods/'+str(p['id'])); deleted.append(p['id'])
    print(json.dumps({'pods_inspected':len(pods),'stale_pods_deleted':deleted,'serverless_touched':False}))
if __name__=='__main__': main()
