"""Reproduce the experiments in jobs.json, then compare every scientific output."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

BASE = Path(__file__).resolve().parent
SAGE = 'sagemath/sagemath@sha256:e068670ae5863b54b2550e72437ec637b0283acb0dc712c8584c124dbf44e667'
PYTHON = 'python@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b'
MIB = 1024**2
WALL = 600
GUARD = """set -eu
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DOT_SAGE=/tmp/dotsage HOME=/tmp
mkdir -p /tmp/dotsage
cd /out
for key in memory.max memory.swap.max cpu.max pids.max; do echo "$key=$(cat /sys/fs/cgroup/$key)"; done > cgroups.txt
uname -m >> cgroups.txt
test "$(cat /sys/fs/cgroup/memory.max)" = "$1"
test "$(cat /sys/fs/cgroup/memory.swap.max)" = 0
test "$(cat /sys/fs/cgroup/cpu.max)" = "100000 100000"
test "$(cat /sys/fs/cgroup/pids.max)" = 48
test "$(wc -l < /proc/net/route)" -le 1
for iface in /sys/class/net/*; do
 test -f "$iface/flags" || continue
 test "${iface##*/}" = lo && continue
 test "$(( $(cat "$iface/flags") & 1 ))" -eq 0
done
shift
"$@" >log.txt 2>&1 && echo 0 >exit.txt || echo $? >exit.txt
cat /sys/fs/cgroup/memory.peak >memory_peak.txt
tar czf - .
"""

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

def docker(*args):
    return subprocess.check_output(['docker', *args], text=True, timeout=25).strip()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536),b''):
            h.update(block)
    return h.hexdigest()

def read_bundle():
    for line in (BASE/'SHA256SUMS').read_text().splitlines():
        digest, name=line.split('  ',1)
        check(sha(BASE/name)==digest, f'Archive member differs: {name}')

def extract(stream, target):
    check(stream.tell()<=100*MIB,'Container output exceeds limit')
    stream.seek(0)
    total=0
    with tarfile.open(fileobj=stream,mode='r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            name=Path(member.name)
            total+=member.size
            check(member.isfile() and len(name.parts)==1 and name.name not in ('.','..') and total<=96*MIB, 'Invalid output member')
            with (target/name.name).open('xb') as dest:
                shutil.copyfileobj(archive.extractfile(member),dest,65536)

def execute(job, work, allowed):
    target=work/job['name']
    target.mkdir(parents=True)
    single=job['source']=='kasner'
    image=PYTHON if single else SAGE
    arch='linux/arm64' if single else 'linux/amd64'
    mem=(160 if single else 1200)*MIB
    lock=Path.home()/'.local/state/math-job.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            time.sleep(1)
    name='twistor-'+uuid.uuid4().hex[:14]
    owner=lock/'owner'
    owner.write_text(json.dumps(dict(executor='supplement-runner',started_at=now(),job_name=job['name'],container=name)))
    try:
        start=time.monotonic()
        vm=json.loads(docker('info','--format','{"cpu":{{.NCPU}},"memory":{{.MemTotal}},"cgroups":"{{.CgroupVersion}}"}'))
        check(vm['cpu']>=1 and vm['memory']>=mem and vm['cgroups']=='2','Insufficient VM resources or cgroup v2 unavailable')
        active=docker('ps','--format','{{.Names}}').splitlines()
        check(not(set(active)-allowed),'Another workload is active. See the runtime instructions.')
        usage=docker('stats','--no-stream','--format','{{.MemUsage}} {{.CPUPerc}}') if active else ''
        limits=[docker('inspect',n,'--format','{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}') for n in active]
        image_id=docker('image','inspect',image,'--format','{{.Id}}')
        mounts=[]
        for destination,source in {'/src': BASE/'code'/job['source'], **{k:work/v for k,v in job['inputs'].items()}}.items():
            check(source.is_dir(),f'Missing job input {source.name}')
            mounts.extend(['--mount',f'type=bind,src={source},dst={destination},readonly'])
        command=['python'] if single else ['sage','-python']
        command+=['/src/'+job['script']]
        if job['script']=='experiment.py':
            command+=[job['entry']]
        docker('create','--platform',arch,'--name',name,'--network','none','--restart=no','--memory',str(mem),'--memory-swap',str(mem),'--cpus','1','--pids-limit','48','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--ulimit','fsize=98304:98304','--ulimit','nofile=128:128','--log-driver','json-file','--log-opt','max-size=128k','--log-opt','max-file=1','--tmpfs','/tmp:rw,nosuid,nodev,size=48m,mode=1777','--tmpfs','/out:rw,noexec,nosuid,nodev,size=96m,mode=1777','--shm-size','4m',*mounts,'--pull','never','--entrypoint','/usr/bin/timeout',image_id,'-s','KILL',str(WALL),'/bin/sh','-ec',GUARD,'guard',str(mem),*command)
        conf=json.loads(docker('inspect',name,'--format','{{json .HostConfig}}'))
        check((conf['Memory'],conf['MemorySwap'],conf['NanoCpus'],conf['PidsLimit'],conf['NetworkMode'],conf['ReadonlyRootfs'],conf['RestartPolicy']['Name'])==(mem,mem,10**9,48,'none',True,'no') and not conf['OomKillDisable'],'Resource guard failed')
        started=now()
        with tempfile.TemporaryFile() as stream:
            proc=subprocess.run(['docker','start','-a',name],stdout=stream,timeout=WALL+8)
            state=json.loads(docker('inspect',name,'--format','{{json .State}}'))
            check(proc.returncode==0 and not state['Running'] and not state['OOMKilled'] and state['ExitCode']==0,repr(state))
            extract(stream,target)
        receipt=dict(start_utc=started,end_utc=now(),job=job['name'],image=image,image_id=image_id,
                     platform=arch,vm=vm,background_limits=limits,background_usage=usage,
                     memory_and_swap=mem,cpu=1,wall_seconds=WALL,pids=48,network='none',elapsed_seconds=time.monotonic()-start,
                     memory_peak_bytes=int((target/'memory_peak.txt').read_text()),container_exit=state['ExitCode'],oom=state['OOMKilled'])
        (target/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        check((target/'exit.txt').read_text().strip()=='0',(target/'log.txt').read_text()[-1800:])
    finally:
        if docker('ps','-aq','--filter','name=^/'+name+'$'):
            if docker('inspect',name,'--format','{{.State.Running}}')=='true':
                docker('kill',name)
            check(docker('inspect',name,'--format','{{.State.Running}}')=='false','Could not confirm termination; retaining lock')
            docker('rm',name)
        owner.unlink()
        lock.rmdir()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,help='a new directory outside the extracted supplement')
    args=parser.parse_args()
    work=args.output.resolve()
    check(work!=BASE and BASE not in work.parents,'Place output outside the extracted supplement')
    read_bundle()
    work.mkdir(parents=True,exist_ok=False)
    jobs=json.loads((BASE/'jobs.json').read_text())
    expected=json.loads((BASE/'expected-sha256.json').read_text())
    allowed=set(os.environ.get('MATH_BACKGROUND_SERVICES','').split())
    totals={};started=now()
    for job in jobs:
        key=job['name']
        print(key,flush=True)
        execute(job,work,allowed)
        folder=work/key
        admin={'receipt.json','cgroups.txt','log.txt','exit.txt','memory_peak.txt'}
        found={p.name:sha(p) for p in folder.iterdir() if p.name not in admin}
        check(found==expected[key],f'Scientific output mismatch in {key}: '+str(sorted(k for k in found.keys()|expected[key].keys() if found.get(k)!=expected[key].get(k))[:15]))
        totals[key]=len(found)
    result=dict(start_utc=started,end_utc=now(),jobs=len(jobs),matching_files=sum(totals.values()),by_job=totals,verified=True)
    (work/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(jobs)} jobs, {sum(totals.values())} scientific files')
if __name__=='__main__':
    main()
