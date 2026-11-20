"""Run one exact check, in a resource-limited Docker container."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

HERE=Path(__file__).resolve().parent
IMAGE='sagemath/sagemath@sha256:e068670ae5863b54b2550e72437ec637b0283acb0dc712c8584c124dbf44e667'
LOCK=Path.home()/'.local/state/math-job.lock'
M=805306368
def docker(*args,secs=15):
    return subprocess.check_output(['docker',*args],timeout=secs,text=True).strip()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def now():
    return datetime.now(timezone.utc).isoformat()
inside="""set -eu
for key in memory.max memory.swap.max cpu.max pids.max; do echo "$key=$(cat /sys/fs/cgroup/$key)" >&2; done
echo "interfaces=$(ls /sys/class/net)" >&2
test "$(cat /sys/fs/cgroup/memory.max)" = 805306368
test "$(cat /sys/fs/cgroup/memory.swap.max)" = 0
test "$(cat /sys/fs/cgroup/cpu.max)" = "100000 100000"
test "$(cat /sys/fs/cgroup/pids.max)" = 48
for iface in /sys/class/net/*; do
    test -f "$iface/flags" || continue
    test "${iface##*/}" = lo && continue
    flags=$(cat "$iface/flags")
    test "$((flags & 1))" -eq 0
done
test "$(wc -l < /proc/net/route)" -le 1
mkdir -p /tmp/home
export DOT_SAGE=/tmp/home/.sage HOME=/tmp/home OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd /work
sage -python check.py "$1" > /tmp/result 2>/tmp/errors || { head -c 4096 /tmp/errors >&2; exit 1; }
test "$(wc -c </tmp/result)" -le 16384
cat /tmp/result
"""
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=['green','exterior','wave','analytic'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    result=out/(args.case+'.json'); receipt=out/(args.case+'-receipt.json')
    if result.exists() or receipt.exists():
        raise SystemExit('Choose a fresh output directory or unrecorded case.')
    # Fail closed at the shared lock. The owner must verify the container exit.
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    try:
        LOCK.mkdir()
    except FileExistsError:
        print('Shared math lock is occupied.',file=sys.stderr); return 75
    name='smooth-twistor-'+uuid.uuid4().hex[:12]
    owner=LOCK/'owner'; started=now()
    owner.write_text(json.dumps(dict(executor='smooth-atlas supplement',started_at=started,job_name=args.case,container=name)))
    created=False; safe=False
    try:
        docker('info','--format','{{.NCPU}} {{.MemTotal}}')
        if docker('ps','-q'):
            raise RuntimeError('Use an idle Docker engine for this reproduction.')
        image_id=docker('image','inspect',IMAGE,'--format','{{.Id}}')
        docker('create','--platform=linux/amd64','--pull=never','--name',name,'--read-only','--network=none',
               '--restart=no','--cpus=1','--memory='+str(M),'--memory-swap='+str(M),
               '--pids-limit=48','--ulimit','fsize=65536:65536','--ulimit','nofile=128:128',
               '--cap-drop=ALL','--security-opt=no-new-privileges','--log-driver=json-file',
               '--log-opt','max-size=64k','--log-opt','max-file=1',
               '--tmpfs','/tmp:rw,nosuid,nodev,size=32m,mode=1777',
               '--shm-size=8m','--mount','type=bind,source='+str(HERE)+',target=/work,readonly',
               '--entrypoint','/usr/bin/timeout',IMAGE,'-s','KILL','150','/bin/sh','-c',inside,'guard',args.case)
        created=True
        cfg=json.loads(docker('inspect',name))[0]['HostConfig']
        assert (cfg['Memory'],cfg['MemorySwap'],cfg['NanoCpus'],cfg['PidsLimit'])==(M,M,1000000000,48)
        assert cfg['NetworkMode']=='none' and cfg['ReadonlyRootfs'] and not cfg.get('OomKillDisable')
        assert cfg['RestartPolicy']['Name']=='no'
        started_process=subprocess.run(['docker','start','-a',name],capture_output=True,text=True,timeout=165)
        if started_process.returncode:
            raise RuntimeError((started_process.stderr+' '+started_process.stdout)[-6000:])
        payload=started_process.stdout.strip()
        state=json.loads(docker('inspect','--format','{{json .State}}',name))
        assert state['Status']=='exited' and state['ExitCode']==0 and not state['OOMKilled']
        parsed=json.loads(payload)
        assert parsed['case']==args.case
        result.write_text(payload+'\n')
        record=dict(case=args.case,started=started,finished=now(),container=name,image=image_id,image_ref=IMAGE,
                    state=state,
                    memory_and_swap_bytes=M,swap_extra_bytes=0,cpus=1,pid_limit=48,
                    internal_timeout_s=150,host_timeout_s=165,network='none',restart='no',
                    tmpfs_bytes=33554432,shm_bytes=8388608,log_limit='64k x 1',file_size_bytes=65536,
                    observed_cgroup=started_process.stderr.strip(),
                    sha256={p.name:sha(p) for p in (HERE/'run.py',HERE/'check.py',HERE/(args.case+'.py'),result)})
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        print(args.case,sha(result))
    finally:
        # A daemon timeout may leave an uncertain create/start. Verify by name
        # before releasing even when create() never returned successfully.
        present=docker('ps','-aq','--filter','name=^/'+name+'$')
        if present:
            status=docker('inspect','--format','{{.State.Status}}',name)
            if status not in ('exited','created'):
                docker('kill',name)
                docker('wait',name)
            status=docker('inspect','--format','{{.State.Status}}',name)
            if status not in ('created','exited'):
                raise RuntimeError('Container end uncertain. Lock retained.')
            docker('rm',name)
        safe=not docker('ps','-aq','--filter','name=^/'+name+'$')
        if safe and json.loads(owner.read_text())['container']==name:
            owner.unlink(); LOCK.rmdir()
        else:
            raise RuntimeError('Container/owner state uncertain. Lock retained.')
    return 0
if __name__=='__main__':
    sys.exit(main())
