import subprocess as sp
import os

if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    gpus = []
    for l in sp.check_output('nvidia-smi  --query-gpu=index,memory.used,memory.free --format=csv | tail -n +2', shell=True).decode('ascii').split('\n'):
        l = l.strip()
        if len(l) == 0:
            continue
        index, used, free  = l.split(',')
        used = int(used.replace(' MiB', ''))
        free = int(free.replace(' MiB', ''))
        if used >= 1000:
            continue
        gpus.append({
            'id': index,
            'free': free
            })

    gpus = gpus[:1]
    print("GPUs:", gpus)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([g['id'] for g in gpus])
