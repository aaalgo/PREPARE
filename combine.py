#!/usr/bin/env python3
import os, sys
import pandas as pd
import pickle
import json
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F

if __name__ == '__main__':

    delta_paths = sys.argv[1:]
    if len(delta_paths) == 0:
        delta_paths = list(glob('model_s*/ch*/*.pkl'))


    ranked = []

    for path in delta_paths:
        with open(path, 'rb') as f:
            _, s = pickle.load(f)

        parent = os.path.dirname(path)
        step = int(parent.rsplit('-', 1)[1])
        #print(path, step)
        with open(os.path.join(parent, 'trainer_state.json'), 'r') as f:
            state = json.load(f)
        loss = None
        for e in state['log_history']:
            if e['step'] == step and 'eval_loss' in e:
                loss = e['eval_loss']
        assert not loss is None
        ranked.append((loss, s))

    ranked.sort()
    ranked = ranked[:10]
    submit = None
    count = 0
    for loss, s in ranked:
        print(loss)
        #t = F.softmax(t, dim=-1)
        s = F.softmax(s, dim=-1)
        #nll = F.nll_loss(torch.log(t+1e-9), test_labels).item()
        print(path)
        if submit is None:
            submit = s
        else:
            submit = submit + s
        count += 1
    print(count)

    submit /= count
    #submit += 1e-3

    submit = submit / torch.sum(submit, dim=1, keepdim=True)

    df = pd.read_csv('data/submission_format.csv')
    df[['diagnosis_control','diagnosis_mci','diagnosis_adrd']] = submit.numpy()
    df.to_csv('submission.csv', index=False)

