import os
import sys
import argparse
import numpy as np
import torch
from data.borderlands import Borderlands
from data.common import Collate
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', default=None)
#parser.add_argument('--direction', default='12', choices=['1', '2', '12'])
parser.add_argument('--data_path', default='borderlands')
parser.add_argument('--model', default='proposed', choices=['baseline_gru', 'baseline', 'proposed'])
args = parser.parse_args()

device = "cpu"

if args.model == 'baseline':
    from hparams import baseline as hparams
    from model.baseline import Baseline
    model = Baseline(hparams)
elif args.model == 'proposed':
    from hparams import proposed as hparams
    from model.proposed import Proposed
    model = Proposed(hparams)

test_dataset = Borderlands(args.data_path, filelist='test.txt')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, collate_fn=Collate(device))

model.load_state_dict(torch.load(args.load_model, map_location='cpu'))
model.to(device)

with torch.no_grad():
    p_gst1, p_gst2, p_lst1, p_lst2 = [], [], [], []
    gst1, gst2, lst1, lst2 = [], [], [], []
    for data in tqdm(test_dataloader):
        sbert1, sbert2, bert1, bert2, _gst1, _gst2, _lst1, _lst2, length1, length2 = data
        _p_gst1, _p_gst2, _p_lst1, _p_lst2 = model(*data)
        gst1.append(_gst1)
        gst2.append(_gst2)
        lst1.append(_lst1)
        lst2.append(_lst2)
        p_gst1.append(_p_gst1)
        p_gst2.append(_p_gst2)
        p_lst1.append(_p_lst1)
        p_lst2.append(_p_lst2)

    gst1 = [j for i in gst1 for j in i]
    gst2 = [j for i in gst2 for j in i]
    lst1 = [j for i in lst1 for j in i]
    lst2 = [j for i in lst2 for j in i]
    p_gst1 = torch.cat(p_gst1, dim=0)
    p_gst2 = torch.cat(p_gst2, dim=0)
    p_lst1 = [j for i in p_lst1 for j in i]
    p_lst2 = [j for i in p_lst2 for j in i]

    for i in range(len(test_dataset)):
        if args.model == 'baseline':
            np.save(test_dataset.path / f'{test_dataset.en[i].wav.stem}.p_duration.npy', p_lst1[i].cpu().numpy())
            np.save(test_dataset.path / f'{test_dataset.zh[i].wav.stem}.p_duration.npy', p_lst2[i].cpu().numpy())
        elif args.model == 'proposed':
            np.save(test_dataset.path / f'{test_dataset.en[i].wav.stem}.p_gst.npy', p_gst1[i].cpu().numpy())
            np.save(test_dataset.path / f'{test_dataset.en[i].wav.stem}.p_lst.npy', p_lst1[i].cpu().numpy())
            np.save(test_dataset.path / f'{test_dataset.zh[i].wav.stem}.p_gst.npy', p_gst2[i].cpu().numpy())
            np.save(test_dataset.path / f'{test_dataset.zh[i].wav.stem}.p_lst.npy', p_lst2[i].cpu().numpy())
