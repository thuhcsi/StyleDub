import os
import sys
import torch
import argparse
from tqdm import tqdm
from data.borderlands import Borderlands
from data.common import Collate
from save import Save

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', default=None)
parser.add_argument('--direction', default='12', choices=['1', '2', '12'])
parser.add_argument('--load_model', default=None)
parser.add_argument('--data_path', default='borderlands')
parser.add_argument('--model', default='proposed', choices=['baseline_gru', 'baseline', 'proposed'])
args = parser.parse_args()

device = "cuda:%d" % args.gpu

if args.model == 'baseline':
    from hparams import baseline as hparams
    from model.baseline import Baseline #, FakeMST
    model = Baseline(hparams)
    #fake = FakeMST(hparams.fake_mst)
    #fake.to(device)
elif args.model == 'proposed':
    from hparams import proposed as hparams
    from model.proposed import Proposed
    model = Proposed(hparams)

train_dataset = Borderlands(args.data_path, filelist='train.txt')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)
test_dataset = Borderlands(args.data_path, filelist='test.txt')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=Collate(device))

if args.load_model:
    model.load_state_dict(torch.load(args.load_model, map_location='cpu'))

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if args.name is None:
    args.name = args.model
else:
    args.name = args.model + '_' + args.name

save = Save(args.name)
save.save_parameters(hparams)

step = 1
for epoch in range(hparams.max_epochs):
    save.logger.info('Epoch %d', epoch)

    batch = 1
    for data in train_dataloader:
        sbert1, sbert2, bert1, bert2, gst1, gst2, lst1, lst2, length1, length2 = data
        p_gst1, p_gst2, p_lst1, p_lst2 = model(*data)

        loss = 0
        if '1' in args.direction:
            gst1_loss = model.gst_loss(p_gst1, gst1)
            lst1_loss = model.lst_loss(p_lst1, lst1)
            save.writer.add_scalar(f'training/gst1_loss', gst1_loss, step)
            save.writer.add_scalar(f'training/lst1_loss', lst1_loss, step)
            save.writer.add_scalar(f'training/direction1_loss', gst1_loss + lst1_loss, step)
            #loss += gst1_loss + lst1_loss
            loss += lst1_loss
        if '2' in args.direction:
            gst2_loss = model.gst_loss(p_gst2, gst2)
            lst2_loss = model.lst_loss(p_lst2, lst2)
            save.writer.add_scalar(f'training/gst2_loss', gst2_loss, step)
            save.writer.add_scalar(f'training/lst2_loss', lst2_loss, step)
            save.writer.add_scalar(f'training/direction2_loss', gst2_loss + lst2_loss, step)
            #loss += gst2_loss + lst2_loss
            loss += lst2_loss

        save.writer.add_scalar(f'training/loss', loss, step)
        save.save_log('training', epoch, batch, step, loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        batch += 1

    save.save_model(model, f'epoch{epoch}')

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

        loss = 0
        if '1' in args.direction:
            gst1_loss = model.gst_loss(p_gst1, gst1)
            lst1_loss = model.lst_loss(p_lst1, lst1)
            save.writer.add_scalar(f'test/gst1_loss', gst1_loss, epoch)
            save.writer.add_scalar(f'test/lst1_loss', lst1_loss, epoch)
            save.writer.add_scalar(f'test/direction1_loss', gst1_loss + lst1_loss, epoch)
            loss += gst1_loss + lst1_loss
        if '2' in args.direction:
            gst2_loss = model.gst_loss(p_gst2, gst2)
            lst2_loss = model.lst_loss(p_lst2, lst2)
            save.writer.add_scalar(f'test/gst2_loss', gst2_loss, epoch)
            save.writer.add_scalar(f'test/lst2_loss', lst2_loss, epoch)
            save.writer.add_scalar(f'test/direction2_loss', gst2_loss + lst2_loss, epoch)
            loss += gst2_loss + lst2_loss

        save.writer.add_scalar(f'test/loss', loss, epoch)
        save.save_log('test', epoch, batch, epoch, loss)
