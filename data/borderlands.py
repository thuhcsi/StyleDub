import numpy as np
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

class Utterance:

    def __init__(self, text):
        with open(text) as f:
            self.text = f.readline().strip('\r\n')
        self.wav = text.parent / f'{text.stem}.wav'
        self.speaker = text.parent / f'{text.stem}.speaker.npy'
        self.bert = text.parent / f'{text.stem}.bert.npy'
        self.sbert = text.parent / f'{text.stem}.sbert.npy'
        self.gst = text.parent / f'{text.stem}.gst.npy'
        #self.gst = text.parent / f'{text.stem}.gst_only.npy'
        self.lst = text.parent / f'{text.stem}.lst.npy'
        #self.lst = text.parent / f'{text.stem}.lst_only.npy'
        #self.lst = text.parent / f'{text.stem}.duration.npy'

class Borderlands(torch.utils.data.Dataset):

    def __init__(self, path, filelist='train.txt'):
        super().__init__()
        self.path = Path(path)

        with open(self.path / filelist) as f:
            self.filelist = f.readlines()

        self.filelist = [i.strip('\r\n') for i in self.filelist]
        self.en = [Utterance(self.path / f'{i}.en.txt') for i in self.filelist]
        self.zh = [Utterance(self.path / f'{i}.zh.txt') for i in self.filelist]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        sbert1 = np.load(self.en[index].sbert)
        sbert2 = np.load(self.zh[index].sbert)
        bert1 = np.load(self.en[index].bert)
        bert2 = np.load(self.zh[index].bert)
        gst1 = np.load(self.en[index].gst)
        gst2 = np.load(self.zh[index].gst)
        lst1 = np.load(self.en[index].lst).astype(np.float32)
        lst2 = np.load(self.zh[index].lst).astype(np.float32)
        try:
            assert bert1.shape[0] == lst1.shape[0]
        except:
            print(self.en[index].text, bert1.shape, lst1.shape)
        try:
            assert bert2.shape[0] == lst2.shape[0]
        except:
            print(self.zh[index].text, bert2.shape, lst2.shape)

        length1 = torch.as_tensor(bert1.shape[0], dtype=torch.int64)
        length2 = torch.as_tensor(bert2.shape[0], dtype=torch.int64)

        return sbert1, sbert2, bert1, bert2, gst1, gst2, lst1, lst2, length1, length2

def process_bert_en(model, tokenizer, utterance: Utterance):
    text = ''.join([i for i in utterance.text.lower() if i in "abcedfghijklmnopqrstuvwxyz' "])
    words = text.split(' ')
    words = [i for i in words if i != '']
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    sbert = outputs.pooler_output[0].detach().numpy()
    bert = outputs.last_hidden_state[0][1:-1].detach().numpy()
    result = []
    start = 0
    for word in words:
        subwords = tokenizer.tokenize(word)
        if len(subwords) > 1:
            result.append(np.mean(bert[start:start+len(subwords)], axis=0, keepdims=False))
        elif len(subwords) == 1:
            result.append(bert[start])
        start += len(subwords)
    try:
        np.save(utterance.bert, np.stack(result))
        np.save(utterance.sbert, sbert)
    except:
        print(utterance.text, utterance.bert)

def process_bert_zh(model, tokenizer, utterance: Utterance):
    text = utterance.text.replace(' ', '')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    sbert = outputs.pooler_output[0].detach().numpy()
    bert = outputs.last_hidden_state[0][1:-1].detach().numpy()
    punctuation = '？，。.！B—\'、FLK·T[]：CEO%60H（；）RAG!D9VIP-《》8",M…QW7“”/S?X'
    bert = [i for character, i in zip(text, bert) if not character in punctuation]
    bert = np.stack(bert)
    try:
        np.save(utterance.bert, bert)
        np.save(utterance.sbert, sbert)
    except:
        print(utterance.text, utterance.bert)

if __name__ == '__main__':
    import sys
    from functools import partial
    from tqdm.contrib.concurrent import process_map, thread_map

    #dataset = Borderlands(sys.argv[1], filelist='train.txt')
    dataset = Borderlands(sys.argv[1], filelist='test.txt')
    print(len(dataset))

    if not dataset.en[0].bert.exists():
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #process_bert_en(model, tokenizer, dataset.en[0])
        thread_map(partial(process_bert_en, model, tokenizer), dataset.en)

    if not dataset.zh[0].bert.exists():
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
        model = AutoModel.from_pretrained("hfl/chinese-bert-wwm")
        #process_bert_zh(model, tokenizer, dataset.zh[0])
        thread_map(partial(process_bert_zh, model, tokenizer), dataset.zh)

    from .common import Collate
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=Collate('cuda:0'), drop_last=True)
    for batch in data_loader:
        for _list in batch:
            print([i.shape for i in _list])
            print([i.dtype for i in _list])
        break
