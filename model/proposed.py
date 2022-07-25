import torch
from torch import nn

from .tacotron import Prenet, CBHG
from .attention import PreservedBidirectionalAttention as Attention

def pad_sequence(sequences, **kwargs):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, **kwargs)

class LocalEncoder(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.prenet = Prenet(hparams.input_dim, sizes=hparams.prenet.sizes)
        self.cbhg = CBHG(hparams.cbhg.dim, K=hparams.cbhg.K, projections=hparams.cbhg.projections)

    def forward(self, inputs, input_lengths=None):
        x = self.prenet(inputs)
        print('prenet', x.isnan().any())
        x = self.cbhg(x, input_lengths)
        print('cbhg', x.isnan().any())
        return x

class Proposed(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.local_encoder_1 = LocalEncoder(hparams.local_encoder)
        self.local_encoder_2 = LocalEncoder(hparams.local_encoder)
        self.local_text_encoder_1 = LocalEncoder(hparams.local_text_encoder)
        self.local_text_encoder_2 = LocalEncoder(hparams.local_text_encoder)
        self.attention = Attention(hparams.attention.k1_dim, hparams.attention.k2_dim, hparams.attention.preserved_k1_dim, hparams.attention.preserved_k2_dim, hparams.attention.dim)
        self.gst_linear_1 = nn.Linear(hparams.gst_linear_1.input_dim, hparams.gst_linear_1.output_dim)
        self.gst_linear_2 = nn.Linear(hparams.gst_linear_2.input_dim, hparams.gst_linear_2.output_dim)
        self.lst_linear_1 = nn.Linear(hparams.lst_linear_1.input_dim, hparams.lst_linear_1.output_dim)
        self.lst_linear_2 = nn.Linear(hparams.lst_linear_2.input_dim, hparams.lst_linear_2.output_dim)
        #self.softmax = nn.Softmax(dim=-1)
        self.mse = nn.MSELoss()

    def forward(self, sbert1, sbert2, bert1, bert2, gst1, gst2, lst1, lst2, length1, length2):
        sbert1, sbert2 = torch.stack(sbert1), torch.stack(sbert2)
        bert1, bert2 = pad_sequence(bert1), pad_sequence(bert2)
        gst1, gst2 = torch.stack(gst1), torch.stack(gst2)
        lst1, lst2 = pad_sequence(lst1), pad_sequence(lst2)
        length1, length2 = torch.stack(length1).cpu(), torch.stack(length2).cpu()
        print([i.isnan().any() for i in [sbert1, sbert2, bert1, bert2, gst1, gst2, lst1, lst2, length1, length2]])
        print([(torch.max(i), torch.min(i)) for i in [bert1, bert2, lst1, lst2]])

        global_features1 = torch.cat([sbert1, gst1], dim=-1)
        global_features2 = torch.cat([sbert2, gst2], dim=-1)
        local_text_features1 = self.local_text_encoder_1(bert1, length1)
        local_text_features2 = self.local_text_encoder_2(bert2, length2)
        local_features1 = self.local_encoder_1(torch.cat([bert1, lst1], dim=-1), length1)
        local_features2 = self.local_encoder_2(torch.cat([bert2, lst2], dim=-1), length2)
        #local_text_features1 = bert1
        #local_text_features2 = bert2
        #local_features1 = torch.cat([bert1, lst1], dim=-1)
        #local_features2 = torch.cat([bert2, lst2], dim=-1)

        print([i.isnan().any() for i in [global_features1, local_features1, local_text_features1]])
        print([i.isnan().any() for i in [global_features2, local_features2, local_text_features2]])
        local_features1to2, local_features2to1, _, _ = self.attention(local_text_features1, local_text_features2, local_features1, local_features2, local_features1, local_features2, length1, length2)

        p_gst1 = self.gst_linear_1(torch.cat([global_features2, sbert1], dim=-1))
        p_gst2 = self.gst_linear_2(torch.cat([global_features1, sbert2], dim=-1))
        p_lst1 = self.lst_linear_1(torch.cat([local_features2to1, local_text_features1], dim=-1))
        p_lst2 = self.lst_linear_2(torch.cat([local_features1to2, local_text_features2], dim=-1))
        p_lst1 = [i[:l] for i, l in zip(p_lst1, length1)]
        p_lst2 = [i[:l] for i, l in zip(p_lst2, length2)]
        return p_gst1, p_gst2, p_lst1, p_lst2

    def gst_loss(self, p_gst, gst):
        gst = torch.stack(gst)
        return self.mse(p_gst, gst)

    def lst_loss(self, p_lst, lst):
        lst = torch.cat(lst, dim=0)
        p_lst = torch.cat(p_lst, dim=0)
        return self.mse(p_lst, lst)

if __name__ == '__main__':
    import sys
    from data.borderlands import Borderlands
    from data.common import Collate
    from hparams import proposed

    device = 'cpu'
    data_loader = torch.utils.data.DataLoader(Borderlands(sys.argv[1]), batch_size=2, shuffle=True, collate_fn=Collate(device))

    model = Proposed(proposed)
    model.to(device)

    for data in data_loader:
        sbert1, sbert2, bert1, bert2, gst1, gst2, lst1, lst2, length1, length = data
        p_gst1, p_gst2, p_lst1, p_lst2 = model(*data)
        print(model.gst_loss(p_gst1, gst1))
        print(model.gst_loss(p_gst2, gst2))
        print(model.lst_loss(p_lst1, lst1))
        print(model.lst_loss(p_lst2, lst2))
        break
