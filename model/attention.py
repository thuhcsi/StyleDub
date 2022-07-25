import torch
from torch import nn

class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.q_layer = nn.Linear(query_dim, attention_dim)
        self.k_layer = nn.Linear(key_dim, attention_dim)
        self.o_layer = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, key, value):
        query = torch.unsqueeze(query, 1)
        score = self.o_layer(self.tanh(self.q_layer(query) + self.k_layer(key)))
        attention_weights = self.softmax(score)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), value)
        context_vector = torch.squeeze(context_vector)
        attention_weights = torch.squeeze(attention_weights)
        return context_vector, attention_weights

class BidirectionalAttention(nn.Module):

    def __init__(self, k1_dim, k2_dim, v1_dim, v2_dim, attention_dim):
        super().__init__()
        self.k1_layer = nn.Linear(k1_dim, attention_dim)
        self.k2_layer = nn.Linear(k2_dim, attention_dim)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

    def forward(self, k1, k2, v1, v2, k1_lengths=None, k2_lengths=None):
        k1 = self.k1_layer(k1)
        k2 = self.k2_layer(k2)
        score = torch.bmm(k1, k2.transpose(1, 2))

        if not k1_lengths is None or not k2_lengths is None:
            mask = torch.zeros(score.shape, dtype=torch.int).detach().to(score.device)
            for i, l in enumerate(k1_lengths):
                mask[i,l:,:] += 1
            for i, l in enumerate(k2_lengths):
                mask[i,:,l:] += 1
            mask = mask == 1
            score = score.clone().masked_fill_(mask, -float('inf'))

        w1 = self.softmax1(score.transpose(1, 2))
        w2 = self.softmax2(score)

        o1 = torch.bmm(w1, v1)
        o2 = torch.bmm(w2, v2)

        w1 = [i[:l2, :l1] for i, l1, l2 in zip(w1, k1_lengths, k2_lengths)]
        w2 = [i[:l1, :l2] for i, l1, l2 in zip(w2, k1_lengths, k2_lengths)]
        score = [i[:l1, :l2] for i, l1, l2 in zip(score, k1_lengths, k2_lengths)]

        return o1, o2, w1, w2, score

class PreservedBidirectionalAttention(nn.Module):

    def __init__(self, k1_dim, k2_dim, preserved_k1_dim, preserved_k2_dim, attention_dim):
        super().__init__()
        self.k1_layer = nn.Linear(k1_dim, attention_dim)
        self.k2_layer = nn.Linear(k2_dim, attention_dim)
        self.preserved_k1_layer = nn.Linear(k1_dim + preserved_k1_dim, attention_dim)
        self.preserved_k2_layer = nn.Linear(k2_dim + preserved_k2_dim, attention_dim)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

    def forward(self, k1, k2, preserved_k1, preserved_k2, v1, v2, k1_lengths=None, k2_lengths=None):
        preserved_k1 = self.preserved_k1_layer(torch.cat([k1, preserved_k1], dim=-1))
        preserved_k2 = self.preserved_k2_layer(torch.cat([k2, preserved_k2], dim=-1))
        k1 = self.k1_layer(k1)
        k2 = self.k2_layer(k2)
        print([i.isnan().any() for i in [k1, k2, preserved_k1, preserved_k2]])

        #public_score = torch.bmm(k1, k2.transpose(1, 2))
        #score_1 = public_score + torch.bmm(preserved_k1, k2.transpose(1, 2))
        #score_2 = public_score + torch.bmm(k1, preserved_k2.transpose(1, 2))
        score_1 = torch.bmm(k1, preserved_k2.transpose(1, 2))
        score_2 = torch.bmm(preserved_k1, k2.transpose(1, 2))
        print([i.isnan().any() for i in [score_1, score_2]])

        if not k1_lengths is None or not k2_lengths is None:
            mask = torch.zeros(score_1.shape, dtype=torch.int).detach().to(score_1.device)
            for i, l in enumerate(k1_lengths):
                mask[i,l:,:] += 1
            for i, l in enumerate(k2_lengths):
                mask[i,:,l:] += 1
            mask = mask == 1
            score_1 = score_1.clone().masked_fill_(mask, -float('inf'))
            score_2 = score_2.clone().masked_fill_(mask, -float('inf'))

        print([i.isnan().any() for i in [score_1, score_2]])
        import sys
        if score_1.isnan().any() or score_2.isnan().any():
            sys.exit()

        w1 = self.softmax1(score_1.transpose(1, 2))
        w2 = self.softmax2(score_2)

        o1 = torch.bmm(w1, v1)
        o2 = torch.bmm(w2, v2)

        w1 = [i[:l2, :l1] for i, l1, l2 in zip(w1, k1_lengths, k2_lengths)]
        w2 = [i[:l1, :l2] for i, l1, l2 in zip(w2, k1_lengths, k2_lengths)]

        return o1, o2, w1, w2
