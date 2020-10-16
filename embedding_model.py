import torch
from torch import nn
import torch.nn.functional as F
from src.utils import TransformerBlock
from src.utils import d

class LocationEncoder(nn.Module):
    def __init__(self, device, embedding_size=[(796,8)], num_numerical_cols=33, output_size=2, layers=[2], p=0.1):
        super().__init__() ##number of unique zip
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)
        self.device = device

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i
        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x_categorical = x[:,0].view(x.shape[0],1).type(torch.LongTensor).to(self.device) ##zip code
        x_numerical = x[:,1:].to(self.device)
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i].to(self.device)))
        out = torch.cat(embeddings, 1)
        out = self.embedding_dropout(out)

        x_numerical = self.batch_norm_num(x_numerical)
        out = torch.cat([out, x_numerical], 1)
        out = self.layers(out)
        return out

class patient_encoder(nn.Module):
    def __init__(self, dic_model_conf, device):
        super(patient_encoder, self).__init__()
        self.device = device
        self.rxdx_embedding = nn.Embedding(dic_model_conf['vocab_size']+1, dic_model_conf['embedding_dim'])
        self.linear_1 = torch.nn.Linear(25600, dic_model_conf['output_dim']) #14464
        self.linear_2 = torch.nn.Linear(2, 2) #14976 for debug  14464
        self.act = nn.Softmax()
        self.embedding_dropout = nn.Dropout(0.1)
        # self.location_encoder = LocationEncoder(device)

    def forward(self, x_rxdx, x_age_gender):
        x_rxdx = x_rxdx.type(torch.LongTensor)
        h1 = self.rxdx_embedding(x_rxdx)   ##[batch size, sequence length, embedding dim]
        h2 = torch.flatten(h1,start_dim=1)
        output_1 = self.linear_1(h2)
        output_2 = self.linear_2(x_age_gender)
        out = torch.cat([output_1, output_2], 1)
        out = self.embedding_dropout(out)
        return out


class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()

  def forward(self, tensors, device):
    result = torch.ones(tensors[0].size()).to(device)
    for t in tensors:
      result *= t
    return t





class UNITE_bnn(nn.Module):
    def __init__(self, dic_model_conf, device):
        self.embed_flag = dic_model_conf['embed']
        super().__init__()
        self.device = device

        self.location_encoder = LocationEncoder(device).to(device)
        self.patient_encoder = CTransformer_embed(dic_model_conf,device).to(device)
        self.fc_age_gender = torch.nn.Linear(2,2)
        self.embed = torch.nn.Linear(20,16)
        self.classifier = torch.nn.Linear(16,2)
        self.embedding_dropout = nn.Dropout(dic_model_conf['dropout'])
        # self.tan = torch.nn.Sigmoid()

    def forward(self,x):
        geographics_features = x[:, :34].to(self.device)
        h_loc = self.location_encoder(geographics_features)
        x_rxdx = x[:, -200:]
        x_rxdx = x_rxdx.type(torch.LongTensor).to(self.device)
        x_age_gender = x[:, 34:36].to(self.device)
        output_2 = self.fc_age_gender(x_age_gender)
        h_pat = self.patient_encoder(x_rxdx)
        all_h = torch.cat([h_loc, h_pat], 1)
        # all_h = h_pat
        all_h = torch.cat([all_h,output_2],1)
        embed = self.embed(all_h)
        embed = self.embedding_dropout(embed)
        embed = self.classifier(embed)
        # embed = self.tan(embed)
        return embed

    def predict_embed(self,x):
        # geographics_features = x[:, :34].to(self.device)
        # h_loc = self.location_encoder(geographics_features)
        x_rxdx = x[:, -200:]
        x_rxdx = x_rxdx.type(torch.LongTensor).to(self.device)
        x_age_gender = x[:, 34:36].to(self.device)
        output_2 = self.fc_age_gender(x_age_gender)
        h_pat = self.patient_encoder(x_rxdx)
        # all_h = torch.cat([h_loc, h_pat], 1)
        # all_h = torch.cat([all_h,output_2],1)
        embed = self.embed(h_pat)
        # outputs = self.classifier(embed)
        return embed

class UNITE_embed(nn.Module):
    def __init__(self, dic_model_conf, device):
        self.embed_flag = dic_model_conf['embed']
        super().__init__()
        self.device = device
        self.location_encoder = LocationEncoder(device).to(device)
        self.patient_encoder = CTransformer_embed(dic_model_conf,device).to(device)
        self.fc_age_gender = torch.nn.Linear(2,2)
        self.embed = torch.nn.Linear(20,16)
        self.classifier = torch.nn.Linear(16,2)
        # self.tan = torch.nn.Sigmoid()

    def forward(self,x):
        geographics_features = x[:, :34].to(self.device)
        h_loc = self.location_encoder(geographics_features.type(torch.float))
        x_rxdx = x[:, -200:]
        x_rxdx = x_rxdx.type(torch.LongTensor).to(self.device)
        x_age_gender = x[:, 34:36].to(self.device)
        output_2 = self.fc_age_gender(x_age_gender.type(torch.float))
        h_pat = self.patient_encoder(x_rxdx)
        all_h = torch.cat([h_loc, h_pat], 1)
        # all_h = h_pat
        all_h = torch.cat([all_h,output_2],1)
        embed = self.embed(all_h)
        if not self.embed_flag:
            embed = self.classifier(embed)
        # embed = self.tan(embed)
        return embed

    def predict_embed(self,x):
        # geographics_features = x[:, :34].to(self.device)
        # h_loc = self.location_encoder(geographics_features)
        x_rxdx = x[:, -200:]
        x_rxdx = x_rxdx.type(torch.LongTensor).to(self.device)
        x_age_gender = x[:, 34:36].to(self.device)
        output_2 = self.fc_age_gender(x_age_gender)
        h_pat = self.patient_encoder(x_rxdx)
        # all_h = torch.cat([h_loc, h_pat], 1)
        # all_h = torch.cat([all_h,output_2],1)
        embed = self.embed(h_pat)
        # outputs = self.classifier(embed)
        return embed

class CTransformer_embed(nn.Module):

    def __init__(self,dic_model_conf,device ):

        super().__init__()
        self.device = device

        emb         = dic_model_conf['emb']
        heads       = dic_model_conf['heads']
        depth       = dic_model_conf['depth']
        seq_length  = dic_model_conf['seq_length']
        num_tokens  = dic_model_conf['vocab_size']
        num_classes = dic_model_conf['num_classes']
        max_pool    = dic_model_conf['max_pool']
        dropout     = dic_model_conf['dropout']
        wide        = dic_model_conf['wide']

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens+1)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)
        # self.norm_layer = nn.LayerNorm(emb)

        # self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

        # self.act = nn.Softmax() .

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # x= x.type(torch.LongTensor).to(self.device)

        try:
            tokens = self.token_embedding(x)
        except:
            print('here')
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=self.device))[None, :, :].expand(b, t, e)
        # positions = self.pos_embedding(torch.arange(t))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        # x = self.toprobs(x)

        # F.log_softmax(x, dim=1)

        return x
