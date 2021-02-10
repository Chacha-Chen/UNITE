import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
import torch.distributions as tdist
import matplotlib.pyplot as plt
import numpy as np
import copy
from torch.utils.data import DataLoader, Dataset
from data_loader import fetch_dataloaders_debug, fetch_dataloaders_nash_all_features,\
    fetch_dataloaders_ALZ_all_features
from config import DIC_VGP_CONF
from evaluation import eval
from config import DIC_CTRANSFORMER_CONF, DIC_STN_CONF
from baseline import CTransformer_embed, STN_embed
from sklearn.cluster import SpectralBiclustering

factor = 1e-1

class RBFKernel(nn.Module):
    def __init__(self, input_dim, parameterize=False):
        super(RBFKernel, self).__init__()
        self.input_dim = input_dim
        self.parameterize = parameterize
        if parameterize:
            self.log_std = nn.Parameter(torch.zeros([1]))
            self.log_ls = nn.Parameter(torch.zeros([self.input_dim]))

    def _square_scaled_dist(self, X, Z=None):
        if self.parameterize:
            ls = self.log_ls.exp()
            scaled_X = X / ls[None, :]
            scaled_Z = Z / ls[None, :]
        else:
            scaled_X = X
            scaled_Z = Z
        X2 = scaled_X.pow(2).sum(1, keepdim=True)
        Z2 = scaled_Z.pow(2).sum(1, keepdim=True)
        XZ = scaled_X @ scaled_Z.t()
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        assert X.shape[1] == Z.shape[1]
        if self.parameterize:
            return (2 * self.log_std - 0.5 * self._square_scaled_dist(X, Z)).exp()
        else:
            return (-0.5 * self._square_scaled_dist(X, Z)).exp()

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class LSTM_embed(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, vocab_size, device):
        super(LSTM_embed, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.num_layers = 1
        self.rxdx_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, z_dim, 1, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.z_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.z_dim).to(self.device)
        h1 = self.rxdx_embedding(x)  ##[batch size, sequence length, embedding dim]
        lstm_out, _ = self.lstm(h1, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        h2 = lstm_out[:, -1, :]  # [batch size, z dim]
        return h2

class DeepRBFKernel(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, parameterize=False, encoder=None):
        super(DeepRBFKernel, self).__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.rbf = RBFKernel(z_dim, parameterize)

    def forward(self, X, Z=None):
        X = X.view(X.size(0), -1)
        if Z is None:
            Z = X
        assert X.shape[1] == Z.shape[1]
        embed_X = self.encoder(X)
        embed_Z = self.encoder(Z)
        return self.rbf(embed_X, embed_Z)

    def encode(self, X):
        X = X.view(X.size(0), -1)
        return self.encoder(X)

    def encodeXrbf(self, X, embed_Z):
        embed_X = self.encode(X)
        return self.rbf(embed_X, embed_Z)  # [200,117]  [128,16]

    def rbf_direct(self, embed_Z):
        embed_X = embed_Z
        return self.rbf(embed_X, embed_Z)

def kernel_compute(X, Z, kernel):
    kxz = kernel(X, Z)
    kzz = kernel(Z)
    kxx = kernel(X)
    kzz = kzz + torch.eye(len(kzz)).to(kzz.device) * factor
    Lz = torch.cholesky(kzz)
    inv_Lz = torch.inverse(Lz)
    kzz_inv = inv_Lz.t() @ inv_Lz
    first = kxz @ kzz_inv
    return kxz, kzz, kxx, inv_Lz, kzz_inv, first

def kernel_compute_embedZ(X, embed_Z, kernel):
    kxz = kernel.encodeXrbf(X, embed_Z)
    kzz = kernel.rbf_direct(embed_Z)
    kxx = kernel(X)
    kzz = kzz + torch.eye(len(kzz)).to(kzz.device) * factor
    Lz = torch.cholesky(kzz)
    inv_Lz = torch.inverse(Lz)
    kzz_inv = inv_Lz.t() @ inv_Lz
    first = kxz @ kzz_inv
    return kxz, kzz, kxx, Lz, kzz_inv, first

def conditional_dist(kxz, kxx, first, u):
    mean = first @ u  # nxc
    var = kxx - first @ kxz.t()  # nxn

    return mean, var

class SparseGP(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, n_induce,
                 parameterize=False, n_class=10, encoder=None):
        super(SparseGP, self).__init__()
        self.n_induce = n_induce
        self.n_class = n_class
        self.deepkernel = DeepRBFKernel(input_dim, z_dim, hidden_dim, parameterize, encoder)
        self.embed_Z = nn.Parameter(torch.rand([n_induce, z_dim]))
        self.final_layer = nn.Linear(z_dim, n_class)
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, y, sample=True):
        if sample:
            u = self.final_layer(self.embed_Z) + torch.rand([self.n_induce, self.n_class]).type(x.dtype).to(x.device)
        else:
            # print(self.embed_Z.shape)
            u = self.final_layer(self.embed_Z)
        kxz, kzz, kxx, Lz, kzz_inv, first = kernel_compute_embedZ(x, self.embed_Z, self.deepkernel)
        m, v = conditional_dist(kxz, kxx, first, u)
        # v = v.clamp(0, v.max().detach())
        if sample:
            f_base = torch.rand([x.size(0), self.n_class]).type(x.dtype).to(x.device)
            v = v + torch.eye(len(v)).to(v.device) * factor
            L = torch.cholesky(v)
            # if DEBUG_FLAG:
            f_base = f_base.type(torch.float)
            f = m + L @ f_base
        else:
            f = m
        logu_diff = (-Lz.diag().log().sum() * 2 + kzz_inv.trace()) * self.n_class + (u.t() @ kzz_inv @ u).diag().sum()
        # kld
        loss = self.ce_loss(f, y) + logu_diff / 2
        return loss

    def predict_logit(self, x):
        u = self.final_layer(self.embed_Z)
        kxz, kzz, kxx, Lz, kzz_inv, first = kernel_compute_embedZ(x, self.embed_Z, self.deepkernel)
        m, cov = conditional_dist(kxz, kxx, first, u)
        return m, cov

    def est_uncertainty(self, x, n_sample=10):
        samples = torch.zeros([n_sample, x.shape[0], self.n_class]).to(x.device)
        for i in range(n_sample):
            x = x.type(torch.float32)
            m, v = self.predict_logit(x)
            x = x.type(torch.float64)
            # u = self.final_layer(self.embed_Z) + torch.rand([self.n_induce, self.n_class]).type(x.dtype).to(x.device)
            # kxz, kzz, kxx, Lz, kzz_inv, first = kernel_compute_embedZ(x, self.embed_Z, self.deepkernel)
            # m, v = conditional_dist(kxz, kxx, first, u)
            f_base = torch.rand([x.size(0), self.n_class]).type(x.dtype).to(x.device)
            v = v + torch.eye(len(v)).to(v.device) * factor
            v = v.clamp(0, v.max())
            L = torch.cholesky(v)
            f = m + L @ f_base
            samples[i] = F.softmax(f)
        sample_mean = samples.mean(0)
        sample_var = samples.var(0)
        return sample_mean, sample_var

    def est_uncertainty_new(self, x, n_sample=10):
        samples = torch.zeros([n_sample, x.shape[0], self.n_class]).to(x.device)
        v_samples = torch.zeros([n_sample,x.shape[0],x.shape[0]]).to(x.device)
        for i in range(n_sample):
            x = x.type(torch.float32)
            m, v = self.predict_logit(x)
            # x = x.type(torch.float64)
            # # u = self.final_layer(self.embed_Z) + torch.rand([self.n_induce, self.n_class]).type(x.dtype).to(x.device)
            # # kxz, kzz, kxx, Lz, kzz_inv, first = kernel_compute_embedZ(x, self.embed_Z, self.deepkernel)
            # # m, v = conditional_dist(kxz, kxx, first, u)
            # f_base = torch.rand([x.size(0), self.n_class]).type(x.dtype).to(x.device)
            # v = v + torch.eye(len(v)).to(v.device) * factor
            # v = v.clamp(0, v.max())
            # L = torch.cholesky(v)
            # f = m + L @ f_base
            samples[i] = m
            v_samples[i] = v
        sample_mean = samples.mean(0)
        var_mean = v_samples.mean(0)
        sample_var = samples.var(0)
        return sample_mean, var_mean


if __name__ == '__main__':

    dataset = 'alz_debug'  ## nash, nash_debug

    dic_model_conf = DIC_CTRANSFORMER_CONF

    if dataset == 'mnist':
        args = {
            'batch_size': 128,
            'test_batch_size': 128
        }
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./', train=False, transform=transforms.ToTensor()),
            batch_size=args['test_batch_size'], shuffle=True)
        args = {
            'input_dim': 784,
            'z_dim': 16,
            'hidden_dim': 256,
            'batch_size': 128,
            'test_batch_size': 128,
            'n_induce': 128,
            'parameterize_kernel': False,
            'n_class': 10,
            'cuda': True,
            'lr': 1e-3,
            'num_epochs': 5,
            'test_frequency': 80,
        }
    elif dataset == 'alz_debug':
        vocab_size = 3767  # 14976 for nash debug # 14464  #14976 3767 for ALZ debug
        my_dic_model_conf = DIC_VGP_CONF
        args = {
            'input_dim': 200,
            'z_dim': 2,
            'hidden_dim': 8,
            'batch_size': 200,
            'test_batch_size': 200,
            'n_induce': 20,
            'parameterize_kernel': False,
            'n_class': 2,
            'cuda': False,
            'lr': 1e-3,
            'num_epochs': 1000,
            'test_frequency': 80,
        }
        dataloaders = fetch_dataloaders_debug(['train', 'test', 'val'], batch_size=args['batch_size'])
        val_loader = dataloaders['val']
        train_loader = dataloaders['train']
        test_loader = dataloaders['test']
    elif dataset == 'nash_all':
        vocab_size = 26706
        my_dic_model_conf = DIC_STN_CONF
        my_dic_model_conf['vocab_size'] = 24285
        my_dic_model_conf['embed'] = True
        my_dic_model_conf['seq_length'] = 200
        args = {
            'input_dim': 236,
            'z_dim': 16,  ##16
            'hidden_dim': 32,
            'batch_size': 256,
            'n_induce': 200,
            'parameterize_kernel': False,
            'n_class': 2,
            'cuda': True,
            'lr': 1e-3,
            'num_epochs': 1000,
            'test_frequency': 10,
        }
        dataloaders = fetch_dataloaders_nash_all_features(['train', 'test', 'val'], batch_size=args['batch_size'])
        train_loader = dataloaders['train']
        test_loader = dataloaders['test']
        val_loader = dataloaders['val']
    elif dataset == 'ALZ_all':
        vocab_size = 30692
        my_dic_model_conf = DIC_STN_CONF
        my_dic_model_conf['vocab_size'] = 30692
        my_dic_model_conf['seq_length'] = 200
        my_dic_model_conf['embed'] = True
        args = {
            'input_dim': 236,
            'z_dim': 16,  ##16
            'hidden_dim': 32,
            'batch_size': 256,
            # 'test_batch_size': 10000,
            'n_induce': 200,
            'parameterize_kernel': False,
            'n_class': 2,
            'cuda': True,
            'lr': 1e-3,
            'num_epochs': 300,
            'test_frequency': 10,
        }
        dataloaders = fetch_dataloaders_ALZ_all_features(['train', 'test', 'val'], batch_size=args['batch_size'])
        train_loader = dataloaders['train']
        test_loader = dataloaders['test']
        val_loader = dataloaders['val']

    if args['cuda']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    print("selected device", device)


    if 'debug' in dataset:
        embed_fn = LSTM_embed(args['input_dim'], args['z_dim'], args['hidden_dim'], vocab_size, device)
    elif dataset == 'nash_all':
        embed_fn = STN_embed(my_dic_model_conf, device)
        model_path = './experiments/0905_nash/stn.pth'
        # model = STN_embed(my_dic_model_conf, device)
        cpt = torch.load(model_path, map_location=torch.device('cuda'))
        embed_fn.load_state_dict(cpt['state_dict'])
    elif dataset == 'ALZ_all':
        embed_fn = STN_embed(my_dic_model_conf, device)
        model_path = './experiments/0905_alz-rxdx/stn.pth'
        cpt = torch.load(model_path, map_location=torch.device('cuda'))
        embed_fn.load_state_dict(cpt['state_dict'])
    else:
        # embed_fn = CTransformer_embed(dic_model_conf, device)
        embed_fn = Encoder(input_dim=args['input_dim'],z_dim=args['z_dim'],hidden_dim=args['hidden_dim'])
    embed_fn = embed_fn.to(device)
    sft = torch.nn.Softmax().to(device)

    sgp = SparseGP(args['input_dim'], args['z_dim'], args['hidden_dim'], args['n_induce'],
                   parameterize=args['parameterize_kernel'], n_class=args['n_class'], encoder=embed_fn)
    if args['cuda']:
        sgp = sgp.to('cuda')
    optimizer = Adam(sgp.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader) / 2, .8)

    best_f1 = 0
    best_model = None
    for epoch in range(args['num_epochs']):
        epoch_loss = 0.
        hit = 0.
        for i, (x, y) in enumerate(train_loader):
            if args['cuda']:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            if epoch < 300:  # todo: set a smaller sample epoch like 50?
                loss = sgp(x, y, sample=False)
            else:
                loss = sgp(x, y, sample=False)
            pred, _ = sgp.predict_logit(x)
            # sgp.est_uncertainty(x)
            pred = pred.argmax(1)
            hit_cur = (pred == y).sum().item()
            if i % 200 == 0:
                # pass
                print(f'current hit: {hit_cur}')
            hit += hit_cur
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        # factor *= 0.1

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        print(f'train acc: {float(hit) / normalizer_train}')

        if epoch % args['test_frequency'] == 0:
            # initialize loss accumulator
            test_loss = 0.
            hit = 0.
            pred_list = []
            y_list = []
            pred_prob_list = []
            for i, (x, y) in enumerate(val_loader):
                if args['cuda']:
                    x = x.cuda()
                    y = y.cuda()
                with torch.no_grad():
                    loss = sgp(x, y, sample=False)
                    test_loss += loss.item()

                    pred, cov = sgp.predict_logit(x)
                    # sgp.est_uncertainty(x)
                    pred_tag = pred.argmax(1)

                    hit += (pred_tag == y).sum().item()
                    pred_list.extend(pred_tag.view(-1).cpu().numpy())
                    y_list.extend(y.view(-1).cpu().numpy())
                    sft_logits = sft(pred)
                    pred_prob = sft_logits[:, 1]
                    pred_prob_list.extend(pred_prob.view(-1).cpu().numpy())

            # report test diagnostics
            y_list = np.array(y_list)
            pred_list = np.array(pred_list)
            pred_prob_list = np.array(pred_prob_list)

            accuracy, precision, recall, mcc, f1, kappa, roc_auc, pr_auc, conf_matrix = eval(y_list, pred_list,
                                                                                             pred_prob_list)
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(sgp.state_dict())
                print("Found new best f1 at {}".format(epoch))
            if epoch % 20 == 0:
                torch.save(best_model, './experiments/0909_alz_debug.pth')

    torch.save(best_model, './experiments/0909_alz_debug.pth')
