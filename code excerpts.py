import numpy as np 
import pandas as pd, random, os
from tqdm import tqdm
import time
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pickle
def set_model_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## data mask 
def get_fc_mask(time, label, num_timestamp):
    '''
        mask is required to get the log-likelihood loss
        mask size is [N, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_timestamp]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored??? label = 1 at t0
            mask[i,int(time[i,0])] = 1 # round t0 and set mask value at the rounded time to 1
        else: #label[i,2]==0: censored...... label = 0 at t0 
            mask[i,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def get_fc_mask2(time, num_timestamp):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category].
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_timestamp]) # for the first loss function                  
    for i in range(np.shape(time)[0]):
        t = int(time[i, 0]) # censoring/event time
        mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask

## IMVLSTM model
class IMVFullLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.W_i = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_f = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_o = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)  #87*32*1
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    @torch.jit.script_method
    
    def forward(self, x):
        # dimension of h: number of observations x number of features x number of cells/units
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units)
        outputs = torch.jit.annotate(List[Tensor], [])
        # loop over time steps 
        for t in range(x.shape[1]):
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            c_t = c_t*f_t + i_t*j_tilda_t.reshape(j_tilda_t.shape[0], -1)
            h_tilda_t = (o_t*torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)

        # calculate attentions, exp/sum(exp)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)                                     
  
     
        g_n = torch.sum(alphas*outputs, dim=1)                                                        
        hg = torch.cat([g_n, h_tilda_t], dim=2)                                                       
        mu = self.Phi(hg)

        betas = torch.tanh(self.F_beta(hg))                                                           
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        
        mean = torch.sum(betas*mu, dim=1)
        mean = torch.nn.functional.softmax(mean,dim=1)
        return mean, alphas, betas

## loss function
class loss_Ranking(torch.nn.Module):
    
    def __init__(self):
        super(loss_Ranking,self).__init__()
        
    def forward(self,pred,t,m1,m2,y):
        
        sigma1 = 0.1
        I_2 = torch.diag(torch.squeeze(y))
        one_vector = torch.ones_like(t)
        
        R = torch.mm(pred, torch.transpose(m2,0,1))
        diag_R = torch.reshape(torch.diag(R), (-1, 1)) 
        R = torch.mm(one_vector, torch.transpose(diag_R,0,1)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = torch.transpose(R,0,1)                   # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        T = torch.nn.functional.relu(torch.sign(torch.mm(one_vector, torch.transpose(t,0,1)) - torch.mm(t, torch.transpose(one_vector,0,1))))        
        T = torch.mm(I_2, T) # only remains T_{ij}=1 when event occured for subject i

        self.LOSS_2 =  torch.mean(torch.mean(T * torch.exp(-R/sigma1)))
                
    
        return  self.LOSS_2


## get minibatch
def f_get_minibatch(idx, mb_size, x, label, time, mask1, mask2):
    x_mb = torch.Tensor(x[idx:idx+mb_size, :])
    y_mb = torch.Tensor(label[idx:idx+mb_size, :])  # censoring(0)/event(1,2,..) label
    t_mb = torch.Tensor(time[idx:idx+mb_size, :])
    m_mb1 = torch.Tensor(mask1[idx:idx+mb_size, :]) #fc_mask
    m_mb2 = torch.Tensor(mask2[idx:idx+mb_size, :]) #fc_mask
    return x_mb, y_mb, t_mb, m_mb1, m_mb2


## train the model   
batch_size=128
n_hidden =8
timeseq=train_x.shape[1]
total_epoch=70
max_valid = -2**32
n_pred=int(max(train_t)+1) 
n_features=train_x.shape[2]

valid_x = torch.Tensor(valid_x)
valid_t = torch.Tensor(valid_t)
valid_m1 = torch.Tensor(valid_m1)
valid_m2 = torch.Tensor(valid_m2)
valid_y = torch.Tensor(valid_y)

model = IMVFullLSTM(n_features, n_pred, n_hidden)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 50, gamma=0.5)

set_model_seed(seed)
for iters in range(total_epoch):
    alphas = []
    betas = []
    avg_loss = 0
    cnt=0
    start = time.process_time()
    for idx in range(0,len(train_x)-batch_size,batch_size):
        x_mb, y_mb, t_mb, m_mb1,m_mb2 = f_get_minibatch(idx, batch_size, train_x, train_y, train_t, train_m1,train_m2)                     
        opt.zero_grad()
        y_pred, a, b = model.forward(x_mb)
        alphas.append(a.detach().cpu().numpy())
        betas.append(b.detach().cpu().numpy())  
        cust_loss=loss_Ranking()
        l =cust_loss(y_pred,t_mb,m_mb1,m_mb2,y_mb)
        l.backward()
        avg_loss += l.item()  
        cnt+=1
        opt.step()
    # step every epoch
    epoch_scheduler.step()
    
    