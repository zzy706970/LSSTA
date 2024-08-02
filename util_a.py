# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:42:10 2022

@author: AA
"""
import numpy as np
import scipy.sparse as sp
import torch

class FlightDataset(torch.utils.data.Dataset):
    def __init__(self, args, delay, time_slot, od, od_a):
        if isinstance(delay, np.ndarray):
            delay = torch.from_numpy(delay).to('cuda:0')
        if isinstance(time_slot, np.ndarray):
            time_slot = torch.from_numpy(time_slot).to('cuda:0')
        # 在此处把nan值变成0
        #delay[torch.isnan(delay)] = 0
        self.delay = delay.float()
        self.time_slot = time_slot.float()
        self.time_slot = self.time_slot.unsqueeze(1).repeat(1, args.num_nodes, 1)
        self.delay = torch.concat([self.delay, self.time_slot], dim=-1)
        self.od = od
        self.od_a = od_a
        self.hist = args.seq_len
        self.pred = args.horizon

    def __len__(self):
        return self.delay.shape[0] - self.hist - self.pred
    
    def __getitem__(self, ts):
        # x, y, x_od_true, y_od, y_od_true
        return (self.delay[ts:ts+self.hist],self.delay[ts+self.hist:ts+self.hist+self.pred], 
                 self.od_a[ts:ts+self.hist], self.od[ts+self.hist:ts+self.hist+self.pred], self.od_a[ts+self.hist:ts+self.hist+self.pred])
    


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def load_data(data_name, ratio = [0.7, 0.1]):
    if data_name == 'US':
        adj_mx = np.load('udata/adj_mx.npy')
        od_power = np.load('udata/od_pair.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(70):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load('udata/udelay.npy')
        wdata = np.load('udata/weather2016_2021.npy')  
    if data_name == 'China':
        adj_mx = np.load('/home/zhangzeyu/flight prediction/MegaCRN-OD/cdata/dist_mx.npy')
        od_power = np.load('/home/zhangzeyu/flight prediction/MegaCRN-OD/cdata/crowdness/daul_od.npy') # 
        od_power = od_power/(od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(50):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load('/home/zhangzeyu/flight prediction/MegaCRN-OD/cdata/delay.npy')
        # data[data<-15] = -15
        wdata = np.load('/home/zhangzeyu/flight prediction/MegaCRN-OD/cdata/weather_cn.npy')
        node_feature = np.load('/home/zhangzeyu/MISTAGCN-main/models/mistagcn/DSTGCN/mx_data/airports_feature.npy')
    time_slots = data.shape[1]
    training_data = data[:, :int(ratio[0]*data.shape[1]) ,:]
    val_data = data[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1]),:]
    test_data = data[:, int((ratio[0] + ratio[1])*data.shape[1]):, :]
    training_w = wdata[:, :int(ratio[0]*data.shape[1])]
    val_w = wdata[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1])]
    test_w = wdata[:, int((ratio[0] + ratio[1])*data.shape[1]):]    
    
    training_od = od_power[:int(ratio[0]*data.shape[0]), :, :]
    val_od = od_power[int(ratio[0]*data.shape[0]):int((ratio[0] + ratio[1])*data.shape[0]), :, :]
    test_od = od_power[int((ratio[0] + ratio[1])*data.shape[1]):, :, :]
    return adj, training_data, val_data, test_data, training_w, val_w, test_w, training_od, val_od, test_od, time_slots

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_wmae(preds, labels, weights, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask * weights
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
        
# DCRNN
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float() # 不等于0是True
    mask /= mask.mean() # 非0个数/总数，相当于乘上 总数/非0个数
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0 # 再把nan值去掉
    return loss.mean() 

def masked_mape_loss(y_pred, y_true): # [b, t, n, 2]
    mask = (y_true != 0).float() # 不等于0是1
    mask /= mask.mean() # 非0个数/总数
    loss = torch.abs(torch.div(y_true - y_pred, y_true)) # 
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0  
    return loss.mean()

def masked_R2(y_pred, y_true):
    mask = (y_true != 0).float() # 不等于0是1
    MSE = mask * (y_pred - y_true)**2 # 
    
    mask1 = mask/mask.mean()
    y_mean = torch.mean(mask1 * y_true)
    m_err = mask * (y_true - y_mean) ** 2
    R2 = 1 - torch.sum(MSE)/torch.sum(m_err)
    return R2

def masked_rmse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'\n In total: {param_count} trainable parameters. \n')
    return