# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:25:29 2022

@author: AA
"""

import torch
import util_a as util
import argparse
#import baseline_methods
import random
import copy
import torch.optim as optim
import numpy as np
import torch.nn as nn

#from baseline_methods import test_error
from model_a import MegaCRN
import logging
import time


'''
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='China',help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='hidden layer dimension', type=float)
parser.add_argument('--h_layers',type=int,default=2,help='number of hidden layer')
parser.add_argument('--in_channels',type=int,default=2,help='input variable')
parser.add_argument("--hidden_channels", nargs="+", default=[128, 64, 32], help='hidden layer dimension', type=int)
parser.add_argument('--out_channels',type=int,default=2,help='output variable')
parser.add_argument('--emb_size',type=int,default=16,help='time embedding size')
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
parser.add_argument('--wemb_size',type=int,default=4,help='covairate embedding size')
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=7,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=True,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')
parser.add_argument('--in_len',type=int,default=12,help='input time series length')      # a relatively long sequence can handle missing data
parser.add_argument('--out_len',type=int,default=3,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')
parser.add_argument('--period1',type=int,default=7,help='the input sequence is longer than one day, we use this periodicity to allocate a unique index to each time point')

args = parser.parse_args()

'''
def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters.')
    return


def get_model():  
    model = MegaCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, rnn_units=args.rnn_units,
                    att_odim=args.att_odim, head=args.head, num_layers=args.num_rnn_layers, lmem_num=args.lmem_num, lmem_dim=args.lmem_dim, 
                    smem_num=args.smem_num, smem_dim=args.smem_dim, cheb_k = args.max_diffusion_step, cl_decay_steps=args.cl_decay_steps, 
                    use_curriculum_learning=args.use_curriculum_learning).to(device)
    return model

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def prepare_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    x0 = x[..., :args.input_dim] # input_dim=3
    y0 = y[..., :args.output_dim] # output_dim=2
    y1 = y[..., args.output_dim:] # y1=2
    # x0 = torch.from_numpy(x0).float()
    # y0 = torch.from_numpy(y0).float()
    # y1 = torch.from_numpy(y1).float()
    return x0.to(device), y0.to(device), y1.to(device) # x, y, y_cov

def fetch_position(p: np.ndarray) -> np.ndarray:
    lat = p[:, 1] # 纬度
    lon = p[:, 0] # 经度
    cos_lat = np.cos((lat / 90) * np.pi / 2)
    sin_lon = np.sin(lon / 180 * np.pi)
    cos_lon = np.cos(lon / 180 * np.pi)
    position = np.stack([cos_lat, sin_lon, cos_lon], axis=1)
    
    return position

def load_data():
    # dataset assertion
    assert args.dataset in ["cdata", "udata"]

    if args.dataset=="cdata":
        delay = np.load("cdata/delay.npy")
        args.num_nodes = delay.shape[0]
        time_slot = delay.shape[1]
        od = np.load("cdata/crowdness/daul_od_s.npy") # 按计划的时间
        od_a = np.load("cdata/crowdness/daul_od.npy") # 按实际的时间
        airports_p = np.load("cdata/airports/airports_position.npy")
        weather = np.load("cdata/weather_cn.npy")
        airports_f = np.load("cdata/airports/airports_feature.npy")

    if args.dataset=="udata":
        delay = np.load("udata/udelay.npy")
        args.num_nodes = delay.shape[0]
        time_slot = delay.shape[1]
        od = np.load("udata/od_pair.npy")
        adj = np.load("udata/adj_mx.npy")
        weather = np.load("udata/weather2016_2021.npy")

    # 处理一下时间
    time_step = np.arange(time_slot).reshape(-1, 1)
    time_d = time_step % args.period_d / (args.period_d - 1)
    time_w = time_step // args.period_d % args.period_w/args.period_w
    time_y = time_step // args.period_d % args.period_y/args.period_y
    time_step = time_d + time_w + time_y # [T, 1]
    # time_workday = np.load('/home/zhangzeyu/flight prediction/MegaCRN-OD/cdata/crowdness/timestep.npy')
    # time_step = np.concatenate([time_d, time_w, time_y], axis=1)
    
    # 处理一下机场特征，航站楼、跑道数量、停机位，最后一维做归一化。
    airports_f = torch.from_numpy(airports_f).float()
    airports_f[:, 2] = airports_f[:, 2]/torch.max(airports_f[:, 2])#[50, 3]
    airports_p = fetch_position(airports_p) #[50, 3]
    airports_p = torch.from_numpy(airports_p).float()
    airports = torch.concat([airports_f, airports_p], dim=-1) # [50, 6]
    # airports = airports_p

    def calculate_normalized_laplacian(adj):
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        eye = np.eye(adj.shape[0])
        normalized_laplacian = eye - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return normalized_laplacian

    weather = np.expand_dims(weather, -1)
    # delay = np.concatenate([delay, weather], axis=-1).transpose(1,0, 2)
    # 不能在这里把nan变成0
    # delay[np.isnan(delay)] = 0
    delay = delay

    # 处理一下OD
    # adj, od = calculate_normalized_laplacian(adj), calculate_normalized_laplacian(od)
    # self.od = self.od / torch.sum(self.od, dim=-1).unsqueeze(-1)
    In = np.eye(50)
    In = np.repeat(np.expand_dims(In, axis=0), od.shape[0], axis=0)
    od = od + In
    for i in range(od.shape[0]):
        od[i] = od[i]/(od[i].max())
    
    od = torch.tensor(od, device=device, dtype=torch.float32, requires_grad=True)
    #print('od形状: {}'.format(od.shape))
    # adj = torch.tensor(adj, device=device, dtype=torch.float32, requires_grad=True)
    od_a = od_a + In
    for i in range(od.shape[0]):
        od_a[i] = od_a[i]/(od_a[i].max())
        
    od_a = torch.tensor(od_a, device=device, dtype=torch.float32, requires_grad=True)

    train_num = int(time_slot * args.train_ratio)
    val_num = int(time_slot * args.val_ratio)
    train_data = delay[:train_num]
    # 做归一化的时候去掉了nan值，这里的均值和方差是没有考虑nan转变为0的
    scaler = StandardScaler(train_data[~np.isnan(train_data)].mean(), train_data[~np.isnan(train_data)].std())
    # 在此处做归一化, nan值还是nan值
    delay = scaler.transform(delay)
    # 和天气拼起来
    delay = np.concatenate([delay, weather], axis=-1).transpose(1,0,2) # [T, N, 3]
    # nan值去掉
    delay[np.isnan(delay)] = 0

    train_dataset = util.FlightDataset(args, delay[:train_num], time_step[:train_num], od[:train_num], od_a[:train_num]) # 返回X，T，Y, X_od, Y_od 
    val_dataset = util.FlightDataset(args, delay[train_num:train_num+val_num], time_step[train_num:train_num+val_num], od[train_num:train_num+val_num], od_a[train_num:train_num+val_num])
    test_dataset = util.FlightDataset(args, delay[train_num+val_num:], time_step[train_num+val_num:], od[train_num+val_num:], od_a[train_num+val_num:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    logger.info(f"{args.dataset} Dataset Load Finished.")
    return train_loader, val_loader, test_loader, scaler, od, airports

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['udata', 'cdata'], default='cdata', help='which dataset to run')
parser.add_argument('--train_ratio', type=float, default=0.7, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.1, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--period_d', type=int, default=36, help='number of time steps per day')
parser.add_argument('--period_w', type=int, default=7, help='number of days per week')
parser.add_argument('--period_y', type=int, default=365, help='number of days per year')
parser.add_argument('--num_nodes', type=int, default=50, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=4, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=2, help='number of output channel')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--num_rnn_layers', type=int, default=2, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=32, help='number of rnn units')
parser.add_argument('--att_odim', type=int, default=8, help='number of attention output units')
parser.add_argument('--head', type=int, default=4, help='number of attention head')

parser.add_argument('--lmem_num', type=int, default=10, help='number of meta-nodes/prototypes')
parser.add_argument('--lmem_dim', type=int, default=32, help='dimension of meta-nodes/prototypes')
parser.add_argument('--smem_num', type=int, default=8, help='number of meta-nodes/prototypes')
parser.add_argument('--smem_dim', type=int, default=32, help='dimension of meta-nodes/prototypes')

parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument('--lamb', type=float, default=0.02, help='lamb value for separate loss')
parser.add_argument('--lamb1', type=float, default=0.02, help='lamb1 value for compact loss')
parser.add_argument('--lamb2', type=float, default=0.1, help='lamb2 value for encoder loss')
parser.add_argument('--lamb3', type=float, default=0.1, help='lamb3 value for decoder loss')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon, 为了增加数值计算的稳定性而加到分母里的项")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default=True, help="use_curriculum_learning")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--test_every_n_epochs', type=int, default=1, help='test_every_n_epochs')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
# parser.add_argument('--seed', type=int, default=100, help='random seed.')
args = parser.parse_args()  

device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

model_name = 'MegaCRN_OD'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
# path = f'./save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'save/{model_name}_{args.dataset}_od&t{timestring}__logging.txt'
modelpt_path = f'save/models/{model_name}_{args.dataset}_od&t{timestring}.pt'
# if not os.path.exists(path): os.makedirs(path)
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

def evaluate(model, val_loader, scaler, airports, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter =  val_loader
        losses = []
        ys_true, ys_pred = [], []
        maes, mapes, mses, mr2 = [], [], [], []
        l_3, m_3, r_3 = [], [], []
        l_6, m_6, r_6 = [], [], []
        l_12, m_12, r_12 = [], [], []
        for x, y, x_od, y_od, y_od_a in data_iter:
            x, y, ycov = prepare_x_y(x, y)
            output, h_att, query, pos, neg, loss_en, loss_de = model(x, ycov, x_od, y_od, y_od_a, airports)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss1 = util.masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            separate_loss = nn.TripletMarginLoss(margin=1.0)
            compact_loss = nn.MSELoss()
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3 + args.lamb2 * loss_en + args.lamb3 * loss_de
            losses.append(loss.item())
            # Followed the DCRNN TensorFlow Implementation
            maes.append(util.masked_mae_loss(y_pred, y_true).item())
            mapes.append(util.masked_mape_loss(y_pred, y_true).item())
            mses.append(util.masked_mse_loss(y_pred, y_true).item())
            # mr2.append(util.masked_R2(y_pred, y_true).item())
            # Important for MegaCRN model to let T come first.
            '''
            y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)
            l_3.append(util.masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
            m_3.append(util.masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
            r_3.append(util.masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
            l_6.append(util.masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
            m_6.append(util.masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
            r_6.append(util.masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
            l_12.append(util.masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
            m_12.append(util.masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
            r_12.append(util.masked_mse_loss(y_pred[11:12], y_true[11:12]).item())
            '''
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        mean_mae, mean_mape, mean_rmse = np.mean(maes), np.mean(mapes), np.sqrt(np.mean(mses))
        Y = torch.concat(ys_true, dim=0) # [2699, 12, 50, 2]
        Yhat = torch.concat(ys_pred, dim=0)
        # print(Y.shape)
        for i in range(args.horizon):
            mr2.append(util.masked_R2(Yhat[:, i, ...], Y[:, i, ...]).item())
        mean_r2 = np.mean(mr2)
        l_3, m_3, r_3 = np.mean(l_3), np.mean(m_3), np.sqrt(np.mean(r_3))
        l_6, m_6, r_6 = np.mean(l_6), np.mean(m_6), np.sqrt(np.mean(r_6))
        l_12, m_12, r_12 = np.mean(l_12), np.mean(m_12), np.sqrt(np.mean(r_12))
        if mode == 'test':
            logger.info('Test dataset: Horizon overall: mae: {:.4f}, R2: {:.4f}, rmse: {:.4f}'.format(mean_mae, mean_r2, mean_rmse))
            # logger.info('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_3, m_3, r_3))
            # logger.info('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_6, m_6, r_6))
            # logger.info('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_12, m_12, r_12))
        return mean_loss, mean_mae, mean_r2, mean_rmse, ys_true, ys_pred
    
def main():
    
    model = get_model()
    print_model(model)
    logger.info('模型encoder使用实际的航班网络情况，decoder使用计划的航班网络情况。')
    logger.info('模型记忆网络大小：lmemory:{}, smemory:{}'.format(args.lmem_num, args.smem_num))
    logger.info('此时loss的权重为：长期的三元组误差 = {}、MSE误差 = {}；短期的图误差和三元组误差 = {}；decoder的误差 = {}'.format(args.lamb, args.lamb1, args.lamb2, args.lamb3))
    logger.info('用{}步预测{}步'.format(args.seq_len, args.horizon))
    # supports = [torch.tensor(i).to(device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    
    train_loader, val_loader, test_loader, scaler, od, airports = load_data()
    airports = airports.to('cuda:0')
    print("start training...",flush=True)
    
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = train_loader
        losses = []
        for x, y, x_od, y_od, y_od_a in data_iter:
            # x_od: [64, 12, 50, 50]
            # print('x_od形状：{}'.format(x_od.shape))
            optimizer.zero_grad() 
            x, y, ycov = prepare_x_y(x, y)
            output, h_att, query, pos, neg, loss_en, loss_de = model(x, ycov, x_od, y_od, y_od_a, airports, y, batches_seen)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss1 = util.masked_mae_loss(y_pred, y_true) # 把0 mask了
            separate_loss = nn.TripletMarginLoss(margin=1.0) # 三元组损失，默认欧式距离
            compact_loss = nn.MSELoss() # 
            loss2 = separate_loss(query, pos.detach(), neg.detach()) # pos: [B, N, d], neg: [B, N, d]
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3 + args.lamb2 * loss_en + args.lamb3 * loss_de
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            # 对所有的梯度乘以一个clip_coef，而且乘的前提是clip_coef一定是小于1的。防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            optimizer.step()
        train_loss = np.mean(losses)
        lr_scheduler.step()
        val_loss, mean_mae, mean_r2, mean_rmse, _, _ = evaluate(model, val_loader, scaler, airports, 'eval')
        # if (epoch_num % args.test_every_n_epochs) == args.test_every_n_epochs - 1:
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, val_mae: {:.4f}, val_r2: {:.4f}, val_rmse: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1, 
                   args.epochs, batches_seen, train_loss, val_loss, mean_mae, mean_r2, mean_rmse, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        test_loss, _, _, _, _, _ = evaluate(model, test_loader, scaler, airports, 'test')
        
        if val_loss < min_val_loss:
            wait = 0
            logger.info('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
            # logger.info('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch_num)
                break
    
    logger.info('=' * 35 + 'Best model performance' + '=' * 35)
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    test_loss, _, _, _, _, _ = evaluate(model, test_loader, scaler, airports, 'test')
    
if __name__ == "__main__":   
    main() 
