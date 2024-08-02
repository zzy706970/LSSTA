import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

def loss_function(A, A_hat, Z, L, alpha=0.02, beta=5.0):
    """
    1阶+2阶损失函数
    :param A:邻接矩阵 [b, N, N]
    :param A_hat:输出的邻接矩阵
    :param Z:中间输出
    :return:
    """
    # 2阶损失
    beta_matrix = torch.ones_like(A) # [b, N, N]
    mask = A != 0 # A中不为0的是True，为0的是False
    beta_matrix[mask] = beta  # 主要目的我理解是让A_hat中非0元素保持非0，不为0的设一个更大的权重，为0的权重为1
    loss_2nd = torch.mean(torch.sum(torch.pow((A - A_hat) * beta_matrix, 2), dim=2)) # 每一个节点的平均误差
    # 1阶损失
    cnt = 0
    for i in range(A.shape[0]):
        loss_1st = alpha * 2 * torch.trace(torch.matmul(torch.matmul(Z[i].transpose(0, 1), L[i]), Z[i])) # ZT: [b, d, N], [b, N, N], [b, N, d] 
        cnt += loss_1st
    loss_1st = cnt / A.shape[0]
    return loss_1st + loss_2nd

def calculate_laplacian(adj):
        '''
        adj: [b, N, N]
        '''
        d = torch.sum(adj, dim=2)
        b = d.shape[0]
        l = []
        for i in range(b):
            l.append(torch.diag(d[i]))
        laplacian = torch.stack(l, dim=0)
        return laplacian

class SDNE(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(SDNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers # 隐层表示的长度列表
        self.num_layers = len(hidden_layers)
        # Build Encoder
        emodules = nn.ModuleList()
        for hidden in hidden_layers:
            emodules.append(nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU()
                ))
            input_dim = hidden
        self.encoder = emodules

        # Build Decoder
        modules = []
        for hidden in reversed(hidden_layers[:-1]):
            modules.append(nn.Linear(input_dim,hidden)) # 此时input_dim是最后一维的隐层长度              
            modules.append(nn.ReLU())
            input_dim = hidden
        modules.append(nn.Sequential(
            nn.Linear(input_dim, self.input_dim),
            nn.ReLU()
        ))
        self.decoder = nn.Sequential(*modules)

    def forward(self, A):
        """
        输入节点的邻接矩阵
        :param A:领接矩阵 [b, t, n, n]
        :return:
        """
        Z = []
        # print(A.shape)
        for i in range(self.num_layers):
            z = self.encoder[i](A)
            Z.append(z) # [num_layers, b, t, n, d]
            A = z
        A_hat = self.decoder(z)
        return A_hat, Z
    

class AGCN(nn.Module): # 扩散GCN
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out)) # 2 is the length of support 
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)# 正太分布初始化
        nn.init.constant_(self.bias, val=0)# 初始化为常数
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []
        for support in supports: # 对于每个图[N, N]
            support_ks = [torch.eye(support.shape[0]).to(support.device), support] # [In, support]
            for k in range(2, self.cheb_k): # 阶数
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) # [In, support, 2 * support ^ 2 - In, ...] 矩阵幂
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,btmc->btnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B,T, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('btni,io->btno', x_g, self.weights) + self.bias  # b, T, N, dim_out
        return x_gconv
    
class Temporal_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, head, flag):
        super(Temporal_Attention, self).__init__()
        self.qff = nn.Linear(input_dim, hidden_dim)
        if flag == 'en':
            self.kff = nn.Linear(input_dim, hidden_dim)
            self.vff = nn.Linear(input_dim, hidden_dim)
        else:
            self.kff = nn.Linear(hidden_dim, hidden_dim)
            self.vff = nn.Linear(hidden_dim, hidden_dim)
        self.trans_x = nn.Linear(input_dim, output_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.ln = nn.LayerNorm(output_dim)

        self.d = hidden_dim // head
        self.head = head
        

    def forward(self, x, key_val=None, Mask=True): 
        '''
        输入x：[b, T_q, N, d*head] , key_val 在encoder中与x相同。
        '''
        query = self.qff(x) # [b, t, N, d]
        # print(key_val.shape) 
        key = self.kff(key_val) # [b, t, N, d]
        value = self.vff(key_val) # [b, t, N, d]
        batch_size = x.shape[0]
        query_steps = x.shape[1]
        kv_steps = key_val.shape[1]
        num_vertexs = x.shape[2]
        device = x.device

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3) # [head*b, N, T_q, d]
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1) # [head*b, N, d, T_k]
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3)# [head*b, N, T_k, d]

        a = torch.matmul(query, key)# [head*b, N, T_q, T_k]
        a /= (self.d ** 0.5)

        if Mask == True:
            # 做mask，把前k个做attention
            mask = torch.ones(query_steps, kv_steps).to(device) # [T,T]
            mask = torch.tril(mask) # [T,T]但是对角线以上的值变成0了
            # mask2 = torch.ones(num_steps, num_steps).to(device)
            # mask2 = torch.tril(mask, diagonal=self.k)
            # mask = mask1 - mask2
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.head * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(a).to(device) # [k*B,N,T,T]里面元素全是负无穷大
            a = torch.where(mask, a, zero_vec)

        a = torch.softmax(a, -1) 

        value = torch.matmul(a, value)# [head*b, N, T_q, d]
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0,2,1,3)#[b, T_q, N, head*d]
        value = self.ff(value) + self.trans_x(x)

        return self.ln(value)
    
    

class AGCN_Block(nn.Module):
    def __init__(self, dim_in, hidden_dim, att_odim, head, node_num, cheb_k, flag):
        '''
        Temporal_Attention: input_dim, hidden_dim, output_dim, head
        '''
        super(AGCN_Block, self).__init__()
        self.head = head
        self.node_num = node_num
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.att_odim = att_odim
        self.attn = Temporal_Attention(self.dim_in, self.hidden_dim, self.att_odim, self.head, flag=flag)
        self.gcn = AGCN(self.att_odim, self.hidden_dim, cheb_k)

    def forward(self, x, supports, key_val=None, Mask=True):
        '''
        x: [b, T, N, c]
        supports: [g1, g2]
        '''
        # 先做attention
        state = self.attn(x, key_val, Mask) # [b, T, N, d], 此时没有输入key, value
        # 再放入GCN中
        state = self.gcn(state, supports) # [b, T, N, d]
        
        return state
    
class AGCN_Encoder(nn.Module):
    def __init__(self, smemory, sdne, w_od,  node_num, dim_in, dim_out, att_odim, head, cheb_k, num_layers):
        '''
        AGCNBlock: dim_in, hidden_dim, att_odim, head, att_steps, node_num, cheb_k
        smemory: num_layers of [M, d]
        '''
        super(AGCN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one layer in the Encoder.'
        self.smemory = smemory # 每一个时间片的模式查询
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        
        self.sdne = sdne
        self.w_od = w_od # 也和层数有关
        #self.w_state = w_state
        
        self.separate_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = nn.MSELoss()
        
        # 几层时间注意力和空间扩散卷积
        self.agcn_blocks = nn.ModuleList()
        self.agcn_blocks.append(AGCN_Block(dim_in, dim_out, att_odim, head, node_num, cheb_k, 'en'))
        for _ in range(1, num_layers):
            self.agcn_blocks.append(AGCN_Block(dim_out, dim_out, dim_out, head, node_num, cheb_k, 'en'))

    def query_smemory(self, layer, h_t:torch.Tensor): # 第几层对应查询第几层的short memory
        query = torch.matmul(h_t, self.smemory[layer]['Wq']) # (B, d) 把隐层表示映射为存储表示
        att_score = torch.softmax(torch.matmul(query, self.smemory[layer]['Memory'].t()), dim=-1)  # alpha: (B, M) 查询矩阵
        value = torch.matmul(att_score, self.smemory[layer]['Memory'])   # (B, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)# (B, 2) 对于每个结点，M中找一下最大的两个位置
        pos = self.smemory[layer]['Memory'][ind[:, 0]] # B, d 在(M, d)中用索引找到d
        neg = self.smemory[layer]['Memory'][ind[:, 1]] # B, d 
        return value, query, pos, neg
    
    def odToEmbedding(self, x_od):
        '''
        x_od: [b, t, n, n]
        '''
        # print(x_od.shape)
        od_hat, xod_emb = self.sdne(x_od)
        return od_hat, xod_emb # [b, t, n, n], [num_layers, b, n, d]
    
    def forward(self, x, x_od, supports):
        #shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        state_list = []
        query_list = []
        pos_list = []
        neg_list = []
        output_hidden = []
        loss_od = 0
        loss_en = 0
        
        od_hat, xod_emb = self.odToEmbedding(x_od) # [b, t, n, n]; 把od矩阵转成embedding 多层的[b, t, N, d]
        for i in range(self.num_layers):
            state = self.agcn_blocks[i](current_inputs, supports[i], current_inputs) # [b, T, N, d]
            # current_inputs = state
            '''
            if i == self.num_layers - 1:
                value, query, pos, neg = self.query_smemory(state)
                state = state + value # [b, N, hidden_dim]
                query_list.append(query)
                pos_list.append(pos)
                neg_list.append(neg)
            '''
    
            layer_od = xod_emb[i] # 当前层的ODembedding：[b, t, N, d]
            for t in range(seq_length):# 对每个时间片分开计算
                current_odemb = layer_od[:, t, ...] # 当前层当前时刻的ODembedding [b, n, d]
                current_state = state[:, t, ...] # 当前层当前时刻的隐层表示[b, n, d]
                # state_weight = torch.tanh(self.w_state(current_state)).transpose(1, 2)
                od_weight = torch.tanh(self.w_od(current_state)).transpose(1, 2) # b, N, d -> b, 1, N
                xod_g = torch.squeeze(torch.matmul(od_weight, current_odemb)) # 全局表示：[b, 1, d]
                value, query, pos, neg = self.query_smemory(i, xod_g) # 查询获得[b, d]
                current_state = current_state + torch.unsqueeze(value, dim=1) # 每一层查完之后加到[b, N, hidden_dim]
                state_list.append(current_state)
                # 每一层每一个时间片做一下查询，都是[b, d]
                query_list.append(query)
                pos_list.append(pos)
                neg_list.append(neg)
                if i == self.num_layers - 1: # 只在最后一层的时候做图的loss
                    current_od = x_od[:, t, ...] # 当前时刻的真实OD
                    current_odhat = od_hat[:, t, ...] # 当前时刻的SDNE生成邻接矩阵[b, n, n]
                    current_lap = calculate_laplacian(current_od)# 当前时刻的拉普拉斯矩阵[b, N, N]
                    loss_od += loss_function(current_od, current_odhat, current_odemb, current_lap) # 算平均loss，[b, N, N]
            current_inputs = torch.stack(state_list, dim=1) # [b, t, n, d]
            output_hidden.append(current_inputs) # numlayers of [b, t, n, d]
            # 每一层算一下查询loss
            query_en = torch.stack(query_list, dim=1) # [b, T, d]
            pos_en = torch.stack(pos_list, dim=1)
            neg_en = torch.stack(neg_list, dim=1)
            loss_en += self.separate_loss(query_en, pos_en.detach(), neg_en.detach())
            if i == self.num_layers - 1:
                loss_en = loss_en / self.num_layers + self.compact_loss(query_en, pos_en.detach())       
            # odloss_list.append(loss_od)

        loss_en = loss_en + loss_od / seq_length
        # query_en = torch.stack(query_list, dim=1)
        # pos_en = torch.stack(pos_list, dim=1)
        # neg_en = torch.stack(neg_list, dim=1)
        # loss_en = self.separate_loss(query_en, pos_en.detach(), neg_en.detach())
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden, loss_en # 最后一层的隐层表示[b, T, N, hidden_dim]
    

class AGCN_Dncoder(nn.Module):
    def __init__(self, smemory, sdne, w_od, node_num, dim_in, dim_out, att_odim, head,  cheb_k, num_layers):
        super(AGCN_Dncoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.smemory = smemory
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        
        self.sdne = sdne
        self.w_od = w_od # 也和层数有关
        # self.w_state = w_state
        
        self.agcn_blocks = nn.ModuleList()
        self.agcn_blocks.append(AGCN_Block(dim_in, dim_out, att_odim, head, node_num, cheb_k, 'de'))
        for _ in range(1, num_layers):
            self.agcn_blocks.append(AGCN_Block(dim_out, dim_out, dim_out, head, node_num, cheb_k, 'de'))
            
        self.w1 = nn.Linear(32, 32)
        self.w2 = nn.Linear(32, 32)
        self.prelu = nn.PReLU()

    def query_smemory(self, layer, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.smemory[layer]['Wq']) # (B, N, d) 把隐层表示映射为存储表示
        att_score = torch.softmax(torch.matmul(query, self.smemory[layer]['Memory'].t()), dim=-1)  # alpha: (B, N, M) 查询矩阵
        value = torch.matmul(att_score, self.smemory[layer]['Memory'])   # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)# (B, N, 2) 对于每个结点，M中找一下最大的两个位置
        pos = self.smemory[layer]['Memory'][ind[:, 0]] # B, N, d 在(M, d)中用索引找到d
        neg = self.smemory[layer]['Memory'][ind[:, 1]] # B, N, d 
        return value, query, pos, neg
    
    def odToEmbedding(self, x_od):
        '''
        x_od: [b, 1, n, n]
        '''
        # print(x_od.shape)
        od_hat, xod_emb = self.sdne(x_od)
        return od_hat, xod_emb # [b, t, n, n], [num_layers, b, n, d]
    
    def forward(self, xt, y_od, y_od_a, init_state, supports):
        # xt: (B, 1, N, D)
        # y_od: (B, N, N)
        # init_state: (num_layers, B, 1, N, hidden_dim), 每一层还是设成一样了
        # print('decoder---------------------------')
        assert xt.shape[2] == self.node_num and xt.shape[3] == self.input_dim
        current_inputs = xt # [b, 1, N, output_dim+yconv_dim]
        output_hidden = []
        loss_od = 0
        device = xt.device
        
        # od_hat, yod_emb = self.odToEmbedding(y_od)
        for i in range(self.num_layers): 
            state = self.agcn_blocks[i](current_inputs, supports[i], key_val=init_state[i], Mask=False) # [b, 1, N, d]
            '''
            if i == self.num_layers - 1:
                value, query, pos, neg = self.query_memory(state.squeeze(1)) # state变成[b, N, d]
                state = state + value # [b, N, hidden_dim] Ht + Mt
                state = state.unsqueeze(1) # [b, 1, N, hidden_dim]
            '''
            if i == 0: #按第一层的隐层表示把y_od变成y_od_f
                S = self.prelu(torch.matmul(self.w1(state.squeeze(1)), self.w2(state.squeeze(1)).transpose(2, 1))).to(device) # [b, N, d] [b, d, N]
                y_od_f = y_od + S # [b, n, n]
                od_hat, yod_emb = self.odToEmbedding(y_od_f)
            layer_od = yod_emb[i] # 当前层的ODembedding[b, n, d]
            od_weight = torch.tanh(self.w_od(state.squeeze(1))).transpose(1, 2) # b, N, d -> b, 1, N
            # state_weight = torch.tanh(self.w_state(state.squeeze(1))).transpose(1, 2)
            yod_g = torch.squeeze(torch.matmul(od_weight, layer_od)) # 全局表示：[b, 1, d]
            value, query, pos, neg = self.query_smemory(i, yod_g) # value [b, d]
            state = state + value.unsqueeze(1).unsqueeze(1).repeat(1, 1, 50, 1) # [b, 1, N, hidden_dim]
            
            if i == self.num_layers - 1:
                current_odemb = layer_od # 最后一层的ODembedding
                current_lap = calculate_laplacian(y_od_a)# 真实矩阵形成的[b, N, N]
                loss_od = loss_function(y_od_a, od_hat, current_odemb, current_lap) # 算平均loss，[b, N, N]
            
            output_hidden.append(state) # num_layers of [b, 1, N, hidden_dim]
            current_inputs = state # [b, 1, N, hidden_dim]
        loss_de = loss_od
        return current_inputs, output_hidden, loss_de # 最后一层的表示[b, n, decoder_dim]
    

class MegaCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, att_odim, head, num_layers=1, cheb_k=3,
                 ycov_dim=2, lmem_num=10, lmem_dim=64, smem_num=8, smem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
        '''
        num_nodes: 结点个数，70/50
        input_dim: 输入维度：3
        output_dim: 输出维度：2
        horizon: 输出时间步：12
        rnn_units: rnn中隐层表示：64
        att_odim: attention和GCN的中间表示：8
        num_layers: rnn层数：2
        cheb_k: 扩散阶数
        ycov_dim: 除了输出维度以外的维度：1
        mem_num: memory中的个数，是个超参，可以调调
        mem_dim: memory中的维度    
        
        AGCN_Encoder/Decoder: memory, node_num, dim_in, dim_out, att_odim, head, cheb_k, num_layers
        '''
        super(MegaCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.att_odim = att_odim
        self.head = head
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # long memory
        self.lmem_num = lmem_num
        self.lmem_dim = lmem_dim
        #self.lmemory = []
        #for i in range(num_layers):
        #    self.lmemory.append(self.construct_long_memory()) # 建立元结点库
        self.lmemory = self.construct_long_memory()
        
        # short memory
        self.smem_num = smem_num
        self.smem_dim = smem_dim
        self.smemory = []
        for i in range(num_layers):
            self.smemory.append(self.construct_short_memory()) # 建立元结点库
            
        # share sdne, w_od, w_air
        self.sdne = SDNE(num_nodes, [self.rnn_units, self.rnn_units])
        self.w_od = nn.Linear(self.rnn_units, 1)
        # self.w_state = nn.Linear(self.rnn_units, 1)
        
        self.w_air = nn.Linear(6, self.rnn_units)

        # encoder
        self.encoder = AGCN_Encoder(self.smemory, self.sdne, self.w_od,
                                    self.num_nodes, self.input_dim, self.rnn_units, self.att_odim, self.head, self.cheb_k, self.num_layers)
        
        # deocoder
        # self.decoder_dim = self.rnn_units + self.lmem_dim # hidden_dim+mem_num
        self.decoder_dim = self.rnn_units
        self.decoder = AGCN_Dncoder(self.smemory, self.sdne, self.w_od, 
                                    self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.att_odim, self.head, self.cheb_k, self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_long_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.lmem_num, self.lmem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.lmem_dim), requires_grad=True)    # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.lmem_num), requires_grad=True) # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.lmem_num), requires_grad=True) # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict.to('cuda:0')
    
    def construct_short_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.smem_num, self.smem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.smem_dim), requires_grad=True)    # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.smem_num), requires_grad=True) # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.smem_num), requires_grad=True) # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict.to('cuda:0')
    
    
    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.lmemory['Wq']) # (B, N, d) 把隐层表示映射为存储表示
        att_score = torch.softmax(torch.matmul(query, self.lmemory['Memory'].t()), dim=-1)  # alpha: (B, N, M) 查询矩阵
        value = torch.matmul(att_score, self.lmemory['Memory'])   # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)# (B, N, 2) 对于每个结点，M中找一下最大的两个位置
        pos = self.lmemory['Memory'][ind[:, :, 0]] # B, N, d 在(M, d)中用索引找到d
        neg = self.lmemory['Memory'][ind[:, :, 1]] # B, N, d 
        return value, query, pos, neg
            
    def forward(self, x, y_cov, x_od, y_od, y_od_a, airports, labels=None, batches_seen=None):
        supports = []
        device = x.device
        
        for i in range(self.num_layers):
            # long
            node_embeddings1 = torch.matmul(self.lmemory['We1'], self.lmemory['Memory']).to(device) # [N, M] * [M, d]
            node_embeddings2 = torch.matmul(self.lmemory['We2'], self.lmemory['Memory']).to(device) # [N, M] * [M, d]
            # short
            node_embeddings3 = torch.matmul(self.smemory[i]['We1'], self.smemory[i]['Memory']).to(device)# [N, M] * [M, d]
            node_embeddings4 = torch.matmul(self.smemory[i]['We2'], self.smemory[i]['Memory']).to(device)# [N, M] * [M, d]
            
            node_embeddingsl1 = torch.concat([node_embeddings1, node_embeddings3], dim=-1).to(device)
            node_embeddingsl2 = torch.concat([node_embeddings2, node_embeddings4], dim=-1).to(device)
            #node_embeddingsl1 = torch.max(torch.stack([node_embeddings1, node_embeddings3], dim=-1), dim=-1)[0].to(device) # [N, d, 2] -> [N, d]
            #node_embeddingsl2 = torch.max(torch.stack([node_embeddings2, node_embeddings4], dim=-1), dim=-1)[0].to(device) # [N, d, 2] -> [N, d]
        
            g1 = F.softmax(F.relu(torch.mm(node_embeddingsl1, node_embeddingsl2.T)), dim=-1).to(device) # 变成[N, N]
            g2 = F.softmax(F.relu(torch.mm(node_embeddingsl2, node_embeddingsl1.T)), dim=-1).to(device)
            supports.append((g1, g2))
            
        # g3 = F.softmax(F.relu(torch.mm(node_embeddings3, node_embeddings4.T)), dim=-1)
        # g4 = F.softmax(F.relu(torch.mm(node_embeddings4, node_embeddings3.T)), dim=-1)
        # supports = [g1,  g2]
        # init_state = self.encoder.init_hidden(x.shape[0])
        
        h_en, state_en, loss_en = self.encoder(x, x_od, supports) # B, T, N, hidden
        '''
        for j in range(self.num_layers):
            h_t = state_en[j][:, -1, :, :] # B, N, hidden (last state)  只要每一层最后一个t时刻的        
            
            h_att, query, pos, neg = self.query_memory(j, h_t) # pos: [B, N, d], neg: [B, N, d]，就是说只做输入最后一个时间的隐层表示
            # h_t = torch.cat([h_t, h_att], dim=-1) # [b, N, hidden+mem_num]
            # 把最后一层都加上查出来的模式
            state_en[j] = state_en[j] + h_att.unsqueeze(1)
        '''
        h_t = h_en[:, -1, :, :] # [b, n, d]
        h_t = h_t + self.w_air(airports).unsqueeze(0) # [b, n, d]
    
        h_att, query, pos, neg = self.query_memory(h_t)
        for j in range(self.num_layers):
            state_en[j] = torch.cat([state_en[j], h_att.unsqueeze(1)], dim=1) # 每一层都和查出来的模式拼一下。
        
        ht_list = state_en # num_layers of [b, T+1, N, hidden]
        go = torch.zeros((x.shape[0], 1, self.num_nodes, self.output_dim), device=x.device) # [b, 1, N, output_dim] 初始化的y1
        out = []
        loss_de_all = 0
        for t in range(self.horizon):
            if t == 0:
                y_ = x[:, -1:, :, self.output_dim:] # 用前一个时间片的天气[b, 1, N, ycov_dim]
            else:
                y_ = y_cov[:, t-1, ...].unsqueeze(1) # [b, 1, N, yconv_dim]
            # print(y_cov.shape)
            h_de, ht_list_t, loss_de = self.decoder(torch.cat([go, y_], dim=-1), y_od[:, t, ...], y_od_a[:, t, ...], ht_list, supports) # 输入为[b, 1, N, output_dim+yconv_dim ] ht_list: [b, t+1, N, hidden_dim]
            loss_de_all += loss_de
            for i in range(self.num_layers):
                ht_list[i] = torch.concat([ht_list[i], ht_list_t[i]] , dim=1) # [b, T+t, N, hidden_dim]
            go = self.proj(h_de) # h_de:[b, 1, n, decoder_dim]
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen): # 随batch增多减小
                    go = labels[:, t, ...].unsqueeze(1) # 其中有一些换成labels
        output = torch.concat(out, dim=1)
        loss_de_all = loss_de_all / self.horizon
        
        return output, h_att, query, pos, neg, loss_en, loss_de_all
    
def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters. \n')
    return

def main():
    import sys
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use")
    parser.add_argument('--num_variable', type=int, default=70, help='number of variables (e.g., 207 in METR-LA, 325 in PEMS-BAY)')
    parser.add_argument('--his_len', type=int, default=12, help='sequence length of historical observation')
    parser.add_argument('--seq_len', type=int, default=12, help='sequence length of prediction')
    parser.add_argument('--channelin', type=int, default=3, help='number of input channel')
    parser.add_argument('--channelout', type=int, default=2, help='number of output channel')
    parser.add_argument('--rnn_units', type=int, default=64, help='number of hidden units')
    parser.add_argument('--att_odim', type=int, default=8, help='number of attention output units')
    parser.add_argument('--head', type=int, default=8, help='number of attention head')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    # MegaCRN: num_nodes, input_dim, output_dim, horizon, rnn_units, att_odim,
    model = MegaCRN(num_nodes=args.num_variable, input_dim=args.channelin, output_dim=args.channelout, horizon=args.seq_len, rnn_units=args.rnn_units, att_odim=args.att_odim, head=args.head).to(device)
    # summary(model, [(args.his_len, args.num_variable, args.channelin), (args.seq_len, args.num_variable, args.channelout)], device=device)
    print_params(model)
    
if __name__ == '__main__':
    main()