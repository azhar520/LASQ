#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from transformers import AutoModel, AutoConfig
from src.common import Triaffine, init_esim_weights, FusionGate, NewFusionGate
# from openhgnn.models import HAN
import torch.nn.functional as F
import torch
import torch.nn as nn
from itertools import accumulate
import math
import json

class GCN(nn.Module):
    def  __init__(self, nfeat, nhid, nclass=0, dropout=0.1):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):    #x特征矩阵,agj邻接矩阵 
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)

        support = torch.einsum('blh,hh->blh', input, self.weight)
        output = torch.einsum('bll,blh->blh', adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    input: (B,N,C_in)
    output: (B,N,C_out)
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征数
        self.out_features = out_features   # 节点表示向量的输出特征数
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # 初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.matmul(inp, self.W)   # [B, N, out_features]
        N = h.size()[1]    # N 图的节点数

        a_input = torch.cat([h.repeat(1,1,N).view(-1, N*N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2*self.out_features)
        # [B, N, N, 2*out_features]
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷


        attention = torch.where(adj>0, e, zero_vec)   # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, dropout=0.6, alpha=0.2, n_heads=8):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout 
        
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)   # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_hid, dropout=dropout,alpha=alpha, concat=False)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))   # 输出并激活
        return x
    
class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        if cfg.tree and cfg.pos:
            output_hidden = 512
            self.gcn = GCN(bert_config.hidden_size + 20, bert_config.hidden_size + 20)
            self.fc = nn.Linear(bert_config.hidden_size + 20, output_hidden)
        elif not cfg.tree and cfg.pos:
            output_hidden = bert_config.hidden_size + 20
            self.gcn = GCN(bert_config.hidden_size + 20, bert_config.hidden_size + 20)
            self.fc = nn.Linear(bert_config.hidden_size + 20, output_hidden)
        elif cfg.tree and not cfg.pos:
            output_hidden = bert_config.hidden_size
            self.gcn = GCN(bert_config.hidden_size, bert_config.hidden_size)
            self.fc = nn.Linear(bert_config.hidden_size, output_hidden)
        else:
            output_hidden = bert_config.hidden_size
            self.gcn = GCN(bert_config.hidden_size + 20, bert_config.hidden_size + 20)
            self.fc = nn.Linear(bert_config.hidden_size + 20, output_hidden)
        self.dense_layers = nn.ModuleDict({
            'ent': nn.Linear(output_hidden, cfg.inner_dim * 2 * 4),
            'rel': nn.Linear(output_hidden, cfg.inner_dim * 4 * 5),
        })

        cfg.inner_dim_sub = cfg['inner_dim_sub_{}'.format(cfg.lang)]
        self.dense_layers_1 = nn.ModuleDict({
            'rel': nn.Linear(output_hidden, cfg.inner_dim_sub * 3 * 5),
        })

        init_esim_weights(self.dense_layers)
        init_esim_weights(self.dense_layers_1)

        self.triaffine = Triaffine(cfg.inner_dim_sub, 1, bias_x=True, bias_y=False)

        h_graph = []
        
        cfg.category = 'tk'

        cfg.meta_paths_dict = {
            'sp-rep': [('tk', 'spk', 'tk'), ('tk', 'rep', 'tk')],
            'rep-sp': [('tk', 'rep', 'tk'), ('tk', 'spk', 'tk')],
            'sp': [('tk', 'spk', 'tk')],
            'rep': [('tk', 'rep', 'tk')],
            'self': [('tk', 'self', 'tk')],
        }

        cfg.hidden_dim = output_hidden
        cfg.out_dim = output_hidden
        cfg.num_heads = [cfg.num_head0]
        # self.han = HAN.build_model_from_args(cfg, h_graph)
        # init_esim_weights(self.han)

        # if cfg.fusion_type == 'gate0':
        #     self.fusion = FusionGate(output_hidden)
        # else:
        #     self.fusion = NewFusionGate(output_hidden)
        # init_esim_weights(self.fusion)
        # self.gan = GAT(bert_config.hidden_size, bert_config.hidden_size)
        self.cfg = cfg
        with open(f"./data/dataset/jsons_{self.cfg.lang}/posDic.json", "r", encoding="utf-8") as f:
            self.posDic = json.loads(f.read())
        self.pos_embedding = nn.Embedding(len(self.posDic), 20, padding_idx=0)
        self.dropout = nn.Dropout(0.5)

    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        """
        output_dim = self.cfg.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.cfg.device)
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((1, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))
        embeddings = embeddings.squeeze(0)
        return embeddings
    
    def get_ro_embedding(self, qw, kw, token_index, token2sents, pos_type):
        if pos_type == 1:
            pos_emb = []
            for i in range(token2sents.shape[0]):
                p = self.custom_sinusoidal_position_embedding(token2sents[i], pos_type)
                pos_emb.append(p)
            pos_emb = torch.stack(pos_emb)
        else:
            position = torch.arange(0, len(token_index[0]), dtype=torch.long, device=self.cfg.device)
            pos_emb = self.custom_sinusoidal_position_embedding(position, pos_type).unsqueeze(0)

        x_cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        x_sin_pos = pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        cur_qw2 = cur_qw2.reshape(qw.shape)
        cur_qw = qw * x_cos_pos + cur_qw2 * x_sin_pos

        y_cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        y_sin_pos = pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        cur_kw2 = cur_kw2.reshape(kw.shape)
        cur_kw = kw * y_cos_pos + cur_kw2 * y_sin_pos

        bsize, qlen, num, dim = cur_qw.shape

        new_kw = cur_kw.permute(0, 2, 1, 3).contiguous().view(bsize * num, qlen, dim)
        new_qw = cur_qw.permute(0, 2, 1, 3).contiguous().view(bsize * num, qlen, dim)

        return new_kw, new_qw

    def classify_matrix(self, kwargs, sequence_outputs, mat_name='ent'):
        utterance_index, token_index = kwargs['utterance_index'], kwargs['token_index']

        token2sents = kwargs['token2sents']
        # if token2sents.shape[1] > 512:
        #     token2sents = token2sents[:, :512]
        #     utterance_index = utterance_index[:, :512]
        #     token_index = token_index[:, :512]
        outputs = self.dense_layers[mat_name](sequence_outputs)

        if mat_name  == 'rel':
            outputs1 = self.dense_layers_1[mat_name](sequence_outputs)

        q_token, k_token, q_utterance, k_utterance = 0, 0, 0, 0
        q1, q2, q3 = 0, 0, 0
        if mat_name == 'ent':
            num = 2
            outputs = torch.split(outputs, self.cfg.inner_dim * num, dim=-1)
            outputs = torch.stack(outputs, dim=-2)
            q_token, k_token = torch.split(outputs, self.cfg.inner_dim, dim=-1)
        elif mat_name == 'rel':
            num = 4
            outputs = torch.split(outputs, self.cfg.inner_dim * num, dim=-1)
            outputs = torch.stack(outputs, dim=-2)
            q_token, q_utterance, k_token, k_utterance = torch.split(outputs, self.cfg.inner_dim, dim=-1)

            num = 3
            outputs1 = torch.split(outputs1, self.cfg.inner_dim_sub * num, dim=-1)
            outputs1 = torch.stack(outputs1, dim=-2)
            q1, q2, q3 = torch.split(outputs1, self.cfg.inner_dim_sub, dim=-1)
            sp0, sp1, sp2, sp3 = q1.shape
            q1 = q1.permute(0, 2, 1, 3).contiguous().view(-1, sp1, sp3)
            q2 = q2.permute(0, 2, 1, 3).contiguous().view(-1, sp1, sp3)
            q3 = q3.permute(0, 2, 1, 3).contiguous().view(-1, sp1, sp3)
        tk_qw, tk_kw = self.get_ro_embedding(q_token, k_token, token_index, token2sents, pos_type=0) # pos_type=0 for token-level relative distance encoding

        ut_qw, ut_kw = 0, 0
        if mat_name != 'ent':
            ut_qw, ut_kw = self.get_ro_embedding(q_utterance, k_utterance, utterance_index, token2sents, pos_type=1) # pos_type=1 for utterance-level relative distance encoding

        return tk_qw, tk_kw, ut_qw, ut_kw, q1, q2, q3

    def get_loss(self, kwargs, logits, input_labels, mat_name):

        nums = logits.shape[-1]
        masks = kwargs['sentence_masks'] if mat_name == 'ent' else kwargs['full_masks']
        # print(masks.shape)
        # if masks.shape[1] > 512:
        #     masks = masks[:, :512, :512]
        # print(masks.shape)
        criterion = nn.CrossEntropyLoss(logits.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))

        active_loss = masks.view(-1) == 1
        active_logits = logits.view(-1, logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)

        return loss
    
    def conduct_triffine(self, qw, kw, q1, q2, q3):
        tri_scores = self.triaffine(q1, q2, q3)
        bi_scores = torch.einsum('bmd,bnd->bmn', qw, kw).contiguous()

        rate = bi_scores
        if self.cfg.soft == 'soft':
            K1 = torch.einsum('bij,bijk->bik', rate, tri_scores.softmax(2))
            K2 = torch.einsum('bjk,bijk->bik', rate, tri_scores.softmax(2))
        else:
            K1 = torch.einsum('bij,bijk->bik', rate, tri_scores)
            K2 = torch.einsum('bjk,bijk->bik', rate, tri_scores)

        K_score = K1 + K2 + bi_scores

        return K_score
    
    def merge_sentence(self, sequence_outputs, dialogue_length):
        res = []
        for i, w in enumerate(dialogue_length):
            res.append(sequence_outputs[i, :w])
        res = torch.cat(res, 0)
        return res
    
    def split_sentence(self, sequence_outputs, dialogue_length):
        bsize = len(dialogue_length)
        res = sequence_outputs.new_zeros([bsize, max(dialogue_length), sequence_outputs.shape[-1]])
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            res[i, :e-s] = sequence_outputs[s:e]
        return res

    def forward(self, **kwargs):
        input_ids, input_masks, input_segments, adj, pos_ids = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments', 'adj', 'pos_id']]
        # if input_ids.shape[1] > 512:
        #     input_ids = input_ids[:, :512]
        #     input_masks = input_masks[:, :512]
        #     input_segments = input_segments[:, :512]
        #     pos_ids = pos_ids[:, :512]
        #     adj = adj[:, :512, :512]
        sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]

        if self.cfg.pos:
            pos_outputs = self.pos_embedding(pos_ids)
            sequence_outputs = torch.cat([sequence_outputs, pos_outputs], dim=-1)
        if self.cfg.tree:
            gcn_outputs = self.gcn(sequence_outputs, adj)
            sequence_outputs = sequence_outputs + gcn_outputs
            sequence_outputs  = self.fc(sequence_outputs)
            sequence_outputs = self.dropout(sequence_outputs)
        # graph_output = self.merge_sentence(sequence_outputs, kwargs['dialogue_length'])
        # h_dict = {'tk': graph_output}
        # graph_output = self.han(hgraphs, h_dict)['tk']

        # graph_output = self.split_sentence(graph_output, kwargs['dialogue_length'])

        # sequence_outputs = self.fusion(sequence_outputs, graph_output)

        mat_names = ['ent', 'rel']
        losses, tags = [], []
        bsize, qlen = sequence_outputs.shape[:2]

        for i in range(len(mat_names)):
            mat_name = mat_names[i]
            tkp, tkq, utp, utq, q1, q2, q3 = self.classify_matrix(kwargs, sequence_outputs, mat_names[i])
            input_labels = kwargs[f"{mat_names[i]}_matrix"]
            if mat_name == 'ent':
                logits = torch.einsum('bmd,bnd->bmn', tkp, tkq).contiguous()
            else:
                logits0 = self.conduct_triffine(tkp, tkq, q1, q2, q3)
                logits1 = torch.einsum('bmd,bnd->bmn', utp, utq).contiguous()
                logits = logits1 + logits0
            
            num = logits.shape[0] // bsize
            logits = logits.view(bsize, num, qlen, qlen).permute(0, 2, 3, 1).contiguous()
            
            loss = self.get_loss(kwargs, logits, input_labels, mat_names[i])
            losses.append(loss)
            tags.append(logits)

        return losses, tags 
