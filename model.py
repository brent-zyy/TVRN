"""SGRAF model"""

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(cap_emb, lengths, batch_first=True)

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] + cap_emb[:, :, cap_emb.size(2)//2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions(batch_size, region_num, emb_size) and raw global image(batch_size, emb_size) 
        l_emb = self.embedding_local(local) # 带dropout的线性层 batch_norm for region_num 
        g_emb = self.embedding_global(raw_global) # 带dropout的线性层batch_norm for emb_sieze 

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1) #(batch_size, region_num, emb_size)
        common = l_emb.mul(g_emb) #(batch_size, region_num, emb_size) 逐元素相乘
        weights = self.embedding_common(common).squeeze(2) # (batch_size, region_num)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = weights.unsqueeze(2) * local
        new_global = new_global.sum(dim=1)
        # new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """
    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_relation(query, context, smooth=9., eps=1e-8):
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    queryT = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, queryT)
    query_norm = torch.norm(query, p=2, dim=2).repeat(1, queryL).view(batch_size_q, queryL, queryL).clamp(min=1e-8)
    source_norm = torch.norm(context, p=2, dim=2).repeat(1, sourceL).view(batch_size_q, sourceL, sourceL).clamp(
        min=1e-8)
    attn = torch.div(attn, query_norm)
    attn = torch.div(attn, source_norm)

    return attn


def get_TAposition(depend, lens):
    temlen = max(lens)
    adj = np.zeros((len(lens), temlen, temlen))
    for j in range(len(depend)):
        dep = depend[j]
        for i, pair in enumerate(dep):
            if i == 0 or pair[0] >= temlen or pair[1] >= temlen:  # 问题：这里为什么将第一个关系舍去了
                continue
            adj[j, pair[0], pair[1]] = 1
            adj[j, pair[1], pair[0]] = 1
        adj[j] = adj[j] + np.eye(temlen)  # 自己到自己的边

    return torch.from_numpy(adj).cuda().float()


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """
    def __init__(self, embed_size, sim_dim, module_name='AVE', sgr_step=3, focal_type='equal'):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        self.v_global_w = VisualSA(embed_size, 0.4, 72)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)
        self.sim_trancon_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.focal_type = focal_type
        
        self.scan_attention = SCAN_attention(embed_size)
        self.fck = nn.Linear(embed_size, embed_size)
        self.fcq = nn.Linear(embed_size, embed_size)
        self.fcv = nn.Linear(embed_size, embed_size)

        self.cap_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)


        if module_name == 'SGR':
            self.SGR_module = nn.ModuleList([GraphReasoning(sim_dim) for i in range(sgr_step)])
        elif module_name == 'SAF':
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError('Invalid input of opt.module_name in opts.py')

        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens, Iou, depend):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # 关系
        img_relaemb = self.scan_attention(img_emb, img_emb, smooth=9.0)

        weight = Iou.cuda().float()

        img_embq = self.fcq(img_emb)
        img_embk = self.fck(img_emb)
        relation = get_relation(img_embq, img_embk)

        img_positionweight = weight.mul(relation) 
        img_positionweight = nn.LeakyReLU(0.1)(img_positionweight)
        img_positionweight = l2norm(img_positionweight, 2)
        img_positionweight = F.softmax(img_positionweight.permute(0,2,1)*9.0, dim=2)

        # 位置+关系来表示上下文
        img_positionemb = torch.bmm(img_positionweight, img_emb) 
        img_positionemb = l2norm(img_positionemb, dim=-1)

        ## img_con代表区域上下文
        img_con = (img_relaemb + img_positionemb) / 2
        img_con = l2norm(img_con, dim=-1)

        ## 如何让图像的区域学习到周围环境信息？直接做逐元素相乘？还是相加？   先做乘法吧    === 作为高阶信息？
        new_img_emb = l2norm(nn.LeakyReLU(0.1)(img_emb.mul(img_con)), dim=-1)
        ## 直接把高阶语义和初级语义拼接？
        # img_emb = l2norm(img_emb + new_img_emb, dim=-1)
        ## 修改成用cat拼接
        # img_emb = torch.cat([img_emb, new_img_emb], 1)  # (batch, 72, 1024)
        img_emb_1 = torch.cat([img_emb, new_img_emb], 1)  # (batch, 72, 1024)

        # 加入GRU的方式
        img_emb_2, _ = self.img_rnn(img_emb_1)
        img_emb_3 = torch.cat([img_emb, img_emb], 1)
        img_emb = l2norm(img_emb_2 + img_emb_3, dim=-1)

        # 做消融实验，去掉GRU模块
        # img_emb = img_emb_1


        # ## gru不加低阶语义
        # img_emb = img_emb_2

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)   #(batch, 1, 1024)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            # cap_i_expand = cap_i.repeat(n_image, 1, 1)

            ## cap_inew代表文本上下文   ===     只是单独抽出上下文 
            cap_con = self.scan_attention(cap_i, cap_i, smooth = 9.0)
            cap_con_expand = cap_con.repeat(n_image,1,1)

            # 图像上下文去扩充文本侧信息    === 每个句子都注意了图像上下文，且复制到了batch次   === 有batch个一样的向量
            # new_cap_i_expand = self.scan_attention(cap_con_expand, img_con,  smooth = 9.0)
            new_cap_i_expand = self.scan_attention(cap_con_expand, img_con,  smooth = 9.0)
            new_cap_i = new_cap_i_expand[0:1, :, :]       ## 取第一个当作新的扩充的句子（因为batch的向量都一样）

            ## 拼接，只需要把两个句子拼接起来就行，所以维度为（1, 2L, 1024） 
            new_cap_i = torch.cat([cap_i, new_cap_i], 1)

            ## 拼接起来的向量要经过一次GRU才能让文本前后都学习到，这样才能使扩充的文本都学习到内容  /   或者直接用self-attention也行，这样也能学习到前后
            # 此时 new_cap_i 为（1, 2L, 1024）  且每个词都学到了上下文信息，相当于此时的文本同时获得了文本上下文和图像上下文
            # new_cap_i = self.scan_attention(new_cap_i, new_cap_i, smooth = 9.0) (1,2L,1024)
            
            #  GRU单元
            cap_i_2 = torch.cat([cap_i, cap_i], 1)
            cap_i, _ = self.cap_rnn(new_cap_i)
            cap_i = l2norm(cap_i_2 + cap_i, dim=-1)

            # 消融实验，去掉GRU单元
            # cap_i = new_cap_i


            # ## 修改为cat
            # cap_i_new, _ = self.cap_rnn(new_cap_i)
            # cap_i = torch.cat([cap_i, cap_i_new], 1) 

            # ## 修改为 不cat, 直接用 cap_i_new
            # cap_i = cap_i_new

            cap_i_expand = cap_i.repeat(n_image, 1, 1) ## 扩充到128维。便于后面cross-attention

            # 新的文本向量的全局
            # cap_ave_i = torch.mean(new_cap_i, 1)
            # cap_glo_i = self.t_global_w(new_cap_i, cap_ave_i)
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # 进行匹配
            # Context_img = self.scan_attention(cap_i_expand, img_emb,  smooth = 9.0)
            # Context_img = func_attention(cap_i_expand, img_emb,  smooth = 9.0, focal_type= self.focal_type, global_emb=img_glo)
            
            # sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            # sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # ## 补充
            # new_Context_img = self.scan_attention(new_cap_i_expand, new_img_emb,  smooth = 9.0)
            
            # new_sim_loc = torch.pow(torch.sub(new_Context_img, new_cap_i_expand), 2)
            # new_sim_loc = l2norm(self.sim_tranloc_w(new_sim_loc), dim=-1)


            #  concat the global and local alignments
            # sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            sim_vec = sim_glo
            
            # sim_emb = torch.cat([sim_emb, new_sim_loc], 1)

            # compute the final similarity vector
            # if self.module_name == 'SGR':
            #     for module in self.SGR_module:
            #         sim_emb = module(sim_emb)
            #     sim_vec = sim_emb[:, 0, :]
            # else:
            #     sim_vec = self.SAF_module(sim_emb)
            

            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SCAN_attention(nn.Module):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    def __init__(self, embed_size):
        super(SCAN_attention, self).__init__()
        self.fcq = nn.Linear(embed_size, embed_size)
        self.fck = nn.Linear(embed_size, embed_size)
        self.fcv = nn.Linear(embed_size, embed_size)


    def forward(self,query, context, smooth, eps=1e-8):
        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        sim_k = self.fck(context)
        sim_q = self.fcq(query)
        sim_v = context
        attn = torch.bmm(sim_k, sim_q.permute(0,2,1))

        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
        
        # --> (batch, queryL, sourceL)
        attn = F.softmax(attn.permute(0,2,1)*smooth, dim=2)
        
        # --> (batch, queryL, d)
        weightedContext = torch.bmm(attn, sim_v)
        weightedContext = l2norm(weightedContext, dim=-1)

        return weightedContext

def func_attention(query, context, smooth, focal_type = None, eps=1e-8, global_emb = None):
    """
    query: (batch, queryL, d)   ==  局部
    context: (batch, sourceL, d)    ==  局部
    opt: parameters
    """
    batch_size, queryL, sourceL = context.size(
        0), query.size(1), context.size(1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL) m张图片，n个单词 query 单词， source 图片
    attn = torch.bmm(context, queryT)     
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    # attn = attn.view(batch_size*queryL, sourceL)
    # attn = nn.Softmax(dim=1)(attn*20)
    # --> (batch, queryL, sourceL)
    # attn = attn.view(batch_size, queryL, sourceL)
    attn = F.softmax(attn*smooth, dim=2)

    ## 上面已经得到了attn
    ## BFAN提出使用一个评分函数 F 来识别与共享语义相关的局部特征，然后将不相关的局部特征从共享语义中去除掉。
    ## BFAN 给出了两种 g 函数的实现，一种实现是使用 √wit，这个方法被称为 prob, 另一种则是平等对待每一个出现的区域，这种方法被称为 equal。

    # Step 2: identify irrelevant fragments     识别不重要的片段    并标记，重要为1，不重要为0， equal和prob都是两种赋权的方法
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if focal_type == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif focal_type == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)

    ## 做了特征融合，得到新的特征为re_attn
    elif focal_type == 'glo':
        re_attn = focal_glo(attn, query, context, global_emb)
    else:
        raise ValueError("unknown focal attention type:", focal_type)
    
    # funcH += eps  BFAN的做法
    # Step 3: reassign attention
    if focal_type == 'equal' or focal_type == 'prob':
        tmp_attn = funcH * attn
        attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
        re_attn = tmp_attn / attn_sum

    ### 将融合得到的新的特征去做attn
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    
    
    weightedContext = l2norm(weightedContext, dim=-1)
    
    return weightedContext

def focal_glo(attn, query, source, glo):
    # Todo 加线性层
    query2glo = torch.bmm(query,glo.unsqueeze(-1))  ## 图像局部注意全局 ==  作为新的图像局部
    query2glo = F.softmax(query2glo*9.0,dim=2)
    
    source2glo = torch.bmm(source,glo.unsqueeze(-1))    ## 文本局部注意全局 ==  作为新的文本局部
    source2glo = F.softmax(source2glo*9.0, dim=2)
    
    source2gloT = source2glo.transpose(1,2).contiguous()
    funcF = torch.bmm(query2glo,source2gloT)       ##  再做新的图像区域和文本区域的权重矩阵 
    
    funcF = attn * funcF        ## 相当于做了特征融合
    # fattn = torch.where(funcF > 0, torch.ones_like(attn),
    #                     torch.zeros_like(attn))
    # return fattn
    # funcF = F.softmax(funcF, dim=2)
    return funcF



def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    #  torch.sum 得到的是注意力权重在当前文本上的和
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    # 
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt 
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size*queryL, sourceL, 1)
    xj = xj.view(batch_size*queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size*queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1-term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn




# class ContrastiveLoss(nn.Module):
#     """
#     Compute contrastive loss
#     """
#     def __init__(self, margin=0, max_violation=False):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.max_violation = max_violation

#     def forward(self, scores):
#         # compute image-sentence score matrix
#         diagonal = scores.diag().view(scores.size(0), 1)
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)

#         # compare every diagonal score to scores in its column
#         # caption retrieval
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#         # compare every diagonal score to scores in its row
#         # image retrieval
#         cost_im = (self.margin + scores - d2).clamp(min=0)

#         # clear diagonals
#         mask = torch.eye(scores.size(0)) > .5
#         if torch.cuda.is_available():
#             I = mask.cuda()
#         cost_s = cost_s.masked_fill_(I, 0)
#         cost_im = cost_im.masked_fill_(I, 0)

#         # keep the maximum violating negative for each query
#         if self.max_violation:
#             cost_s = cost_s.max(1)[0]
#             cost_im = cost_im.max(0)[0]
#         return cost_s.sum() + cost_im.sum()

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.CE = nn.CrossEntropyLoss()
        self.T = 0.05
    def forward(self, scores):
        batch_size = scores.size(0)
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        cost_s /= self.T
        cost_im /= self.T

        labels = torch.Tensor(list(range(batch_size))).long().cuda()

        return (self.CE(cost_im, labels) + self.CE(cost_s, labels)) / 2


class SGRAF(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)
        self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step,opt.focal_type)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            # cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens, Iou, depend):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens, Iou, depend)
        return sims

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, Iou,depend, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens, Iou, depend)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
