#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time   : 2020/9/4 0004$
# @Author : szj


#步骤1∶准备输入
import torch
import numpy as np
import scipy.io as sio

dataset = r'C:\Users\Administrator\Desktop\szj\paper3\experiments\SPF-GZSL-master\data\CUB'

sysfea = 'middle_embed_sae_cub'

# 读取mat数据
sysfea_Mat = sio.loadmat(dataset + "/" + sysfea + ".mat")

### 属性
Attribute = sysfea_Mat['middle_embed_sae_cub'].T #选取mat文件中的变量

NUM_OF_CLASS,NUM_OF_MIDDLE_LAYER = Attribute.shape

Attribute_all = Attribute

## 属性向量归一化
a_max = Attribute_all.max()
a_min = Attribute_all.min()
Attribute_all = (Attribute_all-a_min)/(a_max - a_min) #312*200

alpha = 0.5 #需要调整

x = torch.tensor(Attribute_all, dtype=torch.float32)
x = np.transpose(x)

#步骤2∶初始化权重
w_key = np.random.randn(312, 312) #CUB下的参数
w_query = np.random.randn(312, 312)
w_value = np.random.randn(312, 312)

w_key = torch.tensor(w_key, dtype=torch.float32)
w_query =torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

#步骤3:推导键、查询和值
keys =torch.mm(x, w_key)
querys = torch.mm(x, w_query)
values = torch.mm(x, w_value)


#步骤4∶计算注意力得分
attn_scores = torch.mm(querys, np.transpose(keys))

#步骤5∶计算softmax
from torch.nn.functional import softmax
attn_scores_softmax = softmax(attn_scores, dim=-1)

#步骤6∶将得分和值相乘
weighted_values = values[:, None] * np.transpose(attn_scores_softmax)[:, :, None]


# 步骤7∶求和加权值
weighted_values = torch.tensor(weighted_values, dtype=torch.float32)
output = torch.sum(weighted_values, 0)
middle_embed_am = alpha * output + (1-alpha) * x
# print(middle_embed_am)
np.save(r'C:\Users\Administrator\Desktop\szj\paper3\experiments\SPF-GZSL-master\data\CUB\middle_embed_am_cub.npy', middle_embed_am)
# 得到的数据放入MATLAB中转换为mat文件，并提取unseen对应的行