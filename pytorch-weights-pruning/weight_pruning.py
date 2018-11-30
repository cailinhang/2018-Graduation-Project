import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.utils import  train, test
from models import MLP


# Hyper Parameters
param = {
    'pruning_perc': 90., 'batch_size': 128,  'test_batch_size': 100,
    'num_epochs': 5, 'learning_rate': 0.001, 'weight_decay': 5e-4,
}


# Data loaders
train_dataset = datasets.MNIST(root='../data/',train=True, download=False, 
    transform=transforms.ToTensor())

loader_train = torch.utils.data.DataLoader(train_dataset, 
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST(root='../data/', train=False, download=False, 
    transform=transforms.ToTensor())

loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=param['test_batch_size'], shuffle=True)


# Load the pretrained model
net = MLP()
#net.load_state_dict(torch.load('models/mlp_pretrained1.pkl'))


# Pretraining
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
                                weight_decay=param['weight_decay'])

train(net, criterion, optimizer, param, loader_train)

# Save and load the entire model
#torch.save(net.state_dict(), 'models/mlp_pretrained1.pkl')

print('--- Pretrained network loaded ---')

test(net, loader_test)


from pruning.layers import MaskedLinear
import numpy as np
#1   MaskedLinear(in_features=784, out_features=200, bias=True)
#2   ReLU(inplace)
#3   MaskedLinear(in_features=200, out_features=200, bias=True)
#4   ReLU(inplace)
#5   MaskedLinear(in_features=200, out_features=10, bias=True)

cfg = [50,100,10]

cfg_mask = []

layer_id = 0

for i, m in enumerate(net.modules()): # 遍历 每一层
    if i == 0: # 跳过第一层的 MLP 
        continue
    if isinstance(m, MaskedLinear):
        
        out_channels = m.weight.data.shape[0]
        
        if out_channels == cfg[layer_id]:
            # [1,1,...,1] 全一, 这一层不进行减枝操作
            cfg_mask.append(torch.ones(out_channels))
            layer_id += 1
            continue
        
        weight_copy = m.weight.data.abs().clone()
        weight_copy = weight_copy.cpu().numpy()
        
        L1_norm = np.sum(weight_copy, axis=(1))
        
        if layer_id ==1:
            print('L1_norm.shape ', L1_norm.shape)
            
        # argsort 从小到大排列，提取其的index
        arg_max = np.argsort(L1_norm)
        # arg_max_rev 从大到小
        arg_max_rev = arg_max[::-1][:cfg[layer_id]]
        assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
        #[0, 0,..,0] 全零
        mask = torch.zeros(out_channels)
        #[0,0,,,1,,1,,0] 保留的结点的mask置为1 
        mask[arg_max_rev.tolist()] = 1
        
        cfg_mask.append(mask)
        layer_id += 1
           
new_net = net        
# 输入28*28=784通道
start_mask = torch.ones(784)

layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]

for [m0, m1] in zip(net.modules(), new_net.modules()):
             
    if isinstance(m0, MaskedLinear):
        # 返回所有非零数字的索引
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        # 返回所有非零数字的索引, 即保留的通道channels的下标
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        # 对filters进行裁剪
        # inputs_channels   
        w1 = m0.weight.data[:, idx0.tolist()].clone()
        # output_channels  
        w1 = w1[idx1.tolist(), :].clone()
        
        m1.weight.data = w1.clone()
        
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()        
        start_mask = end_mask
        layer_id_in_cfg += 1
        
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]

num_parameters = sum([param.nelement() for param in new_net.parameters()])            
import os

torch.save({'cfg': cfg, 'state_dict': new_net.state_dict()}, os.path.join('models', 'pruned1.pth.tar'))

# load the checkpoint
checkpoint = torch.load('models/pruned1.pth.tar')
net2 = MLP( checkpoint['cfg'])
net2.load_state_dict(checkpoint['state_dict'])


# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net2.parameters(), lr=param['learning_rate'], 
                                weight_decay=param['weight_decay'])

train(net2, criterion, optimizer, param, loader_train)


# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
test(net2, loader_test)


