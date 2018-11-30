
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.utils import to_var, train, test
from models import ConvNet


# Hyper Parameters
param = {
    'pruning_perc': 50.,
    'batch_size': 128, 
    'test_batch_size': 100,
    'num_epochs': 1,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}
#param1 = {
#    'pruning_perc': 0.,
#    'batch_size': 128, 
#    'test_batch_size': 100,
#    'num_epochs': 7,
#    'learning_rate': 0.001,
#    'weight_decay': 5e-4,
#}


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
net = ConvNet()
net.load_state_dict(torch.load('models/convnet_pretrained1.pkl'))
#if torch.cuda.is_available():
#    print('CUDA ensabled.')
#    net.cuda()

# Pretraining
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(net.parameters(), lr=param1['learning_rate'], 
#                                weight_decay=param['weight_decay'])
#
#train(net, criterion, optimizer, param1, loader_train)

# Save and load the entire model
#torch.save(net.state_dict(), 'models/convnet_pretrained1.pkl')

print("--- Pretrained network loaded ---")
#test(net, loader_test)

from pruning.layers import MaskedLinear, MaskedConv2d
import numpy as np
# MaskedConv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# MaskedConv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# MaskedConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace)
# Linear(in_features=3136, out_features=10, bias=True)

default_cfg = [ 32, 64, 64 ] # 默认的 conv2d 配置
cfg = [ 8, 16, 32 ] # 剪枝后 conv2d 的配置

cfg_mask = []

layer_id = 0

for i, m in enumerate(net.modules()): # 遍历 每一层
    if i == 0: # 跳过第一层的 ConvNet 
        continue
    
    if isinstance(m, MaskedConv2d):
        out_channels = m.weight.data.shape[0]
        
        if out_channels == cfg[layer_id]:
            # [1,1,...,1] 全一, 这一层不进行减枝操作
            cfg_mask.append(torch.ones(out_channels))
            layer_id += 1
            continue
        
        weight_copy = m.weight.data.abs().clone()
        weight_copy = weight_copy.cpu().numpy()
        
        L1_norm = np.sum(weight_copy, axis=(1,2,3))
        
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
                   
new_net = ConvNet()        
# 输入是单通道
start_mask = torch.ones(1)

layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]

for [m0, m1] in zip(net.modules(), new_net.modules()):
    if isinstance(m0, MaskedConv2d):
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
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        # output_channels  
        w1 = w1[idx1.tolist(), :, :, :].clone()
        
        m1.weight.data = w1.clone()
        
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()        
        start_mask = end_mask
        layer_id_in_cfg += 1
        
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
             
    elif isinstance(m0, nn.Linear):
        print('linear layer ')
        if layer_id_in_cfg == len(cfg_mask):# 卷积层conv2d-> 全连接层fc
            print('conv -> linear ')
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            
            # 3136 个 0 [0,0,..,0]                           
            idx0_flatten = np.zeros(m0.weight.data.size(1),dtype='int32')
            # 长宽的乘积
            width_height = m0.weight.data.size(1) // default_cfg[-1]
            
            the_ones= np.ones(width_height,dtype='int32')
            
            for idx in idx0:
                start = idx * width_height                
                idx0_flatten[start : start + width_height] = the_ones
            
            idx0_new = np.squeeze(np.argwhere( idx0_flatten))
            
            if idx0_new.size == 1:
                idx0_new = np.resize(idx0_new, (1,))            
                
            # w.shape (output_channels, reshaped_inputs ) 
            # w (10, 3136=7*7*64) 
            # = (output_channels, size*size*input_channels)
            m1.weight.data = m0.weight.data[:, idx0_new.tolist()].clone()
            #m1.weight.data = m0.weight.data[:, idx0].clone()
                        
            m1.bias.data = m0.bias.data.clone()
            print('m1.weight.data shape ', m1.weight.data.size())
            layer_id_in_cfg += 1
            continue
        
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

num_parameters = sum([param.nelement() for param in new_net.parameters()])   

# prune the weights
#masks = filter_prune(net, param['pruning_perc'])
#net.set_masks(masks)
#print("--- {}% parameters pruned ---".format(param['pruning_perc']))
test(new_net, loader_test)


# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(new_net.parameters(), lr=param['learning_rate'], 
                                weight_decay=param['weight_decay'])

train(new_net, criterion, optimizer, param, loader_train)


# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
test(new_net, loader_test)



# Save and load the entire model
#torch.save(net.state_dict(), 'models/convnet_pruned.pkl')
import os
torch.save({'cfg': cfg, 'state_dict': new_net.state_dict()}, os.path.join('models', 'conv-pruned1.pth.tar'))

checkpoint = torch.load('models/conv-pruned1.pth.tar')
net2 = ConvNet(checkpoint['cfg'])
net2.load_state_dict(checkpoint['state_dict']) 

print(sum([param.nelement() for param in net2.parameters()]))
test(net2, loader_test)