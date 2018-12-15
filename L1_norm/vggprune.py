import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from models import vgg
from load_dataset import load_dataset

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--model', default='./logs/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        
        checkpoint = torch.load(args.model)
        
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        # 加载测试集 cifar10 或 cifar100
    test_loader = load_dataset(dataset=args.dataset,
                                train_batch_size=0, 
                                test_batch_size=args.test_batch_size,
                                kwargs=kwargs, train=False) 
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
             data = Variable(data)   
        target = Variable(target)     
        #data, target = Variable(data, volatile=True), Variable(target)
        
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)
# vgg剪枝后每一层卷积层的通道数channels         
cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]

cfg_mask = []
layer_id = 0

for m in model.modules():# 遍历 每一层
    
    if isinstance(m, nn.Conv2d): # 2d卷积
        
        out_channels = m.weight.data.shape[0]
        if out_channels == cfg[layer_id]:
            # [1,1,...,1] 全一, 这一层不进行减枝操作
            cfg_mask.append(torch.ones(out_channels))
            layer_id += 1
            continue
        
        weight_copy = m.weight.data.abs().clone()
        weight_copy = weight_copy.cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
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
        
    elif isinstance(m, nn.MaxPool2d):
        layer_id += 1


newmodel = vgg(dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()
# 输入3通道, RGB
start_mask = torch.ones(3)

layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]

for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        
        layer_id_in_cfg += 1
        start_mask = end_mask
        
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
            
    elif isinstance(m0, nn.Conv2d):
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
        
    elif isinstance(m0, nn.Linear):
        print('linear layer ')
        
        if layer_id_in_cfg == len(cfg_mask):
            
            print('conv -> linear ')
            
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
        
            m1.weight.data = m0.weight.data[:, idx0].clone()
            # torch.Size([512, 256])
            print('m1.weight.data shape ', m1.weight.data.size())
            m1.bias.data = m0.bias.data.clone()
            
            layer_id_in_cfg += 1
            continue
        m1.weight.data = m0.weight.data.clone()
        
        m1.bias.data = m0.bias.data.clone()
        
    elif isinstance(m0, nn.BatchNorm1d):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
print(newmodel)
model2 = newmodel
acc = test(model2)

num_parameters = sum([param.nelement() for param in model2.parameters()])
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")