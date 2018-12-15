import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from models import resnet
from load_dataset import load_dataset

# Pruning resnet settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=14,
                    help='depth of the resnet')
parser.add_argument('--model', default='./logs/checkpoint-res14.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save pruned model (default: ./logs)')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet(depth=args.depth, dataset=args.dataset)

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


prune_prob = {
    'A': [0.2, 0.2, 0.2],
    'B': [0.6, 0.3, 0.1],
}

layer_id = 1
cfg = []
cfg_mask = []

for m in model.modules():
    
    if isinstance(m, nn.Conv2d):
        
        out_channels = m.weight.data.shape[0]
                
        if layer_id % 2 == 0:
            # stage 层数layer影响剪枝的概率
            if layer_id <= 4:
                stage = 0
            elif layer_id <= 8:
                stage = 1
            else:
                stage = 2
            # 默认v = 'A', 即剪枝概率不随层数变化                
            prune_prob_stage = prune_prob[args.v][stage]
            
            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            
            L1_norm = np.sum(weight_copy, axis=(1,2,3))
            
            num_keep = int(out_channels * (1 - prune_prob_stage))
            
            arg_max = np.argsort(L1_norm)
            
            arg_max_rev = arg_max[::-1][:num_keep]
            
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            
            cfg_mask.append(mask)
            cfg.append(num_keep)
            layer_id += 1
            continue
        
        layer_id += 1

newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)

if args.cuda:
    newmodel.cuda()
# 输入 3通道
start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1

for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        
        if conv_count % 2 == 0:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
        
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue
        
    elif isinstance(m0, nn.BatchNorm2d):
        if conv_count % 2 == 1:
            # Conv2d -> BatchNorm2d -> Relu -> Conv2d 
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue
        # Relu -> Conv2d -> BatchNorm2d  
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
        
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned-res14.pth.tar'))

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
print(newmodel)
model2 = newmodel
acc = test(model2)

print("number of parameters: "+str(num_parameters))
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")