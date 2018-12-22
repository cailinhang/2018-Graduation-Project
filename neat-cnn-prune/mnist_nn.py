import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from random import random

# from __future__ import print_function
import neat1 as neat

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import evaluate_torch
import torchvision.datasets as datasets
import math
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

torch_batch_size = 5

#trainset = torchvision.datasets.CIFAR10(root='../../../../../PyTorch/rethinking-network-pruning-master/cifar/l1-norm-pruning/data.cifar10', train=True,
#                                        download=False, transform=transform)
#trainset.train_data = trainset.train_data[:200]
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=torch_batch_size,
#                                          shuffle=True, num_workers=0)
#
#testset = torchvision.datasets.CIFAR10(root='../../../../../PyTorch/rethinking-network-pruning-master/cifar/l1-norm-pruning/data.cifar10', train=False,
#                                       download=False, transform=transform)
#testset.test_data = testset.test_data[:300]
#testloader = torch.utils.data.DataLoader(testset, batch_size=torch_batch_size,
#                                         shuffle=False, num_workers=0)


# mnist Data loaders
trainset = datasets.MNIST(root='../../../../../PyTorch/data/',train=True, download=False, 
    transform=transforms.ToTensor())

trainset.train_data = trainset.train_data[:100]

trainloader = torch.utils.data.DataLoader(trainset, 
    batch_size=torch_batch_size, shuffle=True)

testset = datasets.MNIST(root='../../../../../PyTorch/data/', train=False, download=False, 
    transform=transforms.ToTensor())

testset.test_data = testset.test_data[:400]

testloader = torch.utils.data.DataLoader(testset, 
batch_size=torch_batch_size, shuffle=True)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
eps = 1e-5
def eval_genomes(genomes, config):

    j = 0
    for genome_id, genome in genomes:
        j += 1
#        if genome.nodes.__contains__(301):
#            print('genome in_nodes ', genome.nodes[301].in_nodes)
#        else: continue             
        #evaluate_batch_size = 1000
        evaluate_batch_size = 20
        val = 0
        hit_count = 0
        #start = int(random() * (len(trainloader) - evaluate_batch_size * torch_batch_size))
        start = 0
        i = 0
        
        for num, data in enumerate(trainloader, start):
            i += 1
            # 得到输入数据
            inputs, labels = data

            # 包装数据
            inputs, labels = Variable(inputs), Variable(labels)
            try:

                net = evaluate_torch.Net(config, genome)
                outputs = net.forward(inputs)
                #print(net)

                #_, predicted = torch.max(outputs.data, 1)
                                
                max_prob, predicted = torch.max(outputs.data, 1)
                #print(predicted)
                #print('max_prob', max_prob)
#                for j in range(len(labels)):
#                    if predicted[j] == labels[j]:
#                       val += math.log(max_prob[j] + eps) 
#                    else:
#                        val += math.log(1 - max_prob[j] + eps)
                    
                hit_count += (predicted == labels).sum().item()

            except Exception as e:
                print(e)
                genome.fitness = 0
            if (i == evaluate_batch_size - 1):
                break

        genome.fitness = hit_count / (evaluate_batch_size * torch_batch_size)
        #genome.fitness = val
        print('{0}: {1:3.3f} {2}'.format(j,genome.fitness,genome_id))

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')

# reset result file
#res = open("result.csv", "w")
#best = open("best.txt", "w")
#res.close()
#best.close()

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner_list = p.run(eval_genomes,1)
winner = winner_list[0]

for i in range(len(winner_list)):
    
    if winner[1] < winner_list[i][1]:
        winner = winner_list[i]

winner = winner[0]        
# Run for up to 300 generations.
# pe = neat.ThreadedEvaluator(4, eval_genomes)
# winner = p.run(pe.evaluate)
# pe.stop()

# Display the winning genome.
#print('\nBest genome:\n{!s}'.format(winner))
s = '\nBest genome:\n{!s}'.format(winner)

#print('winner.size()', winner.size())

correct = 0
total = 0
net = evaluate_torch.Net(config, winner)

#for m in net.children():
#    print(m.weight.data.size())


for data in testloader:
    images, labels = data
    # output.size() [4, 10] 4个样本，10类
    outputs = net.forward(images)
    # 0表示跨行保留每一列，取每一列的max值
    # predicted是 每一列最大值在outputs.data的行下标
    predicted = torch.max(outputs.data, 1)[1]
    correct += (predicted == labels).sum().item()


print("hit %d of %d"%(correct, len(testset)))

num_parameter = sum([param.nelement() for param in net.parameters()])

#for i, param in enumerate(net.parameters()):
#    print(i, ' ' , param.size())

# 画图
import visualize
node_names = {-1:'A', -2: 'B', -3:'C' ,0:'0',
              1:'1',2:'2', 3:'3', 4:'4', 5:"5",
              6:'6', 7:'7', 8:'8', 9:'9'}

visualize.draw_net(config, winner, True, node_names=node_names)

# network pruning

prune_percent = 0.2
print('num_nodes ', len(net.old_nodes))

pruned_time = int (len(net.old_nodes) * prune_percent)

for i in range(pruned_time):
        
    del_node, del_connects, state = net.prune_one_filter()
    if state == False:        
        break
    
    del_node, del_connects, state = net.prune_fc_weight()
    if state == False:
        break
#    if winner.nodes.__contains__(del_node):
#        
#        del winner.nodes[del_node]
#    else:
#        print(del_node , ' already deleted ')
#    for del_con in del_connects:
#        del winner.connections[del_con]
        
    correct = 0
    
    for data in testloader:
        images, labels = data
        # output.size() [4, 10] 4个样本，10类
        outputs = net.forward(images)
        # 0表示跨行保留每一列，取每一列的max值
        # predicted是 每一列最大值在outputs.data的行下标
        predicted = torch.max(outputs.data, 1)[1]
        correct += (predicted == labels).sum().item()


    print("%d time, hit %d of %d"%(i, correct, len(testset)))
#
s1 ='\nBest genome:\n{!s}'.format(winner)
num_parameter1 = sum([param.nelement() for param in net.parameters()])

print('------test net -------')
correct = 0

for data in testloader:
    images, labels = data
    # output.size() [4, 10] 4个样本，10类
    outputs = net.forward(images)
    # 0表示跨行保留每一列，取每一列的max值
    # predicted是 每一列最大值在outputs.data的行下标
    predicted = torch.max(outputs.data, 1)[1]
    correct += (predicted == labels).sum().item()

print("hit %d of %d"%(correct, len(testset)))

# 剪枝后的winner重新生成网络net2
print('----rebuild winner network net2-----')
net2 = evaluate_torch.Net(config, winner)

correct = 0

for data in testloader:
    images, labels = data
    # output.size() [4, 10] 4个样本，10类
    outputs = net2.forward(images)
    # 0表示跨行保留每一列，取每一列的max值
    # predicted是 每一列最大值在outputs.data的行下标
    predicted = torch.max(outputs.data, 1)[1]
    correct += (predicted == labels).sum().item()

print("hit %d of %d"%(correct, len(testset)))

#TODO: wirte model to pytorch files

#node_names = {# -28: '-28', -27: '-27', -26: '-26', -25: '-25', -24: '-24', -23: '-23', -22: '-22', -21: '-21',
#              # -20: '-20', -19: '-19', -18: '-18', -17: '-17', -16: '-16', -15: '-15', -14: '-14', -13: '-13',
#              # -12: '-12', -11: '-11', -10: '-10', -9: '-09', -8: '-08', -7: '-07', -6: '-06',
#              -5: '-05', -4: '-04', -3: '-03', -2: '-02', -1: '-01', 0: '00', 1: '01', 2: '02',
#              3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09'}

# visualize.draw_net(config, winner, True, node_names=node_names)