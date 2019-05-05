import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from random import random

import neat3 as neat

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import evaluate_torch_part
import torchvision.datasets as datasets
import math

from numpy import random
from numpy.random import random, randn, rand
import fine_tune_part

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

trainset.train_data = trainset.train_data[:40]

trainloader = torch.utils.data.DataLoader(trainset, 
    batch_size=torch_batch_size, shuffle=True)

testset = datasets.MNIST(root='../../../../../PyTorch/data/', train=False, download=False, 
    transform=transforms.ToTensor())

testset.test_data = testset.test_data[:4000]

testloader = torch.utils.data.DataLoader(testset, 
batch_size=torch_batch_size, shuffle=True)

has_evaled = {}
cur_generation = 0
eps = 1e-5

def eval_genome(genome_id, genome, config):
    if has_evaled.__contains__(genome_id):
        #print('{0}: {1:3.3f} {2}'.format(j,genome.fitness,genome_id))
        return genome
        #return genome.fitness
    else:
        has_evaled[genome_id] = 1               
    evaluate_batch_size = 400 * 2        
    hit_count = 0        
    start = 0
    i = 0
    #return 0.45
    state = True

    net = evaluate_torch_part.Net(config, genome)        
    
    if random() > 1.0:
        print('prune')
        if random() > 0.5:
            del_node, del_connects, state = net.prune_one_filter()
        
        else:
            del_node, del_connects, state = net.prune_fc_weight()
    
    else: state = False   
                                 
    state_dict = fine_tune_part.retrain(net.state_dict(), config.part_size, 2*int(state) + 20 + int(cur_generation/5))
    
    state_key = [ k for k,v in state_dict.items()]
     
    for idx, p in enumerate(net.parameters()):
        p.data = state_dict[ state_key[idx] ]
       
    net.write_back_parameters(genome)
    
    
    for num, data in enumerate(testloader, start):
        i += 1
        # 得到输入数据
        inputs, labels = data

        # 包装数据
        inputs, labels = Variable(inputs), Variable(labels)
        try:
                    
            net = evaluate_torch_part.Net(config, genome)
                    
            outputs = net.forward(inputs)
            
            #print(net)                
            #_, predicted = torch.max(outputs.data, 1)
                            
            max_prob, predicted = torch.max(outputs.data, 1)
            #print(predicted)                        
            hit_count += (predicted == labels).sum().item()

        except Exception as e:
            print(e)
            genome.fitness = 0
        if (i == evaluate_batch_size - 1):
            break

    genome.fitness = hit_count / (evaluate_batch_size * torch_batch_size)
    return genome
    #return genome.fitness

def eval_genomes(genomes, config):

    j = 0
    global cur_generation
    cur_generation += 1
    for genome_id, genome in genomes:
        j += 1
        if has_evaled.__contains__(genome_id):
            print('{0}: {1:3.3f} {2}'.format(j,genome.fitness,genome_id))
            continue
        else:
            has_evaled[genome_id] = 1

        evaluate_batch_size = 400 * 2
        
        hit_count = 0
        #start = int(random() * (len(trainloader) - evaluate_batch_size * torch_batch_size))
        start = 0
        i = 0
        #"""
        state = True

        net = evaluate_torch_part.Net(config, genome)        
        
        if random() > 1.0:
            print('prune')
            if random() > 0.5:
                del_node, del_connects, state = net.prune_one_filter()
            
            else:
                del_node, del_connects, state = net.prune_fc_weight()
        else: state = False                                        
        state_dict = fine_tune_part.retrain(net.state_dict(), config.part_size , 2*int(state) + 15 + int(cur_generation/5) )
        
        state_key = [ k for k,v in state_dict.items()]
         
        for idx, p in enumerate(net.parameters()):
            p.data = state_dict[ state_key[idx] ]

        net.write_back_parameters(genome)
    
    #fine_tune.retrain(net.state_dict(), 0)
        #"""
        for num, data in enumerate(testloader, start):
            i += 1
            # 得到输入数据
            inputs, labels = data

            # 包装数据
            inputs, labels = Variable(inputs), Variable(labels)
            try:
                        
                net = evaluate_torch_part.Net(config, genome)                
                        
                outputs = net.forward(inputs)
                
                #print(net)

                #_, predicted = torch.max(outputs.data, 1)
                                
                max_prob, predicted = torch.max(outputs.data, 1)
                #print(predicted)                        
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



if __name__ == '__main__':
    # Create the population, which is the top-level object for a NEAT run.
    checkpoint = neat.Checkpointer()
    #config1, init_state = checkpoint.restore_checkpoint('checkpoints/parallel-part-neat-checkpoint-19')
    
    p = neat.Population(config)
    #p = neat.Population(config1, init_state)
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    
    # Run until a solution is found.
    winner_list = p.run(eval_genome,300)
    winner = winner_list[0]
    
    for i in range(len(winner_list)):
        
        if winner[1] < winner_list[i][1]:
            winner = winner_list[i]
    
    winner = winner[0]            


