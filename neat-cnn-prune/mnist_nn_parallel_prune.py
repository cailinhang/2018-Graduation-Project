import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from random import random

import neat1 as neat

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import evaluate_torch
import torchvision.datasets as datasets
import math

from numpy import random
from numpy.random import random, randn, rand
import fine_tune

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

trainset.train_data = trainset.train_data[:400]

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

def eval_genome(genome_id, genome, config, final_elitism_fitness):
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
    
    net = evaluate_torch.Net(config, genome)        
    
                                     
    state_dict = fine_tune.retrain(net.state_dict(), 20 + int(cur_generation/5))
    
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
                    
            net = evaluate_torch.Net(config, genome)
                    
            outputs = net.forward(inputs)
                                                   
            max_prob, predicted = torch.max(outputs.data, 1)
                                
            hit_count += (predicted == labels).sum().item()

        except Exception as e:
            print(e)
            genome.fitness = 0
        if (i == evaluate_batch_size - 1):
            break

    genome.fitness = hit_count / (evaluate_batch_size * torch_batch_size)
    
    # 如果 fitness 不能超过上一代中elitism的最后一名, 但差距不大,
    # 可以考虑剪枝 pruning + retrain
    if genome.fitness < final_elitism_fitness and \
        final_elitism_fitness - genome.fitness < 0.08: 
        
        if random() < 0.1:            
            #print('prune')
            
            net = evaluate_torch.Net(config, genome)      
            
            if random() > 0.5:
                del_node, del_connects, state = net.prune_one_filter()            
            else:
                del_node, del_connects, state = net.prune_fc_weight()
    
            state_dict = fine_tune.retrain(net.state_dict(), 25 )
    
            state_key = [ k for k,v in state_dict.items()]
         
            for idx, p in enumerate(net.parameters()):
                
                p.data = state_dict[ state_key[idx] ]
           
            net.write_back_parameters(genome)
            
            hit_count = 0        
            start = 0
            i = 0
            
            for num, data in enumerate(testloader, start):
                i += 1
        
                inputs, labels = data                        
                inputs, labels = Variable(inputs), Variable(labels)
                try:
                            
                    net = evaluate_torch.Net(config, genome)
                            
                    outputs = net.forward(inputs)
                                                                            
                    max_prob, predicted = torch.max(outputs.data, 1)
                                           
                    hit_count += (predicted == labels).sum().item()
        
                except Exception as e:
                    print(e)
                    genome.fitness = 0
                if (i == evaluate_batch_size - 1):
                    break
                
            genome.fitness = hit_count / (evaluate_batch_size * torch_batch_size)
                
    return genome
    #return genome.fitness
"""
def eval_genomes(genomes, config, final_elitism_fitness):
    print('final_elitism_fitness ', final_elitism_fitness)
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
        start = 0
        i = 0
        
        net = evaluate_torch.Net(config, genome)                
                                       
        state_dict = fine_tune.retrain(net.state_dict(), 15 + int(cur_generation/5) )
        
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
                        
                net = evaluate_torch.Net(config, genome)                
                        
                outputs = net.forward(inputs)
                                                                
                max_prob, predicted = torch.max(outputs.data, 1)
                #print(predicted)                        
                hit_count += (predicted == labels).sum().item()

            except Exception as e:
                print(e)
                genome.fitness = 0
            if (i == evaluate_batch_size - 1):
                break

        genome.fitness = hit_count / (evaluate_batch_size * torch_batch_size)
                                        
        print('{0}: {1:3.3f} {2}'.format(j,genome.fitness,genome_id))
        
        # 如果 fitness 不能超过上一代中elitism的最后一名, 但差距不大,
        # 可以考虑剪枝 pruning + retrain
        if genome.fitness < final_elitism_fitness and \
            final_elitism_fitness - genome.fitness < 0.05: 
            
            if random() < 0.2:            
                print('prune')
                
                net = evaluate_torch.Net(config, genome)      
                
                if random() > 0.5:
                    del_node, del_connects, state = net.prune_one_filter()            
                else:
                    del_node, del_connects, state = net.prune_fc_weight()
        
                state_dict = fine_tune.retrain(net.state_dict(), 25 )
        
                state_key = [ k for k,v in state_dict.items()]
             
                for idx, p in enumerate(net.parameters()):
                    
                    p.data = state_dict[ state_key[idx] ]
               
                net.write_back_parameters(genome)
                
                hit_count = 0        
                start = 0
                i = 0
                
                for num, data in enumerate(testloader, start):
                    i += 1
            
                    inputs, labels = data                        
                    inputs, labels = Variable(inputs), Variable(labels)
                    try:
                                
                        net = evaluate_torch.Net(config, genome)
                                
                        outputs = net.forward(inputs)
                                                                                
                        max_prob, predicted = torch.max(outputs.data, 1)
                                               
                        hit_count += (predicted == labels).sum().item()
            
                    except Exception as e:
                        print(e)
                        genome.fitness = 0
                    if (i == evaluate_batch_size - 1):
                        break
                    
                genome.fitness = hit_count / (evaluate_batch_size * torch_batch_size)
                print('{0}: {1:3.3f} {2}'.format(j,genome.fitness,genome_id))       
"""                    
# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')



if __name__ == '__main__':
    # Create the population, which is the top-level object for a NEAT run.
    checkpoint = neat.Checkpointer()
    #config1, init_state = checkpoint.restore_checkpoint('checkpoints/prune-neat-checkpoint-1')
    
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


