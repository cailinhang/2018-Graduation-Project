import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.utils import train, test
from models_part import ConvNet
import numpy as np
import os

# Hyper Parameters
#param = {    
#    'batch_size': 4, 
#    'test_batch_size': 50,
#    'num_epochs': 20,
#    'learning_rate': 0.001,
#    'weight_decay': 5e-4,
#}

def load_dataset(batch_size=32, test_batch_size=100):    
    # Data loaders
    train_dataset = datasets.MNIST(root='../../../../../PyTorch/data/',train=True, download=False, 
        transform=transforms.ToTensor())
    
    train_dataset.train_data = train_dataset.train_data[:4000]
    
    loader_train = torch.utils.data.DataLoader(train_dataset, 
        batch_size=batch_size, shuffle=False)
    
    test_dataset = datasets.MNIST(root='../../../../../PyTorch/data/', train=False, download=False, 
        transform=transforms.ToTensor())
    
    test_dataset.test_data = test_dataset.test_data[:100] 
    
    loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=test_batch_size, shuffle=False)
    
    return loader_train, loader_test


def retrain(state_dict, part=1, num_epochs=5):
    
    # Hyper Parameters
    param = {    
        'batch_size': 4, 
        'test_batch_size': 50,
        'num_epochs': num_epochs,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
    }
    
    num_cnn_layer =sum( [ int(len(v.size())==4) for d, v in state_dict.items() ] )        

    num_fc_layer = sum( [ int(len(v.size())==2) for d, v in state_dict.items() ] ) 
    
    state_key = [ k for k,v in state_dict.items()]
    
    
    cfg = []
    first = True
    for d, v in state_dict.items():
        #print(v.data.size())    
        if len(v.data.size()) == 4 or len(v.data.size()) ==2:
            if first:
                first = False
                cfg.append(v.data.size()[1]) 
            cfg.append(v.data.size()[0])
    

    assert num_cnn_layer + num_fc_layer == len(cfg) - 1
    
    net = ConvNet(cfg, num_cnn_layer, part)
#    l = list(net.children())
#    for i in range(len(l)):
#        print(i,' ', l[i] )
    masks = []

    for i, p in enumerate(net.parameters()):
        
        p.data = state_dict[ state_key[i] ]
        
        if len(p.data.size()) == 4:
            
            p_np = p.data.cpu().numpy()
            
            masks.append(np.ones(p_np.shape).astype('float32'))
                    
            value_this_layer = np.abs(p_np).sum(axis=(2,3))        
                                    
            for j in range(len(value_this_layer)):
                
                for k in range(len(value_this_layer[0])):
                    
                    if abs( value_this_layer[j][k] ) < 1e-4:
                    
                        masks[-1][j][k] = 0.
                        
        elif len(p.data.size()) == 2:
            
            p_np = p.data.cpu().numpy()
            
            masks.append(np.ones(p_np.shape).astype('float32'))
                    
            value_this_layer = np.abs(p_np)   
                                    
            for j in range(len(value_this_layer)):
                
                for k in range(len(value_this_layer[0])):
                    
                    if abs( value_this_layer[j][k] ) < 1e-4:
                    
                        masks[-1][j][k] = 0.                                        
#    for i in range(len(masks)):
#        print(len(masks[i]), ' ' , len(masks[i][0]))
    net.set_masks(masks)           
    
    ## Retraining    
    loader_train, loader_test = load_dataset()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
                                    weight_decay=param['weight_decay'])
    #if num_epochs > 0:
    #    test(net, loader_test)
    
    train(net, criterion, optimizer, param, loader_train)
    
    for i, p in enumerate(net.parameters()):
        
        state_dict[ state_key[i] ] = p.data
        #print(p.data == state_dict[ state_key[i] ])
    
    #print("--- After retraining ---")
    #test(net, loader_test)
    
    
    #return net.state_dict()
    return state_dict
    

## Retraining    
#loader_train, loader_test = load_dataset()
#
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
#                                weight_decay=param['weight_decay'])
#
#test_acc_list = []
#
#num_epochs = 500
#for t in range(num_epochs ):
#    
#    param['num_epochs'] = 1
#    train(net, criterion, optimizer, param, loader_train)
#
#    #print("--- After retraining ---")
#    
#    test_acc_list.append(test(net, loader_test))
#
#
#import matplotlib.pyplot as plt
#plt.plot(test_acc_list)

## save the net
#save = './models'
#torch.save({'state_dict': net.state_dict()}, os.path.join('./models', '1-retrain.pth.tar'))
#
