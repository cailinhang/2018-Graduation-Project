import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.utils import train, test
from models import ConvNet
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

def load_dataset(batch_size=4, test_batch_size=50):    
    # Data loaders
    train_dataset = datasets.MNIST(root='../../../../../PyTorch/data/',train=True, download=False, 
        transform=transforms.ToTensor())
    
    train_dataset.train_data = train_dataset.train_data[:400]
    
    loader_train = torch.utils.data.DataLoader(train_dataset, 
        batch_size=batch_size, shuffle=False)
    
    test_dataset = datasets.MNIST(root='../../../../../PyTorch/data/', train=False, download=False, 
        transform=transforms.ToTensor())
    
    test_dataset.test_data = test_dataset.test_data[:400] 
    
    loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=test_batch_size, shuffle=False)
    
    return loader_train, loader_test


def retrain(state_dict, num_epochs=5):
    
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
    
    net = ConvNet(cfg, num_cnn_layer)

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
                        
    net.set_masks(masks)           
    
    ## Retraining    
    loader_train, loader_test = load_dataset()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
                                    weight_decay=param['weight_decay'])
    if num_epochs > 0:
        test(net, loader_test)
    
    train(net, criterion, optimizer, param, loader_train)
    
    for i, p in enumerate(net.parameters()):
        
        state_dict[ state_key[i] ] = p.data
        #print(p.data == state_dict[ state_key[i] ])
    
    #print("--- After retraining ---")
    test(net, loader_test)
    
    
    #return net.state_dict()
    return state_dict
    

#model_path = './models/1.pth.tar'
#checkpoint = torch.load(model_path)        
#
#state_dict = checkpoint['state_dict']
#
#
#num_cnn_layer =sum( [ int(len(v.size())==4) for d, v in state_dict.items() ] )        
#
#num_fc_layer = sum( [ int(len(v.size())==2) for d, v in state_dict.items() ] ) 
#
#state_key = [ k for k,v in state_dict.items()]

#cfg = [7, 11, 12, 10, 8, 7, 10]

#new_cfg = []
#first = True
#for d, v in state_dict.items():
#    print(v.data.size())    
#    if len(v.data.size()) == 4 or len(v.data.size()) ==2:
#        if first:
#            first = False
#            new_cfg.append(v.data.size()[1]) 
#        new_cfg.append(v.data.size()[0])
#
#cfg = new_cfg        
##assert new_cfg == cfg
#assert num_cnn_layer + num_fc_layer == len(cfg) - 1
#
#net = ConvNet(cfg, num_cnn_layer)

#net.load_state_dict(checkpoint['state_dict'])

#save = './models'

#torch.save({'cfg': cfg, 'num_cnn_layer': config.num_cnn_layer, 'state_dict': net3.state_dict()}, os.path.join('./models', '1.pth.tar'))

#masks = []
#
#for i, p in enumerate(net.parameters()):
#    
#    p.data = state_dict[ state_key[i] ]
#    
#    if len(p.data.size()) == 4:
#        
#        p_np = p.data.cpu().numpy()
#        
#        masks.append(np.ones(p_np.shape).astype('float32'))
#                
#        value_this_layer = np.abs(p_np).sum(axis=(2,3))        
#        
#        zero_list = []
#        
#        for j in range(len(value_this_layer)):
#            
#            for k in range(len(value_this_layer[0])):
#                
#                if abs( value_this_layer[j][k] ) < 1e-4:
#                
#                    masks[-1][j][k] = 0.
#                    
#net.set_masks(masks)                                

#net3 = evaluate_torch.Net(config, winner_list[13][0])

#conv_list = []
#linear_list = []
#
#for m in net3.modules():
#    if isinstance(m , nn.Conv2d):
#        print(m)
#        conv_list.append(m)
#    elif isinstance(m, nn.Linear):        
#        linear_list.append(m)
#
#idx = 0
#for p in net.parameters():
#    if len(p.data.size()) == 4:
#        print(p.data.size())
#        p.data = conv_list[idx].weight.data
#        
#    elif len(p.data.size()) == 1:        
#        p.data = conv_list[idx].bias.data
#        print(p.data.size())
#        idx += 1
#        if idx == 4:
#            break
#
#idx = 0
#for p in net.parameters():
#    
#    if len(p.data.size()) == 2 or len(p.data.size()) == 4:
#        print(p.data.size())
#        if idx >= 4:
#            p.data = linear_list[idx - 4 ].weight.data
#        idx += 1
#        print(idx)
#        
#    elif len(p.data.size()) == 1:          
#        if idx <= 4:
#            continue     
#        print(p.data.size()) 
#        print(idx)
#        p.data = linear_list[idx - 5 ].bias.data


#loader_train, loader_test = load_dataset()
#
### Retraining
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
#                                weight_decay=param['weight_decay'])
#
#
#train(net, criterion, optimizer, param, loader_train)
#
#print("--- After retraining ---")
#test(net, loader_test)
#
## save the net
#save = './models'
#torch.save({'state_dict': net.state_dict()}, os.path.join('./models', '1-retrain.pth.tar'))
#
