import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d
import numpy as np

class MLP(nn.Module):
    def __init__(self,cfg=None):
        super(MLP, self).__init__()
        
        if cfg == None:
            cfg = [200, 200, 10 ]
                    
        self.linear1 = MaskedLinear(28*28, cfg[0])
        self.relu1 = nn.ReLU(inplace=True)
        
        self.linear2 = MaskedLinear(cfg[0], cfg[1])
        self.relu2 = nn.ReLU(inplace=True)
        
        self.linear3 = MaskedLinear(cfg[1], cfg[2])
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    
class ConvNet(nn.Module):
    def __init__(self,cfg=None, num_cnn_layer=None, part=1):
        super(ConvNet, self).__init__()
        
        self.part = int(part) # add 
        self.part = 1 #　普通卷积
        if cfg == None or num_cnn_layer == None:
            cfg = [1, 7, 11, 12, 10, 8, 7, 10]
            self.cfg = cfg.copy()
            self.num_cnn_layer = 4
        else:
            self.cfg = cfg.copy()
            self.num_cnn_layer = num_cnn_layer            
                
        layer_idx = 0
        for i in range( self.num_cnn_layer ):
            
            if i % self.part == 0:
                self.add_module(str(layer_idx), MaskedConv2d(in_channels = cfg[i], out_channels = cfg[i+1], kernel_size = 3, padding = 1))            
            else:
                self.add_module(str(layer_idx), MaskedConv2d(in_channels = cfg[i], out_channels = cfg[i+1], kernel_size = 1, padding = 0))   
            
            layer_idx += 1
            
            self.add_module(str(layer_idx), nn.ReLU(inplace=True))
            layer_idx += 1
            
            self.add_module(str(layer_idx), nn.MaxPool2d(3))
                
            layer_idx += 1
        
        for i in range( len(cfg)-1-self.num_cnn_layer-1 ):
            self.add_module(str(layer_idx), MaskedLinear(1*1*cfg[i + self.num_cnn_layer ], cfg[i + self.num_cnn_layer +1 ]))
            layer_idx += 1
            self.add_module(str(layer_idx), nn.ReLU(inplace=True))
            layer_idx += 1
        
        self.add_module(str(layer_idx), MaskedLinear(cfg[-2] , cfg[-1] ))
        layer_idx += 1

        
    def forward(self, x):
        l = list(self.children())
#        for i in range(len(l)):
#            print(i, ' ', l[i])
        out = x
        for i in range( self.num_cnn_layer ):
            
            out = l[i*3 + 0 ](out)
            out = l[i*3 + 1 ](out)
            out = l[i*3 + 2 ](out)
                                            
        #print('conv output shape ', out.shape)
        out = out.view(out.size(0), -1)
        #print('reshape shape ',out.shape)
                    
        for i in range( len(self.cfg)-1-self.num_cnn_layer-1 ):
            out = l[2*i +  self.num_cnn_layer*3](out)            
            out = l[2*i + 1 +  self.num_cnn_layer*3 ](out)
        
        out = l[-1](out)

        return out
    
    def forward_with_dropout(self, x):
        l = list(self.children())
        dropout = nn.Dropout(p=0.25)
        
        out = x
        for i in range( self.num_cnn_layer ):
            
            out = l[i*3 + 0 ](out)
            out = l[i*3 + 1 ](out)
            out = l[i*3 + 2 ](out)
                                            
        #print('conv output shape ', out.shape)
        out = out.view(out.size(0), -1)
        #print('reshape shape ',out.shape)
                    
        for i in range( len(self.cfg)-1-self.num_cnn_layer-1 ):
            out = dropout(out)       
            out = l[2*i +  self.num_cnn_layer*3](out)            
            out = l[2*i + 1 +  self.num_cnn_layer*3 ](out)
        
        out = dropout(out)       
        out = l[-1](out)

        return out
    
    def set_masks(self, masks):
        l = list(self.children())                        
        for i in range(self.num_cnn_layer):
            
            assert isinstance(l[3*i], MaskedConv2d)            
            l[3*i].set_mask(torch.from_numpy(masks[i]))
        
        for i in range( len(self.cfg)-1-self.num_cnn_layer):            
            l[3*self.num_cnn_layer + 2*i ].set_mask(torch.from_numpy(masks[i + self.num_cnn_layer]))                
        
#        self.conv1.set_mask(torch.from_numpy(masks[0]))
#        self.conv2.set_mask(torch.from_numpy(masks[1]))
#        self.conv3.set_mask(torch.from_numpy(masks[2]))

