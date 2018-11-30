import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d


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
    def __init__(self,cfg=None):
        super(ConvNet, self).__init__()
        
        if cfg==None:
            cfg = [ 32, 64, 64 ]
        
        self.conv1 = MaskedConv2d(1, cfg[0], kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(cfg[0], cfg[1], kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(cfg[1], cfg[2], kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(7*7*cfg[2], 10)
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        
        out = self.relu3(self.conv3(out))
        #print('conv output shape ', out.shape)
        out = out.view(out.size(0), -1)
        #print('reshape shape ',out.shape)
        out = self.linear1(out)
        return out

