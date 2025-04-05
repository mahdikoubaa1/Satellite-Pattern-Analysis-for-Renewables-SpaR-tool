'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
from collections import OrderedDict
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam
from models.losses import triplet_loss, focal_loss
import torch
class resblk(nn.Module):
    def __init__(self,inp):
        super().__init__()

        self.beforeskip=nn.Sequential(nn.Conv2d(inp,inp,kernel_size=1),
                                      nn.BatchNorm2d(inp),
                                      nn.ReLU(),
                                      nn.Conv2d(inp,inp,kernel_size=1),
                                      nn.BatchNorm2d(inp))
        self.afterskip=nn.ReLU()
    def forward(self, x):
        x1= self.beforeskip(x)
        x=torch.add(x,x1)
        x= self.afterskip(x)
        return x

class contrastive(nn.Module):
    def __init__(self,bands,clayers=[[16,0],[16,0],[5,0]], clr=1e-2,alpha=500):
        super().__init__()
        layers_list= []
        if len(clayers)==0 :
            layers_list.append(('cconv0',nn.Conv2d(bands,2, kernel_size=1)))
            layers_list.append(('cbn0',nn.BatchNorm2d(2)))
        else:
            for i in range (len(clayers)):
                if i==0 :
                    layers_list.append (('cconv0',nn.Conv2d(bands,clayers[0][0], kernel_size=1)))
                    layers_list.append (('cbn0',nn.BatchNorm2d(clayers[0][0])))
                    
                    layers_list.append (('cr0',nn.ReLU()))
                else:
                    for j in range(clayers[i-1][1]):
                        layers_list.append((f'crblk{i-1}-{j}',resblk(clayers[i-1][0])))
                        
                        
                    layers_list.append ((f'cconv{i}',nn.Conv2d(clayers[i-1][0],clayers[i][0], kernel_size=1)))
                    layers_list.append ((f'cbn{i}',nn.BatchNorm2d(clayers[i][0])))
                    layers_list.append ((f'cr{i}',nn.ReLU()))
                if i== (len(clayers)-1):
                    for j in range(clayers[i][1]):
                        layers_list.append((f'crblk{i}-{j}',resblk(clayers[i][0])))
                    
        self.model=nn.Sequential(OrderedDict(layers_list))            
        #self.model=nn.Sequential(
        #                         nn.Conv2d(bands, 512, kernel_size=1),
        #                         nn.BatchNorm2d(512),
        #                         nn.ReLU(),
        #                        
        #                         nn.Conv2d(512, 512, kernel_size=1),
        #                         nn.BatchNorm2d(512),
        #                         nn.ReLU(),
#
        #                        
        #                         nn.Conv2d(512, 16, kernel_size=1),
        #                         nn.BatchNorm2d(16),
        #                         nn.ReLU(),
        #                         nn.Conv2d(16, 2, kernel_size=1),
        #                         nn.BatchNorm2d(2)
#
        #                         )
        
        self.optimizer= Adam(params=self.model.parameters(), lr=clr)

        self.loss=triplet_loss(alpha=alpha)
    def forward(self, x):
        x=self.model(x)
        return x
class pixelwise(nn.Module):
    def __init__(self,bands,layers=[[512,0],[512,0],[16,0]],clayers=[[16,0],[16,0],[5,0]], lr=1e-2, clr=1e-2, alpha=500 ,gamma=0.):
        super().__init__()
        layers_list= []
        self.cont=contrastive(bands=bands,clayers=clayers,clr=clr,alpha=alpha)
        #layers_list.append(('cont0',self.cont))
        if len(layers)==0 :
            layers_list.append(('conv0',nn.Conv2d(bands,2, kernel_size=1)))
            layers_list.append(('bn0',nn.BatchNorm2d(2)))
        else:
            h=0
            for i in range (len(layers)):
                if i==0 :
                    layers_list.append (('conv0',nn.Conv2d(bands,layers[0][0], kernel_size=1)))
                    layers_list.append (('bn0',nn.BatchNorm2d(layers[0][0])))
                    
                    layers_list.append (('r0',nn.ReLU()))
                else:
                    for j in range(layers[i-1][1]):
                        layers_list.append((f'rblk{i-1}-{j}',resblk(layers[i-1][0])))
                        
                        
                    layers_list.append ((f'conv{i}',nn.Conv2d(layers[i-1][0],layers[i][0], kernel_size=1)))
                    layers_list.append ((f'bn{i}',nn.BatchNorm2d(layers[i][0])))
                    layers_list.append ((f'r{i}',nn.ReLU()))
                if i== (len(layers)-1):
                    for j in range(layers[i][1]):
                        layers_list.append((f'rblk{i}-{j}',resblk(layers[i][0])))
                        
                    layers_list.append ((f'conv{i+1}',nn.Conv2d(layers[i][0],2, kernel_size=1)))
                    layers_list.append ((f'bn{i+1}',nn.BatchNorm2d(2)))
        self.model=nn.Sequential(OrderedDict(layers_list))            
        #self.model=nn.Sequential(
        #                         nn.Conv2d(bands, 512, kernel_size=1),
        #                         nn.BatchNorm2d(512),
        #                         nn.ReLU(),
        #                        
        #                         nn.Conv2d(512, 512, kernel_size=1),
        #                         nn.BatchNorm2d(512),
        #                         nn.ReLU(),
#
        #                        
        #                         nn.Conv2d(512, 16, kernel_size=1),
        #                         nn.BatchNorm2d(16),
        #                         nn.ReLU(),
        #                         nn.Conv2d(16, 2, kernel_size=1),
        #                         nn.BatchNorm2d(2)
#
        #                         )
        
        self.optimizer= Adam(params=self.model.parameters(), lr=lr)

        self.loss=focal_loss(gamma=gamma)
    def forward(self, x):
        x=self.model(x)
        return x
