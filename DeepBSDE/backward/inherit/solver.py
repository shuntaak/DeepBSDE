import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import copy

TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6

class Dense(nn.Module):

    def __init__(self,cin,cout, batch_norm=True, activate=True):
        super(Dense, self).__init__()
        self.cin=cin
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.activate = activate
        if batch_norm:
            self.bn = nn.BatchNorm1d(cout,eps=EPSILON, momentum=MOMENTUM)
        else:
            self.bn = None
        nn.init.normal_(self.linear.weight,std=5.0/np.sqrt(cin+cout))

    def forward(self,x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate:
            x = torch.tanh(x)
        return x


class Subnetwork_grad(nn.Module):

    def __init__(self, config):
        super(Subnetwork_grad, self).__init__()
        self._config = config
        self.bn = nn.BatchNorm1d(config.dim,eps=EPSILON, momentum=MOMENTUM)
        self.layers = [Dense(config.num_hiddens[i-1], config.num_hiddens[i],activate=True) for i in range(1, len(config.num_hiddens)-1)]
        self.layers += [Dense(config.num_hiddens[-2], config.num_hiddens[-1], activate=False)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.bn(x)
        x = self.layers(x)
        return x

class Subnetwork_value(nn.Module):

    def __init__(self, config):
        super(Subnetwork_value, self).__init__()
        self._config = config
        self.bn = nn.BatchNorm1d(config.dim,eps=EPSILON, momentum=MOMENTUM)
        self.layers = [Dense(config.num_hiddens[i-1], config.num_hiddens[i],activate=True) for i in range(1, len(config.num_hiddens)-1)]
        self.layers += [Dense(config.num_hiddens[-2], 1, activate=False)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.bn(x)
        x = self.layers(x)
        return x


class FeedForwardModel(nn.Module):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        super(FeedForwardModel, self).__init__()
        self._config = config
        self._bsde = bsde
        self._target = bsde.g_th # initialize target function(U^{(1)}=g, e.g.)
        self._t = bsde.num_time_interval-1

        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time

        # self._y_init = Parameter(torch.Tensor([1]))
        # self._y_init.data.uniform_(self._config.y_init_range[0], self._config.y_init_range[1])
        self._subnetwork_value = Subnetwork_value(config)
        self._subnetwork_grad = Subnetwork_grad(config)
        # self._subnetworkList =nn.ModuleList([Subnetwork(config) for _ in range(self._num_time_interval-1)])


    def forward(self, x, dw):

        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde._delta_t

        y = self._subnetwork_value(x[:, :, self._t])
        z = self._subnetwork_grad(x[:, :, self._t]) 
        #print('y qian', y.max())
        y = y - self._bsde._delta_t * (self._bsde.f_th(time_stamp[self._t], x[:, :, self._t], y, z))
        #print('y hou', y.max())
        add = torch.sum(z * dw[:, :, self._t], dim=1, keepdim=True)

        #print('add', add.max())
        y = y + add

        delta = y-self._target(x[:, :, self._t+1])
        loss = torch.mean(delta**2)
        return loss
    
    def eval_loss(self,x,dw):
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde._delta_t
        net1 = copy.deepcopy(self._subnetwork_value).cuda().eval()
        init = net1.forward(x[:,:,0])[0].item()
        del net1
        y = self._subnetwork_value(x[:, :, self._t])
        z = self._subnetwork_grad(x[:, :, self._t]) 
        #print('y qian', y.max())
        u = y - self._bsde._delta_t * (self._bsde.f_th(time_stamp[self._t], x[:, :, self._t], y, z))
        # print(torch.mean(torch.abs(y)).item(), torch.mean(torch.abs(z)).item())
        #print('y hou', y.max())
        add = torch.sum(z * dw[:, :, self._t], dim=1, keepdim=True)
        #print('add', add.max())
        u = u + add
        delta = u-self._target(x[:, :, self._t+1])
        # print(delta[0].item(),y[0].item(),x[:,:,self._t+1][0].item())
        loss = torch.mean(delta**2)
        # print(x[:,:,0][0].item(),x[:,:,self._t+1][0].item(),y[0].item())
        if self._t > 0:        
            return loss, init
        else:
            return loss, y[0].item()
    
    def target_update(self):
        del self._target
        # yrange = copy.deepcopy(init.item())
        # print(yrange)
        self._target = copy.deepcopy(self._subnetwork_value)
        # del self._subnetwork_value, self._subnetwork_grad
        for param in self._target.parameters():
            param.requires_grad = False    
        # self._y_init = Parameter(torch.Tensor([1]))
        # self._y_init.data.uniform_(self._config.y_init_range[0], self._config.y_init_range[1])
        # self._subnetwork_grad = Subnetwork_grad(self._config)
        # self._subnetwork_value = Subnetwork_value(self._config)
        self._t -= 1




