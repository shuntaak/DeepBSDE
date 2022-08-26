import numpy as np
import torch
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_th(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_th(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")

class AllenCahn(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(AllenCahn, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        return y - torch.pow(y, 3)

    def g_th(self, t, x):
        return 0.5 / (1 + 0.2 * torch.sum(x**2, dim=1, keepdim=True))

# class AllenCahn(Equation):
#     def __init__(self, dim, total_time, num_time_interval):
#         super(AllenCahn, self).__init__(dim, total_time, num_time_interval)
#         # self._x_init = np.zeros(self._dim)
#         self._sigma = np.sqrt(2.0)

#     def sample(self, num_sample):
#         dw_sample = normal.rvs(size=[num_sample,
#                                      self._dim,
#                                      self._num_time_interval]) * self._sqrt_delta_t
#         x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
#         x_sample[:, :, 0] =  normal.rvs(size=[num_sample,
#                                          self._dim, 1])*0.+100
#         x_sample[0, :, 0] =  np.ones(self._dim) * 100.
#         for i in range(self._num_time_interval):
#             x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
#         return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

#     def f_th(self, t, x, y, z):
#         return y - torch.pow(y, 3)
    
#     def g_th(self, t, x):
#         return 0.5 / (1 + 0.2 * torch.sum(x**2, dim=1, keepdim=True))

class PricingDefaultRisk(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(PricingDefaultRisk, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones(self._dim)*100.
        self._sigma = 0.2
        self._delta = 2/3
        self._R = 0.02
        self._mu = 0.02
        self._vh = 50
        self._vl = 70
        self._gh = 0.2
        self._gl = 0.02

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:,:,i] + self._mu*x_sample[:,:,i]*self._delta_t + self._sigma*x_sample[:,:,i]* dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def Q_th(self, t, u):
        q = self._gh*(u<self._vh)+self._gl*(u >= self._vl)+((self._gh-self._gl)*(u-self._vh)/(self._vh-self._vl)+self._gh)*(u >= self._vh)*(u < self._vl)
        return q
        
    def f_th(self, t, x, u, z):
        return -(1-self._delta)*self.Q_th(t,u)*u-self._R*u

    def g_th(self, t, x):
        return torch.min(x, dim=1,keepdim=True).values
        
class HJB(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(HJB, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones(self._dim)*0.
        self._sigma = np.sqrt(2.0)
        self._lam = 1.0

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)
        
    def f_th(self, t, x, u, z):
        return -self._lam*torch.sum((z/self._sigma)**2, dim=1, keepdim=True)

    def g_th(self, t, x):
        return torch.log((1+torch.sum(x**2, dim=1, keepdim=True))/2.)

class HJB1(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(HJB1, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2.0)
        self._lam = 1.0

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] +self._sigma*dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)
        
    def f_th(self, t, x, u, z):
        return -self._lam*torch.sum((z/self._sigma)**2, dim=1, keepdim=True)

    def g_th(self, t, x):
        return torch.sum(x**2, dim=1, keepdim=True)/10.
    
class BS(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(BS, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones([self._dim])
        self._sigma = np.sqrt(2.0)
        self._mu = 1

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]).reshape([num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._mu * x_sample[:,:,i] * self._delta_t + self._sigma * dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)
        
    def f_th(self, t, x, u, z):
        return -self._mu*u

    def g_th(self, t,x):
        return (x-1)*(1+torch.sign(x-1))/2.
    
class Merton(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(Merton, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones([self._dim])
        self._sigma = np.sqrt(2.0)
        self._mu = 0.5
        self._r = 0.2
        self._gamma = 0.5
        self._u = (self._mu-self._r)/(1-self._gamma)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]).reshape([num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] 
            + (self._r+(self._mu-self._r)*self._u)*x_sample[:,:,i]*self._delta_t
            + self._sigma * self._u*x_sample[:,:,i]*dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)
        
    def f_th(self, t, x, u, z):
        return 0

    def g_th(self, t,x):
        return torch.pow(x,self._gamma)