import numpy as np
from common.functions import softmax, cross_entropy
from common.util import *

# Activation function layer
# ReLu layer
class Relu:
    def __init__(self):
        self.mask = None
        
    # forward : if value in x is smaller or equal to 0, convert it to 0.
    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    # backward : if value in x at the forward <= 0, derivative is 0.
    # else, the gradient is propagated without change of value. 
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

# Sigmoid layer

class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self,x):
        out = 1. / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
    
# Affine layer
# output = X * W + b (weight * input + bias)
# X : (# data, input_dim), W : (input_dim, output_dim or hidden_dim), b : output or hidden_dim

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape) 
        return dx
    

class Dropout:
    '''
    Dropout layer
    If train_fig is true, we randomly choose nodes to drop.
    Q. How many are dropped? : following the dropout_ratio.
    More or less may be dropped as we generate random numbers.
    '''
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_fig = True):
        if train_fig:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else: 
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


# Batch Normalization

class BatchNormalization:
    '''
    Reference : http://arxiv.org/abs/1502.03167
    '''
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        # main parameters
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # FNN : 2D, CNN : 4D (input_num, data dim)
        
        # mean and var when running
        self.running_mean = running_mean
        self.running_var = running_var

        # class members used in the backpropagation
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_fig=True):
        self.input_shape = x.shape
        if x.ndim != 2: # For CNN: temporarily convert it to 2D array, like FNN.
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_fig)

        return out.reshape(*self.input_shape)

    def __forward(self,x,train_fig):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_fig:
            mu = x.mean(axis=0) # Sample average
            xc = x - mu
            var = np.mean(xc**2,axis=0) # Sample variance
            std = np.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var       
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 1e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        
        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        # Sum over samples
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        # Why did they keep dgamma and dbeta? : need to see later.
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

# Convolution layer

class Convolution:
    '''
    stride : filter application interval. Default : 1 [1-interval shift]. 
    pad : filling outside of the data with 0. Default : 0 [no filling], value denotes the width of pad
    '''
    def __init__(self, W, b, stride=1, pad=0):
        # Here, W and b is the filter.
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # Used in the backward
        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # Output size estimation
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN,-1).T

        out = np.dot(col,col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

# Pooling layer

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
# Softmax function with loss (cross-entropy) layer

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # Output is considered as the one-hot-encoded one.
        self.t = None
        
    def forward(self,x,t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self,dout):
        batch_size = self.t.shape[0]
        if self.t.shape == self.y.shape:
            dx = (self.y - self.t) / batch_size
        else: # If answer label is not one-hot-encoded data..
            dx = self.y.copy()
            # self.t <- For each data, -1 operation to the output element at answer index.
            dx[np.arange(batch_size), self.t] -= 1  
            dx = dx / batch_size

        
        return dx
