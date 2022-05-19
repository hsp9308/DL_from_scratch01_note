import numpy as np
from common.functions import softmax, cross_entropy

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
        
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        
        return out
    
    def backward(self,dout):
        self.db = np.sum(dout, axis=0) # (N,b.size) => (b.size) : sum over N data
        self.dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)
        
        return dx
    
    
# Softmax function with loss (cross-entropy) layer



class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self,x,t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self,dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
