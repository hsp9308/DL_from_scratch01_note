import numpy as np
import sys, os
sys.path.append(os.pardir)

from common.functions import *
from common.gradient import numerical_grad
from common.layers import *
from collections import OrderedDict

class MultiLayerNet:
    '''
    Implementation of (fully-connected) multi-layer network
    
    input_size : number of input node (MNIST: 784)
    hidden_size : number of hidden nodes, list (i.e. for 3 hidden layers, [100, 50, 50] )
    output_size : number of output node (MNIST : 10)
    activation : activation function, 'sigmoid' or 'relu'
    weight_init_std : weight std value. Default is 0.01.
    
    '''
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu',
                weight_init_std='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        
        self.params = {}
        self.init_weight(weight_init_std)
        
        # Layers defines here
        # Here we use OrderedDict to set order of layers
        activation_layer = {'relu': Relu, 'sigmoid': Sigmoid}
        self.layers = OrderedDict()
        for i in range(1,self.hidden_layer_num+1):
            self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)],
                                            self.params['b'+str(i)])
            self.layers['Activation'+str(i)] = activation_layer[activation]()
        
        ind = self.hidden_layer_num + 1
        self.layers['Affine'+str(ind)] = Affine(self.params['W'+str(ind)],
                                            self.params['b'+str(ind)])
        
        self.last_layer = SoftmaxWithLoss()
        
     
    def init_weight(self, weight_init_std):
        size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1,len(size_list)):
            '''
            To initialize well, we choose one of them.
            'he' :  for relu activation function
            'xavier' : for sigmoid activation function

            '''
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / size_list[i-1])
            elif str(weight_init_std).lower() in ('sigmoid','xavier'):
                scale = np.sqrt(1.0 / size_list[i-1])
            self.params['W'+str(i)] = scale * np.random.randn(size_list[i-1], size_list[i])
            self.params['b'+str(i)] = np.zeros(size_list[i])
    
    def predict(self, x):
        # Forward operation in layers.
        # It does not execute Softmax and Loss layer propagation.
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    
    # t : answer label
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        # Among the output and answer label, pick the largest ones. 
        # As the value hierarchy does not change after softmax operation, 
        # we just take the maximum value index here so that we aviod additional operation.
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1: t = np.argmax(t, axis=1)
        
        accuracy = np.sum( y == t ) / float(x.shape[0])
        return accuracy
    
    def numerical_grad(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}

        for i in range(1,self.hidden_layer_num+2):
            grads['W'+str(i)] = numerical_gradient(loss_W, self.params['W'+str(i)])
            grads['b'+str(i)] = numerical_gradient(loss_W, self.params['b'+str(i)])
        
        return grads
    

    def grad(self, x, t):
        
        self.loss(x, t)
        
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())[::-1]
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}

        for i in range(1,self.hidden_layer_num+2):
            grads['W'+str(i)] = self.layers['Affine'+str(i)].dW
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db

        return grads