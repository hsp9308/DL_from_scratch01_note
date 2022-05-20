import numpy as np
import sys, os
sys.path.append(os.pardir)

from common.functions import *
from common.gradient import numerical_grad
from common.layers import *
from collections import OrderedDict

class MultiLayerNetExtend:
    '''
    Implementation of (fully-connected) multi-layer network
    
    input_size : number of input node (MNIST: 784)
    hidden_size : number of hidden nodes, list (i.e. for 3 hidden layers, [100, 50, 50] )
    output_size : number of output node (MNIST : 10)
    activation : activation function, 'sigmoid' or 'relu'
    weight_init_std : weight std value. Default is 0.01.
          - 'relu' or 'he' : He initialization
          - 'xavier' or 'sigmoid' : Xavier initialization
    weight_decay_lambda : L2 regularization weight
    use_dropout : True or False
    dropout_ratio : dropout ratio
    use_batchnorm : True or False
    
    '''
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu',
                weight_init_std='relu', weight_decay_lambda=0,
                use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        
        self.params = {}
        self.init_weight(weight_init_std)
        
        # Layers defines here
        # Here we use OrderedDict to set order of layers
        activation_layer = {'relu': Relu, 'sigmoid': Sigmoid}
        self.layers = OrderedDict()
        for i in range(1,self.hidden_layer_num+1):
            self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)],
                                            self.params['b'+str(i)])
            # use_batchnorm == True
            # insert gamma and beta layers to rescale the output of Affine layer
            if self.use_batchnorm:
                self.params['gamma' + str(i)] = np.ones(hidden_size_list[i-1])
                self.params['beta' + str(i)] = np.zeros(hidden_size_list[i-1])
                self.layers['BatchNorm' + str(i)] = BatchNormalization(self.params['gamma'+str(i)],
                                                    self.params['beta'+str(i)])
            self.layers['Activation'+str(i)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(i)] = Dropout(dropout_ratio)
        
        ind = self.hidden_layer_num + 1
        self.layers['Affine'+str(ind)] = Affine(self.params['W'+str(ind)],
                                            self.params['b'+str(ind)])
        # Last layer is fully connected.
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
    # L2 Regularization implemented
    def loss(self, x, t):
        y = self.predict(x)

        weight_decay = 0
        for i in range(1,self.hidden_layer_num + 2):
            W = self.params['W'+str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.linalg.norm(W)
        return self.last_layer.forward(y, t) + weight_decay
    
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
            grads['W'+str(i)] = numerical_grad(loss_W, self.params['W'+str(i)])
            grads['b'+str(i)] = numerical_grad(loss_W, self.params['b'+str(i)])

            if self.use_batchnorm and i != self.hidden_layer_num+1:
                grads['gamma' + str(i)] = numerical_grad(loss_W, self.params['gamma' + str(i)])
                grads['beta' + str(i)] = numerical_grad(loss_W, self.params['beta' + str(i)])

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
            grads['W'+str(i)] = self.layers['Affine'+str(i)].dW + self.weight_decay_lambda * self.layers['Affine'+str(i)].W
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db

            if self.use_batchnorm and i != self.hidden_layer_num+1:
                grads['gamma' + str(i)] = self.layers['BatchNorm' + str(i)].dgamma
                grads['beta' + str(i)] = self.layers['BatchNorm' + str(i)].dbeta

        return grads