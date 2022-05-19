import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    '''
    For each data, we calculate the softmax.
    For convenience, we transpose the array. 
    '''
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # Below : single-data case
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(y, t):
    '''
    If input is a single data, we reshape it.
    Else, shape of y and t would be (N, D).
    N : number of data in the batch
    D : label dimension. (i.e. D = 10 for one-hot-encoded MNIST data)

    '''
    if y.ndim == 1:
        t = np.reshape(t,(1, t.size))
        y = np.reshape(y,(1, y.size))

   # if training data is one-hot-encoded vector, convert it to answer index.
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size
#    return -np.sum(t* np.log(y+1e-7)) / batch_size
