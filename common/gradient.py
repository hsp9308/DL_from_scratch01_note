import numpy as np

def numerical_grad(f, x):
    '''
    To deal with the case that x is multi-dimensional array, 
    we adapt nditer here. 
    '''
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        tmp = x[i]
        # f(x+h)
        x[i] = tmp + h
        fxh1 = f(x)
        
        # f(x-h)
        x[i] = tmp - h
        fxh2 = f(x)
        
        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp
        it.iternext()
        
    return grad
