# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:50:15 2018

@author: Amandeep Rathee
"""

# Code block 1 =============================================================

# gradient descent on a single variable function

def loss(x):
    """function to minimise: f(x) = 3x^2"""
    return 3*x**2

def gradient(x):
    """
    returns value of the gradient by feeding x into derivative of the loss function
    the derivative is: f'(x) = 6x
    """
    return 6*x

def gradient_descent(x = 5, alpha=0.05, iterations=100):
    """returns the optimal value of x that minimises the loss"""
    
    print("Initialised x at", x)
    for i in range(iterations):
        print("Iteration #", i+1, sep='')
        print("Loss:", round(loss(x), 2))
        dx = gradient(x)
        print("dx:", round(dx, 2))
        x = x - alpha*dx
        print("x:", round(x, 2))
    print("Function minimised at x =", round(x, 2))
    return x

gradient_descent(alpha=0.1, iterations=200)

# Code block 2 =============================================================

# gradient descent on multivariable function

def loss(x, y):
    """function to minimise: f(x, y) = x^2 + y^2"""
    return x**2 + y**2

def x_gradient(x, y):
    """
    returns value of gradient by feeding x and y into partial derivative of the loss function w.r.t. x
    the derivative of f(x, y) w.r.t. x is d(f(x, y))/dx = 2x + y^2

    """
    return 2*x + y**2

def y_gradient(x, y):
    """returns value of gradient by feeding x and y into partial derivative of the loss function w.r.t. y
    the derivative of f(x, y) w.r.t. y is d(f(x, y))/dy = x^2 + 2y
    """
    return x**2 + 2*y

def gradient_descent(xy = (5, 5), alpha=0.05, iterations=100):
    """returns the optimal value of x and y that minimises the loss function"""
    
    x, y = xy
    print("Initialised x and y at", (x, y))
    for i in range(iterations):
        print("Iteration #", i+1, sep='')
        print("Loss:", round(loss(x, y), 2))
        
        # get gradients of x and y
        dx = x_gradient(x, y)
        dy = y_gradient(x, y)

        print("dx:", round(dx, 2))
        print("dy:", round(dy, 2))
        
        # update x and y
        x = x - alpha*dx
        y = y - alpha*dy
        
        # print updated values
        print("x:", round(x, 2), ",y", round(y, 2))
    
    print("Function minimised at x =", round(x, 2), "and y =", round(y, 2))
    return (x, y)

gradient_descent(xy = (5, 5), alpha=0.1, iterations=200)

# Code block 3 =============================================================

# vector implementation of gradient descent

# W is a (2x1) vector of two elements w1 and w2.
# x is a (1x3) vector of three elements x1, x2 and x3
# The vector W is variable and x is a constant
# W will be updated and x will be treated as a contant throughout the process
# the loss function is: f(W) = sigmoid(Wx)
# the derivative of sigmoid w.r.t. W is: f'(W) = (sigmoid(Wx)*(1-sigmoid(Wx))]*x

import numpy as np

def sigmoid(x):
  return 1. / (1. + np.exp(-1.*x))

def loss(W, x):
    """The loss function is: f(W) = sigmoid(Wx)"""
    assert W.shape[1] == x.shape[0]
    return np.round_(sum(sum(sigmoid(np.dot(W, x)))), 2)

def gradient(W, x):
    """The derivative of f(W) is: f'(W) = [sigmoid(Wx)*(1-sigmoid(Wx))]*x"""
    gradient = np.dot(((sigmoid(np.dot(W, x)))*(1.-sigmoid(np.dot(W, x)))), x.T)
    return gradient
    
def gradient_descent(alpha=0.05, iterations=100):
    
    # initialize the variable W
    W = np.random.random((2, 1))
    W.resize((2, 1))
    initial_W = W
    
    # initialize the constant x
    x = np.array([1.,2.,3.])
    x.resize((1, 3))
    
    # print initial loss
    initial_loss = loss(W, x)
   
    for i, _ in enumerate(range(iterations)):
        
        # find gradient of W
        dw = gradient(W, x)
        
        # update W
        W = W - alpha*dw
        
        # print loss
        if i % (iterations/5.) == 0:
            print("Iteration", i+1, "=======================================")
            print("Updated W:\n", np.round_(W, 2), "\n")
            print("Loss:", np.round_(loss(W, x), 2))
    
    print("==============================================================")
    
    print("Initial W:\n", initial_W, "\n")
    print("Initial loss:", np.round_(initial_loss, 2), "\n\n")
    
    print("Final W:\n", np.round_(W, 2), "\n")
    print("Final loss", np.round_(loss(W, x), 2))

gradient_descent(alpha=0.1, iterations=300)
