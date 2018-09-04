import numpy as np
import math 

def square_hinge_loss(targets, outputs):
  # Write thee square hinge loss here
  loss = 0
  for i in range(len(targets)):
    count += max(1-target[i]*outputs[i],0)
  return loss

def logistic_loss(targets, outputs):
  # Write thee logistic loss loss here
  loss = np.sum( np.log(1+np.exp(-targets*outputs)) )
  return loss

def perceptron_loss(targets, outputs):
  # Write thee perceptron loss here
  count = 0
  for i in range(len(targets)):
    count+=max(0,-targets[i]*outputs[i])
  return count

def L2_regulariser(weights):
    # Write the L2 loss here
  count=0
  for i in weights:
    count+=i**2
  loss = np.sum(weights[1:]**2)
  return count

def L4_regulariser(weights):
    # Write the L4 loss here
  loss = np.sum(weights[1:]**4)
  return loss
    #return 0.0

def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
  gradient = np.zeros(weights.shape)
  for i in range(len(weights)):
    gradient[i] = np.sum(2*(1-targets*outputs)*inputs[:,i])
  weights = weights-5*gradient
  return gradient

def logistic_grad(weights,inputs, targets, outputs):
  # Write thee logistic loss loss gradient here
  gradient = np.zeros(weights.shape)
  a = []
  count = 0
  for i in range(len(targets)):
    count += targets[i]*outputs[i]
  for i in range(len(weights)):
    gradient[i] = np.sum( -inputs[:,i]*targets*np.exp(-1*targets*outputs)/(1+np.exp(-1*targets*outputs)) )
  #print weights
  logic = np.exp(-1*outputs*targets)/(1+np.exp(-1*outputs*targets))
  a=[]
  for k in range(len(weights)):
    c = 0
    for l in range(len(targets)):
      c+=logic[l]*targets[l]*inputs[l][k]
    a.append(c)
  #print count-(targets*outputs)
  for i in range(len(weights)):
    if math.isnan(gradient[i])!=True:
     weights[i] = weights[i]-5*gradient[i]
  #print gradient - a
  return gradient


def perceptron_grad(weights,inputs, targets, outputs):
  # Write thee perceptron loss gradient here
  gradient = np.zeros(weights.shape)
  wt = weights*outputs
  for i in range(len(weights)):
    gradient[i] = np.sum((targets*outputs)*inputs[:,i])
  return gradient

def L2_grad(weights):
    # Write the L2 loss gradient here
    gradient = np.zeros(weights.shape)
    gradient[0] = 0
    gradient[1:] = 2*weights[1:]
    weights -= 5*gradient
    return gradient

def L4_grad(weights):
    # Write the L4 loss gradient here
    gradient = np.zeros(weights.shape)
    gradient[0] = 0
    gradient[1:] = 4*weights[1:]**3
    weights -= 5*gradient
    return gradient

loss_functions = {"square_hinge_loss" : square_hinge_loss, 
                  "logistic_loss" : logistic_loss,
                  "perceptron_loss" : perceptron_loss}

loss_grad_functions = {"square_hinge_loss" : square_hinge_grad, 
                       "logistic_loss" : logistic_grad,
                       "perceptron_loss" : perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
                         "L4": L4_regulariser}

regularizer_grad_functions = {"L2" : L2_grad,
                              "L4" : L4_grad}
