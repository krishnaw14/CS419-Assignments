import numpy as np
import math 

def square_hinge_loss(targets, outputs):
  # Write thee square hinge loss here
  targets[targets<1] = -1
  hinge_loss = targets*outputs
  hinge_loss[hinge_loss>1] = 0 
  hinge_loss[hinge_loss<=1] = 1- hinge_loss[hinge_loss<=1]
  loss = np.sum(hinge_loss**2)
  return loss

def logistic_loss(targets, outputs):
  # Write thee logistic loss loss here
  targets[targets<1] = -1
  #print("targets = ", targets)
  loss = np.sum( np.log(1+np.exp(-targets*outputs)) )
  print(loss)
  return loss

def perceptron_loss(targets, outputs):
  # Write thee perceptron loss here
  targets[targets<1] = -1
  loss = targets*outputs
  loss[loss>=0] = 0
  loss[loss<0] *=(-1)
  return np.sum(loss)

def L2_regulariser(weights):
    # Write the L2 loss here
  loss = np.sum(weights[1:]**2)
  return 4*loss

def L4_regulariser(weights):
    # Write the L4 loss here
  loss = np.sum(weights[1:]**4)
  return 5*loss
    #return 0.0

def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
  gradient = np.zeros(len(weights), dtype=np.float32)
  hinge = 1-targets*outputs
  hinge[hinge<0] = 0 

  for i in range(len(weights)):
    gradient[i] = np.sum(2*(hinge)*inputs[:,i])
  #weights -= .1*gradient
  return gradient

def logistic_grad(weights,inputs, targets, outputs):
  # Write thee logistic loss loss gradient here

#   gradient = np.zeros(len(weights), dtype=np.float32)
#   for i in range(len(weights)):
#     gradient[i] = np.sum( -1.0*inputs[:,i]*targets*np.exp(targets*outputs)/(1+np.exp(targets*outputs)) )
  gradient = np.zeros(weights.shape, dtype=np.float32)
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
     weights[i] = weights[i]-.1*gradient[i]
  #print gradient - a
  return gradient


def perceptron_grad(weights,inputs, targets, outputs):
  # Write thee perceptron loss gradient here

  gradient = np.zeros(len(weights), dtype=np.float32)
  flag = targets*outputs
  flag[flag>=0] = 0
  flag /= (-outputs)
  for i in range(len(weights)):
    gradient[i] = np.sum(flag*inputs[:,i])
  return gradient

def L2_grad(weights):
    # Write the L2 loss gradient here
    gradient = np.zeros(len(weights), dtype=np.float32)
    gradient[0] = 0
    gradient[1:] = 2*4*weights[1:]
    #weights -= .1*gradient
    return gradient

def L4_grad(weights):
    # Write the L4 loss gradient here
    gradient = np.zeros(len(weights), dtype=np.float32)
    gradient[0] = 0
    gradient[1:] = 5*4*weights[1:]**3
    #weights -= .1*gradient
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
