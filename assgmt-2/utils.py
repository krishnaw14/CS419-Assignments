import numpy as np
import math
 

def square_hinge_loss(targets, outputs):
  # Write thee square hinge loss here
  count = 0
  for i in range(len(targets)):
    count+=(1-targets[i]*outputs[i])**2
  return count

def logistic_loss(targets, outputs):
  # Write thee logistic loss loss here
  count = 0
  for i in range(len(targets)):
    count += math.log(1+math.exp(-targets[i]*outputs[i]))
  return count

def perceptron_loss(targets, outputs):
  # Write thee perceptron loss here
  count = 0
  for i in range(len(targets)):
    count+=max(0,-targets[i]*outputs[i])
  return count

def L2_regulariser(weights):
    # Write the L2 loss here
  count = 0
  for i in weights:
    count+=i**2
  return count

def L4_regulariser(weights):
    # Write the L4 loss here
  count = 0
  for i in weights:
    count+=i**4
  return count
    #return 0.0

def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
  x = 0
  count = 0
  for i in range(len(targets)):
    x+=targets[i]*weights[i]
  for i in range(len(outputs)):
    if outputs[i]*x<1:
      count-=outputs[i]*targets[i]
    else:
      count+=0
  return count

def logistic_grad(weights,inputs, targets, outputs):
  # Write thee logistic loss loss gradient here
    return 1.00

def perceptron_grad(weights,inputs, targets, outputs):
  # Write thee perceptron loss gradient here
  print(weights)
  print(type(weights))
  return np.random.random(11)

def L2_grad(weights):
    # Write the L2 loss gradient here
    gradient = 2*weights[1:]
    return gradient

def L4_grad(weights):
    # Write the L4 loss gradient here
    gradient = 4*weights[1:]**3
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
