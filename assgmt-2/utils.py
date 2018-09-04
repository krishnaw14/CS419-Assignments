import numpy as np
 

def square_hinge_loss(targets, outputs):
  # Write thee square hinge loss here
  loss = np.sum( (1-targets*outputs)**2 )
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
  loss = np.sum(weights[1:]**2)
  return loss

def L4_regulariser(weights):
    # Write the L4 loss here
  loss = np.sum(weights[1:]**4)
  return loss
    #return 0.0

def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
  #print(weights)
  gradient = np.zeros(len(weights), dtype=np.float32)
  #print(inputs.shape)
  for i in range(len(weights)):
    gradient[i] = np.sum(2*(1-targets*outputs)*inputs[:,i])
  #print(gradient)
  return gradient

def logistic_grad(weights,inputs, targets, outputs):
  # Write thee logistic loss loss gradient here
  gradient = np.zeros(len(weights), dtype=np.float32)
  for i in range(len(weights)):
    gradient[i] = np.sum( -1.0*inputs[:,i]*targets*np.exp(targets*outputs)/(1+np.exp(targets*outputs)) )
  return gradient

def perceptron_grad(weights,inputs, targets, outputs):
  # Write thee perceptron loss gradient here
  gradient = np.zeros(len(weights), dtype=np.float32)
  for i in range(len(weights)):
    gradient[i] = np.sum(targets*inputs[:, i])
  return gradient

def L2_grad(weights):
    # Write the L2 loss gradient here
    gradient = np.zeros(len(weights), dtype=np.float32)
    gradient[0] = 0
    gradient[1:] = 2*weights[1:]
    return gradient

def L4_grad(weights):
    # Write the L4 loss gradient here
    gradient = np.zeros(len(weights), dtype=np.float32)
    gradient[0] = 0
    gradient[1:] = 4*weights[1:]**3
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
