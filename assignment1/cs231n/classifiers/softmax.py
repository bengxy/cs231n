import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A qnumpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in xrange(num_train):
    pred_i = X[i,:].dot(W)
    bound = np.max(pred_i)
    pred_i -= bound   # avoid overflow
    #exp
    exp_i = np.exp(pred_i)
    #norm
    sum_i  = np.sum(exp_i)
    #norm_i =  exp_i/sum_i
    loss = loss -pred_i[y[i]] + np.log(sum_i)
    for j in xrange(num_classes):
        dW[:,j] += (exp_i[j]/sum_i - (y[i] == j))*X[i,:]

  loss  /= num_train
  dW /= num_train
    
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pred_i = X.dot(W)
  #bound = np.matrix( np.max(pred_i, axis=1) )
  #print(bound.shape)
  bound =np.amax(pred_i, axis=1)
  bound = bound.reshape((len(bound), 1))
  pred_i = pred_i - bound
  exp_pred_i = np.exp(pred_i)
  sum_i = np.sum(exp_pred_i, axis=1) # exp_sum
  loss_all = - pred_i[np.arange(X.shape[0]), y].T +  np.log( sum_i)
  
  #score = pred_i[np.arange(4), [0]*4].T + np.log( np.sum(pred_i, axis=1) )
  loss = np.mean(loss_all)
  loss += 0.5 * reg *np.sum(W*W)  
  
  res = exp_pred_i/(sum_i.reshape( (sum_i.shape[0],-1) ) )
  tmp  = np.zeros_like(res)
  tmp[np.arange(X.shape[0]), y] = -1
  res = res +tmp
 
  dW = X.T.dot(res)

  dW /= X.shape[0]
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

