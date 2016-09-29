import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=5,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.01,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    W1 = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
    b1 = np.zeros(num_filters)
    W4 = np.random.randn(num_filters, num_filters, filter_size, filter_size) * weight_scale
    b4 = np.random.randn(num_filters)
    W2 = np.random.randn(num_filters * (input_dim[1]/4) * (input_dim[2]/4), hidden_dim) * weight_scale
    b2 = np.zeros(hidden_dim)
    W3 = np.random.randn(hidden_dim, num_classes) * weight_scale
    b3 = np.zeros(num_classes)

    self.params['W1'] = W1
    self.params['b1'] = b1
    self.params['W2'] = W2
    self.params['b2'] = b2
    self.params['W3'] = W3
    self.params['b3'] = b3
    self.params['W4'] = W4
    self.params['b4'] = b4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out1, cache4 = conv_relu_pool_forward(out1, W4, b4, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    out3, cache3 = affine_forward(out2, W3, b3)
    scores = out3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(out3, y)
    dtemp, dW3, db3 = affine_backward(dout, cache3)
    dtemp, dW2, db2 = affine_relu_backward(dtemp, cache2)
    dtemp, dW4, db4 = conv_relu_pool_backward(dtemp, cache4)
    _, dW1, db1 = conv_relu_pool_backward(dtemp, cache1)

    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4))

    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4

    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    grads['b4'] = db4
    grads['W4'] = dW4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  