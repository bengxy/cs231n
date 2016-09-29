import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
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
    
    F = num_filters
    C = input_dim[0]
    HH = WW = filter_size
    
    self.params['W1'] = np.random.normal(scale= weight_scale, size=(F, C, HH, WW))
    self.params['b1'] = np.zeros(F)
    self.params['W2'] = np.random.normal(scale = weight_scale, size=(F*input_dim[1]*input_dim[1]/4, hidden_dim)) # ? fix pooling size? 2*2 strike 2
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(scale = weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
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
    cnn_out, cnn_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #relu_out, relu_cache = relu_forward(conv_out)
    #pool_out, pool_cache = max_pool_forward_fast(relu_out, pool_param)
    N = X.shape[0]
    cnn_shape = cnn_out.shape
    res = cnn_out.reshape((N, -1)) # to D dim * N
    
    layer1_out, layer1_cache  = affine_relu_forward(res, W2, b2)
    layer2_out, layer2_cache = affine_forward(layer1_out, W3, b3)
    
    #simplify  remove batch and dropout
    #layer1_bn_out, layer1_bn_cache = batchnorm_forward
    #layer1_relu_out, layer1_relu_cache = drop_out_forward
    
    
    
    scores = layer2_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    #print( scores.shape)
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dlayer2_out = softmax_loss(layer2_out, y)
    dlayer1_out, dW3, db3 = affine_backward(dlayer2_out, layer2_cache)
    dres , dW2, db2 = affine_relu_backward(dlayer1_out, layer1_cache)
    
    dcnn_out = dres.reshape(cnn_shape) # D*N to pool * N
    #dcnn_out
    dX, dW1, db1 = conv_relu_pool_backward(dcnn_out, cnn_cache)
    
    loss += 0.5 * self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    grads['W1'] = dW1 + self.reg*sum(W1*W1)
    grads['W2'] = dW2 + self.reg*sum(W2*W2)
    grads['W3'] = dW3 + self.reg*sum(W3*W3)
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
     
    return loss, grads
  
