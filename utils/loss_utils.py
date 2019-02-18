from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import torch
import torch.utils.serialization


def l1_loss(predictions, targets):
  """Implements tensorflow l1 loss.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.abs(predictions- targets))
  loss = tf.div(loss, total_elements)
  return loss

def l2_loss(predictions, targets):
  """Implements tensorflow l2 loss, normalized by number of elements.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.square(predictions-targets))
  loss = tf.div(loss, total_elements)
  return loss

class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return h_tv/count_h + w_tv/count_w

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    
def vae_loss(z_mean, z_logvar, prior_weight=1.0):
  """Implements the VAE reguarlization loss.
  """
  total_elements = (tf.shape(z_mean)[0] * tf.shape(z_mean)[1] * tf.shape(z_mean)[2]
      * tf.shape(z_mean)[3])
  total_elements = tf.to_float(total_elements)

  vae_loss = -0.5 * tf.reduce_sum(1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
  vae_loss = tf.div(vae_loss, total_elements)
  return vae_loss

def bilateral_loss():
  #TODO
  pass


def L1Loss(tensora, tensorb):
    s = tensora.size()
    eles = s[0] * s[1] * s[2] * s[3]
    return torch.abs(tensora - tensorb).sum()/eles

def L1Loss_pixel_wise(tensora, tensorb):
    s = tensora.size()
    eles = s[0] * s[1] * s[2] * s[3]
    return torch.sum(torch.abs(tensora - tensorb), 1) / s[1]