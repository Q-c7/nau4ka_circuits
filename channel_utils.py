#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf  # tf 2.x
import math
import cmath
import numpy as np


# In[9]:


import general_utils as util


# In[10]:


def convert_2qmatrix_to_channel(U):
    """
    Converts 2-qubit gate U into a quantum channel U x U*

    Args:
        U: complex valued tf tensor of shape (4, 4)

    Returns:
        complex valued tf tensor of shape (4, 4, 4, 4).
    """
    phi = tf.tensordot(U, tf.math.conj(U), axes=0)
    phi = tf.transpose(phi, perm=(0, 4, 1, 5, 2, 6, 3, 7))
    phi = tf.reshape(phi, (4, 4, 4, 4))
    return phi

def convert_1qmatrix_to_channel(U):
    """
    Converts 1-qubit gate U into a quantum channel U x U*

    Args:
        U: complex valued tf tensor of shape (2, 2)

    Returns:
        complex valued tf tensor of shape (4, 4).
    """
    phi = tf.tensordot(U, tf.math.conj(U), axes=0)
    phi = tf.transpose(phi, perm=(0, 2, 1, 3))
    phi = tf.reshape(phi, (4, 4))
    return phi

def convert_params_to_channel(A):
    """
    Converts a batch_size of parameter matrices A into a quantum channel representation A * A^dagger
    Sadly, matrices need to be the same dimension...

    Args:
        A: complex valued tf tensor of shape (bs, dim^2, dim^2)
        'dim' corresponds to the number of qubits this quantum channel should be applied to.

    Returns:
        complex valued tf tensor of shape (bs, dim^2, dim^2).
    """
    choi = A @ tf.linalg.adjoint(A)
    dim = A.get_shape()[-2]
    dim = tf.cast(tf.math.sqrt(tf.cast(dim, dtype=A.dtype)), dtype=tf.int32)
    bs_shape = A.get_shape()[:-2]

    # corresponding quantum channel
    choi = tf.reshape(choi, (-1, dim, dim, dim, dim))
    phi = tf.transpose(choi, (0, 2, 4, 1, 3))
    phi = tf.reshape(phi, (-1, dim ** 2, dim ** 2))
    phi = tf.reshape(phi, (*bs_shape, dim ** 2, dim ** 2))
    return phi

def convert_channel_to_params(phi, eps=1e-5):
    """
    Converts a batch_size of quantum channel matrices Phi into a parameter representation
    Phi = A * A^dagger, gets converted into A
    Take care since the decomposition is not unique and one Phi has multiple A's 
    Again, matrices 'Phi' need to be the same dimension...

    Args:
        Phi: complex valued tf tensor of shape (bs, dim^2, dim^2)
        'dim' corresponds to the number of qubits this quantum channel should be applied to.

    Returns:
        complex valued tf tensor of shape (bs, dim^2, dim^2).
    """
    dim = phi.get_shape()[-2]
    dim = tf.cast(tf.math.sqrt(tf.cast(dim, dtype=phi.dtype)), dtype=tf.int32)
    bs_shape = phi.get_shape()[:-2]

    choi = tf.reshape(phi, (-1, dim, dim, dim, dim))
    choi = tf.transpose(choi, (0, 3, 1, 4, 2))
    choi = tf.reshape(choi, (-1, dim ** 2, dim ** 2))
    lmbd, u, _ = tf.linalg.svd(choi)
    lmbd = tf.cast(lmbd, dtype=u.dtype)
    A = u * tf.math.sqrt(lmbd[:, tf.newaxis])
    A = tf.reshape(A, (*bs_shape, dim ** 2, dim ** 2))
    return A