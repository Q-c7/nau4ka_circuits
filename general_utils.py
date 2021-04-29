#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf  # tf 2.x
import math
#import cmath


# In[9]:


def matr_Ry(angle):
    return tf.constant([[math.cos(angle/2), -math.sin(angle/2)],
                        [+math.sin(angle/2), math.cos(angle/2)]], dtype=tf.complex64)

def matr_Rx(angle):
    return tf.constant([[math.cos(angle/2) + 0j, -1j * math.sin(angle/2)],
                        [-1j * math.sin(angle/2), math.cos(angle/2) + 0j]], dtype=tf.complex64)

def matr_Rz(angle):
    return tf.constant([[math.cos(angle/2) -1j * math.sin(angle/2), 0],
                        [0, math.cos(angle/2) + 1j * math.sin(angle/2)]], dtype=tf.complex64)


# In[10]:


@tf.function
def kron(A, B):
    """
    Returns Kronecker product of two square matrices.

    Args:
        A: complex valued tf tensor of shape (dim1, dim1)
        B: complex valued tf tensor of shape (dim2, dim2)

    Returns:
        complex valued tf tensor of shape (dim1 * dim2, dim1 * dim2),
        kronecker product of two matrices
    """

    dim1 = A.shape[-1]
    dim2 = B.shape[-1]
    AB = tf.transpose(tf.tensordot(A, B, axes=0), (0, 2, 1, 3))
    return tf.reshape(AB, (dim1 * dim2, dim1 * dim2))


# In[11]:


def tf_ravel_multi_index(multi_index, dims):
    """
    Converts a batch of bitstrings into a batch of single int numbers.

    Args:
        multi_index: 2D array of shape (batch_size, dims.shape[0])
        Is effectively a batch of 1D arrays, each array containing several digits to encode a bitstring.
        dims: a tuple representing dimensions (aka numeral system) for each digit in 1D array multi_index[k].

    Returns:
        1D array of type int32 of shape (batch_size)
    """
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    return tf.reduce_sum(multi_index * strides, axis=-1)


# In[12]:


def simple_qc(circ, length):
    state = tf.constant([1] + (2 ** length-1) * [0], dtype=tf.complex64)
    #print(state)
    state = tf.reshape(state, length * (2,))
    for gate, sides in circ:
        if sides[0] > sides[1]:
            min_index = sides[1]
            max_index = sides[0]
            first_edge = length-1
            second_edge = length-2
        else:
            min_index = sides[0]
            max_index = sides[1]
            first_edge = length-2
            second_edge = length-1
        new_ord = tuple(range(min_index)) + (first_edge,) + tuple(range(min_index, max_index-1)) + (second_edge,) + tuple(range(max_index-1, length-2))
        #print(new_ord)
        #print("sides23:", [sides, [2, 3]])
        state = tf.tensordot(state, gate, axes=[sides, [2, 3]]) #sides
        #print(state)
        state = tf.transpose(state, new_ord)
        #print(state)
    return state


# In[13]:


def convert_44_to_2222(U):
    new_U = tf.reshape(U, (2, 2, 2, 2))
    new_U = tf.transpose(new_U, (1, 0, 3, 2))
    return new_U

def swap_legs(U):
    return tf.transpose(U, (1, 0, 3, 2))


# In[20]:


Hadamard = tf.constant([[1, 1],
                        [1, -1]], dtype=tf.complex64) / math.sqrt(2)

S = matr_Rz(math.pi/2)

T = matr_Rz(math.pi/4)

E = tf.eye(2, dtype=tf.complex64)

CZ_44 = tf.constant([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]], dtype=tf.complex64)

E_44 = tf.eye(4, dtype=tf.complex64)

CNOT_44 = tf.constant([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]], dtype=tf.complex64) 
RNOT_44 = tf.constant([[1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0]], dtype=tf.complex64)


# In[25]:


CNOT = convert_44_to_2222(CNOT_44)
CZ = convert_44_to_2222(CZ_44)
RNOT = convert_44_to_2222(RNOT_44)
big_E = convert_44_to_2222(E_44)

