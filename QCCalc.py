#!/usr/bin/env python
# coding: utf-8

# ## Basic functions

# In[1]:


import tensorflow as tf  # tf 2.x
import tensornetwork as tn
tn.set_default_backend("tensorflow")
from tensornetwork import ncon
import general_utils as util
import channel_utils as c_util
import math
import cmath
import numpy as np


# In[8]:


class QCEvaluator:

    def __init__(self, gates, n):
        self.gates = gates
        self.circuits = {}
        self.n = n
        self.in_states = n * [tf.constant([1, 0, 0, 0], dtype=tf.complex64)]
        
        #(n - 1) * [tf.constant([1, 0], dtype=tf.complex64)] + [tf.constant([0, 1], dtype=tf.complex64)]
        
    def add_circuit(self, tn_template, name):
        self.circuits[name] = tn_template
        #template = [gates_order, ncon]
        
    def evaluate (self, samples, name): #indices.shape = (bs, n)
        """
        Evaluates probabilities of obtaining each sample in 'samples' as the circuit's action upon |0>^self.n 
        Yields array of probabilities, one for each bitstring in 'samples'

        Args:
            name: name of the circuit in the QuantumCircuits class instance
            
            samples: batch of bitstrings represented by 2D array of type int32 of shape (bs, self.n)

        Returns:
            1D array of type complex64 of shape (bs)
            
        """
        out_tensors = tf.one_hot(tf.multiply(samples, 3), 4, dtype=tf.complex64) #(bs, n, 4); n - нумерует тензор
        out_tensors = [out_tensors[:, i, :] for i in range(self.n)]
        
        #print("aaa", out_tensors)
        
        tensors, net_struc, con_order, out_order = self.circuits[name]
        tensors = out_tensors + self.in_states + [self.gates[i] for i in tensors]
        #print(net_struc)
        #print(len(tensors))
        for i in np.arange(len(net_struc)):
            for j in np.arange(len(net_struc[i])):
                if isinstance(net_struc[i][j], int):
                    if net_struc[i][j] < 0:
                        #print('CCC', net_struc[i][j])
                        net_struc[i][j] = 'out' + str(-net_struc[i][j])
        net_struc = [[-1, 'out' + str(i)] for i in range(1, self.n+1)] + [[i] for i in range(1, self.n+1)] + net_struc
        #print(len(net_struc))
        con_order = ['out' + str(i) for i in range(1, self.n+1)] + [i for i in range(1, self.n+1)] + con_order
        
        return ncon(tensors, net_struc, con_order, (-1,))
    
    def sample_next_qubit(self, name, prev_samples, bs_override=1000):
        """
        Creates a batch of samples for a circuit with name 'name' for a single qubit no. L
        Sample shape is (bs,), but the final goal is to make a huge array (bs, self.n)
        Since we employ 'qubit-by-qubit' sample generation, L is defined as (prev_samples.shape[1] + 1)
        if prev_samples = None, this means we start the generation from 1st qubit

        Args:
            name: name of the circuit in the QuantumCircuits class instance
            
            prev_samples: Is a int32 (but can be bool) tensor of shape (bs, (L-1))
            Contains previously generated bitstrings for first (L-1) qubits
            Also can be None
            
            bs_override: since prev_samples can be None, we need to determine sample size when starting from scratch
            default value is 1000

        Returns:
            1D array of type int32 of shape (bs)
        """
        
        '''
        this block defines the qubit l we need to sample, extracts the batch size and prev_samples is they exist
        we use one_hot(prev_samples) for out (l-1) qubits which are already sampled, and connect them to '-1'
        we use tf.eye(4) for qubit number l which we sample at the moment and conenct it to '-2'
        '''
        if prev_samples != None:
            l = prev_samples.shape[1] + 1
            bs = prev_samples.shape[0]
            one_hot_prev_samples = tf.one_hot(tf.multiply(prev_samples, 3), 4, dtype=tf.complex64) #bs, l, 4
            slices = [one_hot_prev_samples[:, i, :] for i in range(0, l-1)] #1..(l-1)
            out_tensors = slices + [tf.eye(4, dtype = tf.complex64)] #slices & target
            out_new_order = (-1, -2)
        else:
            l = 1
            bs = bs_override
            slices = None
            out_tensors = [tf.eye(4, dtype = tf.complex64)] #slices & target
            out_new_order = (-2,)
            
        #print(slices)
        
        '''
        now we take care about plugs - qubits after l which are going to be sampled later
        '''
    
        plugs = (self.n - l) * [tf.constant([1, 0, 0, 1], dtype=tf.complex64)] #(l+1)..n
        out_tensors = out_tensors + plugs #add plugs
    
        #in_states = self.n * [tf.constant([1, 0], dtype=tf.complex64)] #inputs are now self.in_states
        
        '''
        we unpack a tensor network template, then add slices, add target qubit, add plugs, and finally input legs
        '''
        
        tensors, net_struc, con_order, out_order = self.circuits[name]
        
        tensors = out_tensors + self.in_states + [self.gates[i] for i in tensors]
        
        #print(out_tensors)
        
        for i in np.arange(len(net_struc)):
            for j in np.arange(len(net_struc[i])):
                if isinstance(net_struc[i][j], int):
                    if net_struc[i][j] < 0:
                        net_struc[i][j] = 'out' + str(-net_struc[i][j])
    
        #slices & target
        net_struc = [[-1, 'out' + str(i)] for i in range(1, l)] + [[-2, 'out' + str(l)]] +        [['out' + str(i)] for i in range (l+1, self.n+1)] + [[i] for i in range(1, self.n+1)] + net_struc 
        #plugs & inputs & circuit
        
        con_order = ['out' + str(i) for i in range(1, self.n+1)] + [i for i in range(1, self.n+1)] + con_order
        
        psi = ncon(tensors, net_struc, con_order, out_new_order)
        #big_p = tf.abs(psi) ** 2 
        psi = tf.abs(psi)
        #print('PSI', psi)
        
        if prev_samples != None:
            big_p = tf.concat([psi[:,0][tf.newaxis], psi[:,3][tf.newaxis]], axis=0)
            big_p = tf.transpose(big_p)
            big_p = big_p / tf.reduce_sum(big_p, axis=1, keepdims=True)
            #print(l, big_p)
            log_probs = tf.math.log(big_p) #OMG
            eps = -tf.math.log(-tf.math.log(tf.random.uniform(log_probs.shape)))
            samples = (tf.argmax(log_probs + eps, axis=-1, output_type=tf.int32))
        else:
            big_p = tf.concat([psi[0][tf.newaxis], psi[3][tf.newaxis]], axis=0)
            big_p = big_p / tf.reduce_sum(big_p, keepdims=True)
            #print(l, big_p)
            log_probs = tf.math.log(big_p)
            eps = -tf.math.log(-tf.math.log(tf.random.uniform((bs, 2))))
            samples = (tf.argmax(log_probs + eps, axis=-1, output_type=tf.int32))
    
        return samples
    
    def make_full_samples(self, name, bs_override=1000):
        """
        Creates a batch of all-qubit samples for a circuit with name 'name'
        Sample shape is (bs, self.n)
        Actively uses 'sample_next_qubit' for sample generation

        Args:
            name: name of the circuit in the QuantumCircuits class instance
            
            bs_override: since prev_samples can be None, we need to determine sample size when starting from scratch
            default value is 1000

        Returns:
            2D array of type int32 of shape (bs, self.n)
        """
        next_samples = self.sample_next_qubit(name, None, bs_override)
        big_samples = next_samples[:, tf.newaxis]
        #print(big_samples)
        for i in range(1, self.n):
            next_samples = self.sample_next_qubit(name, big_samples)
            big_samples = tf.concat([big_samples, next_samples[:, tf.newaxis]], axis=1)
        return big_samples

