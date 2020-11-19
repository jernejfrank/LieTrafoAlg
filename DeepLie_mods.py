import numpy as np
import sympy as sp
from sympy import Symbol, expand
import matplotlib.pyplot as plt


import tensorflow as tf

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential


from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

def getKronPowers(state, order, dim_reduction = True):
    """Calculates Kroneker powers of state vector
       with dimension reduction

    e.g. for (x y) and order=2 returns:
    1, (x y), (x x*y y)  

    Returns:
    list of numpy arrays, index corresponds to power 
    """
    powers = [state]
    index = [np.ones(len(state), dtype=bool)]
    for i in range(order-1):
        state_i = np.kron(powers[-1], state)
        reduced, red_ind = reduce(state_i)
        if dim_reduction:
            powers.append(reduced)
        else:
            powers.append(state_i)
            
        index.append(red_ind)

    powers.insert(0, np.array([1]))
    index.insert(0, [True])
    return powers, index 

def reduce(state):
    state_str = state.astype(str)
    reduced_state = []
    unique = []

    index = []
    for variable, variable_str in zip(state, state_str):
        if variable_str not in unique:
            unique.append(variable_str)
            index.append(True)
            reduced_state.append(variable)
        else:
            index.append(False)

    return np.array(reduced_state), index


def fill_matrices(state_powers, expr, M):
    coefs = [e.as_coefficients_dict() for e in expr]

    for k, state_power in enumerate(state_powers):
        for i, coef_dict in enumerate(coefs):
            for j, variable in enumerate(state_power):
                M[k][i, j] += coef_dict[variable]
    return M


def sum(X, Y, k):
    res = []
    for x, y in zip(X,Y):
        res.append(x+k*y)
    return res


def printlist(l):
    for el in l:
        print(el)
    return

class LieMapBuilder:
    def __init__(self, state, right_hand_side, order=3):
        """Initialization of Lie map builder

        Arguments:
        state -- 1d numpy array of sympy Symbols
        right_hand_side -- iterable  sympy expressions (polynomial)
        order -- order of nonlinearity of the resulting map 

        TO DO:
        authomatic expand right_hand_side into polynomial series
        """
        if len(state)!=len(right_hand_side):
            raise Exception('incorrect system dimension')

        self.StateSize = len(state)
        if order < 1:
            order = 1 # linear system at least
        print(state)
        print(right_hand_side)

        self.Order = order
        self.X, self.index = getKronPowers(state, order=order) 
        _,self.index_full = getKronPowers(state, order=order, dim_reduction=False)     
        
        self.P = []
        for X in self.X:
            self.P.append(np.zeros((self.StateSize, len(X))))

        fill_matrices(self.X, right_hand_side, self.P)
        
        return
    
    def right_hand_side_maps(self, R, verbose=False):
        dR = []
        n = self.StateSize
        for X in self.X:
            dR.append(np.zeros((n, len(X))))
        
        dR[0]+=self.P[0]
     
        X = np.zeros((self.StateSize,), dtype=object)
        for R_k, X0_k in zip(R, self.X):
            X += np.dot(R_k,X0_k)

        
        # Xk = [np.array([1])]
        Xk_list = [np.array([1])]
        for k in range(1, self.Order+1):
            tmp = np.kron(Xk_list[-1], X)[self.index[k]]
            Xk_list.append(tmp)
        for k, Xk in enumerate(Xk_list):
            # print 'X%s expanding' % k
            for i, xk in enumerate(Xk):
                Xk[i] = expand(xk)
                #cprint i, '/', len(Xk)

        
        for k in range(1, self.Order+1):
            fill_matrices(self.X, np.dot(self.P[k], Xk_list[k]), dR)


        # for k in xrange(1, self.Order+1):
        #     Xk = np.kron(Xk, X)[self.index[k]]
        #     if verbose:
        #         print 'X%s is calculated'%k
        #     for i, xk in enumerate(Xk):
        #         Xk[i] = expand(xk)
        #         print i, '/', len(Xk)
        #     if verbose:
        #         print 'X%s is expanded'%k
        #     fill_matrices(self.X, np.dot(self.P[k], Xk), dR)
        #     if verbose:
        #         print 'dR[%s] is calculated'%k
        
        return dR

    def getInitR(self):
        R = []
        for X in self.X:
            R.append(np.zeros((self.StateSize, len(X))))
        
        R[1] = np.eye(self.StateSize)
        return R

    def propogate(self, h = 0.1, N = 10, R=None, verbose = True):
        if R == None:
            R = self.getInitR()
        
        for i in range(N):
#             print i, N

            k1 = self.right_hand_side_maps(R, verbose=verbose)

            k2 = self.right_hand_side_maps(sum(R, k1, h/2), verbose=verbose)
            

            k3 = self.right_hand_side_maps(sum(R, k2, h/2), verbose=verbose)

            k4 = self.right_hand_side_maps(sum(R, k3, h/2), verbose=verbose)

            k12 = sum(k1, k2, 2)
            k34 = sum(k4, k3, 2)
            k = sum(k12, k34, 1)
            
            R = sum(R, k, h/6)
        return R

    def convert_weights_to_full_nn(self, R):
        W = []

        m = 1

        for ind, Rk in zip(self.index_full, R):
            w = np.zeros((self.StateSize, m))
            m*=self.StateSize
            w[:, ind] = Rk
            W.append(w.T)

        return W

class LieLayer(Layer):
    def __init__(self, output_dim, order=1, **kwargs):
        self.output_dim = output_dim
        self.order = order
        super(LieLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_dim = input_dim
        nsize = 1
        self.W = []
        self.nsizes = [nsize]
        for i in range(self.order+1):
            initial_weight_value = np.zeros((nsize, self.output_dim))
            nsize*=input_dim
            self.nsizes.append(nsize)
            self.W.append(K.variable(initial_weight_value))

        self.W[1] = (K.variable(np.eye(N=input_dim, M=self.output_dim)))
        self.trainable_weights = self.W
        return


    def call(self, x, mask=None):
        ans = self.W[0]
        tmp = x
        x_vectors = tf.expand_dims(x, -1) 
        for i in range(1, self.order+1):
            ans = ans + K.dot(tmp, self.W[i])
            if(i == self.order):
                continue
            xext_vectors = tf.expand_dims(tmp, -1)
            # x_extend_matrix = tf.batch_matmul(x_vectors, xext_vectors, adj_x=False, adj_y=True)
            x_extend_matrix = tf.matmul(x_vectors, xext_vectors, adjoint_a=False, adjoint_b=True)
            tmp = tf.reshape(x_extend_matrix, [-1, self.nsizes[i+1]])
        
        return ans


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)