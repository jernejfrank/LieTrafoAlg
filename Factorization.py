import numpy as np

import sympy
from sympy import *
from operator import mul
import mpmath as mp
from mpmath import fac
import functools
from sympy import init_printing
from sympy.core.decorators import _sympifyit, call_highest_priority
from IPython.display import display_latex
init_printing()
import copy
import collections

from functools import reduce  # Required in Python 3
import operator
#akin to built - in sum() but for product
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

from LieOperator import *

sym_x, sym_y, sym_z, sym_px, sym_py, sym_pz = sympy.symbols('x y z p_x p_y \delta')

def factorization(taylor): #Dragt-Finn factorization from taylor map vector
    LieProduct = []
    degree = 0
    
    for polynomial in taylor:
        comp_degree = poly(polynomial).total_degree() #highest degree of hom poly in Lie map
        if degree <= comp_degree:
            degree = comp_degree

    for i in range(2,degree+2):
        T = taylor_to_lie(taylor, i) #coeff match to get hom poly in Lie product maps
        LieProduct.append(T) #Lie maps product as array
        taylor = transform_taylor(T, taylor, i, degree) #adjust higher order taylor coeff for next coeff extraction

        if i > 5:
            print('Implemented only to 5th order so far.')
            break
            
    return LieProduct



def taylor_to_lie(taylor,degree): #decide if linear or nonlinear
    
    if degree == 2:
        f = deg2_lie(taylor) #linear case
        
    else:
        f = degN_lie(taylor,degree) #higher order maps
    
    return f



def deg2_lie(taylor): #extract matrix from linear part of the taylor polynomial
    I = np.identity(6)
    R = []
    i = len(taylor)

    for polynomial in taylor:
        p = poly(polynomial, sym_x,sym_px,sym_y,sym_py,sym_z,sym_pz)
        monomials = p.monoms()
        coeffs = p.coeffs()
        
        Matrow = []
        for j,row in enumerate(I):
            try:
                index = monomials.index(tuple(row))
                Matrow.append(coeffs[index])
            except:
                if j < i: #make sure the matrix stays quadratic
                    Matrow.append(0)

        R.append(Matrow)
        
    
    R = np.asarray(R, dtype='float') #convert to numpy so linalg can invert it
    
    return R



def degN_lie(taylor, degree): #higher order Lie maps
    _epstemp = sympy.symbols('e')
    variables = (sym_x, sym_px, sym_y, sym_py, sym_z, sym_pz)
    derivatives = (1,0,3,2,5,4) #order: d/dpx, d/dx, d/dpy, d/dy, d/dz, d/dz
    f = poly(0, sym_x,sym_px,sym_y,sym_py,sym_z,sym_pz) #Lie map hom poly generated from taylor
    
    for var,polynomial in enumerate(taylor):
        p = poly(polynomial, sym_x,sym_px,sym_y,sym_py,sym_z,sym_pz) #make it a sympy polynomial
        order = [sum(mon) for mon in p.monoms()] #array of monomial degrees
        for index,monomial in enumerate(p.monoms()): #iterate over all monomials in taylor
            if order[index] == (degree-1): #check hom level -> derivative is order - 1
#                 print(variables,monomial)
                mon = prod(a**b for a,b in zip(variables,monomial)) #reconstruct monomial
                if (f.coeff_monomial(mon * variables[derivatives[var]])) == 0: #avoid double sum of same coeff
                    power = monomial[derivatives[var]] #normalize derivative power
                    f = f + (p.coeffs()[index] / (power+1)) * mon * variables[derivatives[var]] * (-1)**(var)
                    
    return f.subs(_epstemp,0)



def transform_taylor(ham, taylor, hom_order, degree=3): #adjust higher order coeffs
    #getaround for .subs() being iterative
    sym_x1, sym_y1, sym_z1, sym_px1, sym_py1, sym_pz1 = sympy.symbols('x_1 y_1 z_1 p_{x1} p_{y1} \delta_1')
    variables = (sym_x, sym_px, sym_y, sym_py, sym_z, sym_pz)
    new_variables = (sym_x1, sym_px1, sym_y1, sym_py1, sym_z1, sym_pz1)
    
    if hom_order == 2: #linear case needs more checking
        R_inv = np.linalg.inv(ham)
        vec = [new_variables[i] for i in range(len(R_inv))]
        
        new_coords = np.dot(R_inv, vec) #exp(-:G_2:) z_1 = z_0 + higher orders
        int_taylor = [polynomial.subs([(i,j) for i,j in zip(variables,new_coords)]) for polynomial in taylor]
        taylor = [polynomial.subs([(i,j) for i,j in zip(vec,variables)]) for polynomial in int_taylor]
    else: #higher order
        LiePoly = LieOperator(-ham,[sym_x,sym_y,sym_z],[sym_px,sym_py,sym_pz]) #use hom poly ham of degree = hom_order
        mod_taylor = taylorize(LiePoly, degree+1) #create symplectic jet to adjust the coeffs of taylor polynomial
        taylor = [old_poly - new_poly for old_poly,new_poly in zip(taylor,mod_taylor)] #exp(-:G_n:)z -> subtract from taylor poly the symplectic jet
    
    return taylor

def taylorize(LieHam, degree): #Apply Lie map to get taylor map vector on 6d vector
    taylor_maps = []
    
    for i in LieHam.indep_coords:
        fct = LieHam.LieMap(i,degree).doit()
        fct = truncate(fct,degree)
        
        taylor_maps.append(fct.expand())
        
    for i in LieHam.indep_mom:
        fct = LieHam.LieMap(i,degree).doit()
        fct = truncate(fct,degree)
        
        taylor_maps.append(fct.expand())
    
    # reorder taylor maps to x, px, y, py, z, pz
    if len(LieHam.indep_coords) == 2:
        taylor_maps[1], taylor_maps[2] = taylor_maps[2], taylor_maps[1] #swtich y and px
        
    elif len(LieHam.indep_coords) == 3:
        taylor_maps[1], taylor_maps[3] = taylor_maps[3], taylor_maps[1] #swtich y and px
        taylor_maps[2], taylor_maps[3] = taylor_maps[3], taylor_maps[2] #switch z and y
        taylor_maps[3], taylor_maps[4] = taylor_maps[4], taylor_maps[3] #switch z and py
    
    return taylor_maps



def truncate(LieHam,degree): #cutoff Hamiltonian at specified degree
    _epstemp = sympy.symbols('e')
    fct = LieHam.ham
    
    for i in LieHam.indep_coords:
        fct = fct.subs(i,i*_epstemp)
        
    for i in LieHam.indep_mom:
        fct = fct.subs(i,i*_epstemp)
    
    fct = fct.expand() + O(_epstemp**degree)
    fct = fct.removeO().subs(_epstemp,1)
    
    return fct


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
        reduced, red_ind = cust_reduce(state_i)
        if dim_reduction:
            powers.append(reduced)
        else:
            powers.append(state_i)
            
        index.append(red_ind)

    powers.insert(0, np.array([1]))
    index.insert(0, [True])
    return powers, index 

def cust_reduce(state):
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


def taylor_to_weight_mat(_taylor):
    degree = 0
    for polynomial in _taylor:
            comp_degree = poly(polynomial).total_degree() #highest degree of hom poly in Lie map
            if degree <= comp_degree:
                degree = comp_degree

    if len(_taylor) == 2:
        coords = [sym_x,sym_px]
    elif len(_taylor) == 4:
        coords = [sym_x,sym_px,sym_y,sym_py]
    elif len(_taylor) == 6:
        coords = [sym_x,sym_px,sym_y,sym_py,sym_z,sym_pz]
    else:
        raise TypeError('The dimension of the Taylor map vector does not match the phase space.')

    state_vectors, index = getKronPowers(coords,degree,dim_reduction=False)
    
#     print(_taylor)
#     print(coords)
    
    #print(state_vectors)

    W = []

    for i in range(degree+1):
        if i == 0:
#             print("Displacement not yet programmed.")
            continue

        w_sub = np.zeros((len(_taylor),len(state_vectors[i])))

        for j, taylor_row in enumerate(_taylor):
            taylor_sub = poly(taylor_row, state_vectors[1])

            for k, (state_vector, flag) in enumerate(zip(state_vectors[i],index[i])):
                coeff = taylor_sub.coeff_monomial(state_vector)

                if coeff != 0 and flag == True:
#                     print(coeff)
                    w_sub[j,k] = coeff
        W.append(w_sub.T)
    
    return W