#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:45:08 2020

@author: manal
"""

import numpy as np
# from GPy import *
import GPy
from GPy.kern.src.stationary import Stationary
import matplotlib.pyplot as plt

class Sinus(Stationary):
    """
    Sinus kernel:

    .. math::

       k(r) = \sigma^2 \sin ( r )
       
       r(x, x') = \\sqrt{ \\sum_{q=1}^Q \\frac{(x_q - x'_q)^2}{\ell_q^2} }

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Sinus'):
        super(Sinus, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
    
    def K_of_r(self, r):
        return self.variance * np.sin(r)


    def dK_dr(self, r):
        return self.variance * np.cos(r)
    
    def dK_dX(self, X, X2, dimX):
        r = self._scaled_dist(X, X2)
        dK_dr = self.dK_dr(r)
        dist = X[:,None,dimX]-X2[None,:,dimX]
        lengthscale2inv = (np.ones((X.shape[1]))/(self.lengthscale**2))[dimX]
        return lengthscale2inv*dist*r**(-1)*dK_dr
        # return self.variance*lengthscale2inv*np.cos(r)*r**(-1)*dist
    
    def dK_dX2(self,X,X2,dimX2):
        return -1.*self.dK_dX(X,X2, dimX2)
    
    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        if X2 is None:
            X2 = X
        r = self._scaled_dist(X, X2)
        dist = X[:,None,:]-X2[None,:,:]
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)
        K = self.K_of_r(r)
        dK_dr = self.dK_dr(r)
        l1 = lengthscale2inv[dimX]
        l2 = lengthscale2inv[dimX2]
        d1 = dist[:,:,dimX]
        d2 = dist[:,:,dimX2]
        return (dimX!=dimX2)*self.variance*d1*l1*d2*l2*(r*np.sin(r)+np.cos(r))*r**(-3) + (dimX==dimX2)*self.variance*l1*(-r**2*np.cos(r)+r*d1**2*l1*np.sin(r)+d1**2*l1*np.cos(r))*r**(-3)
  

#%%

import func
# import math

def test_kern_Sinus():
    """
    .. math::
        \dfrac{-\dfrac{cos(r)}{r} + \dfrac{(x_a - x_b)^2}{lx^2}*\dfrac{sin(r)}{r^2} + \dfrac{(x_a - x_b)^2}{lx^2}*\dfrac{cos(r)}{r^3}}{lx^2}
    Returns
    -------
    None.

    """
    x0train = np.linspace(-5,5,100).reshape(-1,1)
    x1train = np.linspace(-2,2,100).reshape(-1,1)
    x2train = np.linspace(0,9,100).reshape(-1,1)
    x3train = np.linspace(8,18,100).reshape(-1,1)
    xtrain = np.hstack((x1train, x3train))#, x2train))#, x3train))
    
    k0 = Sinus(1)#, active_dims=1)
    k0_K = k0.K(x1train, x3train)
    dk0 = k0.dK_dX(x1train, x3train, 0)
    dk0_2 = k0.dK2_dXdX2(x1train, x3train, 0, 0)
    
    # # k0 = Cosin(1, active_dims=0)
    # k0 = Sinus(2)#, active_dims=1)
    # k0_K = k0.K(xtrain,xtrain)
    # dk0 = k0.dK_dX(xtrain, xtrain, 1)
    # dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 1, 1)
    
    l = np.array([1, 1])
    
    K = np.zeros((len(x0train),len(x0train)))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = func.f_kern(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)
    
    dK = np.zeros((len(x0train),len(x0train)))
    for i in range(dK.shape[0]):
        for j in range(dK.shape[1]):
            dK[i,j] = func.dkdx(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)
    
    dK2 = np.zeros((len(x0train),len(x0train)))
    for i in range(dK2.shape[0]):
        for j in range(dK2.shape[1]):
            dK2[i,j] = func.d2kdxdx0(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)
    
    print(np.isnan(dK[0,0]))
    # plt.figure()
    # plt.imshow(K)
    # plt.figure()
    # plt.imshow(k0_K)
    plt.figure()
    plt.plot(K[49,:])
    plt.title('K sympy' '\n' 'order 2')
    plt.figure()
    plt.plot(k0_K[49,:])
    plt.title('K' '\n' 'order 2')
    
    # plt.figure()
    # plt.imshow(dK)
    # plt.figure()
    # plt.imshow(dk0)
    plt.figure()
    plt.plot(dK[49,:])
    plt.title('1st derivative sympy' '\n' 'order 2')
    plt.figure()
    plt.plot(dk0[49,:])
    plt.title('1st derivative' '\n' 'order 2')
    
    # plt.figure()
    # plt.imshow(dK2)
    # plt.figure()
    # plt.imshow(dk0_2)
    plt.figure()
    plt.plot(dK2[49,:])
    plt.title('2nd derivative sympy' '\n' 'order 2')
    plt.figure()
    plt.plot(dk0_2[49,:])
    plt.title('2nd derivative' '\n' 'order 2')

    print(np.isclose(k0_K, K, rtol=1e-6),'\n')
    print(np.isclose(dk0, dK, rtol=1e-6),'\n')
    print(np.isclose(dk0_2, dK2, rtol=1e-6))
    # print(dk0_2)
    # print(id)
    

test_kern_Sinus()







