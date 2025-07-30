#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:06:37 2021

@author: pp423
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def create_x_0(eps,n,prior, mu_1=0,sigma_1=0):
    # Creates original, unknown data vector x_0
    x_0 = np.zeros(n)
    if prior=="Discrete":
        for i in range(len(x_0)):
            rand = np.random.random_sample()
            if rand < eps:
                #Entry takes value 1 or -1
                x_0[i] = np.sign(np.random.random_sample() - 0.5)
    elif prior=="Bernoulli-Gaussian":
        for i in range(len(x_0)):
            rand = np.random.random_sample()
            if rand < eps:
                #Entry takes value 1 or -1
                x_0[i] = np.random.normal(mu_1, sigma_1) 
    return x_0

def create_x_0_non0_mean(n, a, prob):
    # Creates original, unknown data vector x_0
    x_0 = np.ones(n)*(-a)
    for i in range(len(x_0)):
        rand = np.random.random_sample()
        if rand < prob:
            x_0[i] = a
    return x_0

def create_W(lam, omega, rho=0):
    #Creates a random lambda, omega base matrix
    C = lam
    R = lam + omega - 1
    
    W = rho*(1/(lam - 1))*np.ones((R,C))
    for c in range(C):
        for r in range(c, c+omega): #(c+w) index NOT included
            W[r,c]=(1-rho)*(1/omega)
    
    return W


def create_A_sc(W,N, delta_hat):
    R, C = np.shape(W)
    M = int(delta_hat*N)
    m, n = M*R, N*C
    A_sc = np.zeros((m,n))
    for c in range(C):
        for r in range(R):
            A_sc[M*r:(M*r+M), N*c:(N*c + N )] = np.random.normal(0, np.sqrt((1/M)*W[r,c]), (M, N))
    return A_sc


def create_A_iid(m,n):
    A_iid = np.random.normal(0,1/np.sqrt(m),size=(m,n))
    return A_iid

def create_y(A, x_0, sigma):
    m,n = np.shape(A)
    w = np.random.normal(0, sigma, size=m)
    y = np.dot(A,x_0) + w
    return y

