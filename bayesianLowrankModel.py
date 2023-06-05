# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:43:24 2017

@author: 
"""

__author__ = 'an'
import numpy as np
from numpy.matlib import repmat
from normwish1 import *
import scipy.special as ssp

def bayesianLowrankModel(data,params,gammas,K,R,W):
    D = data.shape[1];
    N =data.shape[0];
    #K = (params['a']).shape[0];
    a0 = D;
    beta0 = 1;
    mean0 = np.mean(data,axis=0);
    B0 = .1 * D * np.cov(data.T);
    pob = np.zeros((N, K))
    A=params['A']
    B = params['B_1']
    lambdaR=np.random.rand(1,R)*20;

    count=0;
    while count<1:
        #convenience variables first
        A = np.dot(gammas,B);
        B = np.dot(gammas.T,A);

        Ns = np.sum(gammas,axis=0) + 1e-10;
        mus = np.zeros((K,D))
        sigs = np.zeros((D,D,K))
        mus = np.dot(gammas.T , data) /(repmat(Ns,D,1).T)
        for i in range(K):
            diff0 = data - repmat(mus[i,:],N,1);
            diff1 = repmat(np.sqrt(gammas[:,i]),D,1).T * diff0;
            sigs[:,:,i] = np.dot(diff1.T , diff1);

        params['beta'] = Ns + beta0;
        params['a'] = Ns + a0;
        tempNs=repmat(Ns,D,1).T * mus
        for k in range(K):
            if k>1:
                params['mean'][k,:] = ( tempNs[k]+ beta0 * mean0) / (repmat(Ns[k] + beta0,D,1).T)
            else:
                params['mean'][k,:] = ( tempNs[k]+ beta0 * mean0) / (repmat(Ns[k] + beta0,D,1).T)

        #for one dimension

        for i in range(K):
            diff = mus[i,:] - mean0
            params['B'][:,:,i] = sigs[:,:,i] + Ns[i] * beta0 * np.dot(diff,diff.T) / (Ns[i]+beta0) + B0

        # print(params['mean'][:,:])
        for i in range(K):
            #eq_log_Vs[i] = ssp.psi(params['g'][i,0]) - ssp.psi(params['g'][i,0]+params['g'][i,1]);
            #eq_log_1_Vs[i] = ssp.psi(params['g'][i,1]) - ssp.psi(params['g'][i,0]+params['g'][i,1]);
            #log_V_prob[i] = eq_log_Vs[i] + np.sum(eq_log_1_Vs[np.arange(i)])
            pob[:,i] = normwish(data,params['mean'][i,:],params['beta'][i],params['a'][i],params['B'][:,:,i]);
        gammas=pob
        tempP =np.dot (A,B.T);
        tempP = tempP / repmat(np.sum(tempP, axis=1)+0.01, K, 1).T;
        gammas = np.exp(tempP+ gammas);
        pob=-1*abs(pob)
        gammas = gammas / repmat(np.sum(gammas, axis=1), K, 1).T;
        # print(pob)
        # print(gammas)
        # for k in range(N):
        #     print(pob[k,:])
        # ############################################################################################################
        u_SVD, s_SVD, v_SVD = np.linalg.svd(gammas, full_matrices=True)


        D_u=(u_SVD.shape)[0]
        D_s=(s_SVD.shape)[0]
        S=s_SVD+60
        s_SVD = np.zeros((D_u, D_s))
        s_SVD[:D_s, :D_s] = np.diag(S)
        #s_SVD = s_SVD + 10.00

        #s_SVD = s_SVD + abs(s_SVD);
        #s_SVD = s_SVD / 2;

        gammas=np.dot(u_SVD, np.dot(s_SVD, v_SVD))
        gammas = (gammas + abs(gammas))/2;
        # ######################################################################

        # print(gammas)
        gammas = gammas / repmat(np.sum(gammas, axis=1), K, 1).T;

        # cho = 0;
        # lambdaRate = 0.01;
        # if cho == 1:
        #     count1 = 10
        #     while count1 != 0:
        #         gammas1 = (1 - lambdaRate)* gammas;
        #         W1 = repmat((repmat(np.sum(W, 1),1,1)).T, 1, K);
        #         gammas2 = lambdaRate*(np.dot(W , gammas));
        #         temp=(gammas2)/ W1
        #         #print(temp.shape)
        #         gammas = gammas1 + (gammas2)/ W1;
        #
        #         if (count1% 10) == 0:
        #             lambdaRate = 0.9 * lambdaRate;
        #         count1 = count1 - 1;
        # gammas = gammas / repmat(np.sum(gammas, axis=1), K, 1).T;
        #computing the loss, we neglect the normalization term
        tempP = np.exp(tempP);
        tempP = tempP / repmat(np.sum(tempP, axis=1), K, 1).T;
        count=count+1;
        temp1=np.dot(A,B)
        temp1 = temp1 / (repmat(np.sum(temp1, axis=1), K, 1).T+0.1);
        temp3=0
        for i in range(K):
            temp3=temp3+np.log(params['beta'][i])-D*params['a'][i]/2

        temp2=0
        for i in range(K):
            temp2=temp2-((a0-1)/2)*(np.linalg.slogdet(params['B'][:,:,i])[1])
        loss=gammas*pob-gammas*np.log(gammas+1)+gammas*temp1
        loss=np.sum(loss+temp2-temp3)
        tempP=loss
        # print(loss+lossA+lossB)
    return gammas,params,tempP
