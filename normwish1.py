import numpy as np
import scipy.special as ssp
from numpy.matlib import repmat

def normwish(x,mu,lambda1,a,B):
    #print('hello world')
    N,D = x.shape
    Gamma_bar = 0.5 * a * np.linalg.inv(B+np.eye(D) )
    d = x - np.squeeze(repmat(mu,N,1)).reshape(N,D);
    logprob = -.5 * D * np.log(2*np.pi) - .5 * (np.linalg.slogdet(.5*B))[1] +\
    .5 * np.sum(ssp.psi(0.5 * (a +1- np.arange(D)))) - 0.5 * D / lambda1- \
    np.sum((np.dot(d , Gamma_bar))*d,axis=1);
    logprob.reshape(N,1)
    return logprob
