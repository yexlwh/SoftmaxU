__author__ = 'ye'
import numpy as np
from numpy.matlib import repmat

def lr_init(testData,K,R):
    params={}
    num,dim=testData.shape
    gammas = np.random.rand(num,K)
    temp=repmat(np.sum(gammas,axis=1),K,1)
    temp1=temp.transpose()
    gammas = gammas /temp1
    params['eq_alpha'] = 50
    params['beta'] = np.zeros((K,1))
    params['a'] = np.zeros((K,1))
    params['meanN'] = np.zeros((dim,K))
    params['B'] = np.zeros((dim,dim,K))
    params['sigma'] = np.zeros((dim,dim,K))
    params['mean'] = np.zeros((K,dim))
    params['g'] = np.zeros((K,2))
    params['ll'] = -np.inf
    params['A']=np.random.rand(num,R)*10;
    params['B_1']=np.random.rand(K,R)*10;
    return params,gammas
def lr_init_withlabel(testData,label,K,R):
    params={}
    num,dim=testData.shape
    gammas = np.random.rand(num,K)
    temp=repmat(np.sum(gammas,axis=1),K,1)
    temp1=temp.transpose()
    gammas = gammas /temp1
    params['eq_alpha'] = 50
    params['beta'] = np.zeros((K,1))
    params['a'] = np.zeros((K,1))
    params['meanN'] = np.zeros((dim,K))
    params['B'] = np.zeros((dim,dim,K))
    params['sigma'] = np.zeros((dim,dim,K))
    params['mean'] = np.zeros((K,dim))
    uniqueLabel=np.unique(label)
    for i in uniqueLabel:
      count=0
      for j in range(num):
        if i==label[j]:
          count=count+1
          params['mean'][i,:]=params['mean'][i,:]+testData[j,:]
      params['mean'][i,:]=params['mean'][i,:]/count
    for i in uniqueLabel:
      count=0
      for j in range(num):
        if i==label[j]:
          count=count+1
          params['B'][:,:,i]=params['B'][:,:,i]+(testData[j,:]-params['mean'][i,:]).T*(testData[j,:]-params['mean'][i,:])
      params['B'][:,:,i]=params['B'][:,:,i]/count
    params['g'] = np.zeros((K,2))
    params['ll'] = -np.inf
    params['A']=np.random.rand(num,R)*10;
    params['B_1']=np.random.rand(K,R)*10;
    return params