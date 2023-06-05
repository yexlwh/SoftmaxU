"""
All rights reserved.
"""
#export CUDA_VISIBLE_DEVICES=1
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi, thrC,post_proL
import scipy.io as sio
import math
from lr_init import *
from sklearn.manifold import spectral_embedding
from bayesianLowrankModel import *
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

K2=40
KD=30

class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp

        return loss

class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        #super(FcNet,self)
        self.hidden1=nn.Linear(KD,40)
        #self.hidden1 = torch.nn.ModuleList(self.hidden1)
        #self.hidden2=nn.Linear(40,50)
        #self.hidden2 = torch.nn.ModuleList(self.hidden2)
        self.hidden3=nn.Linear(40,K2)
        #self.hidden3 = torch.nn.ModuleList(self.hidden3)
    def forward(self, x):
        x1 = F.relu(self.hidden1(x))
        #x2 = F.relu(self.hidden2(x1))
        x3 = torch.softmax(self.hidden3(x1),dim=0)
        return x3
    def loss_fn(self, xt, yt,regZero):
        loss = F.mse_loss(yt, xt, reduction='mean')+1*F.mse_loss(xt*xt, regZero, reduction='mean')
        return loss
        

def train(model,  # type: DSCNet
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Nz,Dz,_,_=x.shape
    print(Nz)
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    FcModel=FcNet()
    FcModel=FcModel.to(device)
    
    #FCNN1=FullyConnectedNuralNetwork()
    K1=K2
    numlabel=30
    numlabel_val=40
    data_ohe=np.zeros((Nz,K1))
    
    of=OneHotEncoder(sparse=False).fit(y.reshape(-1,1))
    data_ohe1=of.transform(y.reshape(-1,1))
    shapeOhe1=data_ohe1.shape
    data_ohe=np.zeros((Nz,K1))
    data_ohe[:,0:shapeOhe1[1]]=data_ohe1[:,0:shapeOhe1[1]]
    
    
    select=np.array([0,72,144])#,216,288,360,432,502,576,648,720,792,864,936,1008#,216,288,360,432,502,576,648#,432,502,576,648,720,792,864,936,1008,1080,1152,1224
    print(select.shape)
    Ns = select.shape
    Ns=Ns[0]
    print(Ns)
    y_train=np.zeros((Ns*numlabel,K1))
    f_train=np.zeros((Ns*numlabel,KD))
    print(Ns)
    
    #disReal = kneighbors_graph(x, Knum, mode='connectivity', include_self=False)
    #maps1 = spectral_embedding(disReal, n_components=90)
    flag=0
    optimizer2 = optim.Adam(FcModel.parameters(), lr=0.001)
       
    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            #y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            
            C=post_proL(C, K1, dim_subspace, ro)
            maps = spectral_embedding(C+C.T, n_components=KD)
            if flag==0 and epoch<25:
                y_train=np.zeros((Ns*numlabel,K1))
                f_train=np.zeros((Ns*numlabel,KD))
                for i in range(Ns):
                    y_train[i*numlabel:i*numlabel  + numlabel, :] = data_ohe[select[i]:select[i] + numlabel, :]
                    f_train[i*numlabel:i*numlabel  + numlabel, :] = maps[select[i]:select[i] + numlabel, :]
                #flag=1
                per=np.random.permutation(Ns*numlabel)
                #print(per)
                y_train=y_train[per,:]
                f_train=f_train[per,:]
                pMaps=torch.tensor(maps, dtype=torch.float32, device=device)
    
                if not isinstance(f_train, torch.Tensor):
                    f_train = torch.tensor(f_train, dtype=torch.float32, device=device)
                if isinstance(y_train, torch.Tensor):
                    y_train = y_train.to('cpu').numpy()
                if not isinstance(y_train, torch.Tensor):
                    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
                y_train = y_train.to(device)
            
            
            
            numIter=0
            while numIter<100:
              lossT=0
              for j in range(Ns):
                x_train1 = FcModel(f_train[j*numlabel:j*numlabel  + numlabel,:])
                y_train1 = y_train[j*numlabel:j*numlabel  + numlabel,:]
                regZero=np.zeros((numlabel,K2))
                if not isinstance(regZero, torch.Tensor):
                    regZero = torch.tensor(regZero, dtype=torch.float32, device=device)
                loss1 = FcModel.loss_fn(x_train1, y_train1,regZero)
                optimizer2.zero_grad()
                loss1.backward()
                optimizer2.step()
                lossT=lossT+loss1.item()
              #print(lossT)
              numIter=numIter+1
            
            
            gammas_y=FcModel.forward(pMaps).detach().to('cpu').numpy()
            gammas_y = gammas_y / repmat(np.sum(gammas_y,axis=1),K1,1).T
            gammasC=gammas_y
            gammas_y_temp=FcModel.forward(pMaps)
            yPredStageOne=torch.argmax(gammas_y_temp,dim=1).detach().to('cpu').numpy()
            params = lr_init_withlabel(maps,yPredStageOne,K1,K1)
            numIter=0
            W=0
            #params,gammas = vdpmm_init(maps,K1)
            while numIter<100:
              lossT=0
              for j in range(Ns):
                x_train1 = FcModel(f_train[j*numlabel:j*numlabel  + numlabel,:])
                y_train1 = y_train[j*numlabel:j*numlabel  + numlabel,:]
                regZero=np.zeros((numlabel,K2))
                if not isinstance(regZero, torch.Tensor):
                    regZero = torch.tensor(regZero, dtype=torch.float32, device=device)
                loss1 = FcModel.loss_fn(x_train1, y_train1,regZero)
                optimizer2.zero_grad()
                loss1.backward()
                optimizer2.step()
                lossT=lossT+loss1.item()
                
              gammas_y=FcModel.forward(pMaps).detach().to('cpu').numpy()
              numIter=numIter+1
             
              gammasC, params, P = bayesianLowrankModel(maps, params, gammasC, K1, K1, W)
              gammas_y = gammas_y / repmat(np.sum(gammas_y,axis=1),K1,1).T
              data_oheGammas=np.zeros((Nz,K1))
    
              gammas_yTemp=FcModel.forward(pMaps)
              yGammasTemp=torch.argmax(gammas_yTemp,dim=1).detach().to('cpu').numpy()
              
              ofGammas=OneHotEncoder(sparse=False).fit(yGammasTemp.reshape(-1,1))
              data_oheGammas1=ofGammas.transform(yGammasTemp.reshape(-1,1))
              shapeOheGammas1=data_oheGammas1.shape
              data_oheGammas=np.zeros((Nz,K1))
              data_oheGammas[:,0:shapeOheGammas1[1]]=data_oheGammas1[:,0:shapeOheGammas1[1]]
              #gammas_y=data_oheGammas*0.5
              
              #print('begin:')
              #print(gammas_y[1,:])
              #print(gammasC[1,:])
              gammasC=gammasC+gammas_y
              #print(gammasC[1,:])
              gammasC = gammasC / repmat(np.sum(gammasC,axis=1),K1,1).T
              #print(gammas_y[1,:])
              #print(gammasC[1,:])
              
              
              
            posGaussian=gammasC
            
            if isinstance(posGaussian, torch.Tensor):
                posGaussian = posGaussian.to('cpu').numpy()
            if not isinstance(posGaussian, torch.Tensor):
                posGaussian = torch.tensor(posGaussian, dtype=torch.float32, device=device)
            posGaussian = posGaussian.to(device)
            
            y_pred=torch.argmax(posGaussian,dim=1).detach().to('cpu').numpy()

            #[Nz,Dz]=posGaussian.shape

            #temp=np.max(posGaussian,axis=1)
            #temp.shape=(Nz,1)
            #storeIndex=np.zeros((1,Nz),dtype='int32')
            #print(storeIndex.shape)
            #for i in range(Nz):
            #    index1=np.where(temp[i]==posGaussian[i,:])
            #    storeIndex[0,i]=int(index1[0][0])
            #y_pred=np.squeeze(storeIndex)
            #y_pred = kmeans.fit_predict(maps)
            print('Unique label',(np.unique(y_pred)).shape)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                  (epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred)))
            Np=y.shape
            yTemp = np.zeros(y.shape)
            y_pred_temp = np.zeros(y.shape)
            print(Ns)
            #print(y)
            #print(y_pred)
            for j in range(Np[0]):
                #print(j)
                if y[j]<Ns:
                    yTemp[j]=0
                else:
                    yTemp[j]=1
                if y_pred[j]<Ns:
                    y_pred_temp[j]=0
                else:
                    y_pred_temp[j]=1
                    
            
            np.set_printoptions(threshold=np.inf)
            #print(yTemp)
            #print(y_pred_temp)
            print('F1 score',f1_score(yTemp, y_pred_temp))
            
            #y_pred1=FcModel.forward(f_train)
            #y1=torch.argmax(y_pred1,dim=1).detach().to('cpu').numpy()
            print(y_pred.shape)
            y1=np.zeros((Ns*(numlabel+numlabel_val)))
            y2=np.zeros((Ns*(numlabel+numlabel_val)))
            for j in range(Ns):
                y1[j*(numlabel+numlabel_val):j*(numlabel+numlabel_val)  + (numlabel+numlabel_val)] = y_pred[select[j]:select[j] + (numlabel+numlabel_val)]
                y2[j*(numlabel+numlabel_val):j*(numlabel+numlabel_val)  + (numlabel+numlabel_val)] = y[select[j]:select[j] + (numlabel+numlabel_val)]
            #y2=torch.argmax(y_train,dim=1).detach().to('cpu').numpy()
            countC=0
            for j in range(Ns*(numlabel+numlabel_val)):
                if y1[j]==y2[j]:
                    countC=countC+1
    
    
    
    


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='coil20',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db
        # load data
    data = sio.loadmat('COIL20.mat')
    xOriginal, y = data['fea'], data['gnd']
        
    x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        

        # network and optimization parameters
    num_sample = x.shape[0]
    channels = [1, 15]
    kernels = [3]
    epochs = 100
    weight_coef = 1.0
    weight_selfExp = 75

    # post clustering parameters
    alpha = 0.04  # threshold of C
    dim_subspace = 12  # dimension of each subspace
    ro = 8  #
    warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")

    dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels)
    dscnet.to(device)

    # load the pretrained weights which are provided by the original author in
    # https://github.com/panji1990/Deep-subspace-clustering-networks
    ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % db)
    dscnet.ae.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")
    #y_pred = KMeans(n_clusters=40).fit_predict(x1)
    #print(nmi(y_pred,y))

    train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
          alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
    torch.save(dscnet.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)
