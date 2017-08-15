#!/usr/bin/env python


########################################################################
# confirm that pre-trained classifier exits as result/model.npz(defailt)
# to train classifier, use train_model.py.
########################################################################

from __future__ import print_function

try:
    import matplotlib 
    matplotlib.use('Agg')
    from matplotlib import pylab as plt
    
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
from chainer import cuda
from chainer.dataset import convert
import numpy as np
from chainer import serializers
import os
import mytools
import model

# add bias term to x. output shape: (x.shape[0],x.shape[1]+1)
def add_bias(x):
    assert isinstance(x,chainer.Variable)
    xp=cuda.get_array_module(x)
    x_bias = chainer.Variable(xp.ones((x.shape[0],1)).astype(xp.float32))
    return F.concat((x,x_bias), axis=1)

# sigmoid function for xp.array
def sigmoid(x):
    sigmoid_range = 34.538776394910684
    xp=cuda.get_array_module(x)
    x = xp.maximum(-sigmoid_range,xp.minimum(sigmoid_range,x))
    return 1.0/(1.0+xp.exp(-x))

#calculate H
def calc_H(train_x,train_y,clf,batchsize=100,xp=np):
    N = train_x.shape[0]
    acc=0
    H = xp.zeros((train_x.shape[1]+1,train_x.shape[1]+1)) # param_num x param_num
    for i in range(0,N,batchsize):
        x = chainer.Variable(xp.array(train_x[i:i+batchsize]).astype(xp.float32))
        t = xp.array(train_y[i:i+batchsize]).astype(xp.int32)
        y = clf(x)
        x += np.random.standard_normal(x.shape)*0.3
        # add bias term            
        x = add_bias(x)
        xxT = x.data.reshape(x.shape[0],1,x.shape[1]) * x.data.reshape(x.shape[0],x.shape[1],1)
        H += xp.sum(y.data.reshape(-1,1,1)*(1-y.data).reshape(-1,1,1)*xxT,axis=0)/N
        #calc accuracy
        pred_class = 2*(y.data.reshape(-1)>0.5)-1
        acc += xp.sum(pred_class==t)
    acc /= float(N)
    return H,acc

def calc_H_inv(H,lmd=0):
    H = H + lmd * np.eye(H.shape[0]) # add damping term
    H=np.matrix(H)
    H_inv=np.linalg.inv(H)
    H_inv=np.array(H_inv)
    return H_inv

# calculate s_test
def calc_s_test(x,y,H_inv,clf,xp=np):
    test_x = chainer.Variable(x.reshape(1,-1))
    test_t = chainer.Variable(y.reshape(1,1).astype(xp.float32))
    thetax = clf(test_x,apply_sigmoid=False)
    test_x = add_bias(test_x)
    return -(test_t.data * sigmoid(-test_t.data * thetax.data)*xp.sum(test_x.data*H_inv,axis=1)).reshape(1,-1)

# calculate Euclidean distance between train_x(shape: (N,x_dim)) and x(shape: (x_dim))
def calc_distance(train_x,x,xp=np):
    x=x.reshape(1,-1)
    return xp.sqrt(xp.sum((train_x-x)**2,axis=1))

# calculate nabla_L for training set
def calc_nabla_L(train_x,train_y,clf,batchsize=100,xp=np):
    N = train_x.shape[0]
    nabla_L= xp.zeros((N,train_x.shape[1]+1))
    y = xp.zeros(N)
    for i in range(0,N,batchsize):
        x = chainer.Variable(xp.array(train_x[i:i+batchsize]).astype(xp.float32))
        t = xp.array(train_y[i:i+batchsize].reshape(-1,1))
        thetax = clf(x,apply_sigmoid=False).data
        x = add_bias(x).data
        nabla_L[i:i+t.shape[0]] = - t * sigmoid(-t*thetax) * x # shape: (batchsize,x_dim+1)
        y[i:i+t.shape[0]] = sigmoid(thetax.reshape(-1)) # output value of classifier
    return nabla_L,y


# calculate dL = - 1/N Lup,loss(z_test,z)
def calc_dL(s_test,nabla_L,total_data_num,xp=np):
    return  xp.sum(s_test.reshape(1,-1)*nabla_L,axis=1).reshape(-1)/total_data_num


def main():
    parser = argparse.ArgumentParser(description='influence function')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_path', '-m', default='result/model.npz',
                        help='path to the model file(.npz)')
    parser.add_argument('--p', '-l', default='result/model.npz',
                        help='path to the model file(.npz)')
    parser.add_argument('--dont_log', action='store_false')
    args = parser.parse_args()

    image_size=28
    batchsize=100
    class0=4
    class1=9
    test_id=[0,1,10,40,41,2,4,26,39,93]
    '''
    class0=1
    class1=7
    test_id=[103,112,120,174,179,100,102,118,171,191]
    '''
    
    lmd=0 # damping term to calculate inverse matrix of H if H is not PD.
    show_rank3=True
    
    # load classifier
    clf = model.LogisticRegression()
    serializers.load_npz(args.model_path, clf)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        clf.to_gpu()  # Copy the model to the GPU
    
    xp = cuda.cupy if args.gpu>=0 else np

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train = mytools.extract_2classes(train,class0,class1)
    test  = mytools.extract_2classes(test,class0,class1)
    train_x,train_y = convert.concat_examples(train,-1)
    test_x ,test_y  = convert.concat_examples(test ,-1)
    train_x=2*train_x-1
    test_x=2*test_x-1

    # add noise to digit image to avoid getting Hessian rank deficient.
    train_x = mytools.add_noise(train_x,0.1,seed=0)
    test_x = mytools.add_noise(test_x,0.1,seed=0)
    
    train_x0=train_x[train_y==-1]
    train_y0=train_y[train_y==-1]
    train_x1=train_x[train_y==1]
    train_y1=train_y[train_y==1]
    
    N = len(train_y)
    print('train num: ' +str(N))
    print('test num: '  +str(len(test_y)))
    
    #calculate H
    if os.path.isfile(args.model_path+'_H.npy'):
        print('load H from ' + args.model_path+'_H.npy')
        H=np.load(args.model_path+'_H.npy')
    else:
        print('calculate H')
        H ,acc=calc_H(train_x,train_y,clf,batchsize=batchsize,xp=xp)    
        print('train accuracy: '+str(acc))
        np.save(args.model_path+'_H.npy',np.array(H))
    
    # calculate inverse matrix of H
    print('calculate H_inv')
    H_inv = calc_H_inv(H,lmd)
    
    # calculate nabla_L for all training data
    nabla_L0,y0 = calc_nabla_L(train_x0,train_y0,clf,batchsize=batchsize,xp=xp)
    nabla_L1,y1 = calc_nabla_L(train_x1,train_y1,clf,batchsize=batchsize,xp=xp)
    
    s=0.7 # size of a MNIST image in a result image.
    if show_rank3:
        plt.figure(figsize=(s*14, s*len(test_id)))
    else:
        plt.figure(figsize=(s*6, s*len(test_id)))
    
    for i in range(len(test_id)):
        print('calculate dL for test data '+str(test_id[i]))
        #calc s_test    
        s_test = calc_s_test(test_x[test_id[i]],test_y[test_id[i]],H_inv,clf,xp=xp)
          
        # calc dL    
        dL0=calc_dL(s_test,nabla_L0,N,xp=xp)
        dL1=calc_dL(s_test,nabla_L1,N,xp=xp)
        
        # draw image
        if show_rank3:
            arg0 = np.argsort(-dL0)
            arg1 = np.argsort(-dL1)
            
            imgs=[test_x[test_id[i]],                                       #test image
                  train_x0[arg0[0]],train_x0[arg0[1]],train_x0[arg0[2]],    # class0 helpful 
                  train_x0[arg0[-1]],train_x0[arg0[-2]],train_x0[arg0[-3]], # class0 harmful 
                  train_x1[arg1[0]],train_x1[arg1[1]],train_x1[arg1[2]],    # class1 helpful 
                  train_x1[arg1[-1]],train_x1[arg1[-2]],train_x1[arg1[-3]]] # class1 harmful
        else:
            imgs=[test_x[test_id[i]],          #test image
                  train_x0[np.argmax(dL0)],    # class0 helpful 
                  train_x0[np.argmin(dL0)],    # class0 harmful 
                  train_x1[np.argmax(dL1)],    # class1 helpful 
                  train_x1[np.argmin(dL1)]]    # class1 harmful
        plt.subplot(len(test_id),len(imgs)+1, 1+(len(imgs)+1)*i)
        plt.axis('off')
        plt.title('test'+str(test_id[i]))
        for j in range(len(imgs)):      
            plt.subplot(len(test_id),len(imgs)+1, 2+j+(len(imgs)+1)*i)
            plt.axis('off')
            plt.imshow(imgs[j].reshape(image_size,image_size), cmap='gray', interpolation='nearest')
        
        # log result
        if not args.dont_log:
            d0 = calc_distance(train_x0,test_x[test_id[i]],xp=xp)
            d1 = calc_distance(train_x1,test_x[test_id[i]],xp=xp)
            mytools.log_influence(args.model_path+'_test'+str(test_id[i])+'_class0.log',dL0,train_y0,y0,d0)
            mytools.log_influence(args.model_path+'_test'+str(test_id[i])+'_class1.log',dL1,train_y1,y1,d1)
    plt.savefig(args.model_path+"_image.png", dpi=100)
    
if __name__ == '__main__':
    main()