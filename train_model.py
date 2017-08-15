#!/usr/bin/env python

from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,training
from chainer.training import extensions
from chainer.dataset import convert
import numpy as np
from chainer import serializers
import mytools
import model

def evaluate(test_x,test_y,predictor,batchsize=100):
    xp = predictor.xp
    acc=0
    for i in range(0,len(test_y),batchsize):
        x = chainer.Variable(xp.array(test_x[i:i+batchsize]),volatile = 'on')
        t = xp.array(test_y[i:i+batchsize])
        y = predictor(x).data.reshape(-1)
        pred_class = 2*(y>0.5) - 1
        acc += xp.sum(pred_class==t)
    acc /= float(len(test_y))
    return acc

def main():
    parser = argparse.ArgumentParser(description='train ')
    parser.add_argument('--iteration', '-i', type=int, default=100000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=1000,
                        help='Frequency of logging')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='output directry')
    parser.add_argument('--model_name', '-mn', default='model.npz',
                        help='output model name')
    args = parser.parse_args()

    batchsize=100
    # define two classes to discriminate
    class0=4
    class1=9

    log_freq=args.frequency
    clf = model.LogisticRegression()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        clf.to_gpu()  # Copy the model to the GPU
    xp = cuda.cupy if args.gpu>=0 else np
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(clf)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train = mytools.extract_2classes(train,class0,class1)
    test  = mytools.extract_2classes(test,class0,class1)
    train_x,train_y = convert.concat_examples(train,-1)
    test_x ,test_y  = convert.concat_examples(test ,-1)
    # x:[0,1]->[-1,1]
    train_x = 2*train_x-1
    test_x = 2*test_x-1
    
    # add noise to digit image to avoid getting Hessian rank deficient.
    train_x = mytools.add_noise(train_x,0.1,seed=0)
    test_x = mytools.add_noise(test_x,0.1,seed=0)
    
    N = len(train_y)
    print('train num: ' +str(N))
    print('test num: '  +str(len(test_y)))
    
    #initialize log files
    if not mytools.mkdir(args.out):
        print('could not create directory')
        return
    log_filename=args.out+'/'+args.model_name+'_train_log.txt'
    mytools.log(['iteration','loss','train_accuracy','test_accuracy'],log_filename,initialize=True)
    perm = np.random.permutation(N)
    mean_loss=0
    idx=0

    # main loop 
    for itr in range(args.iteration):
        mean_loss=0
        denominator=0
        # load batch and update index
        if idx+batchsize > N:
            idx=0
            perm = np.random.permutation(N)
        x = chainer.Variable(xp.array(train_x[perm[idx:idx+batchsize]]).astype(xp.float32))
        t = chainer.Variable(xp.array(train_y[perm[idx:idx+batchsize]]).astype(xp.float32))
        t = t.reshape(-1,1)
        idx += batchsize

        # predict
        # cross entropy loss
        thetax = clf(x,apply_sigmoid=False)
        loss = F.sum( F.log(1 + F.exp(-t*thetax)))/t.shape[0]
        # update model
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()
        
        mean_loss += loss.data * t.shape[0]     
        denominator += t.shape[0]
        # report
        if itr%log_freq==0:
            mean_loss /= denominator
            train_acc = evaluate(train_x,train_y,clf)
            test_acc = evaluate(test_x,test_y,clf)
            mytools.log([itr,mean_loss,train_acc,test_acc],log_filename)
            print("iteration: "+str(itr)+"  loss:"+str(mean_loss)+"  train_acc:"+str(train_acc)+"  test_acc:"+str(test_acc))
            mean_loss=0
            denominator=0

        if itr%(log_freq*10)==0:
            serializers.save_npz(args.out+'/'+args.model_name, clf)  
            
if __name__ == '__main__':
    main()