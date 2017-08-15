import numpy as np
import os
from chainer import cuda

def extract_2classes(tupledata,class0,class1):
    assert isinstance(tupledata[0],tuple),"data type error: extract_2clases"
    out=[]
    for data in tupledata:
        label=data[1]
        if label==class0:
            out.append((data[0],-1))
        elif label==class1:
            out.append((data[0],1))
    return out
def add_noise(x,std,seed=0):
    xp = cuda.get_array_module(x)
    xp.random.seed(seed)
    return x + xp.random.standard_normal(x.shape).astype(xp.float32) * std
def mkdir(path):
    if os.path.isdir(path):
        return True
    elif os.path.isfile(path):
        return False
    else:
        os.mkdir(path)
    return True
def log(data_list,filename,initialize=False):
    if initialize:
        f = open(filename,'w')
    else:
        f = open(filename,'a')
    sp = ','
    ln = os.linesep
    for data in data_list:
        f.write(str(data)+sp)
    f.write(ln)
    f.close()
def log_influence(path,L,t,y,d):
    f=open(path,'w')
    sp = ','
    ln = os.linesep
    f.write("label"+sp+"pred_y"+sp+"distance"+sp+"dL"+ln)
    for i in range(len(L)):
        f.write(str(t[i])+sp+str(y[i])+sp+str(d[i])+sp+str(L[i])+ln)
    f.close()