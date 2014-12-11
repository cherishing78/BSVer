import scipy.io as sio
import numpy as np
import os
from noloop import *

p1='../../WDRef/lbp_WDRef.mat'
data=sio.loadmat(p1)
x=data['lbp_WDRef']
#Substracted tge mean of all the faces.
x=x-x.mean(0)

p2='../../WDRef/id_WDRef.mat'
data=sio.loadmat(p2)
label=data['id_WDRef']

# For Debug
x=x[0:50,:]
label=label[0:50,:]

Su,Se=Noloop(x,label)
x1=np.transpose(x[0,:])
x2=np.transpose(x[1,:])

print Su,Se
