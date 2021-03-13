import numpy as np
import cv2 
import os
from conv import *
import multiprocessing
from multiprocessing import Pool
from itertools import product
from numba import njit
from functools import partial
import math
import sklearn
from sklearn import linear_model

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        images.append(cv2.imread(os.path.join(folder,filename),0))
    return images

def load_data(folder):
    images=[]
    n=len(os.listdir(folder))
    #print(n)
    output=[]
    iters = 0
    for filename in os.listdir(folder):
        path=folder+"\\"+filename
        pictures = load_images_from_folder(path)
        for pics in pictures:
            images.append(pics)
            y=np.zeros((n,1))

            y[iters,:] =1
            y.reshape(1,n)
           
            output.append(y)
        iters += 1
    
    return images,output

        
def convert(l):
    return (*l,)
def data_preprocessing(data,reshape_dim):
    for i in range(0,len(data)):
        data[i]=ConvNet(cv2.resize(data[i]/255,reshape_dim,interpolation=cv2.INTER_AREA))
        data[i]=data[i].reshape(data[i].size,1)
    return data
def prepare(data,reshape_dim,i):
    data[i]=ConvNet(cv2.resize(data[i]/255,reshape_dim,interpolation=cv2.INTER_AREA))
    data[i]=data[i].reshape(data[i].size,1)
def prepare_2(data):
    data=ConvNet(cv2.resize(data/255,(256,256),interpolation=cv2.INTER_AREA))
    data=data.reshape(data.size,1)
    return data
def parallel(data,reshape_dim):
    process=[]
    for i in range(len(data)):
       p=multiprocessing.Process(target=prepare,args=(data,reshape_dim,i))
       process.append(p)
    for x in process:
        x.start()
    for x in process:
        x.join()
    for i in data:
        print(i.shape)
    return data
def square(x):
    return x**2
def parallel_2(data,reshape_dim):
    x=0
    pool=Pool(4)
    x=pool.map(prepare_2,data)
    print(x)
    pool.close()
    pool.join()
    return x
def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True,initial=-np.inf))
    return e_Z / e_Z.sum(axis = 0)

def predict(X,weights):
    return softmax(weights.T@X)
def cross_entropy(y_hat, y):
    return - np.log(y_hat[range(len(y_hat)), y])
def update_weights(features,output,weights,learning_rate):
    predicted=predict(features,weights)
    print(features.shape)
    print(weights.shape)
    print(predicted.shape)
   #print(np.linalg.norm(predicted-output))
    weights=weights-learning_rate*(((output-predicted)@features.T).T)
    
    return weights
def Adam(features,output,weights,lr,t,beta1=0.9,beta2=0.999,epsilon=1e-08):
    #print(features.shape)
    #print(output.shape)
    #print(weights)
    #print(type(weights))
    predicted=predict(features,weights)
    g=(-(output-predicted)@features.T).T
    m=np.zeros(weights.shape)
    v=np.zeros(weights.shape)

    m=beta1*m+(1-beta1)*g
    v=beta2*v+(1-beta2)*(g*g)
    m_hat=m/(1-(beta1**(t+1)))
    v_hat=v/(1-(beta2**(t+1)))
    #print(m_hat,v_hat)
    #print(type(((lr*m_hat)/(np.sqrt(v_hat)+epsilon)).T))
    weights=weights-((lr*m_hat)/(np.sqrt(v_hat)+epsilon))
 
    return weights
def softmax_regression(data,output,learning_rate,epoch):
    data_hat=np.array(data)
    data_hat=data_hat.reshape(data_hat.shape[0],data_hat.shape[1]).T
    output_hat=np.array(output)
    output_hat=output_hat.reshape(output_hat.shape[0],output_hat.shape[1]).T
    pre_weights=0
    weights=np.zeros((len(data[0]),len(output[0])))
    model=linear_model.LogisticRegression(C=1e5,solver='lbfgs',multi_class='multinomial')
    """for i in range(epoch):
        predicted=predict(data_hat,weights)
        print(np.linalg.norm(predicted-output_hat))
        #for n in np.random.permutation(len(output)):
        weights=Adam(data_hat,output_hat,weights,learning_rate,i)
            #if np.linalg.norm(weights-pre_weights)<0.0001:
             #   print(i)
              #  break"""
    return weights
def softmax_regression_2(data,output,x1,x2,x3):
    output=np.asarray(output)
    output=output.reshape(output.shape[0],output.shape[1]).T
    output=output.reshape(-1)
    data=np.asarray(data)
    data=data.reshape(data.shape[0],data.shape[1]).T
    weights=np.zeros((len(data),len(output)))
    model=sklearn.linear_model.LogisticRegression(C=1e5,solver='lbfgs',multi_class='multinomial')
    model.fit(data,output)
    y1=model.predict(x1)
    y2=model.predict(x2)
    y3=model.predict(x3)
    #for i in range(epoch):
    #    weights=update_weights(data,output,weights,learning_rate)
    return y1,y2,y3
def CNN(data,output,lr,epoch):
    k1=np.random.rand(3,3)
    k2=np.random.rand(3,3)
    k3=np.random.rand(3,3)
    k4=np.random.rand(3,3)
    k5=np.random.rand(3,3)
    k6=np.random.rand(3,3)
    k7=np.random.rand(3,3)
    k8=np.random.rand(3,3)


    pool=Pool(4)
    conv1=pool.map(partial(conv_layer,kernel=k1),data)
    pool.close()
    pool.join()
    conv1[conv1<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv1)
    pool.close()
    pool.join()
    m1=[i[0] for i in m1_]
    pos1=[i[1]for i in m1_]
    u1=[i[2]for i in m1_]
    r1=[i[3]for i in m1_]


    pool=Pool(4)
    conv2=pool.map(partial(conv_layer,kernel=k2),m1)
    pool.close()
    pool.join()
    conv2[conv2<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv2)
    pool.close()
    pool.join()
    m2=[i[0] for i in m1_]
    pos2=[i[1]for i in m1_]
    u2=[i[2]for i in m1_]
    r2=[i[3]for i in m1_]

    pool=Pool(4)
    conv3=pool.map(partial(conv_layer,kernel=k3),m2)
    pool.close()
    pool.join()
    conv3[conv3<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv3)
    pool.close()
    pool.join()
    m3=[i[0] for i in m1_]
    pos3=[i[1]for i in m1_]
    u3=[i[2]for i in m1_]
    r3=[i[3]for i in m1_]

    pool=Pool(4)
    conv4=pool.map(partial(conv_layer,kernel=k4),m3)
    pool.close()
    pool.join()
    conv4[conv4<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv4)
    pool.close()
    pool.join()
    m4=[i[0] for i in m1_]
    pos4=[i[1]for i in m1_]
    u4=[i[2]for i in m1_]
    r4=[i[3]for i in m1_]

    pool=Pool(4)
    conv5=pool.map(partial(conv_layer,kernel=k5),m4)
    pool.close()
    pool.join()
    conv5[conv5<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv5)
    pool.close()
    pool.join()
    m5=[i[0] for i in m1_]
    pos5=[i[1]for i in m1_]
    u5=[i[2]for i in m1_]
    r5=[i[3]for i in m1_]

    pool=Pool(4)
    conv6=pool.map(partial(conv_layer,kernel=k6),m5)
    pool.close()
    pool.join()
    conv6[conv6<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv6)
    pool.close()
    pool.join()
    m6=[i[0] for i in m1_]
    pos6=[i[1]for i in m1_]
    u6=[i[2]for i in m1_]
    r6=[i[3]for i in m1_]

    pool=Pool(4)
    conv7=pool.map(partial(conv_layer,kernel=k7),m6)
    pool.close()
    pool.join()
    conv7[conv7<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv7)
    pool.close()
    pool.join()
    m7=[i[0] for i in m1_]
    pos7=[i[1]for i in m1_]
    u7=[i[2]for i in m1_]
    r7=[i[3]for i in m1_]

    pool=Pool(4)
    conv8=pool.map(partial(conv_layer,kernel=k8),m7)
    pool.close()
    pool.join()
    conv8[conv8<=0]=0
    pool=Pool(4)
    m1_=pool.map(max_pooling_,conv1)
    pool.close()
    pool.join()
    m8=[i[0] for i in m1_]
    pos8=[i[1]for i in m1_]
    u8=[i[2]for i in m1_]
    r8=[i[3]for i in m1_]

    



    


   

    
def train(folder,reshape_dim,learning_rate,epoch):
    data,output=load_data(folder)
    #data=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    #print(output)
    #print(output[0].shape)
    #print(data[0].shape)
    #print(data[1])
    data=parallel_2(data,reshape_dim)
    weights=softmax_regression(data,output,learning_rate,epoch)
    return weights


def train_with_sklearn(folder,reshape_dim,x1,x2,x3):
    data,output=load_data(folder)
    data=parallel_2(data,reshape_dim)
    y1,y2,y3=softmax_regression_2(data,output,x1,x2,x3)
    return y1,y2,y3
