import numpy as np
import cv2 
import os
from conv import *
import multiprocessing
from multiprocessing import Pool
from itertools import product
from functools import partial
import math
from optimizer import *
import copy
def softmax(Z):
        e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True,initial=-np.inf))
        return e_Z / e_Z.sum(axis = 0)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        images.append(cv2.imread(os.path.join(folder,filename),0))
    return images

def load_data(folder):
    #USE THIS FUNCTION IF YOU HAVE YOUR OWN DATA FILES WITH MANY TRAIN IMAGES
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
            y.reshape(n,1)
           
            output.append(y)
        iters += 1
    
    return images,output

def divide_picture(filename,reshape_dim):
    img=cv2.imread(filename,0)
    division=img.shape[0]/reshape_dim[0]
    inp=[np.hsplit(rows,division) for rows in np.vsplit(img,division)]
    features=[]
    output=[]
    
    for i in inp:
        for j in i:
            features.append(j)
    iters=0
    for i in range(1,2501):
        
        out=np.zeros((10,1))
        out[iters]=1
        output.append(out)
        if i%250==0:
            iters+=1
    #print(len(features))
    #print(len(output))
    return features,output
def one_hot_encoding(arr):
    y=np.zeros(arr.shape)
    m_index=nanargmax(arr)
    y[m_index]=1
    return y
class Activation:
    def softmax(Z):
        e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True,initial=-np.inf))
        return e_Z / e_Z.sum(axis = 0)
    def sigmoid(Z):
        return 1/(1+np.exp(-Z))
    def relu(Z):
        return np.maximum(0,Z)
    def tanh(Z):
        return np.tanh(Z)
    def elu(z,alpha):
	    return z if z >= 0 else alpha*(e^z -1)
    def leakyrelu(z, alpha):
	    return max(alpha * z, z)    

def iFilter(size, scale = 1.0):
    '''
    Initialize filter using a normal distribution with and a 
    standard deviation inversely proportional the square root of the number of units
    '''
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def iWeight(size):
    '''
    Initialize weights with a random normal distribution
    '''
    return np.random.standard_normal(size=size) * 0.01


#THE LAST LAYER OF THE MODEL
class ActivationLayer:
    def __init__(self,features,output,weights,bias,name,optimizer,lr,iters,runtime_args,bias_args,optimize_args=None):
        self.features = features
        self.fcn=[]
        self.output = output
        self.name = name
        self.optimizer = globals()[optimizer]
        self.lr = lr
        self.iters=iters
        self.weights=weights
        self.bias = bias
        self.optimize_args=optimize_args
        self.runtime_args=runtime_args
        self.bias_args=bias_args
        self.backprop=[]
        self.bias_grad=[]
        self.grad=[]
    def reshape_data(self):
        for i in copy.copy(self.features):
            i=i.reshape(i.size,1)
            self.fcn.append(i)
    def loss(self):
        loss=0
        for i in range(len(self.fcn)):
            predicted=self.weights.T@self.fcn[i]
            predicted=Activation.softmax(predicted)
            y=-np.sum(self.output[i]*np.log(predicted))
            loss+=y
        return loss/len(self.output)

    def fit(self,t):
        grad=[]
        bias_grad=[]
        #print('fuck ',self.features[0].shape)
        if self.weights is None:
            self.weights=iWeight((len(self.fcn[0]),len(self.output[0])))
            self.bias=0
        if self.name=="softmax":
            for i in range(len(self.features)):
                predicted=self.weights.T@self.fcn[i]+self.bias
                predicted=Activation.softmax(predicted)
                g1=predicted-self.output[i]
                g=self.fcn[i]@g1.T
                gb=np.sum(g1)
                grad.append(g)
                bias_grad.append(gb)
                self.grad.append(g1)
                

        args=self.optimize_args
        if self.optimizer is not None and self.weights is not None:
            if args==None:
                self.weights,self.runtime_args=self.optimizer(grad,self.weights,self.lr,t,self.runtime_args)
                self.bias,self.bias_args=self.optimizer(bias_grad,self.bias,self.lr,t,self.bias_args)
            else:
                self.weights,self.runtime_args=self.optimizer(grad,self.weights,self.lr,t,self.runtime_args,*args)
                self.bias,self.bias_args=self.optimizer(bias_grad,self.bias,self.lr,t,self.bias_args,*args)
    def feed_backward(self):
        for i in self.grad:
            x=self.weights@i
            x=x.reshape(self.features[0].shape)
            #print('grad shape: ',i.shape)
            #print('x: ',type(x))
            self.backprop.append(x)
        


#LAYER OF THE MODEL
class Layer:
    def __init__(self,func,grad_func,args,weights=None,bias=None,optimizer=None,optimize_args=None,lr=0,diff=False):
        self.func=globals()[func]
        self.grad_func=globals()[grad_func]
        self.args=args
        self.weights=weights
        self.backprop=None
        self.result=None 
        self.foward_result=None
        self.grad=None
        self.bias_grad=None
        self.dconv_prev=None
        self.lr = lr
        self.optimize_args=optimize_args
        self.runtime_args=[]
        self.diff=diff
        self.features=None
        self.bias=bias
        self.bias_args=[]
        if optimizer is not None:
            self.optimizer= globals()[optimizer]
        else:
            self.optimizer=None
    def feed_forward(self,prev_result):
        self.features=prev_result
        if self.args==None:
            if self.weights is not None:
                w=[self.weights]*len(prev_result)
                b=[self.bias]*len(prev_result)
                p=Pool(os.cpu_count())
                self.result=p.starmap(self.func,zip(prev_result,w,b))
                p.close()
                p.join()
                if self.diff==True:
                    self.foward_result=[i[0]for i in self.result]
                else:
                    self.foward_result=self.result
            if self.weights is None:
                p=Pool(os.cpu_count())
                self.result=p.map(self.func,prev_result)
                p.close()
                p.join()
                if self.diff==True:
                    self.foward_result=[i[0]for i in self.result]
                else:
                    self.foward_result=self.result
        else:
            if self.weights is None:
                a=[self.args]*len(prev_result)
                p=Pool(os.cpu_count())
                self.result=p.starmap(self.func,zip(prev_result,a))
                p.close()
                p.join()
                if self.diff==True:
                    self.foward_result=[i[0]for i in self.result]
                else:
                    self.foward_result=self.result
            if self.weights is not None:
                a=[self.args]*len(prev_result)
                w=[self.weights]*len(prev_result)
                b=[self.bias]*len(prev_result)
                self.result=p.starmap(self.func,zip(prev_result,a,w,b))
                p.close()
                p.join()
                if self.diff==True:
                    self.foward_result=[i[0]for i in self.result]
                else:
                    self.foward_result=self.result
    def feed_backward(self,dconv_prev):
        #print(len(dconv_prev))
        #for i in dconv_prev:
        #    print(i.shape)
        if self.args==None:
            if self.weights is None:
                p=Pool(os.cpu_count())
                backward=p.starmap(self.grad_func,zip(dconv_prev,self.features))
                p.close()
                p.join()
            else:
                w=[self.weights]*len(self.result)
                p=Pool(os.cpu_count())
                backward=p.starmap(self.grad_func,zip(dconv_prev,self.features,w))
                p.close()
                p.join()
        else:
            if self.weights is None:
                a=[self.args]*len(dconv_prev)
                p=Pool(os.cpu_count())
                backward=p.starmap(self.grad_func,zip(dconv_prev,self.features,a))
                p.close()
                p.join()
            else:
                a=[self.args]*len(dconv_prev)
                w=[self.weights]*len(dconv_prev)
                backward=p.starmap(self.grad_func,zip(dconv_prev,self.features,a,w))
                p.close()
                p.join()
        if self.weights is None:
            self.backprop=backward
        else:
            self.backprop=[i[0] for i in backward]
            self.grad=[i[1]for i in backward]
            self.bias_grad=[i[2]for i in backward]
    def optimize(self,t):
        args=self.optimize_args
        if self.optimizer is not None and self.weights is not None:
            if args==None:
                self.weights,self.runtime_args=self.optimizer(self.grad,self.weights,self.lr,t,self.runtime_args)
                self.bias,self.bias_args=self.optimizer(self.bias_grad,self.bias,self.lr,t,self.bias_args)
            else:
                self.weights,self.runtime_args=self.optimizer(self.grad,self.weights,self.lr,t,self.runtime_args,*args)
                self.bias,self.bias_args=self.optimizer(self.bias_grad,self.bias,self.lr,t,self.bias_args,*args)
        return self

#backward=grad_func(dconv_prev,*self.args,self.result,self.weights)

class Model:
    def __init__(self,filename,load_method,reshape_dim):
         
        self.weights=[]
        self.bias=[]
        self.layers = []
        self.curr_backprop = None
        self.curr_forward = None
        self.iters=0
        self.activation_weights=None
        self.activation_bias=None
        self.activation=None
        self.lr=None 
        self.epoch=None
        self.optimizer=None
        self.runtime_args=[]
        self.bias_args=[]
        self.reshape_dim= reshape_dim
        self.loss=0
        self.pre_loss=None
        self.count=0
        self.best_weights=[]
        self.best_bias=[]
        self.best_aw=None
        self.best_ab=None
        self.best_loss=None
        if load_method=="load from folder":
            #TAKE TRAIN DATA FROM A FOLDER
            self.features,self.output=load_data(filename)
        if load_method=="divide picture":
            #TAKE TRAIN DATA FROM AN IMAGE
            self.features,self.output=divide_picture(filename,reshape_dim)
    
    def Add(self,func_name,grad_func,args,weight,bias,optimizer,optimize_args,lr,diff):
        self.weights.append(weight)
        new_layer=Layer(func_name,grad_func,args,weight,bias,optimizer,optimize_args,lr,diff)
        self.layers.append(new_layer)
    def reshape_data(self):
        for i in range(len(self.features)):
            self.features[i]=cv2.resize(self.features[i],self.reshape_dim,interpolation=cv2.INTER_AREA)
    def normalized(self):
        for i in range(len(self.features)):
            self.features[i]=self.features[i]/255
    def feed_forward(self):
        self.reshape_data()
        for i in range(len(self.layers)):
            #print(i)
            if i==0:
                self.layers[i].feed_forward(self.features)
                self.curr_forward=self.layers[i].foward_result
                self.curr_forward=ReLU(self.curr_forward)
            else:
                self.layers[i].feed_forward(self.curr_forward)
                self.curr_forward=self.layers[i].foward_result
                self.curr_forward=ReLU(self.curr_forward)
    def feed_backward(self):
        for layer in self.layers[::-1]:
            layer.feed_backward(self.curr_backprop)
            self.curr_backprop=layer.backprop
            self.curr_backprop=ReLU(self.curr_backprop)
        
    def optimize_weights(self):
        p=Pool(os.cpu_count())
        a=[self.iters]*len(self.layers)
        new_layer=p.starmap(Layer.optimize,zip(self.layers,a))
        p.close()
        p.join()
        self.layers=new_layer
    def optimize_2(self):
        for layer in self.layers:
            layer.optimize(self.iters)
    def compile(self,activation,optimizer,lr,epoch):
        self.activation=activation
        self.optimizer=optimizer
        self.lr=lr
        self.epoch=epoch
        
    def fit(self):
        for i in range(self.epoch):
            print('epoch: ',self.iters)
            print('LOSS: ',self.loss)
            self.feed_forward()
            softmax_layer=ActivationLayer(self.curr_forward,self.output,self.activation_weights,self.activation_bias,self.activation,self.optimizer,self.lr,self.iters,self.runtime_args,self.bias_args)
            softmax_layer.reshape_data()
            softmax_layer.fit(self.iters)
            self.loss=softmax_layer.loss()
            self.activation_weights=softmax_layer.weights
            self.activation_bias=softmax_layer.bias
            self.runtime_args=softmax_layer.runtime_args
            softmax_layer.feed_backward()
            self.curr_backprop=softmax_layer.backprop
            self.feed_backward()
            self.optimize_2()
            
            for r in self.layers:
                self.weights.append(r.weights)
                self.bias.append(r.bias)
            
            if i==0:
                self.pre_loss=self.loss
            else:
                if self.loss>=self.pre_loss:
                    #break
                    self.count+=1
                    self.pre_loss=self.loss
                else:
                    self.count=0
                    self.pre_loss=self.loss
            if self.iters==0:
                self.best_weights=self.weights
                self.best_bias=self.bias
                self.best_aw=self.activation_weights
                self.best_ab=self.activation_bias
                self.best_loss=self.loss

            else:
                if self.loss<=self.best_loss:
                    self.best_weights=self.weights
                    self.best_bias=self.bias
                    self.best_aw=self.activation_weights
                    self.best_ab=self.activation_bias
                    self.best_loss=self.loss
            if self.count==10:
                break
            self.iters+=1
    def predicted(self,features):
        result = []
        self_copy=copy.copy(self)
        self_copy.features=features
        for i in range(len(self_copy.layers)):
            self_copy.layers[i].weights=self.best_weights[i]
            self_copy.layers[i].bias=self.best_bias[i]
        self_copy.activation_weights=self.best_aw
        self_copy.activation_bias=self.best_ab
        self_copy.reshape_data()
        self_copy.feed_forward()
        func=globals()[self.activation]
        for i in self_copy.curr_forward:
            i=i.reshape(i.size,1)
            predicted=func(self.activation_weights.T@i)
            result.append(predicted)
        return result
    

        







def run_model(filename,load_method,test_data,reshape_dim):
    model=Model(filename,load_method,reshape_dim)
    model.Add("conv_layer","convbackprop",None,iFilter((3,3)),1e-8,"Adam",None,0.003,False)  
    model.Add("max_pooling","maxpool_backprop",None,None,None,None,None,None,True)
    model.compile(activation="softmax",optimizer="Adam",lr=0.003,epoch=200)    
    model.fit()
    result=model.predicted(test_data)
    for r in result:
        r=one_hot_encoding(r)
        if r[0]==1:print("ZERO")
        if r[1]==1:print("ONE")
        if r[2]==1:print("TWO")
        if r[3]==1:print("THREE")
        if r[4]==1:print("FOUR")
        if r[5]==1:print("FIVE")
        if r[6]==1:print("SIX")
        if r[7]==1:print("SEVEN")
        if r[8]==1:print("EIGHT")
        if r[9]==1:print("NINE")
        
    
    return model.weights,model.activation_weights,result





        
    
    






    


