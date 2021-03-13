import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2

#THIS 2 FUNCTIONS CAN HELP YOU IN THE FUTURE, BUT IT IS DONT HELP MUCH WITH THAT PROJECT
def tile_array(a, b0, b1):
    r, c = a.shape                                    
    rs, cs = a.strides                                
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) 
    return x.reshape(r*b0, c*b1) 
                       
                       
def upsample(img,under_pad,right_pad):
    new_img=tile_array(img,2,2)
    if under_pad==True:
        new_img=np.delete(new_img,-1,0)
    if right_pad==True:
        new_img=np.delete(new_img,-1,axis=1)
    return new_img
###

#CONVOLUTION LAYER
def conv_layer(img,kernel,bias):
    row,col= img.shape
    s=row-2
    k=col-2
    new_img=[]
   
    for r in range(0,row-2):
        for c in range(0,col-2):
            take_out=np.array([[img[r,c],img[r,c+1],img[r,c+2]],
                              [img[r+1,c],img[r+1,c+1],img[r+1,c+2]],
                              [img[r+2,c],img[r+2,c+1],img[r+2,c+2]]])
            
            take_out=take_out*kernel
            take_out=np.sum(take_out)
           
            new_img.append(take_out)
    new_img=np.array(new_img)
    new_img=np.reshape(new_img,(s,k))
    new_img=new_img+bias
    #print(new_img.shape)
    return new_img

#RELU FUNCTION
def ReLU(img):
    return np.maximum(0,img)

#MAX POOLING LAYER
def max_pooling(img):
    img,under_pad,right_pad=pool_pad(img)
    row,col=img.shape
    new_img=[]
    for r in range(0,row-1,2):
        for c in range(0,col-1,2):
            arr=np.array([[img[r,c],img[r,c+1]],
                          [img[r+1,c],img[r+1,c+1]]])
            take_out=np.max(arr)
            new_img.append(take_out)
    new_img=np.array(new_img)
    new_img=np.reshape(new_img,(row//2,col//2))
    return [new_img,under_pad,right_pad]
#PADDING
def pool_pad(img):
    right_pad=False
    under_pad=False
    row,col=img.shape
    new_row=row
    if row % 2 !=0:
        new_row=row+1
        new_img=np.zeros((new_row,col))
        under_pad=True
        for r in range(0,row):
            for c in range(0,col):
                new_img[r,c]=img[r,c]
    else:
        new_img=img
    if col%2 !=0:
        new_col=col+1
        new_new_img=np.zeros((new_row,new_col))
        right_pad=True
        for r in range(0,new_row):
            for c in range(0,col):
                new_new_img[r,c]=new_img[r,c]

    else:
        new_new_img=new_img
    return new_new_img,under_pad,right_pad

def pad(img):
    s=img.shape
    
    i=s[0]
    i=i+4
    y=s[1]
    y=y+4
    
    pad_img=np.zeros((i,y))
    for r in range(2,i-2):
        for c in range(2,y-2):
            x=img[r-2,c-2]
            pad_img[r,c]=pad_img[r,c]+img[r-2,c-2]
    return pad_img
def unpooling(img,pos,under_pad=False,right_pad=False):
    img_list=[]


    print(img_list)
    row,col=img.shape
    for r in range(0,row):
        for c in range(0,col):
            img_list.append(img[r,c])
    
    out_row=row*2
    out_col=col*2
    new_img=np.zeros((out_row,out_col))
    now_iter=0
    for r in range(0,out_row-1,2):
        for c in range(0,out_col-1,2):
            now_pos=pos[now_iter]
            now_max=img_list[now_iter]
           
            arr=np.array([[new_img[r,c],new_img[r,c+1]],
                          [new_img[r+1,c],new_img[r+1,c+1]]])
            if now_pos==[0,1]:
                new_img[r,c+1]=now_max
            if now_pos==[1,0]:
                new_img[r+1,c]=now_max
            if now_pos==[0,0]:
                new_img[r,c]=now_max
            if now_pos==[1,1]:
                
                new_img[r+1,c+1]=now_max
            now_iter+=1
    if under_pad==True:
        new_img=np.delete(new_img,-1,0)
    if right_pad==True:
        new_img=np.delete(new_img,-1,axis=1)
    return new_img
            
def transpose_conv(img,kernel):
    new_img=pad(img)
    new_img=conv_layer(new_img,kernel)
    return new_img

def convbackprop(dconv,conv,kernel):
    row,_=conv.shape
    f,_=kernel.shape
    dout=np.zeros(conv.shape)
    dkernel=np.zeros(kernel.shape)
    dbias=0
    curr_y=out_y=0
    while curr_y+f<=row:
        curr_x=out_x=0
        while curr_x+f<=row:
            dkernel+=dconv[out_y,out_x]*conv[curr_y:curr_y+f,curr_x:curr_x+f]
            dout[curr_y:curr_y+f,curr_x:curr_x+f]+=dconv[out_y,out_x]*kernel
            curr_x+=1
            out_x+=1
        curr_y+=1
        out_y+=1
    dbias=np.sum(dconv)
    return (dout,dkernel,dbias)
    
def nanargmax(array):
    idx=np.nanargmax(array)
    idxs=np.unravel_index(idx,array.shape)
    return idxs

def maxpool_backprop(dpool,orig):
    orig,up,rp=pool_pad(orig)
    dout=np.zeros(orig.shape)
    rows,cols=orig.shape
    for r in range(0,rows-1,2):
        for c in range(0,cols-1,2):
            x=r//2
            y=c//2
            (a,b)=nanargmax(orig[r:r+2,c:c+2])
            z=dpool[x,y]
            dout[r+a,c+b]=z
    if up==True:
        dout=np.delete(dout,-1,0)
    if rp==True:
        dout=np.delete(dout,-1,axis=1)
    return dout




    



