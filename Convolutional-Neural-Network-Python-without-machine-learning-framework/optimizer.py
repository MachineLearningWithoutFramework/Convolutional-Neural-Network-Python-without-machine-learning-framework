import numpy as np
import math






#BATCH GRADIENT DESCENT
def GD(grad,weights,lr,t,args):
    g=sum(grad)/len(grad)
    weights=weights-lr*g
    return weights,[]


#STOCHASTIC GRADIENT DESCENT
def SGD(grad,weights,lr,t,args):
    for g in grad:
        weights=weights-lr*g
    return weights,[]


#MOMENTUM
def Momentum(grad,weights,lr,t,args,beta=0.9):
    g=sum(grad)/len(grad)
    v=None
    if args==[]:
        v=np.zeros(weights.shape)
    else:
        v=args[0]
    v=beta*v+(1-beta)*g
    weights=weights=lr*v
    return weights,[v]



#AdaGrad
def AdaGrad(grad,weights,lr,t,args,epsilon=1e-8):
    g=sum(grad)/len(grad)
    v=None
    if args==[]:
        v=np.zeros(weights.shape)
    else:
        v=args[0]
    v=v+g**2
    weights=weights-(lr/(np.sqrt(v+epsilon)))*g
    return weights,[v]




#RMSProp
def RMSProp(grad,weights,lr,t,args,beta=0.9,epsilon=1e-8):
    #g=sum(grad)/len(grad)
    v=None
    if args==[]:
        if type(weights) is int:
            v=0
        else:
            v=np.zeros(weights.shape)
    else:
        v=args[0]
    for g in grad:
        v=beta*v+(1-beta)*(g**2)
        weights=weights-(lr/(np.sqrt(v+epsilon)))*g
    return weights,[v]





#Adam
def Adam(grad,weights,lr,t,args,c=2,beta1=0.85,beta2=0.999,epsilon=1e-08):
    g=sum(grad)/len(grad)
    a=np.linalg.norm(g)
    m=None 
    v=None
    

    if args==[] and type(weights) is float or type(weights) is int:
       m=0
       v=0
    elif args==[]:
        m=np.zeros(weights.shape)
        v=np.zeros(weights.shape)
    else:
        m=args[0]
        v=args[1]
    #if a>c:
    # g=c*g/a
    m=beta1*m+(1-beta1)*g
    v=beta2*v+(1-beta2)*g*g
    m_hat=m/(1-beta1**(t+2))
    v_hat=v/(1-beta2**(t+2))
    weights-=lr*m_hat/np.sqrt(v_hat+epsilon)
    return weights,[m,v]



    