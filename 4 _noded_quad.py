import numpy as np
from math import *
from sympy import *
import matplotlib.pyplot as plt


m=10 #horizontal
n=10 #vertical
coord=np.array([[0,0],[1,0],[1,1],[0,1]])
tvec1=np.array([[100],[0]])
E=200
nu = 0
gp = np.array([-0.774597,0.774597,0])
w = np.array([5/9,5/9,8/9])
C = (E/((1+nu)*(1-2*nu)))*np.array([[1-nu,nu,0],[nu,1-nu,0],[0,0,0.5*(1-2*nu)]])


def mesh(m,n,coord):
    NoN=(m+1)*(n+1)
    NoE=m*n
    NpE=4

    ## Nodes

    NL=np.zeros([NoN,2])
    X_diff=(coord[1,0]-coord[0,0])/m       
    Y_diff=(coord[2,1]-coord[0,1])/n
    
    N=0
    for i in range(1,n+2):
        for j in range(1,m+2):
            NL[N,0]=coord[0,0]+(j-1)*X_diff
            NL[N,1]=coord[0,1]+(i-1)*Y_diff
            N+=1

    ##elements
    EL=np.zeros([NoE,NpE])
    for i in range(1,n+1):
        for j in range(1,m+1):
            if j==1:

                EL[(i-1)*m+j-1,0]=(i-1)*(m+1)+j
                EL[(i-1)*m+j-1,1]=EL[(i-1)*m+j-1,0]+1
                EL[(i-1)*m+j-1,3]=EL[(i-1)*m+j-1,0]+(m+1)
                EL[(i-1)*m+j-1,2]=EL[(i-1)*m+j-1,3]+1

            else:
                EL[(i-1)*m+j-1,0]=EL[(i-1)*m+j-2,1]
                EL[(i-1)*m+j-1,3]=EL[(i-1)*m+j-2,2]
                EL[(i-1)*m+j-1,1]=EL[(i-1)*m+j-1,0]+1
                EL[(i-1)*m+j-1,2]=EL[(i-1)*m+j-1,3]+1
    
    ECG=np.zeros([NoE,4,3])
    for i in range(1,NoE+1):
        ECG[i-1,:,0]=EL[i-1,:]
        for j in range(0,4):
            ECG[i-1,j,1:]=NL[int(ECG[i-1,j,0])-1,:]
        
    return NL,EL,ECG   

def Stiff(g,h,C,ele):

    B1 = np.array([[1,0,0,0],[0,0,0,1],[0,1,1,0]])
    J=0.25*np.dot(np.array([[-(1-h),(1-h),(1+h),-(1+h)],[-(1-g),-(1+g),(1+g),(1-g)]]),ele)
    B2 = np.zeros((4,4))    
    B2[np.ix_([0,1],[0,1])] = np.linalg.inv(J)
    B2[np.ix_([2,3],[2,3])] = np.linalg.inv(J)
    B3 = 0.25*np.array([[-(1-h),0,(1-h),0,(1+h),0,-(1+h),0],
                     [-(1-g),0,-(1+g),0,(1+g),0,(1-g),0],
                     [0,-(1-h),0,(1-h),0,(1+h),0,-(1+h)],
                     [0,-(1-g),0,-(1+g),0,(1+g),0,(1-g)]])

    B = np.dot(np.dot(B1,B2),B3)
    detj = np.linalg.det(J)
    return detj*np.dot(np.dot(transpose(B),C),B)

def GlobalStiff(ECG,gp,w,m,n,C):
    K=np.zeros((2*(m+1)*(n+1),2*(m+1)*(n+1)))
    for k in range(0,m*n):
        ele=ECG[k,:,1:]
        nv=ECG[k,:,0]
        K_val=np.zeros((8,8))
        for i in range(0,len(gp)):
            for j in range(0,len(gp)):
                K_val = K_val + Stiff(gp[i],gp[j],C,ele)*w[i]*w[j]
        pv = np.array([int(2*nv[0]-1),int(2*nv[0]),int(2*nv[1]-1),int(2*nv[1]),int(2*nv[2]-1),int(2*nv[2]),int(2*nv[3]-1),int(2*nv[3])])
        
        for l in range(0,len(pv)):
            for p in range(0,len(pv)):
                K[pv[l]-1,pv[p]-1]=K[pv[l]-1,pv[p]-1]+K_val[l,p]
    
    return K

def loadvec(edg,q,ele,tvec):
    x=ele[:,0]
    y=ele[:,1]

    if (edg == 1):
        le=((x[1]-x[0])**2 + (y[1]-y[0])**2 )**0.5
        N = np.array([[0.5*(1-q),0,0.5*(1+q),0,0,0,0,0],
                      [0,0.5*(1-q),0,0.5*(1+q),0,0,0,0]])
    elif (edg == 2):
        le=((x[2]-x[1])**2 + (y[2]-y[1])**2 )**0.5
        N = np.array([[0,0,0.5*(1-q),0,0.5*(1+q),0,0,0],
                      [0,0,0,0.5*(1-q),0,0.5*(1+q),0,0]])
    elif (edg == 3):
        le=((x[3]-x[2])**2 + (y[3]-y[2])**2 )**0.5
        N = np.array([[0,0,0,0,0.5*(1-q),0,0.5*(1+q),0],
                      [0,0,0,0,0,0.5*(1-q),0,0.5*(1+q)]])
    elif (edg == 4):
        le=((x[3]-x[0])**2 + (y[3]-y[0])**2 )**0.5
        N = np.array([[0.5*(1-q),0,0,0,0,0,0.5*(1+q),0],
                      [0,0.5*(1-q),0,0,0,0,0,0.5*(1+q)]])
        
    return 0.5*le*np.dot(transpose(N),tvec)

def GlobalLoadvec(ECG,gp,w,m,n,tvec):
    f=np.zeros((2*(m+1)*(n+1),1))
    
    for k in range(1,n+1):
        ele = ECG[k*m-1,:,1:]
        nv = ECG[k*m-1,:,0]
        f_val = np.zeros((8,1))
        for i in range (0,len(gp)):
            f_val = f_val + loadvec(2,gp[i],ele,tvec)*w[i]

        pv = np.array([int(2*nv[0]-1),int(2*nv[0]),int(2*nv[1]-1),int(2*nv[1]),int(2*nv[2]-1),int(2*nv[2]),int(2*nv[3]-1),int(2*nv[3])])
        for l in range(0,len(pv)):
            f[pv[l]-1,0]=f[pv[l]-1,0]+f_val[l,0]
    return f

def constraints(NL,m,n):
    tot_nods=(m+1)*(n+1)
    boun_nods=n+1
    nv=np.arange(1,tot_nods+1)
    del_nods=np.zeros((boun_nods,1))
    #rem_nods=np.zeros((tot_nods-boun_nods,1))
    for i in range(1,boun_nods+1):
        del_nods[i-1]=(i-1)*(m+1)+1

    rem_nods=np.array(list(set(nv).difference(set(del_nods.reshape(len(del_nods),)))))
    rem_nods=rem_nods.reshape(len(rem_nods),1)
    
    return del_nods,rem_nods

def RedGStiff(dn,K,m,n):
    dof_dn =np.zeros((2*(n+1),1))
    for i in range(0,n+1):
        dof_dn[2*i]= 2*dn[i]-1
        dof_dn[2*i+1] = 2*dn[i]
    tr=0
    Kr=K
    for j in range(0,len(dof_dn)):
        num=int(dof_dn[j]-tr-1)
        Kr=np.delete(Kr,num,0)
        Kr=np.delete(Kr,num,1)
        tr+=1
    return Kr

def RedGLoad(dn,f,m,n):
    dof_dn =np.zeros((2*(n+1),1))
    for i in range(0,n+1):
        dof_dn[2*i]= 2*dn[i]-1
        dof_dn[2*i+1] = 2*dn[i]
    tr=0
    fr=f
    for j in range(0,len(dof_dn)):
        num=int(dof_dn[j]-tr-1)
        fr=np.delete(fr,num,0)
        tr+=1
    return fr

NL,EL,ECG=mesh(m,n,coord)
K=GlobalStiff(ECG,gp,w,m,n,C)
f=GlobalLoadvec(ECG,gp,w,m,n,tvec1)
dn,rn=constraints(NL,m,n)
Kr = RedGStiff(dn,K,m,n)
fr=RedGLoad(dn,f,m,n)
ur=np.dot(np.linalg.inv(Kr),fr)

print(ur)

"""x=np.arange(0,5)
x[0:4]=coord[:,0]
x[4]=coord[0,0]
y=np.arange(0,5)
y[0:4]=coord[:,1]
y[4]=coord[0,1]

plt.plot(x,y)
plt.show()"""

