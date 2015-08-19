
import os
import csv
import math
import numpy as np
from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['replen','attrho','eta','reprho','mvector']


# rho is tanh(a dx) * exp(-b dx)
# the inflexion point is located at (1/2a)ln(2a/b + sqrt((2a/b)^2+1)



workDir = '/home/ctorney/workspace/moveRules/'
# Define data and stochastics


## interaction function - this is a square wave function which switches from reprho to attrho at distance replen (all these parameters are inferred 
## and more complex interaction functions can be used)
replen = Uniform('replen',lower=0,upper=5) # this is the repulsion zone
reprho = Uniform('reprho',lower=0,upper=1) # this is the strength of repulsion
attrho = Uniform('attrho',lower=0,upper=1) # this is the strength of attraction

eta = Uniform('eta',lower=0,upper=1) # this is the autocorrelation or persistence in direction


## data pre-processing - this section calculates change in heading and relative distances and orientations between individuals
dt = 1
allDF = pd.DataFrame()
for trial in np.arange(0,1):
    print(trial)
    fileimportname= workDir + '/data.csv'
    
    df = pd.read_csv(fileimportname)
    df['trial']=trial
    df['dtheta']=np.NaN
    for index, row in df.iterrows():
        thisTime =  row['time']
        thisID = row['id']
        thisTheta = row['angle']
        nextTime = df[(np.abs(df['time']-(thisTime+dt))<1e-6)&(df['id']==thisID)]
        if len(nextTime)==1:
            df.ix[index,'dtheta'] = nextTime.iloc[0]['angle'] -  thisTheta 
    allDF = allDF.append(df)

allDF = allDF[pd.notnull(allDF['dtheta'])]
allData = allDF.values
mvector = np.copy(allData[:,6])
mvector[mvector<-pi]=mvector[mvector<-pi]+2*pi
mvector[mvector>pi]=mvector[mvector>pi]-2*pi
dsize = len(mvector)
# first find the maximum number of neighbours that any individual could observe
maxN=0
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisID = allData[thisRow,1]
    window = allData[(allData[:,0]==thisTime)&(allData[:,1]!=thisID),:]
    if len(window)>maxN:
        maxN=len(window)#

dparams = np.zeros((dsize,maxN,2)).astype(np.float32) # dist, angle
rhos = np.zeros((dsize,maxN)).astype(np.float32) # dist, angle
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisID = allData[thisRow,1]
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = (allData[thisRow,4])
    window = allData[(allData[:,0]==thisTime)&(allData[:,1]!=thisID),:]
    ncount = 0
    for w in window:
        xj = w[2]
        yj = w[3]
        dparams[thisRow,ncount,0] = (((thisX-xj)**2+(thisY-yj)**2))**0.5
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        dparams[thisRow,ncount,1] = math.atan2(math.sin(angle), math.cos(angle))
        ncount+=1

## end of data pre-proc


@stochastic(observed=True)
def moves(dr=replen,rr=reprho,ar=attrho, ep=eta,value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    
    lambdas = np.zeros_like(mvector) # these are the headings without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*lambda
    #lambdas[np.abs(mvector)>pi]=pi
    lambdas[np.abs(mvector)>(1-ep)*pi]=pi
    lambdas[np.abs(mvector)<(1-ep)*pi]=mvector[np.abs(mvector)<(1-ep)*pi]/(1-ep)
    
    # first calculate all the rhos
    rhos = np.zeros_like(dparams[:,:,0])
    rhos[dparams[:,:,0]>dr]=ar
    rhos[dparams[:,:,0]<dr]=-rr
    
    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
    nc = np.sum(np.abs(rhos),1) # normalizing constant

    wwc = (np.abs(rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((lambdas-dparams[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
    wwc = np.sum(wwc,1)/nc
    wwc[np.isinf(wwc)]=1/(2*pi)
    return np.sum(np.log(wwc))
