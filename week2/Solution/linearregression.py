#filename : linearregression.py
#author : Prajwal T R
#date last modified : Thu Feb  6 21:26:52 2020
#comments : linear regression single variable week 2 data : ex1data1.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iterations = 1500
alpha = 0.01

def plot_graph(x,y,type,plots):
    for i in range(plots):
        plt.plot(x[i],y[i],type[i])
    plt.show()
def gradientdescent(x,y,sample,theta):
    for i in range(iterations):
        theta[0] = theta[0] - ((alpha/m)*((np.dot(x,theta)-y)*x[:,0]).sum())
        theta[1] = theta[1] - ((alpha/m)*((np.dot(x,theta)-y)*x[:,1]).sum())
    return theta
def cost_calc(x,y,theta):
    cal_cost = (1/(2*m))*(((np.dot(x,theta)-y)**2).sum())
    return cal_cost
if __name__=="__main__":
    data = pd.read_csv("../Question/ex1data1.csv",names=["x","y"])
    ones = np.ones(data.shape[0],dtype=float) #add bias or ones
    data.insert(0,"ones",ones) #adding ones to data at index 0
    x = data[['ones','x']].to_numpy(dtype=float) # features
    y = data['y'].to_numpy(dtype=float) #target variable
    m = x.shape[0] #sample length
    plot_graph([x[:,1]],[y],['ro'],1) #plot initial graph
    theta = np.zeros(2,dtype=float) #initial to zero
    #gradientdescent
    theta = gradientdescent(x,y,m,theta)
    #calculate cost
    cost = cost_calc(x,y,theta) # todo :  obtained cost : 4.48311 actual cost : 32.07 fix !
    print("obtained cost : %.5f" % cost)
    yf = np.dot(x,theta)
    plot_graph([x[:,1],x[:,1]],[y,yf],["ro",""],2)
