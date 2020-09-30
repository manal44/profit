# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:10:48 2020

@author: khall
"""

#import numpy as np
#import matplotlib.pyplot as plt
#import datetime as dt
#from pandas_datareader import DataReader
#import pandas as pd
##chercher panda et taper datareader pour l'installer
##pip
##from matplotlib.pyplab import rcParams
##rcParams
##matplotlib inline
#
#tickers = ['IBM']
#portfolio_returns = pd.DataFrame() #tableau dont on peut attribuer des label/nom de colonnes
#start, end = dt.datetime(2018,9,1),dt.datetime(2018,10,1) #date de début et de fin
#portfolio_returns=DataReader(tickers,'yahoo',start,end).loc[:,'Close']#.pct_change()[1,:]
###donne la valeur journalière de cloture entre la date de début et de fin
###pourcentage change: applique un rendement: 
##print(portfolio_returns)
##
###les log rendement suivent des gaussiennes indépendantes

import numpy as np 
import pandas as pd 
from pandas_datareader import DataReader 
from datetime import datetime 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
#matplotlib inline
spy = DataReader('SPY',  'yahoo', datetime(2013,1,1), datetime(2015,1,1)) #print(spy) spy_returns = pd.DataFrame(np.diff(np.log(spy['Adj Close'].values)))
# creates a table of dataframe countaining the values of SP&500 from (2013,1,1) to (2015,1,1)
# the tables' columns are: ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
# the first column is 'Date' though. It's however considered as an index and not a column.
#print(spy)
# spy_returns=pd.DataFrame(np.diff(np.log(spy['Adj Close'].values))) #creates a new table of dataframe which countains the diff of log of values of the column 'Adj Close'
# spy_returns.index=spy.index[1:spy.index.shape[0]] # to take all the lines except the first one
# spy_returns.columns=['S&P500 Returns'] # spy_returns.columns returns the list of the columns' names except the first one 'date' which is considered as an index 
#print(spy_returns)
# plt.figure(figsize=(15,5))
# plt.plot(spy_returns)
# plt.ylabel('Returns')
# plt.title('S&P500 Returns')
# plt.show()
# spy_returns.hist(bins=30)
#print(pd.DataFrame.shape)

# Extraction of the Close values
y_all = spy['Close'].values
    
# Plot of the Close values
plt.figure(figsize=(15,5))
plt.plot(y_all)
plt.ylabel('Close values')
plt.title('S&P500 Close Values')
plt.show()
plt.hist(y_all, bins=30)

# import numpy.random as npr
# n=756
# r=0.05 #taux d'intérêt du marché
# vol=0.11 #volatilité
# T=3
# m=0
# sigma=0.2
# p=np.pi

# dt=T/n
# G=npr.normal(0,1,(n,1))
# log_ret_BS=(r-0.5*vol**2)*dt+vol*np.sqrt(dt)*G
# # plt.hist(log_ret_BS,bins=30)


# import GPy
# nm = 268
# nm2 = 269#504
# x = np.linspace(0,2,nm).reshape((nm,1))
# y_all = spy['Close'].values
# y = y_all[0:nm].reshape((nm,1))
# x_new = np.linspace(0,2,nm2).reshape((nm2,1))
# kern = GPy.kern.Matern32(1,lengthscale= 0.01)
# m = GPy.models.GPRegression(x,y,kern,noise_var=0.001**2)
# # m = GPy.models.GPRegression(x,y,kern,noise_var=0.001**2)
# m.optimize()
# e,f = m.predict(x_new)
# # plt.figure()
# # plt.plot(x,y)
# # plt.plot(x_new,e)
# # plt.figure()
# # plt.hist(y_all,bins=30)
# # plt.figure()
# # plt.hist(e,bins=30)
# # plt.figure()
# # m.plot()
# print(y_all[nm]-e[nm2-1])

# noise variance of the prediction is higher than the noise variance of the actual values (should be exactly the same if we calculate it)
# use Matern
# Matern52 is smoother than Matern32
# use Matern52 for derivative_observations (pb à partir de 2nd derivative of matern52)
# pb à partir de 1st derivative of Matern32 so derivative_observations are impossible
# RBF does over fitting
# Brownian motion not very 'appropriated' because countains already a lot of noise in it

# x = np.array(range(503))+1
# a = np.zeros(504)
# a2 = np.zeros(504)
# a3 = np.zeros(504)
# y_all = spy['Close'].values
# b = np.linspace(0,2,21504)

# for i in x:
#     nm = i
#     nm2 = i+1
#     x = b[0:i].reshape((nm,1))
#     y = y_all[0:nm].reshape((nm,1))
#     x_new = b[0:nm2].reshape((nm2,1))
    
#     # kern = GPy.kern.Matern32(1,lengthscale= 0.01)
#     # m = GPy.models.GPRegression(x,y,kern,noise_var=0.001**2)
#     # m.optimize()
#     # e,f = m.predict(x_new)
    
#     # kern2 = GPy.kern.Matern52(1,lengthscale= 0.01)
#     # m2 = GPy.models.GPRegression(x,y,kern2,noise_var=0.001**2)
#     # m2.optimize()
#     # e2,f = m2.predict(x_new)
    
#     kern3 = GPy.kern.Brownian(1)
#     m3 = GPy.models.GPRegression(x,y,kern3,noise_var=0.001**2)
#     m3.optimize()
#     e3,f = m3.predict(x_new)

#     # a[i]=y_all[nm]-e[nm]
#     # a2[i]=y_all[nm]-e2[nm]
#     a3[i]=y_all[nm]-e3[nm]

# # plt.figure(1)
# # plt.plot(b[5:i],a[5:i])
# # plt.title('Matern32')

# # plt.figure(2)
# # plt.plot(b[5:i],a2[5:i])
# # plt.title('Matern52')

# plt.figure(3)
# plt.plot(b[26:i],a3[26:i])
# plt.title('Brownian Motion')

# # plt.figure(4)
# # plt.hist(a[5:i])
# # plt.title('Matern32')

# # plt.figure(5)
# # plt.hist(a2[5:i])
# # plt.title('Matern52')

# plt.figure(6)
# plt.hist(a3[26:i])
# plt.title('Brownian Motion')





# p = 200 # 504
# h = 1./p
# d = np.zeros(p-1)
# for i in range(p-1):
#     d[i] = (y_all[i]-y_all[i+1])*h

# d = d.reshape((-1,1))
# x = np.array(range(p-1))+1
# a4 = np.zeros(p)
# b = np.linspace(0,2,p)

# for i in x:
#     nm = i
#     nm2 = i+1
#     x = b[0:i].reshape((nm,1))
#     y = y_all[0:nm].reshape((nm,1))
#     x_new = b[0:nm2].reshape((nm2,1))
    
#     kern4 = GPy.kern.Matern52(1,lengthscale= 0.01)
#     kern5 = GPy.kern.DiffKern(kern4,0)
#     gauss = GPy.likelihoods.Gaussian(variance=0.01**2)
#     m4 = GPy.models.MultioutputGP(X_list=[x, x], Y_list=[y, d], kernel_list=[kern4, kern5], likelihood_list = [gauss, gauss])
#     m4.optimize()
#     e4,f = m4.predict(x_new)

#     a4[i]=y_all[nm]-e4[nm]
    
# plt.figure(7)
# plt.plot(b[0:i],a4[0:i])
# plt.title('Matern32: derivative observations')

# plt.figure(8)
# plt.hist(a4[0:i])
# plt.title('Matern32: derivative observations')
