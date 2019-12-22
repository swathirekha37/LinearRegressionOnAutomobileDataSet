# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:23:19 2019

@author: Surface
"""

import numpy 
import pandas as pd
import sklearn
from sklearn.metrics import r2_score

colnames=["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders",'engine-size',"fuel-system",'bore','stroke','compression-ratio','horsepower','peak-rpm',"city-mpg","highway-mpg","price"]
data=pd.read_csv("C:/Users/Surface/Desktop/automobile/imports-85.data", names=colnames)

cleandata=data.loc[(data['normalized-losses']!='?') & (data['num-of-doors']!='?') & (data['bore']!= '?') & (data['stroke']!='?') & (data['horsepower']!='?') & (data['peak-rpm']!='?') & (data['price']!='?')]
#X=cleandata[['normalized-losses','wheel-base','length',"width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg"]]
X=cleandata.iloc[:,18:25]
y=cleandata["fuel-type"]
y[y=="diesel"]=1
y[y=="gas"]=0


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)

#Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)
y_pred=lr.predict(X)
r2_score(y,y_pred)          #We got 95% acuuracy. This model proves that it suits perfectly for furistic data.
y_pred=lr.predict([[3.01,3.40,23.00,106,4800,26,27]])
y_pred.astype(int)
a=y_pred[0].astype(int)
print(" ")
if a==1:
    print("The vehicle possess diesel")
elif a==0:
    print("The vehicle possess gas")


