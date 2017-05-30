# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:49:17 2017

@author: L
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import pickle
import os

class kdd_gbdt:
    def __init__(self,name='kdd gbdt'):
        self.name = name
    
    def build(self):
        self.save_name = 'should the name of saved model'
        self.param = 'paramlist'
        self.model = ''
        self.test_size = 0.2
        self.save_path = r'../model/'
        self.train_times = 10
        
    def choose(self,_X,_Y,Nminedge,Nmaxedge,Nstep,Dminedge,Dmaxedge,Dstep):
        model = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
        #search
        param_grid = {'n_estimators': [i for i in range(Nminedge,Nmaxedge,Nstep)], 
                                       'max_depth': [j for j in range(Dminedge,Dmaxedge,Dstep)]}
        
        grid_search = GridSearchCV(estimator = model,param_grid = param_grid, scoring='roc_auc',iid=False,cv=5)
        grid_search.fit(_X, _Y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print (para, val)
        return best_parameters["n_estimators"],best_parameters["max_depth"]
        
    def train(self,_X,y):        
        model = GradientBoostingRegressor()
        
        for i in range(self.train_times):
            # 随机抽取20%的测试集
            X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=self.test_size)
            model.fit(X_train,y_train)
            
            y_pred = model.predict(X_test)
            y_predprob = model.predict_proba(X_test)[:,1]
            print ("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred))
            print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))
        self.model = model
        return y_pred
    
    def predict(self,_x,load=False):
        if load:
            print('load:',self.save_param)
            self.model = load(self.save_param)
        else:
            print(self.model)
        return self.model.predict(_x)
    
    
    def save_model(self):
        if os.path.exists(self.save_path):
            f = open(''.join([self.save_path,self.save_name,'.pkl']),'wb')
            pickle.dump(self.model,f)
            f.close()
        else:
            print('path does not exist')
            
    def load_model(self):
        if os.path.exists(self.save_path):
            f = open(''.join([self.save_path,self.save_name,'.pkl']),'rb')
            self.model = pickle.load(f)
            f.close()
        else:
            print('path doesnot exist')
            sys.exit()