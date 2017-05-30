# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:11:25 2017

@author: L
"""

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import pickle
import os
from time_series_analysis_p2 import *

class search_arima:
        
    def choose(self,_X,_Y,Nminedge,Nmaxedge,Nstep,Dminedge,Dmaxedge,Dstep):
        model = ARIMA_V1()
        #search
        param_grid = {'n_estimators': [i for i in range(Nminedge,Nmaxedge,Nstep)], 
                                       'max_depth': [j for j in range(Dminedge,Dmaxedge,Dstep)]}
        
        grid_search = GridSearchCV(estimator = model,param_grid = param_grid, scoring='roc_auc',iid=False,cv=5)
        grid_search.fit(_X, _Y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print (para, val)
        return best_parameters["n_estimators"],best_parameters["max_depth"]
    