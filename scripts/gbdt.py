# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:49:17 2017

@author: L
http://www.cnblogs.com/en-heng/p/6907839.html
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
import sklearn

import pickle
import os

class kdd_gbdt:
    def __init__(self,name='kdd gbdt'):
        self.name = name
        self.save_name = 'should the name of saved model'
        self.params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        self.model = ''
        self.test_size = 0.2
        self.save_path = r'../model/'
        self.train_times = 10
        
    def choose(self,_X,_Y):
        model = GradientBoostingRegressor(n_estimators=500)
        param_grid = {'learning_rate': [0.1],
                      'max_depth': [10],
                      'min_samples_leaf': [3],
                      # 'max_features': [1.0, 0.3, 0.1] ## not         possible in our example (only 1 fx)
          }
        #search
#        param_grid = {'n_estimators': [i for i in range(Nminedge,Nmaxedge,Nstep)], 
#                                       'max_depth': [j for j in range(Dminedge,Dmaxedge,Dstep)]}
        print('begin search')
        grid_search = GridSearchCV(estimator = model,param_grid = param_grid,n_jobs=4)
        print('fit')
        grid_search.fit(_X, _Y)
        print('done')
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print (para, val)
        return best_parameters
    
    def select(self,_X,_Y,X_test ,y_test):
        select = SelectKBest(k='all')
        model = GradientBoostingRegressor()
        steps = [('feature_selection', select),
        ('random_forest', model)]
        pipeline = Pipeline(steps)
        pipeline.fit(_X, _Y )
        ### call pipeline.predict() on your X_test data to make a set of test predictions
        y_prediction = pipeline.predict( X_test )
#        print(y_prediction)
        ### test your predictions using sklearn.classification_report()
        report = mean_squared_error( y_test, y_prediction )
        ### and print the report
        print("MSE: %.4f" % report)
        print(model)
        
        
    def train(self,_X,y): 
        '''
        input Series
        '''
        model = GradientBoostingRegressor(**self.params) #注意这里放置参数的方式
        # 随机抽取20%的测试集
        X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2)
        model.fit(X_train,y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE: %.4f" % mse)
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
            
    def plot(self,Xtest,y_test,y_pred):
        ###############################################################################
        # Plot training deviance
        
        # compute test set deviance
        test_score = np.zeros((self.params['n_estimators'],), dtype=np.float64)
        
        for i, y_pred in enumerate(self.model.staged_predict(X_test)):
            test_score[i] = self.model.loss_(y_test, y_pred)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Deviance')
        plt.plot(np.arange(self.params['n_estimators']) + 1, self.model.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(self.params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        
        ###############################################################################
        # Plot feature importance
        feature_importance = self.model.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, Xtest.columns[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()