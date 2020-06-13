import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import joblib

class mainFunc():
    def __init__(self):
        #self.proc_LoadModel()
        
        self.proc_LoadData()
        self.proc_TreatData()
        
        self.proc_LinearTraining()
        
        #self.proc_DisplayData()         #multivariate regressions are kind of hard to graph
        
        self.proc_TestData()
        
        #self.proc_SaveModel()
        
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/homeprices.csv")
        print(self.dataBin)
        print()
        
    def proc_TreatData(self):
        dummyVars = pd.get_dummies(self.dataBin['town'])
        self.dataBin = pd.concat([self.dataBin,dummyVars], axis='columns')
        self.dataBin = self.dataBin.drop(['town','west windsor'], axis='columns')
        
        print(self.dataBin)
        print()
        
    def proc_LinearTraining(self):
        X = self.dataBin[['area','monroe township','robinsville']]
        y = self.dataBin.price
        
        self.reg = linear_model.LinearRegression()
        self.reg.fit(X, y)
        
        print('coef: ', self.reg.coef_)
        print('intercep: ', self.reg.intercept_)
        print()
        
    
    def proc_DisplayData(self):

        plt.xlabel("year")
        plt.ylabel("per capita income (US$)")
        plt.show()
        
    def proc_TestData(self):

        vx = 2800
        vy = 0
        vz = 1
        
        print("Area: ", vx, "LocA: ", vy, "LocB: ", vz)
        print("Price: ", self.reg.predict([[vx,vy,vz]]))
        
        print()

        
    def proc_SaveModel(self):
        joblib.dump(self.reg, "models/trainedModel_02.joblib")
    
    def proc_LoadModel(self):
        self.reg = joblib.load("models/trainedModel_02.joblib")
    
mainFunc()